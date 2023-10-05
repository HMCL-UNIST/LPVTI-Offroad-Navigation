#!/usr/bin/env python3
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2023 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>, Sanghun Lee <sanghun17@unist.ac.kr>
  @date: September 10, 2023
  @copyright 2023 Ulsan National Institute of Science and Technology (UNIST)
  @brief: Torch version of util functions
"""

from re import L
import rospy
import time
import threading
import os
import numpy as np
import torch

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Joy
from visualization_msgs.msg import MarkerArray

from nav_msgs.msg import Odometry
from hmcl_msgs.msg import vehicleCmd
from sensor_msgs.msg import Image, CameraInfo
from pvti_offroad.common.pytypes import AUCModelData, VehicleCommand, VehicleState, SimData
from pvti_offroad.common.file_utils import *

from collections import deque
from dynamic_reconfigure.server import Server
from pvti_offroad.cfg import predictorDynConfig
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import tf 
from pvti_offroad.common.utils_batch import predicted_trajs_visualize
from pvti_offroad.common.utils import gen_random_model_state, preprocess_image_depth
from pvti_offroad.map.gpgridmap_batch import GPGridMap
from pvti_offroad.simulation.vehicle_model import VehicleModel
from gazebo_msgs.msg import ModelState, ModelStates
from copy import deepcopy

class DataLogger:
    def __init__(self):     
        
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)                           
        self.dt = self.t_horizon / self.n_nodes*1.0        
        
        self.local_map = GPGridMap(dt = self.dt)
        self.vehicle_model = VehicleModel(dt = self.dt, N_node = self.n_nodes, map_info = self.local_map)
        self.cur_state = VehicleState()
        self.u_buffer = deque() 
        self.cur_vx_buffer = deque()
        
        for i in range(self.n_nodes+1):
            u_tmp = VehicleCommand()
            u_tmp.ax = 0.0
            u_tmp.steer = 0.0
            self.u_buffer.append(u_tmp)
        
        
        self.obj_poses = []
        self.tree_respwan = []
        self.input_select_idx = 0
        '''
        ############### Vehicle specific ############### 
        '''
        self.init_odom = None
        self.vehicle_state = VehicleState()
        self.ax_max = 1.5
        self.delta_max =  25 * np.pi /180.0 # 0.25 # radian

        self.ax_cmd = 0.0
        self.delta_cmd = 0.0

        ####################### Vehicle Specific End ############### 
        self.map_available = False
        self.cur_odom = Odometry()
        self.cur_depth_msg = None
        self.cur_color_msg  = None
        self.joy_msg = None
        
        '''
        ###############  Dataset related variables ############### 
        '''        
        self.auc_dataset = []
        self.data_save = False
        self.save_buffer_length = 500
        self.cum_data_save_length = 0
        ############### Dataset related variables End ########################
        

        '''
        ###############  Camera related variables ############### 
        '''                
        listener = tf.TransformListener()
        listener.setUsingDedicatedThread(True)  # Use dedicated thread for the listener

        camera_info_topic = "/camera_name/color/camera_info"  # Replace with your actual camera_info topic
        self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        
        self.bridge = CvBridge()
        print("Camera Init Done")
        ############### Camera related variables End ########################

        
        self.ini_pub_and_sub()
        self.init_timer_callback()     

        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()
    


        
    ''' 
        ###############  ROS pub and sub ###############  
    '''
    def ini_pub_and_sub(self):       
        self.sim_model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=2)    
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)
  
        # pub
        self.cmd_pub = rospy.Publisher('/acc_cmd',vehicleCmd, queue_size = 1)
        
        
        self.nominal_predicted_trj_publisher = rospy.Publisher("/nominal_pred_trajectory", MarkerArray, queue_size=2)    

        
        
        self.obj_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.obj_callback, queue_size= 10)


        self.node_hz_pub = rospy.Publisher('data_logging_hz', Header, queue_size=1)
        self.node_hz = Header()        
        self.node_hz.stamp= rospy.Time.now()
        self.node_hz.seq = 0        
        # Subscribers     
        self.local_map_sub = rospy.Subscriber('/traversability_estimation/global_map', GridMap, self.gridmap_callback)
        self.odom_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.odom_callback)

        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
             
        
        depth_camera_sub = Subscriber('/camera_name/depth/image_raw', Image)
        color_camera_sub = Subscriber('/camera_name/color/image_raw', Image)        
        self.ts = ApproximateTimeSynchronizer(
        [depth_camera_sub, color_camera_sub],
        queue_size=10, slop=0.2, allow_headerless=True
        )
        self.ts.registerCallback(self.msg_filter_callback)        
        ############### ROS pub and sub  End ########################

    def odom_callback(self,msg):
        if self.init_odom is None:
            self.init_odom = deepcopy(msg)

        self.cur_odom = msg
        
    def init_timer_callback(self):
        ## controller callback
        self.cmd_hz = 5
        self.cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.cmd_callback)         

    def joy_callback(self,msg):
        self.joy_msg = msg
        
    def gridmap_callback(self,msg):                     
       
        if self.map_available is False:
            self.map_available = True        
        
        
        self.local_map.set_map(msg)        

    def obj_callback(self,msg):              
        if len(self.obj_poses) > 0:  
            return             
            
        for i in range(len(msg.name)):            
            if 'tree' in msg.name[i]:            
                obj_pose = np.array([msg.pose[i].position.x,msg.pose[i].position.y])                
                self.obj_poses.append(obj_pose)
        if len(self.obj_poses) > 0:
            self.obj_poses = np.stack(self.obj_poses)            
    
    def check_vehicle_within_map(self):
        
        self.reset_simulation()

    def set_tree_pose(self,pose):
        new_model = ModelState()
        new_model.model_name = 'oak_tree'        
        new_model.pose.position.x = pose[0]
        new_model.pose.position.y = pose[1]
        new_model.pose.position.z = 0.0
        new_model.pose.orientation.w = 1
        new_model.pose.orientation.x = 0
        new_model.pose.orientation.y = 0
        new_model.pose.orientation.z = 0
        self.sim_model_state_pub.publish(new_model)        
        self.tree_respwan = False

    
    
    
    def is_near_obs(self, model: ModelState):
        if len(self.obj_poses) < 1:
            return False
        # ifself.obj_poses 
        dist_thres = 3.0
        model_pose = np.array([model.pose.position.x,model.pose.position.y]).transpose()
        distances = np.linalg.norm(self.obj_poses - model_pose, axis=1)
        close_objects = np.where(distances < dist_thres)[0]
        if close_objects.size > 0:
            return True
        
        return False

    def reset_simulation(self):        
        near_obs = True        
        loop_count = 100
        while near_obs and loop_count > 0:            
            pose = None
            if self.reset_to_init: 
                pose = self.init_odom
            new_model = gen_random_model_state(pose)            
            near_obs = self.is_near_obs(new_model)               
            loop_count-=1
        self.sim_model_state_pub.publish(new_model)        
        self.clear_buffer()
        return

    def dyn_callback(self,config,level):        
        self.input_select_idx = config.input_select_idx
        self.sim_reset = config.sim_reset
        self.reset_to_init = config.reset_to_init
        if self.sim_reset:
            self.save_buffer_in_thread()             
            self.reset_simulation()            
            
        self.tree_respwan = config.tree_respwan
        if self.tree_respwan is False:
            faraway_pose_ = [-100,100]
            self.set_tree_pose(faraway_pose_)
        self.data_save = config.logging_vehicle_states
        self.save_now = config.save_now
        self.ax_cmd = config.ax
        self.delta_cmd = config.delta
        if self.save_now and self.data_save:
            self.save_now = False
            self.save_buffer_in_thread()

        if config.clear_buffer:
            self.clear_buffer()        
        print("dyn reconfigured")
        
        return config
    
    def update_node_hz(self):
        self.node_hz.stamp= rospy.Time.now()
        self.node_hz.seq +=np.random.randint(5)+1
        if self.node_hz.seq > 1000:
            self.node_hz.seq = 0
        self.node_hz_pub.publish(self.node_hz)
                         
    def datalogging(self,cur_data):                   
        self.auc_dataset.append(cur_data.copy())     
        self.cum_data_save_length+=1              
        if len(self.auc_dataset) > self.save_buffer_length:
            self.save_buffer_in_thread()
      
    def save_buffer_in_thread(self):
        # Create a new thread to run the save_buffer function
        t = threading.Thread(target=self.save_buffer)
        t.start()
    
    def clear_buffer(self):
        if len(self.auc_dataset) > 0:
            self.auc_dataset.clear()            
        rospy.loginfo("states buffer has been cleaned")

    def save_buffer(self):        
        if len(self.auc_dataset) ==0:
            return
        self.data_save = False
        real_data = SimData(len(self.auc_dataset.copy()), self.auc_dataset.copy(), None)        
        create_dir(path=train_dir)        
        pickle_write(real_data, os.path.join(train_dir, str(self.vehicle_state.header.stamp.to_sec()) + '_'+ str(len(self.auc_dataset))+'.pkl'))
        rospy.loginfo("states data saved")        
        self.clear_buffer()

    


    def msg_filter_callback(self,depth_msg,color_msg):
        self.cur_depth_msg = depth_msg
        self.cur_color_msg = color_msg
        if self.local_map.is_within_map(self.cur_odom) is False:
            self.reset_simulation()
        
        

    def check_if_vehicle_is_stop(self,state:VehicleState):
        is_stop = False
        stop_check_duration =4.0
        self.cur_vx_buffer.append(self.cur_state.local_twist.linear.x)            
        if len(self.cur_vx_buffer) > stop_check_duration/self.dt:                
            vx_data = [self.cur_vx_buffer[i] for i in range(len(self.cur_vx_buffer))]
            vx_mean = np.mean(vx_data)
            if vx_mean < 0.1:
                is_stop = True                
            self.cur_vx_buffer.popleft()
        if is_stop:
            self.cur_vx_buffer.clear()
        return is_stop
              
          
    def cmd_callback(self,event):
       
        start_time = time.time()                        
        if self.cur_odom is None or self.cur_depth_msg is None or self.cur_color_msg is None:
            return
        tmp_odom = deepcopy(self.cur_odom)

        if len(self.auc_dataset) > 0:
            prev_time = self.auc_dataset[-1].header.stamp.to_sec()
            time_diff = tmp_odom.header.stamp.to_sec() - prev_time                            
            print(f"Od  ometry time_diff in Seconds: {time_diff:.5f}")                     
            
            
        self.node_hz_pub.publish(self.node_hz)

        num_old_data = self.n_nodes-1
        old_data = [self.u_buffer[i] for i in range(1, num_old_data + 1)]
        
        self.pred_u = old_data.copy()        
        self.cur_u = self.pred_u[0].copy()
        
        self.cur_state = VehicleState()
        self.cur_state.update_from_msg(tmp_odom, self.cur_u)        
        self.vehicle_state = self.cur_state
        

        batch_u = torch.stack([torch.tensor([u.ax, u.steer]) for u in self.pred_u])
        batch_u = batch_u.unsqueeze(dim=0)
                    
      
        batch_xhat, batch_u , pred_states = self.vehicle_model.dynamics_predict(self.cur_state, batch_u)
        
        
        cur_data = AUCModelData()
        
        
        depth_img, color_img, concat_image = preprocess_image_depth(self.bridge,self.cur_depth_msg, self.cur_color_msg)                
        
        cur_data.update_from_pred_u(tmp_odom,pred_states.clone(), batch_u.clone(), depth_img.clone(),color_img.clone())
        cur_data.update_with_xhat_and_img(tmp_odom, batch_xhat, concat_image)

        nominal_path_color = [0,1,0,1]            
        nominal_pred_traj_marker = predicted_trajs_visualize(pred_states,nominal_path_color)        
        self.nominal_predicted_trj_publisher.publish(nominal_pred_traj_marker) 
        

        if self.data_save:            
            self.datalogging(cur_data.copy())
        print("buffer filled length = " + str(len(self.auc_dataset)))
        for i in range(int(1/self.cmd_hz / self.dt)+1):            
            self.u_buffer.popleft()
        
        if len(self.u_buffer) < self.n_nodes+2:
            self.gen_input_sequence()      
             
        cmd_msg = vehicleCmd()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.acceleration =  self.pred_u[0].ax
        cmd_msg.steering =  self.pred_u[0].steer
        self.cmd_pub.publish(cmd_msg)        

        end_time = time.time()
       
        return 
    
    
    def gen_input_sequence(self):        
        u_tmp = VehicleCommand()        
     
        action_samples = self.vehicle_model.action_sampling(self.cur_state)
        rand_idx = self.input_select_idx 
        
        action_sample = action_samples[rand_idx,:,:].cpu().numpy()       
        if action_sample[0,0] < 0.0:
            for i in range(action_sample.shape[0]):        
                u_tmp.ax = action_sample[i,0]
                u_tmp.steer = action_sample[i,1]                
                self.u_buffer.append(u_tmp)   
        else:
            for j in range(1):     
                for i in range(action_sample.shape[0]):        
                    u_tmp.ax = action_sample[i,0]
                    u_tmp.steer = action_sample[i,1]                
                    self.u_buffer.append(u_tmp)   
        
            


def main():
    rospy.init_node("data_logger")    
    DataLogger()

if __name__ == "__main__":
    main()




 
    


