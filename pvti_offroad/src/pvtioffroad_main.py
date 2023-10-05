#!/usr/bin/env python
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
import numpy as np
import math 
import torch
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from hmcl_msgs.msg import Waypoints, vehicleCmd
from visualization_msgs.msg import MarkerArray
from autorally_msgs.msg import chassisState
from sensor_msgs.msg import Joy
from grid_map_msgs.msg import GridMap
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from pvti_offroad.simulation.vehicle_model import VehicleModel
from pvti_offroad.common.utils_batch import get_variance_ellipoid_pred_traj,  predicted_trajs_visualize
from pvti_offroad.common.utils import gen_random_model_state, preprocess_image_depth
from pvti_offroad.map.gpgridmap_batch import GPGridMap
from dynamic_reconfigure.server import Server
from pvti_offroad.cfg import predictorDynConfig
from pvti_offroad.common.pytypes import VehicleState
from pvti_offroad.common.file_utils import *
from sensor_msgs.msg import Image
from pvti_offroad.AUC_estimate import AUCEStimator
from collections import deque
from gazebo_msgs.msg import ModelState
from copy import deepcopy

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class AUCPlanner:
    def __init__(self):        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_hz = rospy.get_param('~prediction_hz', default=5.0)
        self.ctrl_hz = rospy.get_param('~ctrl_hz', default=10.0)
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)                   
        self.input_random = rospy.get_param('~input_random', default=False)                              
        self.dt = self.t_horizon / self.n_nodes*1.0        
         # x, y, psi, vx, vy, wz, z ,  roll, pitch 
         # 0  1  2     3  4   5   6    7, 8   
        self.bridge = CvBridge()
        self.cur_state = VehicleState()
        self.local_map = GPGridMap(dt = self.dt)
        self.auc_model = AUCEStimator(dt = self.dt, N_node = self.n_nodes, model_path='singl_aucgp_snapshot.pth')
        
        self.vehicle_model = VehicleModel(dt = self.dt, N_node = self.n_nodes, map_info = self.local_map)        
        
        self.init_odom = None
        self.odom_msg = None
        self.depth_msg = None
        self.color_msg = None                
        self.dyn_srv = Server(predictorDynConfig, self.dyn_callback)   
        self.odom_available   = False 
        
        self.map_available = False
        self.joy_msg = None
        # Thread for optimization
        self.vehicleCmd = vehicleCmd()
        self._thread = threading.Thread()        
        self.map_thread_lock = threading.Lock()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        
        status_topic = "/is_data_busy"        
        var_pred_traj_topic_name = "/var_pred_trajectory"
        nominal_pred_traj_topic_name = "/nominal_pred_trajectory" 
        mean_pred_traj_topic_name = "/gpmean_pred_trajectory"         
        best_pred_traj_topic_name = "/best_gplogger_pred_trajectory" 
        local_traj_topic_name = "/local_traj"        
        # Publishers        
        self.cmd_pub = rospy.Publisher('/acc_cmd',vehicleCmd, queue_size = 1)
        self.local_traj_pub = rospy.Publisher(local_traj_topic_name, Waypoints, queue_size=2)        
        self.var_predicted_trj_publisher = rospy.Publisher(var_pred_traj_topic_name, MarkerArray, queue_size=2)        
        self.mean_predicted_trj_publisher    = rospy.Publisher(mean_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.nominal_predicted_trj_publisher = rospy.Publisher(nominal_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.best_trj_publisher = rospy.Publisher(best_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        self.nominal_predicted_trj_publisher = rospy.Publisher(nominal_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.sim_model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=2)    
        
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        self.local_map_sub = rospy.Subscriber('/traversability_estimation/global_map', GridMap, self.gridmap_callback)               
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)               
        
        odom_sub = Subscriber('/ground_truth/state', Odometry)        
        depth_camera_sub = Subscriber('/camera_name/depth/image_raw', Image)
        color_camera_sub = Subscriber('/camera_name/color/image_raw', Image)                
        self.ts = ApproximateTimeSynchronizer(
        [odom_sub, depth_camera_sub, color_camera_sub],
        queue_size=10, slop=0.2, allow_headerless=True
        )
        self.ts.registerCallback(self.msg_filter_callback)     
        
        self.u_buffer = deque() 
        
        self.ctrl_timer = rospy.Timer(rospy.Duration(1/self.ctrl_hz), self.cmd_timer)         
        self.planner_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.planner_timer)     
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()
    


    def joy_callback(self,msg):
        self.joy_msg = msg

    
    def post_process_paths(self,best_path_idx, pred_states, pred_residual_mean):
            norminal_states = pred_states[:,:,:].clone()        
            pred_mean_states =pred_states[:,:,:].clone()        
            # states has ->  [x, y, psi, vx, vy, wz, z, roll, pitch]
            # residual has -> [x, y, vx, vy , wz] 
            pred_mean_states[:,1:,:2] += pred_residual_mean[:,:,:2]
            pred_mean_states[:,1:,3] += pred_residual_mean[:,:,2]
            pred_mean_states[:,1:,4] += pred_residual_mean[:,:,3]
            pred_mean_states[:,1:,5] += pred_residual_mean[:,:,4]
            best_states =  pred_states[best_path_idx,:,:].clone().unsqueeze(dim=0)
            return norminal_states.cpu().numpy(), pred_mean_states.cpu().numpy(), best_states.cpu().numpy()
    
    def msg_filter_callback(self, odom_msg: Odometry, depth_msg: Image,color_msg: Image):
        if self.init_odom is None:
            self.init_odom = deepcopy(odom_msg)        

        self.odom_msg = odom_msg
        self.depth_msg= depth_msg
        self.color_msg = color_msg
    
    
    def planner_timer(self,event):
        if self.odom_msg is None or self.depth_msg is None or self.color_msg is None:
            return
        odom_msg = self.odom_msg
        depth_msg = self.depth_msg
        color_msg = self.color_msg
        
        if self.map_available is False:
            rospy.loginfo("Map is not available yet")
            return
        start_time = time.time()

        self.cur_state.update_odom(odom_msg)        
        batch_u = self.vehicle_model.action_sampling(self.cur_state)
        batch_xhat, batch_u , pred_states= self.vehicle_model.dynamics_predict(self.cur_state, batch_u)
        #################################        
        _, _, image = preprocess_image_depth(self.bridge, depth_msg, color_msg)                                
        # residual -> delta vector of [x, y, vx, vy , wz] , x and y in global and vx vy wz in local 
        pred_residual_mean, pred_residual_std = self.auc_model.pred(batch_xhat[:,:self.n_nodes-1,:], image)
        best_path_idx, best_path, total_trav_costs, pred_pose_mean_seq = self.local_map.compute_best_path(pred_states.clone(), pred_residual_mean.clone(), pred_residual_std.clone(), self.goal_pose)
        # states has ->  [x, y, psi, vx, vy, wz, z, roll, pitch]
        # residual has -> [x, y, vx, vy , wz] 
        norminal_states, pred_mean_states, best_states = self.post_process_paths(best_path_idx, pred_states, pred_residual_mean)
        ############################        
        self.path_visualization(norminal_states, pred_mean_states,best_states, pred_residual_std.cpu().numpy())
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        if self.goal_pose is None:
            return        
        dist_to_goal =  np.sqrt((odom_msg.pose.pose.position.x - self.goal_pose[0])**2 + (odom_msg.pose.pose.position.y - self.goal_pose[1])**2)
        if len(self.u_buffer) ==0:            
            for i in range(2):
                cmd_msg = vehicleCmd()
                cmd_msg.header.stamp = rospy.Time.now()                
                cmd_msg.acceleration =  batch_u[best_path_idx,i,0]
                if self.joy_msg is None:
                    return 
                if dist_to_goal < 1.5 or self.joy_msg.buttons[4] ==0:
                    cmd_msg.acceleration = -3.0
                cmd_msg.steering =  batch_u[best_path_idx,i,1]                
                self.u_buffer.append(cmd_msg)
              
    def path_visualization(self, norminal_path, pred_pose_mean_seq,best_path,pred_residual_std):
        nominal_path_color = [0,1,0,0.3]            
        gpmean_path_color = [0,0,1,0.3]        
        var_elips_color = [0,0.5,1,0.1]        
        best_path_color = [1,0,0,1.0]        
        nominal_pred_traj_marker = predicted_trajs_visualize(norminal_path,nominal_path_color)        
        mean_pred_traj_marker = predicted_trajs_visualize(pred_pose_mean_seq,gpmean_path_color)        
        var_elips_color = get_variance_ellipoid_pred_traj(pred_pose_mean_seq, pred_residual_std,var_elips_color)          
        best_traj_marker = predicted_trajs_visualize(best_path,best_path_color)
        
        self.nominal_predicted_trj_publisher.publish(nominal_pred_traj_marker) 
        self.mean_predicted_trj_publisher.publish(mean_pred_traj_marker)    
        self.var_predicted_trj_publisher.publish(var_elips_color)
        self.best_trj_publisher.publish(best_traj_marker)    

    def reset_simulation(self):        
        pose = None
        if self.reset_to_init: 
            pose = self.init_odom
        new_model = gen_random_model_state(pose)                                    
        self.sim_model_state_pub.publish(new_model)                
        return
    
    def dyn_callback(self,config,level):   
        self.sim_reset = config.sim_reset
        self.reset_to_init = config.reset_to_init
        if self.sim_reset:
            self.reset_simulation()            

        scale_data = {"dist_heuristic_cost_scale":  config.dist_heuristic_cost_scale,                         
                        "rollover_cost_scale": config.rollover_cost_scale,
                        "model_error_weight": config.model_error_weight,                         
                        "local_map_cost_weight": config.local_map_cost_weight,
                        "error_std_scale": config.error_std_scale,
                        "error_mean_scale": config.error_mean_scale}        
        
        self.local_map.set_scales(scale_data)
        print("weight rescaled")
        return config

    def goal_callback(self,msg):        
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]
        
    def gridmap_callback(self,msg):                     
        if self.map_available is False:
            self.map_available = True
        with self.map_thread_lock:
            self.local_map.set_map(msg)
                
    
    def cmd_timer(self,timer):                
        if len(self.u_buffer) > 0:
            cmd_msg = self.u_buffer.popleft()
            self.cmd_pub.publish(cmd_msg)                   
        return
    

###################################################################################
def main():
    rospy.init_node("pvti_planner")    
    AUCPlanner()
if __name__ == "__main__":
    main()
