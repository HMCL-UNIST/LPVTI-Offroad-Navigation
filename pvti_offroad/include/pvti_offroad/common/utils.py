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



import math
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
import torch
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import tf
import cv2
from gazebo_msgs.msg import ModelState, ModelStates

def gen_random_model_state(odom = None):        

    new_model = ModelState()
    new_model.model_name = 'Alpha'        
    map_x_max = 15.0
    map_y_max = 15.0
    new_pose = np.random.rand(3,1) 
    new_model.pose.position.x = new_pose[0] *map_x_max*2- map_x_max 
    new_model.pose.position.y = new_pose[1]*map_y_max*2- map_y_max
    new_model.pose.position.z = 0.5                  
    quat = euler_to_quaternion(0, 0, new_pose[2])
    new_model.pose.orientation.w = quat[0]
    new_model.pose.orientation.x = quat[1]
    new_model.pose.orientation.y = quat[2]
    new_model.pose.orientation.z = quat[3]
    if odom is not None:
        new_model.pose = odom.pose.pose             

    return new_model 


def normalize_depth(depth):    
    depth = np.clip(depth,0.0, 16.0) 
    norm_depth = depth/16.0
    
    return  norm_depth

def normalize_color(img):    
    norm_color = img/255    
    return norm_color


def unnormalize_color(img):    
    norm_color = img*255
    return norm_color


def preprocess_image_depth(bridge,depth_msg, color_msg):

    depth_image = np.copy(bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
    replacement_distance = 16.0  # if nan -> maximum distance as 16
    # Find NaN values in the image
    depth_nan_indices = np.isnan(depth_image)
    # Replace NaN values with the replacement distance
    if len(depth_image[depth_nan_indices]) > 0:
        depth_image[depth_nan_indices] = replacement_distance

    color_image = np.copy(bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough"))
    replacement_color = [0, 0, 0]  # if non -> White
    # Find NaN values in the image
    nan_indices = np.isnan(color_image)
    # Replace NaN values with the replacement color
    if len(color_image[nan_indices]) > 0:
        color_image[nan_indices] = replacement_color
    depth_image = normalize_depth(torch.tensor(depth_image)).unsqueeze(dim=2)
    color_image = normalize_color(torch.tensor(color_image))
    image = torch.cat((color_image,depth_image), dim=2)    
    return depth_image, color_image, image.permute(2,0,1)


def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)

    
def b_to_g_rot(r,p,y): 
    row1 = torch.transpose(torch.stack([torch.cos(p)*torch.cos(y), -1*torch.cos(p)*torch.sin(y), torch.sin(p)]),0,1)
    row2 = torch.transpose(torch.stack([torch.cos(r)*torch.sin(y)+torch.cos(y)*torch.sin(r)*torch.sin(p), torch.cos(r)*torch.cos(y)-torch.sin(r)*torch.sin(p)*torch.sin(y), -torch.cos(p)*torch.sin(r)]),0,1)
    row3 = torch.transpose(torch.stack([torch.sin(r)*torch.sin(y)-torch.cos(r)*torch.cos(y)*torch.sin(p), torch.cos(y)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(r)*torch.cos(p)]),0,1)
    rot = torch.stack([row1,row2,row3],dim = 1)
    return rot

def np_b_to_g_rot(r,p,y): 
    row1 = np.array([np.cos(p)*np.cos(y), -1*np.cos(p)*np.sin(y), np.sin(p)])
    row2 = np.array([np.cos(r)*np.sin(y)+np.cos(y)*np.sin(r)*np.sin(p), np.cos(r)*np.cos(y)-np.sin(r)*np.sin(p)*np.sin(y), -np.cos(p)*np.sin(r)])
    row3 = np.array([np.sin(r)*np.sin(y)-np.cos(r)*np.cos(y)*np.sin(p), np.cos(y)*np.sin(r)+np.cos(r)*np.sin(p)*np.sin(y), np.cos(r)*np.cos(p)])
    rot = np.stack([row1,row2,row3])
    return rot


def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def wrap_to_pi_torch(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return (((angle + torch.pi) % (2 * torch.pi)) - torch.pi)
    
    



def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    # if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
    q_norm = np.sqrt(np.sum(q ** 2))
    # else:
    #     q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    
    return unit_quat(np.array([qw, qx, qy, qz]))



def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = False):
    local_vel = np.array([0.0, 0.0, 0.0])
    local_ang_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        global_ang_val = np.array([odom.twist.twist.angular.x,odom.twist.twist.angular.y,odom.twist.twist.angular.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
        local_ang_vel = inv_rot_mat_.dot(global_ang_val)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
        local_ang_vel[0] = odom.twist.twist.angular.x
        local_ang_vel[1] = odom.twist.twist.angular.y
        local_ang_vel[2] = odom.twist.twist.angular.z

    return local_vel, local_ang_vel


def traj_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        quat_tmp = euler_to_quaternion(0.0, 0.0, traj[i,3])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 255, 0)
        marker_ref.color.a = 0.2
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.2, 0.2, 0.15)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs

def predicted_distribution_traj_visualize(x_mean,x_var,y_mean,y_var,mean_predicted_state,color):
    marker_refs = MarkerArray() 
    for i in range(len(x_mean)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = x_mean[i]
        marker_ref.pose.position.y = y_mean[i]
        marker_ref.pose.position.z = mean_predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state[i,2])             
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5        
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        marker_ref.scale.x = 2*np.sqrt(x_var[i])
        marker_ref.scale.y = 2*np.sqrt(y_var[i])
        marker_ref.scale.z = 1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def predicted_trj_visualize(predicted_state,color):        
    marker_refs = MarkerArray() 
    for i in range(len(predicted_state[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predicted_state[i,0] 
        marker_ref.pose.position.y = predicted_state[i,1]              
        marker_ref.pose.position.z = predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5                
        marker_ref.scale.x = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.y = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.z = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def ref_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states_"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 0, 255)
        marker_ref.color.a = 0.5
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.1, 0.1, 0.1)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs



def multi_predicted_distribution_traj_visualize(x_mean_set,x_var_set,y_mean_set,y_var_set,mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(len(x_mean_set)):
        for i in range(len(x_mean_set[j])):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "gplogger_ref"+str(i)+str(j)
            marker_ref.id = j*len(x_mean_set[j])+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = x_mean_set[j][i]
            marker_ref.pose.position.y =  y_mean_set[j][i]
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])             
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (246, 229, 100 + 155/(len(x_mean_set)+0.01)*j)
            marker_ref.color.a = 0.5        
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.1 #2*np.sqrt(x_var_set[j][i])
            marker_ref.scale.y = 0.1 #2*np.sqrt(y_var_set[j][i])
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
    return marker_refs


def mean_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 0
            marker_ref.color.g = 255
            marker_ref.color.b = 0 
            marker_ref.color.a = 0.5    
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.1
            marker_ref.scale.y = 0.1
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs



def nominal_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 255
            marker_ref.color.g = 0
            marker_ref.color.b = 0 
            marker_ref.color.a = 1.0    
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.05
            marker_ref.scale.y = 0.05
            marker_ref.scale.z = 0.05
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs

def dist3d(point1, point2):
    """
    Euclidean distance between two points 3D
    :param point1:
    :param point2:
    :return:
    """
    x1, y1, z1 = point1[0:3]
    x2, y2, z2 = point2[0:3]

    dist3d = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    return math.sqrt(dist3d)

def gaussianKN2D(rl=4,cl=5, rsig=1.,csig=2.):
    """
    creates gaussian kernel with side length `rl,cl` and a sigma of `rsig,csig`
    """
    rx = np.linspace(-(rl - 1) / 2., (rl - 1) / 2., rl)
    cx = np.linspace(-(cl - 1) / 2., (cl - 1) / 2., cl)
    gauss_x = np.exp(-0.5 * np.square(rx) / np.square(rsig))
    gauss_y = np.exp(-0.5 * np.square(cx) / np.square(csig))
    kernel = np.outer(gauss_x, gauss_y)
    return kernel / (np.sum(kernel)+1e-8)



def torch_path_to_marker(path):
    path_numpy = path.cpu().numpy()
    marker_refs = MarkerArray() 
    marker_ref = Marker()
    marker_ref.header.frame_id = "map"  
    marker_ref.ns = "mppi_ref"
    marker_ref.id = 0
    marker_ref.type = Marker.LINE_STRIP
    marker_ref.action = Marker.ADD     
    marker_ref.scale.x = 0.1 
    for i in range(len(path_numpy[0,:])):                
        point_msg = Point()
        point_msg.x = path_numpy[0,i] 
        point_msg.y = path_numpy[1,i]              
        point_msg.z = path_numpy[3,i] 
        
        color_msg = ColorRGBA()
        color_msg.r = 0.0
        color_msg.g = 0.0
        color_msg.b = 1.0
        color_msg.a = 1.0
        marker_ref.points.append(point_msg)
        marker_ref.colors.append(color_msg)    
    marker_refs.markers.append(marker_ref)
    return marker_refs


def get_rect_vertext(dist,angle):
    p1 = np.array([[1,1], [-1,1], [-1,-1], [1,-1]])    
    p1 = p1*dist    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])    
    rotated_vector = np.dot(rotation_matrix, np.transpose(p1))
    return np.transpose(rotated_vector)


def convert_to_local_frame(target_position, odom_msg, cam_info,  patch_in_meter = 1.0):
    
    fx = cam_info.fx
    fy = cam_info.fy
    cx = cam_info.cx
    cy = cam_info.cy    
    assert len(cam_info.distortion) == 5, "assuming radial distortion but got something else"
    distortion_coeffs = cam_info.distortion  # Assuming radial distortion
    
    
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])    
    
    trans = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
    rot = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]        
    robot_to_world_matrix =  tf.TransformerROS().fromTranslationRotation(trans, rot)    
    world_to_robot_matrix = np.linalg.inv(robot_to_world_matrix)    
    # Transform from robot base to camera local frame

    robot_to_camera_matrix = cam_info.R_camera_to_base    
    # Combine transformations: world -> robot -> camera_local
    # P_global = R(base->map) R(camera->base) P_camera
    ''' 
     P_camera = R(base->camera) R(map->base) P_global
    '''    
    world_to_camera_local_matrix = np.dot(robot_to_camera_matrix, world_to_robot_matrix )
    # Transform the point
    camera_local_point = np.dot(world_to_camera_local_matrix, np.array([target_position[0], target_position[1], target_position[2], 1.0]))    
    camera_local_point_in_cframe = np.array([-1*camera_local_point[1], -1*camera_local_point[2], camera_local_point[0]])        
    p1 = get_rect_vertext(patch_in_meter, 0.0)        
    ## compute the vertex point to make cube estimation TODO:: now we have 4 vertext, but can be make it faster by having only 2 vertex
    point_3d = []
    for i in range(4):
        tmp = camera_local_point_in_cframe+ np.array([p1[int(i),0], 0, p1[int(i),1]])
        tmp[2] = max(tmp[2],0.0)
        point_3d.append(tmp.copy())
      
    point_3d = np.array(point_3d)
    (projected_points, _) = cv2.projectPoints(point_3d, np.zeros(3), np.zeros(3), camera_matrix, distortion_coeffs)


    return projected_points


def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Calculate angles of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort points based on angles in clockwise order
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    return sorted_points



def get_mask(original_image,boundary_points):
    # Create an empty mask with the same shape as the original image
    # mask = np.zeros_like(original_image, dtype=np.uint8)
    # # Set pixel values to 0 for the unmasked area
    # # Draw a filled polygon on the mask using the boundary points
    # cv2.fillPoly(mask, [np.array(boundary_points)], color=(0, 0, 255))
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [boundary_points], 255)  # Create a white polygon as a mask
    # Create an inverse mask for unmasked area
    # Create an inverse mask for unmasked area
    inverse_mask = cv2.bitwise_not(mask)
    # Create a copy of the original image
    result_image = original_image.copy()
    # Set pixel values to 0 for the unmasked area
    result_image[inverse_mask != 0] = 0    

    return result_image


def get_attented_img(img,target_position, odom_msg: Odometry, cam_info,  patch_in_meter = 1.0):
    vertex = convert_to_local_frame(target_position, odom_msg, cam_info,  patch_in_meter)    
    height = img.shape[0] # 480
    width = img.shape[1] # 640 
    vertex = np.squeeze(vertex).astype(int)  # Convert vertex to the correct format
    vertex[:, 0] = np.clip(vertex[:, 0], 0, width - 1)  # Subtract 1 from width
    vertex[:, 1] = np.clip(vertex[:, 1], 0, height - 1)  # Subtract 1 from height
    vertex = sort_points_clockwise(vertex)    
    img = get_mask(img,vertex)
    
    return img

def get_attented_img_given_aucData(target_poses, auc_data, camera_info):
    
    color = auc_data.color
    depth = auc_data.depth
    color_imgs = []
    depth_imgs = []
    for i in range(len(target_poses)):
        att_color_tmp = get_attented_img(color,target_poses[i,:], auc_data.vehicle.odom, camera_info,  patch_in_meter = 1.0)
        color_imgs.append(att_color_tmp.copy())
        att_depth_tmp = get_attented_img(depth,target_poses[i,:], auc_data.vehicle.odom, camera_info,  patch_in_meter = 1.0)
        depth_imgs.append(att_depth_tmp.copy())

    color_imgs = np.array(color_imgs)
    color_imgs = np.transpose(color_imgs,(0,3,1,2))
    depth_imgs = np.array(depth_imgs)
    depth_imgs = depth_imgs[:,np.newaxis]
    
    return color_imgs, depth_imgs

