"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
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
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: Compute uncertainty-aware traversability cost given the predictive path distributions
"""
from this import d
import numpy as np
import torch
from nav_msgs.msg import Odometry
from pvti_offroad.common.utils_batch import dist2d, gaussianKN2D, wrap_to_pi_torch

class GPGridMap:
    def __init__(self, device = "cuda", dt = 0.1):
        self.dt = dt
        self.vehicle_model = None        
        self.map = None
        self.elev_map = None
        self.normal_vector = None

        self.definite_map_x_max = 30.0 
        self.definite_map_y_max = 30.0 

        self.right_corner_x = None
        self.right_corner_y = None
        self.map_resolution = None
        self.map_info = None
        self.c_size = None
        self.r_size = None
        self.kernel_size = 5
        self.surface_normal_x_idx = None
        self.surface_normal_y_idx = None
        self.surface_normal_z_idx = None
        self.elevation_idx = None    

        self.map_ready = False
        
        
        self.dist_heuristic_cost_scale = 1.0        
        self.model_error_weight  = 0.0
        self.local_map_cost_weight = 0.0
        self.rollover_cost_scale = 0.0
        
        self.kernel_dist_cost_scale = 1.0
        self.prediction_diff_cost_scale = 1.0

        self.error_mean_scale = 1.0
        self.error_std_scale = 1.0
        
        self.torch_device  = device

    def is_within_map(self,odom:Odometry):              
        if odom.pose.pose.position.x > 17 or odom.pose.pose.position.y > 20:
            return False
        if odom.pose.pose.position.x < -8 or odom.pose.pose.position.y < -12:
            return False
        return True    
        pose = torch.tensor([[odom.pose.pose.position.x,odom.pose.pose.position.y]]).to(device=self.torch_device)

        if self.is_ready() is False:
            return False
        pose = torch.tensor([[odom.pose.pose.position.x,odom.pose.pose.position.y]]).to(device=self.torch_device)
        idx = self.pose2idx(pose)
        if idx == -1:
            return False        
        return True
    
    def is_ready(self):
        return self.map_ready
    
    def set_scales(self,scale_data):        
        self.dist_heuristic_cost_scale   = scale_data['dist_heuristic_cost_scale']                
        self.model_error_weight      = scale_data['model_error_weight']        
        self.local_map_cost_weight      = scale_data['local_map_cost_weight']        
        self.rollover_cost_scale = scale_data['rollover_cost_scale']
        self.error_std_scale = scale_data['error_std_scale']
        self.error_mean_scale = scale_data['error_mean_scale']

    def set_map(self,map):
        self.map_info = map.info           
        self.map = map 
        self.elevation_idx = self.map.layers.index("elevation")                
        self.geo_traversability_idx = self.map.layers.index("terrain_traversability")    

        self.c_size =torch.tensor(map.data[self.elevation_idx].layout.dim[0].size).to(device=self.torch_device)             
        self.r_size = torch.tensor(map.data[self.elevation_idx].layout.dim[1].size ).to(device=self.torch_device)  
        self.map_resolution = torch.tensor(self.map_info.resolution).to(device=self.torch_device)    
         
        self.right_corner_x = torch.tensor(self.map_info.pose.position.x + self.map_info.length_x/2).to(device=self.torch_device)          
        self.right_corner_y = torch.tensor(self.map_info.pose.position.y + self.map_info.length_y/2).to(device=self.torch_device)  
        
        
        self.surface_normal_x_idx = self.map.layers.index("surface_normal_x")            
        self.surface_normal_y_idx = self.map.layers.index("surface_normal_y")            
        self.surface_normal_z_idx = self.map.layers.index("surface_normal_z")   
            
        self.map.data[self.geo_traversability_idx].data = np.nan_to_num(self.map.data[self.geo_traversability_idx].data, nan = 0.0)           
        self.map.data[self.elevation_idx].data = np.nan_to_num(self.map.data[self.elevation_idx].data, nan = 0.0)           
        self.normal_vector_x = np.nan_to_num(self.map.data[self.surface_normal_x_idx].data, nan = 0.0)   
        self.normal_vector_y = np.nan_to_num(self.map.data[self.surface_normal_y_idx].data, nan = 0.0)   
        self.normal_vector_z = np.nan_to_num(self.map.data[self.surface_normal_z_idx].data, nan = 1.0)   

        self.terrain_type_idx = self.map.layers.index("terrain_type")
        self.map.data[self.terrain_type_idx].data = np.nan_to_num(self.map.data[self.terrain_type_idx].data, nan = 0.0)       
        
        self.update_torch_map()
        

    def update_torch_map(self):     
        self.geotraversableMap_torch = 1-torch.from_numpy(self.map.data[self.geo_traversability_idx].data).to(device=self.torch_device)                          
        self.terrainMap_torch        = torch.from_numpy(self.map.data[self.terrain_type_idx].data).to(device=self.torch_device)                                  
        self.elevationMap_torch      = torch.from_numpy(self.map.data[self.elevation_idx].data).to(device=self.torch_device)                    
        self.normal_vector_x_torch   = torch.from_numpy(self.normal_vector_x).to(device=self.torch_device)            
        self.normal_vector_y_torch   = torch.from_numpy(self.normal_vector_y).to(device=self.torch_device)            
        self.normal_vector_z_torch   = torch.from_numpy(self.normal_vector_z).to(device=self.torch_device)            
        self.map_ready = True

    def normalize_vector(self,vector):        
        return vector / torch.sqrt(torch.sum(vector**2,1)).view(-1,1).repeat(1,3)     

    def get_normal_vector_idx(self,idx):
        assert self.surface_normal_x_idx is not None, "surface normal x is not available"        
        if torch.sum(idx >= self.r_size*self.c_size) + torch.sum(idx < 0) > 0:      
            default_normal_vector = torch.zeros(len(idx),3)    
            default_normal_vector[:,2] = 1.0
            return default_normal_vector           
        else:
            normal_vector = torch.vstack((torch.index_select(self.normal_vector_x_torch,0,idx.squeeze()), 
                                            torch.index_select(self.normal_vector_y_torch,0,idx.squeeze()),
                                                torch.index_select(self.normal_vector_z_torch,0,idx.squeeze())))          
            normal_vector = torch.transpose(normal_vector,0,1)            
            return self.normalize_vector(normal_vector)

    def get_normal_vector(self,pose):
        if self.map is None or self.surface_normal_x_idx is None:
            return torch.Tensor([0,0,1])
        idx = self.pose2idx(pose)
        return self.get_normal_vector_idx(idx)                
    

    def get_terrain_type_idx(self,idx):
        if self.terrain_type_idx is None:
            return torch.zeros(len(idx))
        if torch.sum(idx >= self.r_size*self.c_size) > 0:
            print("idx out of bound")            
            return torch.zeros(len(idx))       
        return torch.index_select(self.terrainMap_torch,0,idx.squeeze())
        
       
    def get_terrain_type(self,pose):        
        idx = self.pose2idx(pose)
        if torch.sum(idx < 0) > 0:
            return torch.zeros(len(idx))
        return self.get_terrain_type_idx(idx)

    
    def get_elevation_idx(self,idx):
        if self.elevation_idx is None:
            return torch.zeros(len(idx))
        if torch.sum(idx >= self.r_size*self.c_size) > 0:
            print("idx out of bound")            
            return torch.zeros(len(idx))
        else: 
            return torch.index_select(self.elevationMap_torch,0,idx.squeeze())
           
    
    def get_elevation(self,pose):
        idx = self.pose2idx(pose)
        if torch.sum(idx < 0) > 0:
            return torch.zeros(len(idx))
            
        return self.get_elevation_idx(idx)        

  
    def get_rollpitch(self,pose):
        if not torch.is_tensor(pose):
            pose = torch.tensor(pose)  

        idx =self.pose2idx(pose)
        yaw = pose[:,2]
        default_rpy = torch.zeros(len(pose),3).to(device=self.torch_device) 
        default_rpy[:,2] = yaw
        if self.map is None:
            return default_rpy

        if torch.sum(idx) < 0:
            return default_rpy
        if self.surface_normal_x_idx is None:
            return default_rpy    
        normal_vector = self.get_normal_vector_idx(idx)             
        yaw_vec = torch.hstack([torch.cos(yaw).view(-1,1),torch.sin(yaw).view(-1,1),torch.zeros(len(yaw)).view(-1,1).to(device=self.torch_device)])
        yaw_vec[:,2] = -1*(normal_vector[:,0]*yaw_vec[:,0]+normal_vector[:,1]*yaw_vec[:,1])/(normal_vector[:,2]+1e-10)
        yaw_vec = self.normalize_vector(yaw_vec)                
        ry = torch.asin(yaw_vec[:,2])
        rz = torch.acos(yaw_vec[:,0] / ( torch.cos(ry)+1e-5))
        rx = torch.acos(normal_vector[:,2]/ (torch.cos(ry)+1e-5))
        roll = -1*rx
        pitch = -1*ry 
        yaw = rz 
        roll = wrap_to_pi_torch(roll)
        pitch = wrap_to_pi_torch(pitch)
        # assert roll is not np.nan, "idx is out of bound"                    
        return torch.hstack([roll.view(-1,1), pitch.view(-1,1), yaw.view(-1,1)])
        
        
    def pose2idx(self,pose):  
        grid_c_idx = ((self.right_corner_x - pose[:,0].view(-1,1)) / self.map_resolution).int()
        grid_r_idx = ((self.right_corner_y - pose[:,1].view(-1,1)) / self.map_resolution).int()
        if torch.sum(grid_c_idx >= self.c_size) > 0:
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device) 
        if torch.sum(grid_r_idx >= self.r_size) > 0:            
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device)         
        idx = grid_c_idx + grid_r_idx*self.r_size 
        if torch.sum(idx >= self.c_size*self.r_size) + torch.sum(idx < 0) > 0:
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device)                  
        return idx.int()


    def idx2grid(self,idx):
        r_idx  = (idx/self.c_size).int()
        c_idx  = (idx%self.c_size).int()
        return torch.transpose(torch.vstack((r_idx.squeeze(),c_idx.squeeze())),0,1)

    
    def get_sub_map_idx(self,idx,local_map_size):        
        map = self.geotraversableMap_torch        
        ## To do, currently the kernel size is manually fixed as 5.. 
        # need to implement auto generation of kernel given the local_map_size
        rc15 = map[idx.long()-2-self.c_size*2] 
        rc14 = map[idx.long()-1-self.c_size*2] 
        rc13 = map[idx.long()-self.c_size*2] 
        rc12 = map[idx.long()+1-self.c_size*2] 
        rc11 = map[idx.long()+2-self.c_size*2]        

        rc25 = map[idx.long()-2-self.c_size] 
        rc24 = map[idx.long()-1-self.c_size]
        rc23 = map[idx.long()-self.c_size] 
        rc22 = map[idx.long()+1-self.c_size]
        rc21 = map[idx.long()+2-self.c_size]

        rc35 = map[idx.long()-2] 
        rc34 = map[idx.long()-1]
        rc33 = map[idx.long()]
        rc32 = map[idx.long()+1] 
        rc31 = map[idx.long()+2]

        rc45 = map[idx.long()-2+self.c_size]
        rc44 = map[idx.long()-1+self.c_size]
        rc43 = map[idx.long()+self.c_size] 
        rc42 = map[idx.long()+1+self.c_size]
        rc41 = map[idx.long()+2+self.c_size] 

        rc55 = map[idx.long()-2+self.c_size*2]
        rc54 = map[idx.long()-1+self.c_size*2]
        rc53 = map[idx.long()+self.c_size*2]
        rc52 = map[idx.long()+1+self.c_size*2]
        rc51 = map[idx.long()+2+self.c_size*2]

        rc1 = torch.hstack([rc15,rc14,rc13,rc12,rc11])
        rc2 = torch.hstack([rc25,rc24,rc23,rc22,rc21])
        rc3 = torch.hstack([rc35,rc34,rc33,rc32,rc31])
        rc4 = torch.hstack([rc45,rc44,rc43,rc42,rc41])
        rc5 = torch.hstack([rc55,rc54,rc53,rc52,rc51])
        output_submaps = torch.stack([torch.transpose(rc1,0,1),torch.transpose(rc2,0,1),torch.transpose(rc3,0,1),torch.transpose(rc4,0,1),torch.transpose(rc5,0,1)])
        output_submaps = torch.permute(output_submaps,(2,0,1))
        
        return output_submaps


    def get_sub_map(self,pose,local_map_size):
        idx = self.pose2idx(pose)
        return self.get_sub_map_idx(idx,local_map_size)
        
    
    
    def get_model_error_mahalanbis_distance(self, error_mean, error_std):        
        diagonal_matrices = torch.diag_embed(error_std, offset=0, dim1=-2, dim2=-1)        
        cholesky_decomposition = torch.linalg.cholesky(diagonal_matrices)                
        inverse_diagonal_matrices = torch.cholesky_inverse(cholesky_decomposition)        
        mahalanobis_distances = torch.matmul(torch.matmul(error_mean.unsqueeze(-2), inverse_diagonal_matrices) ,error_mean.unsqueeze(-1)).squeeze()        
        return mahalanobis_distances

    def get_model_error_cost(self,error_mean, error_std):
        mean_error_cost = torch.pow(error_mean,2) * self.error_mean_scale
        std_error_cost = torch.pow(error_std,2) *self.error_std_scale
        return torch.sum(mean_error_cost + std_error_cost,dim=2)
    

    def get_total_trav_cost(self, model_error_cost, local_map_cost):        
        return model_error_cost + self.local_map_cost_weight/ (1 + torch.exp(-1*local_map_cost))

    def compute_best_path(self,nominal_states, error_mean, error_std, goal = None):
        pose_error_mean  = error_mean[:,:,:2]
        pose_error_std  = error_std[:,:,:2]        
        model_error = self.get_model_error_cost(pose_error_mean, pose_error_std)
        seq_length = model_error.shape[1]
        batch_size = model_error.shape[0]
        pred_pose_mean_seq = (nominal_states[:,1:,:2]+ error_mean[:,:,:2])
        pred_pose_mean = pred_pose_mean_seq.view(-1,2)
        
        xsig = error_std[:,:,0].view(-1,1)
        ysig = error_std[:,:,1].view(-1,1)        
       
        local_map_sizes = torch.tensor([self.kernel_size,self.kernel_size]).to(device= self.torch_device).int()            
        submaps = self.get_sub_map(pred_pose_mean,local_map_sizes)        
        gpkernel = gaussianKN2D(local_map_sizes, rsig=ysig,csig=xsig)
        gpkernel_tensor = torch.stack(gpkernel,dim=0)
        submap_dot_gpkernel = torch.bmm(submaps,gpkernel_tensor)
        local_map_cost = torch.sum(submap_dot_gpkernel,dim=(1,2))
        local_map_cost = local_map_cost.view(-1, seq_length)
        local_map_cost = torch.sum(local_map_cost,dim=-1)                
        
        model_error = torch.sum(model_error,dim=-1)        
        normalized_model_error_cost = (model_error-torch.min(model_error))  / (torch.max(model_error)-torch.min(model_error)+1e-20)        
        model_error_costs = normalized_model_error_cost * self.model_error_weight 
        
        
        normalized_local_map_cost = (local_map_cost-torch.min(local_map_cost))  / (torch.max(local_map_cost)-torch.min(local_map_cost)+1e-20)        
        local_map_costs = normalized_local_map_cost         
        
        total_trav_costs = self.get_total_trav_cost(model_error_costs, local_map_costs)
     
        if goal is not None:                        
            goal_torch = torch.tensor(goal).to(device = self.torch_device).repeat(batch_size,1) 
            dists = dist2d(pred_pose_mean_seq[:,-1,:],goal_torch)
            dists_cost = dists                        
            normalized_dist_cost = (dists_cost-torch.min(dists_cost))  / (torch.max(dists_cost)-torch.min(dists_cost)+1e-20)        
            total_trav_costs += normalized_dist_cost *self.dist_heuristic_cost_scale        
            

        best_path_idx = torch.argmin(total_trav_costs)          
        
        best_path = nominal_states[best_path_idx,:,:]
        
        return best_path_idx, best_path, total_trav_costs, pred_pose_mean_seq

           