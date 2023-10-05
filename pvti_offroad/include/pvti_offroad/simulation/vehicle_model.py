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
from pvti_offroad.common.utils import b_to_g_rot, wrap_to_pi_torch
from pvti_offroad.common.pytypes import VehicleState
import torch

class VehicleModel:
    def __init__(self, dt = 0.2, N_node = 10, map_info = None, n_sample = 100):        
        self.local_map = map_info
        self.N = N_node
        self.m = 25
        self.width = 0.45        
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt 
        self.delta_rate_max = 50*math.pi/180.0 # rad/s
        self.rollover_cost_scale = 10.0        
        self.Q = torch.tensor([0.5, 0.5, 0.5])           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## dyn_state = [x,y,z,vx,vy,wz,roll,pitch,yaw]
        self.state_dim = 9
        ## inputs = [ax, delta]
        self.input_dim = 2        
        self.horizon = self.dt* N_node        
        # delta and ax        
        self.ax_max = 1.0
        self.ax_min = -1.0
        self.delta_min = -1* 25*torch.pi/180.0    
        self.delta_max = 25*torch.pi/180.0

        self.cur_state = VehicleState()


    def compute_slip(self, x,u):           
        try: # when u is for batch
            delta = u[:, 1]  
            clip_vx = torch.max(torch.hstack((x[:,3].view(-1,1),torch.ones(len(x[:,3])).to(device=self.device).view(-1,1))),dim =1).values   
            alpha_f = delta - (x[:,4]+self.Lf*x[:,5])/clip_vx
            alpha_r = (-x[:,4]+self.Lr*x[:,5])/clip_vx       
        except IndexError: # when u is single
            delta = u[1]
            vx = x[3]
            alpha_f = delta - (x[4]+self.Lf*x[5])/vx
            alpha_r = (-x[4]+self.Lr*x[5])/vx       
            
        return alpha_f, alpha_r
      
    def compute_normal_force(self,x,u,roll,pitch):     
        try:
            ax = u[:, 0]  # when u is for batch
        except IndexError:
            ax = u[0]  # when u is single
            ax = torch.tensor(ax)
        Fzf = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L + self.h*self.m/self.L*(ax+self.g*torch.sin(pitch))
        Fzr = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L - self.h*self.m/self.L*(ax+self.g*torch.sin(pitch))
        return Fzf, Fzr
    
    def dynamics_predict(self,cur_state:VehicleState, batch_u = None):            
        x =  cur_state.odom.pose.pose.position.x
        y =  cur_state.odom.pose.pose.position.y
        psi = cur_state.euler.yaw 
        vx = cur_state.local_twist.linear.x        
        vy = cur_state.local_twist.linear.y
        wz = cur_state.local_twist.angular.z
        z =  cur_state.odom.pose.pose.position.z
        roll = cur_state.euler.roll
        pitch = cur_state.euler.pitch
        
        x =torch.tensor([x,y,psi,vx,vy,wz,z,roll,pitch]).to(device=self.device) 
        
        if batch_u.shape[0] > 1:
            x = x.repeat(batch_u.shape[0],1)
        else:
            x = x.unsqueeze(dim=0)
        
        usets = batch_u.to(device=self.device)
            
        # return self.torch_dynamics_predict(cur_state = x, batch_u = usets)
        return self.torch_2d_predict(cur_state = x, batch_u = usets)
    
    def torch_dynamics_predict(self,cur_state:torch.tensor, batch_u = torch.tensor):            
        #  x, y, psi, vx, vy, wz, z, roll, pitch 
        
        assert len(batch_u.shape) ==3, 'uset is not in batch'
        batch_size = batch_u.shape[0]
        
        roll_state = cur_state.clone()
        # x_set = x_.repeat(n_sample,1,1)  #  [ number_of_sample, number_of_batch, number_of_states] 
        
        pred_states = []
        pred_states.append(roll_state.clone())
        ####
        # N is the number of step to predict                 
        # same for nominal dynamics 
        tmp_x_set_nominal = roll_state.clone()                
        ## ## Update for each prediction step
        for i in range(self.N-1): 
            px = tmp_x_set_nominal[:,0]
            py = tmp_x_set_nominal[:,1]
            yaw = tmp_x_set_nominal[:,2]
            vx = torch.max(tmp_x_set_nominal[:,3],torch.ones(len(tmp_x_set_nominal[:,3])).to(0)*0.05)                        
            vy = tmp_x_set_nominal[:,4]
            wz = tmp_x_set_nominal[:,5]
            z = tmp_x_set_nominal[:,6]
            usets = batch_u[:,i,:]
            axb = usets[:,0]
            delta = usets[:,1]

            if self.local_map.is_ready():            
                pose = torch.transpose(torch.vstack([px,py,yaw]),0,1)
                rpy_tmp = self.local_map.get_rollpitch(pose)                               
                roll = rpy_tmp[:,0]
                pitch = rpy_tmp[:,1]
                tmp_x_set_nominal[:,6] = self.local_map.get_elevation(pose)                
            else:                  
                roll = tmp_x_set_nominal[:,7]
                pitch = tmp_x_set_nominal[:,8]                
            
            rot_base_to_world = b_to_g_rot(roll,pitch,yaw).double()            
            
            Fzf, Fzr = self.compute_normal_force(tmp_x_set_nominal,usets,roll,pitch)
            alpha_f, alpha_r = self.compute_slip(tmp_x_set_nominal,usets)
            Fyf = Fzf * alpha_f            
            Fyr =  Fzr * alpha_r            
            local_vel = torch.hstack([vx.view(-1,1),vy.view(-1,1),torch.zeros(len(vx)).to(device=self.device).view(-1,1)]).view(-1,3,1).double()            
            vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
            
            vxw = vel_in_world[:,0]
            
            vyw = vel_in_world[:,1]
            
            vzw = vel_in_world[:,2] 
            

            roll_state[:,0] += self.dt*vxw
            roll_state[:,1] += self.dt*vyw
            roll_state[:,2] += self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*wz)
            roll_state[:,3] += self.dt*axb 
            roll_state[:,3] = torch.max(torch.hstack((roll_state[:,3].view(-1,1),torch.zeros(len(roll_state[:,3])).to(device=self.device).view(-1,1))),dim =1).values           
            roll_state[:,4] += self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-vx*wz) 
            roll_state[:,5] += self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz) 
            roll_state[:,6] += self.dt*vzw
            pred_states.append(roll_state.clone())            
            tmp_x_set_nominal = roll_state.clone()           

        pred_states = torch.stack(pred_states,dim =1)
        xhat = self.get_batch_xhat(pred_states)        
        
        
        return xhat, batch_u, pred_states
    
    def torch_2d_predict(self,cur_state:torch.tensor, batch_u = torch.tensor):            
        #  x, y, psi, vx, vy, wz, z, roll, pitch 
        
        assert len(batch_u.shape) ==3, 'uset is not in batch'
        batch_size = batch_u.shape[0]        
        roll_state = cur_state.clone()        
        
        pred_states = []
        pred_states.append(roll_state.clone())
        ####
        # N is the number of step to predict                 
        # same for nominal dynamics 
        tmp_x_set_nominal = roll_state.clone()                
        ## ## Update for each prediction step
        for i in range(self.N-1): 
            px = tmp_x_set_nominal[:,0]
            py = tmp_x_set_nominal[:,1]
            yaw = tmp_x_set_nominal[:,2]
            vx = torch.max(tmp_x_set_nominal[:,3],torch.ones(len(tmp_x_set_nominal[:,3])).to(0)*0.05)                        
            vy = tmp_x_set_nominal[:,4]
            wz = tmp_x_set_nominal[:,5]
            z = tmp_x_set_nominal[:,6]
            usets = batch_u[:,i,:]
            axb = usets[:,0]
            delta = usets[:,1]

            if self.local_map.is_ready():            
                pose = torch.transpose(torch.vstack([px,py,yaw]),0,1)
                rpy_tmp = self.local_map.get_rollpitch(pose)                               
                roll = rpy_tmp[:,0]
                pitch = rpy_tmp[:,1]
                tmp_x_set_nominal[:,6] = self.local_map.get_elevation(pose)                
            else:                  
                roll = tmp_x_set_nominal[:,7]
                pitch = tmp_x_set_nominal[:,8]                
            
            pitch = torch.zeros(pitch.shape).to(self.device)
            roll = torch.zeros(roll.shape).to(self.device)

            rot_base_to_world = b_to_g_rot(roll,pitch,yaw).double()            
            
            Fzf, Fzr = self.compute_normal_force(tmp_x_set_nominal,usets,roll,pitch)
            alpha_f, alpha_r = self.compute_slip(tmp_x_set_nominal,usets)
            Fyf = Fzf * alpha_f            
            Fyr =  Fzr * alpha_r            
            local_vel = torch.hstack([vx.view(-1,1),vy.view(-1,1),torch.zeros(len(vx)).to(device=self.device).view(-1,1)]).view(-1,3,1).double()            
            vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
            
            vxw = vel_in_world[:,0]
            
            vyw = vel_in_world[:,1]
            
            

            roll_state[:,0] += self.dt*vxw
            roll_state[:,1] += self.dt*vyw
            roll_state[:,2] += self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*wz)
            roll_state[:,3] += self.dt*axb 
            roll_state[:,3] = torch.max(torch.hstack((roll_state[:,3].view(-1,1),torch.zeros(len(roll_state[:,3])).to(device=self.device).view(-1,1))),dim =1).values           
            roll_state[:,4] += self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-vx*wz) 
            roll_state[:,5] += self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz) 
            
            pred_states.append(roll_state.clone())            
            tmp_x_set_nominal = roll_state.clone()           

        pred_states = torch.stack(pred_states,dim =1)
        xhat = self.get_batch_xhat(pred_states)        
        
        
        return xhat, batch_u, pred_states
    
        
       


    def get_batch_xhat(self, pred_states):    
        # pred_states : x, y, psi, vx, vy, wz, z, roll, pitch 
        # hxat : x, y, vx, vy, wz,  roll, pitch, yaw        
        batch_size = pred_states.shape[0]
        xhat = torch.zeros(batch_size,pred_states.shape[1],8).to(self.device)
        xhat[:,:,0] = pred_states[:,:,0] # x -> x 
        xhat[:,:,1] = pred_states[:,:,1] # y -> y 
        xhat[:,:,2] = pred_states[:,:,3] # psi -> vx 
        xhat[:,:,3] = pred_states[:,:,4] # vx -> vy 
        xhat[:,:,4] = pred_states[:,:,5] # vy -> wz  
        xhat[:,:,5] = pred_states[:,:,7] # wz -> roll  
        xhat[:,:,6] = pred_states[:,:,8] # z -> pitch
        xhat[:,:,7] = pred_states[:,:,2] # roll -> yaw 
        
        cur_states_x = xhat[:,0,0].clone()
        cur_states_y = xhat[:,0,1].clone()
        
        
        world_del_x = xhat[:,:,0].transpose(0,1) - cur_states_x.repeat(pred_states.shape[1],1)
        world_del_y = xhat[:,:,1].transpose(0,1) - cur_states_y.repeat(pred_states.shape[1],1)
        
        
        xhat[:,:,0] = world_del_x.transpose(0,1).clone() # delta px in local 
        xhat[:,:,1] = world_del_y.transpose(0,1).clone() # delta py in local 
        
        del_yaw = wrap_to_pi_torch(xhat[:,:,-1].transpose(0,1) - xhat[:,0,-1])
        
        xhat[:,:,-1] = del_yaw.transpose(0,1).clone()
     
        return xhat



    
    def action_sampling(self, cur_state: VehicleState):
        
        batch_num = 20
        batch_u = torch.zeros([batch_num,9,2]).to(self.device)

        ax_low_limit = self.ax_min
        ax_high_limit = self.ax_max
        batch_ax = torch.rand([batch_num,9]).to(self.device)
        batch_ax = (ax_high_limit - ax_low_limit) * batch_ax + ax_low_limit
        
        delta_low_imit = self.delta_min 
        delta_high_imit = self.delta_max         
        batch_delta = torch.rand([batch_num,9]).to(self.device)
        batch_delta = (delta_high_imit - delta_low_imit) * batch_delta + delta_low_imit

        batch_u[:,:,0] = batch_ax
        batch_u[:,:,1] = batch_delta
        
        for i in range(batch_num):
            batch_u[i,:,1] = -0.25+0.025*i     

        odd_indices = torch.arange(1, batch_u.shape[0], step=2).to(self.device)
        even_indices = torch.arange(0, batch_u.shape[0], step=2).to(self.device)

        if cur_state.local_twist.linear.x < 1.0:
            batch_u[odd_indices,:,0] = 1.0
            batch_u[even_indices,:,0] = -1.0
        else:
            batch_u[odd_indices,:,0] = 1.0            
            batch_u[even_indices,:,0] = -1.0
            

        return batch_u
    