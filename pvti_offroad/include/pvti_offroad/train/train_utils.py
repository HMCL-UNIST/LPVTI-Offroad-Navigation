#!/usr/bin/env python3
import os
import pickle
import random
from typing import List
import numpy as np
from pvti_offroad.common.file_utils import * 
from pvti_offroad.common.pytypes import AUCModelData, SimData 
import torch 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def get_residual_state(auc_datas : List[AUCModelData]):
    ''' 
    state residual in vehicle reference frame 
    '''
    gt_states = torch.tensor([[data.vehicle.odom.pose.pose.position.x, 
                                data.vehicle.odom.pose.pose.position.y,                   
                                data.vehicle.local_twist.linear.x,
                                data.vehicle.local_twist.linear.y,
                                data.vehicle.local_twist.angular.z
                                ] for data in auc_datas]) 
   
    cur_data = auc_datas[0].copy()
    
    pred_out_states = torch.tensor([[data.odom.pose.pose.position.x, 
                                data.odom.pose.pose.position.y,                   
                                data.local_twist.linear.x,
                                data.local_twist.linear.y,
                                data.local_twist.angular.z] for data in cur_data.pred_vehicles]) 
    residual_state = gt_states[1:,:] - pred_out_states[1:,:]         
    
    return residual_state

def get_action_set(auc_data:AUCModelData):
    action_set = torch.tensor([[data.u.ax, data.u.steer] for data in auc_data.pred_vehicles])    
    return action_set[:-1,:]

def get_cur_vehicle_state_input(auc_data:AUCModelData):
    
    '''
    # we use local velocity and pose as vehicle current state         
    TODO: see if vz, wx, wy can be ignored and independent of estimation process
    '''
    return torch.tensor([auc_data.vehicle.local_twist.linear.x,
                    auc_data.vehicle.local_twist.linear.y,                                        
                    auc_data.vehicle.local_twist.angular.z,
                    auc_data.vehicle.euler.pitch,
                    auc_data.vehicle.euler.roll])


class AUCDataset(Dataset):    
    def __init__(self, input_d, output_d):  
         
        (states,pred_actions, xhat, colors, depths, concat_image)  = input_d 
        
        (gt_state_residual) = output_d 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = states
        self.pred_actions = pred_actions
        self.colors = colors
        self.depths = depths
        self.xhat = xhat
        self.concat_image= concat_image

        self.gt_state_residual = gt_state_residual       
        
        self.states_mean = None
        self.states_std = None
        self.pred_actions_mean = None
        self.pred_actions_std = None
        self.gt_state_residual_mean = None
        self.gt_state_residual_std = None
        self.gt_state_residual_max = None
        self.gt_state_residual_min = None
        self.data_noramlize(states, pred_actions, gt_state_residual)
        assert len(self.states) == len(self.pred_actions) , "All input and output must have the same length"
    
    def get_norm_stats(self):
        stat_dict = {
                'states_mean':self.states_mean,
                'states_std':self.states_std,
                'pred_actions_mean':self.pred_actions_mean,
                'pred_actions_std':self.pred_actions_std,
                'gt_state_residual_mean':self.gt_state_residual_mean,
                'gt_state_residual_std':self.gt_state_residual_std,
                'gt_state_residual_max':self.gt_state_residual_max,
                'gt_state_residual_min':self.gt_state_residual_min
        }

        return stat_dict

    def min_max_scaling(self,data,max,min):
        range_val = max-min+1e-10
        return (data-min)/range_val
        
    def normalize(self,data, mean, std):        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def standardize(self, normalized_data, mean, std):        
        data = normalized_data*std+mean        
        return data
    
    def data_noramlize(self, states, pred_actions, gt_state_residual):
        self.states_mean, self.states_std = self.normalize_each(states)
        self.pred_actions_mean, self.pred_actions_std = self.normalize_each(pred_actions)
        self.gt_state_residual_mean, self.gt_state_residual_std = self.normalize_each(gt_state_residual)

        self.gt_state_residual_max, self.gt_state_residual_min = self.get_min_max(gt_state_residual)
        
    def get_min_max(self,x):
        stacked_tensor = torch.stack(x, dim=0)
        max_tensor = torch.max(stacked_tensor, dim=0)
        min_tensor = torch.min(stacked_tensor, dim=0)
        return max_tensor, min_tensor
        
    def normalize_each(self, x):
        stacked_tensor = torch.stack(x, dim=0)
        # Calculate mean and standard deviation along dimension 1
        mean_tensor = torch.mean(stacked_tensor, dim=0)
        std_tensor = torch.std(stacked_tensor, dim=0)
        return mean_tensor, std_tensor

        

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        assert self.states_mean is not None, 'input, output should be normlized'        
        states = self.states[idx]
        pred_actions = self.pred_actions[idx]
        colors = self.colors[idx]
        depths = self.depths[idx]    
        xhat = self.xhat[idx]
        concat_image = self.concat_image[idx]
        
        gt_state_residual = self.gt_state_residual[idx]
        
        input_d = (states,pred_actions,xhat, colors, depths, concat_image)  
        output_d = (gt_state_residual) 

        return input_d, output_d

class SampleGenerator():
 

    def __init__(self, abs_path, data_path= None, elect_function=None):
   
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.ego_check_time = []
        self.check_time_diff = []
        self.valid_check_time_diff = []        
        self.ax_sample = []
        self.delta_sample = []
        self.sequence_length = 10
        self.N = self.sequence_length
        self.dt = 0.2 
        
        self.X_data = []
        self.Y_data = []


        self.plot_validation_result = False
        data_loaded = False

        if data_path is not None :            
           data_loaded = self.preprocessed_data_load(data_path)        
            
        if data_loaded is False:
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')                        
                        scenario_data: SimData = pickle.load(dbfile)
                        N = len(scenario_data.auc_data) # scenario_data.N
                        camera_info = scenario_data.camera_info                        
                        if N < 12:
                            print("file skipped " + str(filename)+ ' at time step ' + str(i))
                            continue
                    
                        for i in range(N-self.sequence_length-1):         
                            
                            time_diff =scenario_data.auc_data[i+1].header.stamp.to_sec()-scenario_data.auc_data[i].header.stamp.to_sec()                        
                            if self.validate_time_diff(time_diff) is False:
                                print("time diff file skipped " + str(filename) + ' at time step ' + str(i))
                                continue
                            

                            pred_actions = get_action_set(scenario_data.auc_data[i])
                            self.ax_sample.append(pred_actions[0,0])
                            self.delta_sample.append(pred_actions[0,1])
                            
                            ################ GET Predicted states for input
                            # scenario_data.auc_data[i]
                            xhat = scenario_data.auc_data[i].xhat.squeeze()
                            concat_image = scenario_data.auc_data[i].image
                            #############################################################
                            if len(scenario_data.auc_data[i:i+self.N]) < self.N:
                                continue 
                            
                            residual_state = get_residual_state(scenario_data.auc_data[i:i+self.N])
                            
                            if self.residual_validation(None, residual_state) is False:
                                print("residual validation file skipped " + str(filename)+ ' at time step ' + str(i))
                                continue
                            
                            cur_state = get_cur_vehicle_state_input(scenario_data.auc_data[i])
                            if self.state_validation(scenario_data.auc_data[i]) is False:                            
                                continue
                            
                            input_data = {}
                            input_data['state'] = torch.tensor(cur_state).cpu()
                            input_data['pred_actions'] = torch.tensor(pred_actions).cpu()
                            color_img = np.transpose(scenario_data.auc_data[i].color,(2,0,1))
                            depth = scenario_data.auc_data[i].depth
                            depth_img = depth[np.newaxis,:]
                            
                            input_data['color'] = torch.tensor(color_img).cpu().float()  
                            input_data['depth'] = torch.tensor(depth_img).cpu().float()  
                            input_data['concat_image'] = torch.tensor(concat_image).cpu().float()   
                            input_data['xhat'] = torch.tensor(xhat).cpu().float()                        

                            output_data= {}
                            

                            output_data['gt_state_residual'] = torch.tensor(residual_state).cpu().float()                        
                          
                            if self.detect_nan_from_data(output_data) or self.detect_nan_from_data(input_data):
                                print(str(i) + " NAN is included ..")
                                continue

                            self.X_data.append(input_data)
                            self.Y_data.append(output_data)

                        dbfile.close()
        
            if self.plot_validation_result:     

                self.plot_action_samples()
                self.plot_state_validation()
                self.plot_residual_list()
                
                self.plotTimeDiff()
            print('Generated Dataset with', len(self.X_data), 'samples!')
            self.preprocessed_data_save()
    
    def resample_image(self,image):
        return image
    
    def detect_nan_from_data(self, dict_obj):
        def has_nan(tensor):
            return torch.isnan(tensor).any().item()
        has_nan_values = any(has_nan(tensor) for tensor in dict_obj.values())
        return has_nan_values

    def get_dataset(self,ratio = 0.8):        
        state = [d['state'] for d in self.X_data]
        pred_action = [d['pred_actions'] for d in self.X_data]
        pred_pose = [d['pred_actions'] for d in self.X_data]

        color = [d['color'] for d in self.X_data]             
        depth = [d['depth'] for d in self.X_data]
        xhat = [d['xhat'] for d in self.X_data]
        concat_image = [d['concat_image'] for d in self.X_data]
        gt_state_residual = [d['gt_state_residual'] for d in self.Y_data]
        
    
        input_d = (state,pred_action,xhat, color, depth,concat_image)
        output_d = (gt_state_residual)
        auc_dataset = AUCDataset(input_d, output_d)
        return auc_dataset
        
    def preprocessed_data_load(self,path = None):
        if path is None:
            return False        
        loaded = torch.load(path)     
        self.X_data = loaded['input']
        self.Y_data = loaded['ouput']
        print('Loaded Dataset with', len(self.X_data), 'samples!')
        return True
    
    def preprocessed_data_save(self,data_dir = None):
        if data_dir is None:
            data_dir = preprocessed_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_path = os.path.join(data_dir, f'preprocessed_data_{len(self.X_data)}.pth')                
        # Create a checkpoint dictionary including both model state and args
        checkpoint = {
            'input': self.X_data,
            'ouput': self.Y_data
        }
        
        torch.save(checkpoint, file_path)
        print(f"preprocessed_data_ saved at epoch {str(file_path)}")

    def state_validation(self,state):
        if not hasattr(self,'state_list'):
            self.state_list = []            
            self.valid_state_list = []
            return        
        
        self.state_list.append(abs(state.vehicle.local_twist.linear.x.cpu().numpy()))        
        is_valid = True                
   

        if is_valid:
            self.valid_state_list.append(abs(state.vehicle.local_twist.linear.x.cpu().numpy()))        
        else:
            self.valid_state_list.append(0.0)
        return is_valid

    def plot_action_samples(self):
        ax = torch.stack(self.ax_sample).view(-1)
        delta = torch.stack(self.delta_sample).view(-1)
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        labels= ['ax', 'delta']        
        axs[0].hist(ax, bins=20, alpha=0.5, label='ax')
        axs[1].hist(delta, bins=20, alpha=0.5, label='delta')
        axs[0].legend()        
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    def plot_state_validation(self):
        if not hasattr(self,'residual_list'):            
            return       
        residual_list =np.array(self.state_list)
        valid_state_list = np.array(self.valid_state_list)
      
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        labels= ['true', 'filtered']        
        axs.plot(residual_list, label=labels[0])
        axs.plot(valid_state_list, label=labels[1])        
        axs.set_title(f"Vx")            
        axs.set_ylabel("VX")
        axs.legend()
        plt.tight_layout()
        plt.show()


    def residual_validation(self,pred_pose,gt_state_residual):
        if not hasattr(self,'residual_list'):
            self.residual_list = []            
            self.valid_residual_list = []
            return        
        max_tmp = torch.mean(gt_state_residual, dim=0)
        
        self.residual_list.append(max_tmp)        
        is_valid = True        
        
        if is_valid:
            self.valid_residual_list.append(max_tmp)        
        else:
            self.valid_residual_list.append(np.zeros(max_tmp.shape))
        return is_valid
        
    def plot_residual_list(self):
        if not hasattr(self,'residual_list'):            
            return       
        residual_list = torch.stack(self.residual_list).cpu().numpy()
        
        fig, axs = plt.subplots(5, 1, figsize=(10, 5 * 5))
        labels= ['px', 'py', 'vx', 'vy', 'wz']
        for i in range(5):
            axs[i].plot(residual_list[:, i], label=labels[i])            
            axs[i].set_title(f"Residuals for Array {i}")            
            axs[i].set_ylabel("Residual Value")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        
    
    def validate_time_diff(self,time_diff):
        is_valid = False
        self.check_time_diff.append(time_diff)
        if time_diff < self.dt*1.8 and time_diff > self.dt*0.2:
            self.valid_check_time_diff.append(time_diff)                            
            is_valid = True
        else:
            self.valid_check_time_diff.append(0.0)                            
            is_valid = False

        return is_valid
    
    def plotTimeDiff(self):
        all_time_diff = np.array(self.check_time_diff)
        valid_time_diff = np.array(self.valid_check_time_diff)
        plt.plot(all_time_diff)
        plt.plot(valid_time_diff,'*')
        plt.show()
    
    def reset(self, randomize=False):
        if randomize:
            random.shuffle(self.samples)
        self.counter = 0

    def getNumSamples(self):
        return len(self.X_data)

    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return self.samples[self.counter - 1]

  
    def useAll(self, ego_state, tar_state):
        return True

