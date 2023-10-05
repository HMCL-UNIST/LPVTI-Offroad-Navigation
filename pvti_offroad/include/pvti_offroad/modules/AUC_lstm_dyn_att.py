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

import torch
import torch.nn as nn
import os 
import math

'''
Input: state, attentioned predicted images, action prediciton 
Output: Hidden states for residual predicted positions 
'''
def ff(x,k,s):
    return (x-k)/s+1
def rr(y,k,s):
    return (y-1)*s+k

class AUCLSTMModel(nn.Module):    
    def __init__(self, args):
        super(AUCLSTMModel, self).__init__()
        
        
        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim'] 
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']        
        self.lstm_hidden_size = args['lstm_hidden_size']        
        self.output_residual_dim = args['output_residual_dim']
        self.conv_to_feedforwad_dim = args['conv_to_feedforwad_dim']
        self.distributed_train = args['distributed_train']
        if self.distributed_train:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0    
        

        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        self.auc_lstm_hidden_size = args['lstm_hidden_size']                   
        self.auc_lstm_input_dim = 4
        
        
        
        self.auc_output_fc_hidden = args['auclstm_out_fc_hidden_size']        
        self.auclstm_output = args['auclstm_output_dim']
        
        self.init_input_size = self.input_grid_width*self.input_grid_height + self.input_state_dim + self.input_action_dim        
       
        self.image_conv = nn.Sequential(
        nn.utils.spectral_norm(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=2, padding=1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3),        
        nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3),
        nn.utils.spectral_norm(nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3)
        ).to(self.gpu_id) # Optional pooling layer
        
        self.auc_conv_out_size = self._get_conv_out_size(self.image_conv,self.input_grid_height,self.input_grid_width, input_channels= 4 )        
        

        
        self.image_fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.auc_conv_out_size, 30)),        
                nn.ReLU(),                                    
                nn.utils.spectral_norm(nn.Linear(30, 20)),        
                nn.ReLU(),                                    
                nn.utils.spectral_norm(nn.Linear(20, 8))                               
        ).to(self.gpu_id) 

        self.input_to_image_ff = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(8, 20)),
                nn.ReLU(),       
                nn.utils.spectral_norm(nn.Linear(20, 30)), 
                nn.ReLU(), 
                nn.utils.spectral_norm(nn.Linear(30, self.auc_conv_out_size))
        ).to(self.gpu_id) 
        
        self.auc_lstm = nn.LSTM(input_size=16,  
                    hidden_size=self.auc_lstm_hidden_size,
                    num_layers=2,
                    batch_first=True).to(self.gpu_id)
        
        
        self.test_fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(180, 70)),                                
                nn.BatchNorm1d(70),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Linear(70, 45))                                           
        ).to(self.gpu_id) 
     

    def _get_conv_out_size(self, model, width,height, input_channels = 4):
        dummy_input = torch.randn(1, input_channels, width,height, requires_grad=False).to(self.gpu_id).float()         
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
    def pred_from_single_image(self,input_bag):
        
        xhat, image = input_bag
        batch_size = xhat.shape[0]
        seq_size = xhat.shape[1]
        single_image_features = self.image_conv(image.unsqueeze(dim=0))
        image_features = single_image_features.repeat(batch_size,1,1,1)
        image_features = image_features.view(batch_size,-1)
        
        att_img_featurs = torch.zeros(xhat.shape).to(self.gpu_id).float()
        for i in range(seq_size):
            Qry = image_features.unsqueeze(dim=-1).clone()            
            att_key = self.input_to_image_ff(xhat[:,i,:]).unsqueeze(dim=-1)
            att_weight = torch.softmax(Qry@att_key.transpose(-2,-1)/math.sqrt(Qry.size(-1)), dim=-1)
            att_features = (att_weight @ Qry).squeeze()
                
            att_features = self.image_fc(att_features)
            att_img_featurs[:,i,:] = att_features.clone()

        lstm_input = torch.cat((att_img_featurs,xhat),dim=2)
        

        h0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()
        c0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()        
        output, (h,c) = self.auc_lstm(lstm_input,(h0,c0))

        
        output_flatten = output.reshape(batch_size,-1)
        dir_pred = self.test_fc(output_flatten)
        
        aug_pred = dir_pred.reshape(dir_pred.shape[0],-1,5)
        temporal_encoding = torch.ones(aug_pred.shape[0], aug_pred.shape[1], 1).to(self.gpu_id).float() 
        for i in range(seq_size-1):
            temporal_encoding[:,i,:] = i
        aug_pred = torch.cat((aug_pred, temporal_encoding), dim =2)
     
        return aug_pred

        

    def forward(self, input):        
        xhat, images = input 
        batch_size = xhat.shape[0]
        seq_size = xhat.shape[1]
        image_features = self.image_conv(images)
        image_features = image_features.view(batch_size,-1)        
        att_img_featurs = torch.zeros(xhat.shape).to(self.gpu_id).float()
        for i in range(seq_size):
            Qry = image_features.unsqueeze(dim=-1).clone()            
            att_key = self.input_to_image_ff(xhat[:,i,:]).unsqueeze(dim=-1)
            att_weight = torch.softmax(Qry@att_key.transpose(-2,-1)/math.sqrt(Qry.size(-1)), dim=-1)
            att_features = (att_weight @ Qry).squeeze()                
            att_features = self.image_fc(att_features)
            att_img_featurs[:,i,:] = att_features.clone()

        lstm_input = torch.cat((att_img_featurs,xhat),dim=2)
        

        h0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()
        c0 = torch.zeros(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.gpu_id).float()        
        output, (h,c) = self.auc_lstm(lstm_input,(h0,c0))

        output_flatten = output.reshape(batch_size,-1)
        dir_pred = self.test_fc(output_flatten)
        
        aug_pred = dir_pred.reshape(dir_pred.shape[0],-1,5)
        temporal_encoding = torch.ones(aug_pred.shape[0], aug_pred.shape[1], 1).to(self.gpu_id).float() 
        for i in range(seq_size-1):
            temporal_encoding[:,i,:] = i
        aug_pred = torch.cat((aug_pred, temporal_encoding), dim =2)
     
        return aug_pred
    