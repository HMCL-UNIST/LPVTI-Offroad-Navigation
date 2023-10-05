import torch
import gpytorch
from pvti_offroad.gp_encoder.gpytorch_models import IndependentMultitaskGPModelApproximate
from pvti_offroad.modules.AUC_lstm_dyn_att import AUCLSTMModel
import os


class GPAUC(gpytorch.Module):    
    """LSTM-based Contrasiave Auto Encoder"""
    def __init__(
        self, args, train_norm_stat = None):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(GPAUC, self).__init__()        
        self.args = args

        self.train_norm_stat = train_norm_stat

        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim']
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']
        
        self.lstm_hidden_size = args['lstm_hidden_size']
        self.output_residual_dim = args['output_residual_dim']
        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        self.auc_lstm_hidden_size = args['lstm_hidden_size']
        
        

        self.distributed_train = args['distributed_train']
        if self.distributed_train:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0   
        
        # dimensions        
        
        self.gp_input_size = args['output_residual_dim']
        self.gp_output_dim =  args['output_residual_dim']
        self.auclstm_output = args['auclstm_output_dim']
                
        self.seq_len = args['n_time_step']

        
        self.rnn = AUCLSTMModel(args)        
        
        self.gp_layer = IndependentMultitaskGPModelApproximate(inducing_points_num=300,
                                                                input_dim=6,
                                                                num_tasks=5 )  # Independent
        
        
            
    def outputToReal(self, batch_size, pred_dist):
        with torch.no_grad():
            
            standardized_mean = pred_dist.mean.view(batch_size,-1,pred_dist.mean.shape[-1])
            standardized_stddev = pred_dist.stddev.view(batch_size,-1,pred_dist.mean.shape[-1])
            return standardized_mean, standardized_stddev
            

            
    def get_hidden(self,input_data):
        aug_pred = self.rnn(input_data)        
        return aug_pred
        

    def get_pred(self,input_data):
        aug_pred = self.rnn.pred_from_single_image(input_data)
        exp_aug_pred = aug_pred.reshape(-1,6)   
        return self.gp_layer(exp_aug_pred)

    def forward(self, input_data):    
        # current vehicle state, pred_action , RGB-D normalized image (4 channel)        
        aug_pred = self.rnn(input_data)
        # exp_dir_pred = dir_pred.reshape(dir_pred.shape[0],-1,5)        
        # # remap to [batch , sqeucen, feature]  -> [batch x sqeucen, feature + 1 (temporal encoding)]        
        exp_aug_pred = aug_pred.reshape(-1,6)                        
        return self.gp_layer(exp_aug_pred)
                
