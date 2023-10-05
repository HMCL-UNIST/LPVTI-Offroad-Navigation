
#!/usr/bin/env python3
from pvti_offroad.common.file_utils import *
import torch
import os
from pvti_offroad.gp_encoder.gp_encoderModel import GPAUC
import gpytorch  

class AUCEStimator:
    def __init__(self, dt = 0.2, N_node = 10, model_path = 'singl_aucgp_snapshot.pth') -> None:
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        assert model_path is not None, 'need to load proper model first'
        self.model = None
        self.likelihood = None
        file_path = os.path.join(snapshot_dir, model_path)                            
        assert os.path.exists(file_path), f"Cannot find GPAUC model at {file_path}"
        print("Loading snapshot")
        self.load_model(file_path)        
        
        

        
    def load_model(self,file_path):     
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(file_path, map_location=loc)
        # TODO: need to save args while training   
        self.args = snapshot["Args"]        
        self.model = GPAUC(args=self.args)        
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model.to(self.gpu_id).float()
        self.model.eval()
        self.likelihood = snapshot["Liklihood"]
        self.likelihood.eval()
        self.model.train_norm_stat = snapshot["Norm_stat"]        
        print(f"Model has been loaded {file_path}")
        

    def pred(self, batch_xhat:torch.tensor, image:torch.tensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            batch_size = batch_xhat.shape[0]
            input_to_model = (batch_xhat.to(self.gpu_id), image.to(self.gpu_id))
            gpoutput = self.model.get_pred(input_to_model)
            normalized_pred_residual = self.likelihood(gpoutput)
            pred_residual_mean,pred_residual_std = self.model.outputToReal(batch_size,normalized_pred_residual)            
            return pred_residual_mean , pred_residual_std
    
        

# if __name__ == "__main__":    
#     auc_estimator = AUCEStimator(model_path = '/home/racepc/offroad_sim_data/models/island_grass/dist_process_1770.pth', dt = 0.2, N_node = 10)
    