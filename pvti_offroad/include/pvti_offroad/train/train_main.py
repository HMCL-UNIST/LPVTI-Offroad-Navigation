
#!/usr/bin/env python3
from pvti_offroad.common.file_utils import *
import torch
from torch.utils.data import Dataset, DataLoader
from pvti_offroad.train.train_utils import SampleGenerator
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import gpytorch
from tensorboardX import SummaryWriter
import time
import datetime
from pvti_offroad.gp_encoder.gp_encoderModel import GPAUC

    
GP_train = False
train_all = True
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        liklihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"{train_log_dir}/single_process_{current_time}"

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model = model.to(self.gpu_id).float()
        self.likelihood = liklihood.to(self.gpu_id).float()               
        self.train_data = train_data                     
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer, num_data=len(self.train_data.dataset)*9).to(self.gpu_id) 
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = os.path.join(snapshot_dir, snapshot_path)     
     
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)
            
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.model = DDP(self.model, device_ids=[self.gpu_id])
     

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.likelihood = snapshot["Liklihood"]
        self.model.train_norm_stat = snapshot["Norm_stat"]        
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _plot_model_performance_to_tensorboard(self,epoch, mu, sigma, ground_truth_pred_pose_residuals):        
        # writer.add_histogram('Normal Distribution', data, global_step=0)
        ground_truth_pred_pose_residuals = ground_truth_pred_pose_residuals.cpu()
        mu = mu.cpu()
        mean_error = (mu- ground_truth_pred_pose_residuals)        
        unpack_temporal_mean_error = mean_error.view(-1,9,mean_error.shape[1])        
        # unpack_temporal_mean_error = mean_error.view(mean_error.shape[0],9,-1)
        # writer.add_histogram('Normal Distribution', data, global_step=0)                    
        for sequence_idx in range(unpack_temporal_mean_error.shape[1]):
            for feature_idx in range(unpack_temporal_mean_error.shape[2]):
                sequence_feature_data = unpack_temporal_mean_error[:, sequence_idx, feature_idx]
                self.writer.add_histogram(f'Feature_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)
        if sigma is not None:
            unpack_temporal_sigma = sigma.view(-1,9,sigma.shape[1])
            for sequence_idx in range(unpack_temporal_sigma.shape[1]):
                for feature_idx in range(unpack_temporal_sigma.shape[2]):
                    sequence_feature_data = unpack_temporal_sigma[:, sequence_idx, feature_idx]
                    self.writer.add_histogram(f'Sigma_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)

 
    def _run_batch(self, source, targets,epoch):
        
        (state,action_predictions, xhat, colors, depths, concat_image)  = source                      
        intput_source = (xhat[:,:9,:].to(self.gpu_id), concat_image.to(self.gpu_id))

        with gpytorch.settings.use_toeplitz(False):
            
            self.model.train()
            self.likelihood.train()            
            self.optimizer.zero_grad()
     
            gpoutput = self.model(intput_source) 
            gt = targets.reshape(-1,targets.shape[-1]).to(self.gpu_id)                        
            loss = -self.mll(gpoutput, gt)   
            loss.backward()
            self.optimizer.step()                        
            self.writer.add_scalar("loss_total", float(loss), epoch)               
          
        for name, param in self.model.named_parameters():            
            self.writer.add_histogram(name + '/grad', param.grad, global_step=epoch)         
            if 'weight' in name:
                self.writer.add_histogram(name + '/weight', param, epoch)
            if 'bias' in name:
                self.writer.add_histogram(name + '/bias', param, epoch)
     
        with torch.no_grad(), gpytorch.settings.fast_pred_var():        
            
            mx = gpoutput.mean.cpu()
            std = gpoutput.stddev.detach()                        
            self._plot_model_performance_to_tensorboard(epoch, mx, std, gt)
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data))[0][0].shape[0]
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if hasattr(os.environ,"LOCAL_RANK"):
            self.train_data.sampler.set_epoch(epoch)
        total_loss = 0.0
        count = 0
        for source, targets in self.train_data:                            
            loss_bath = self._run_batch(source, targets,epoch)
            total_loss+=loss_bath
            count+=1
            print(f" Current batch count = {count} | Epoch {epoch} | Batchsize: {b_sz} | BATCH LOSS: {loss_bath:.6f}")


        avg_loss_non_torch = total_loss / (count+1)        
        self.writer.add_scalar('LTATT Loss/Train', avg_loss_non_torch, epoch + 1)        
        print(f" Epoch {epoch} | Batchsize: {b_sz} | AVG_LOSS: {avg_loss_non_torch:.6f}")
        

 

    def _save_snapshot(self, epoch):        
    
        if hasattr(os.environ,"LOCAL_RANK"):
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
        else:
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "EPOCHS_RUN": epoch,
                "Args": self.model.args,
                "Liklihood": self.likelihood,
                "Norm_stat": self.model.train_norm_stat
            }
        
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:                
                if train_all:
                    self._save_snapshot(epoch)
                else:
                    if GP_train:
                        self._save_snapshot(epoch)
                


def load_train_objs(args):
    preprocessed_dataset_load = False
    preprocessed_data_path = os.path.join(preprocessed_dir, f'preprocessed_data_669.pth')                    
    dirs = [train_dir] 
    if preprocessed_dataset_load:
        data_path = preprocessed_data_path 
        sampGen = SampleGenerator(dirs, data_path = data_path)
    else:
        sampGen = SampleGenerator(dirs)
    train_set = sampGen.get_dataset()

    model = GPAUC(args= args, train_norm_stat= train_set.get_norm_stats())

    liklihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=5)  
                
    lr = 0.01
    lr_gp = 0.01
    
    optimizer = Adam([        
            {'params': model.gp_layer.parameters(), 'lr': lr_gp},
            {'params': model.rnn.parameters(), 'lr': lr },                
            {'params': liklihood.parameters(), 'lr': lr_gp}
            ], lr=lr)
    
    return train_set, model, optimizer, liklihood


def prepare_dataloader(dataset: Dataset, batch_size: int):
    if hasattr(os.environ,"LOCAL_RANK"):
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=False,sampler=DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=True)
    return dataloader


def main(save_every: int, total_epochs: int, batch_size: int):
    # ddp_setup()
    args = {'input_grid_width':153,
            'input_grid_height':128,
            'n_time_step':10, 
            'lstm_hidden_size': 20,              
            'input_state_dim':5, # [vx, vy, wz, roll, pitch] 
            'input_action_dim':2, # [ax, delta]  
            'conv_to_feedforwad_dim': 20,               
            'batch_size':2,
            'num_epochs': 2,
            'output_residual_dim': 5, # [delx, dely, del_vx, del_vy, del_wz]
            'distributed_train': False,
            'arnp_train': True,
            'auclstm_output_dim': 5,
            'auclstm_out_fc_hidden_size': 28
            }
    
    snapshot_path = 'singl_aucgp_snapshot.pth'
    
    dataset, model, optimizer,liklihood = load_train_objs(args)
    train_data = prepare_dataloader(dataset, batch_size)    
    trainer = Trainer(model, train_data, optimizer, liklihood, save_every, snapshot_path)
    trainer.train(total_epochs)
    

if __name__ == "__main__":
    save_every = 50
    total_epochs = 10000
    batch_size = 160
    main(save_every, total_epochs, batch_size)