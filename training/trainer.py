import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import shutil
import time

# # Assuming these are imported from sibling modules
# from config import Config
from utils.logging import TensorBoardLogger # Placeholder for basic logging setup

class Trainer:
    """
    A unified Trainer class to handle the deep learning training lifecycle.
    """
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 optimizer: Optimizer, scheduler, criterion,**args):
        
        # Core Components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 如果有多张GPU,在to device前
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.epochs = args['epochs']
        log_dir='./experiments/logs/'
        self.exp_name=args['model_name']+'_'+args['pde_name']+'_'+args['loss']
        self.logger=TensorBoardLogger(log_dir,self.exp_name)
        self.checkpoint_dir=f'./experiments/checkpoints'
        if not os.path.exists('./experiments/checkpoints'):
            os.makedirs('./experiments/checkpoints')

    def _save_checkpoint(self, epoch: int):
        """Saves the current model and optimizer state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.exp_name}_best.pth")
        
        # State dictionary to save
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(state, checkpoint_path)
        
        # if is_best:
        #     best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        #     shutil.copyfile(checkpoint_path, best_path)
        #     self.logger.info(f"New best model saved to {best_path}")

    def _load_checkpoint(self, path: str):
        """Loads model and optimizer state from a checkpoint path."""
        # if not os.path.exists(path):
        #     self.logger.warning(f"No checkpoint found at {path}. Starting training from epoch 1.")
        #     return

        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Resuming training from epoch {self.start_epoch}")

    def train(self, epoch: int):
        """Runs a single training loop over the training data."""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for x,y,grid in self.train_loader:
            x,y,grid = x.to(self.device),y.to(self.device),grid.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x,grid)
            
            loss = self.criterion(output, y.reshape(output.shape))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
        self.scheduler.step()
            # --- Per-Iteration Logging ---
            # global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            # if batch_idx % self.log_interval == 0:
            #     self.writer.add_scalar('Loss/Train_Iter', loss.item(), global_step)
            #     self.logger.info(
            #         f"Epoch {epoch}/{self.cfg.training.epochs} | "
            #         f"Batch {batch_idx}/{len(self.train_loader)} | "
            #         f"Loss: {loss.item():.4f}"
            #     )
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        end_time = time.time()
        
        # --- Per-Epoch Logging ---
        self.logger.log_scalar('Loss/Train_Epoch', avg_loss, epoch)
        self.logger.info(f"Epoch {epoch} finished. Avg Train Loss: {avg_loss:.4f}. Time: {end_time - start_time:.2f}s")
        return avg_loss

    def validate(self, epoch: int):
        """Runs validation and returns the average loss."""
        self.model.eval()
        
        metrics=[1,2,float('inf')]
        total_loss = torch.zeros(len(metrics))
        avg_loss = torch.zeros(len(metrics))
        
        with torch.no_grad():
            for x,y,grid in self.train_loader:
                x,y,grid = x.to(self.device),y.to(self.device),grid.to(self.device)
                output = self.model(x,grid).reshape(y.shape)
                for i in range((len(metrics))):
                    if metrics[i]==1:
                        total_loss[i]+=torch.norm(output-y,p=1,dim=1).sum().item()
                    elif metrics[i]==2:
                        total_loss[i]+=torch.norm(output-y,p=2,dim=1).sum().item()
                    elif metrics[i]==float('inf'):
                        total_loss[i]+=torch.max(torch.abs(output-y),dim=1).values.sum().item()
            
        for i in range(len(metrics)):        
            avg_loss[i] = total_loss[i] / len(self.val_loader.dataset)
            
            # --- Per-Epoch Logging ---
            self.logger.log_scalar(f'L{metrics[i]}Loss/Validation', avg_loss[i], epoch)
            self.logger.info(f"L{metrics[i]}Loss: {avg_loss[i]:.4f}")
        
        return avg_loss

    def run(self, resume_path=''):
        """The main training loop orchestrator."""
        
        if resume_path:
            self._load_checkpoint(resume_path)
        
        pbar=tqdm(range(self.start_epoch,self.epochs), dynamic_ncols=True, smoothing=0.05)
        for epoch in pbar:
            # 1. Train
            train_loss=self.train(epoch)
            
            # 2. Validate
            val_l1,val_l2,val_linf = self.validate(epoch)
            pbar.set_description(f"[Epoch {epoch}] train_loss: {train_loss:.5e} val_loss: {val_l2:.5e}")
            # 3. Checkpointing
            is_best = val_l2 < self.best_val_loss
            if is_best:
                self.best_val_loss = val_l2
                self._save_checkpoint(epoch)
        
        self.logger.info("Training complete. Shutting down TensorBoard writer.")
        self.logger.close()





    