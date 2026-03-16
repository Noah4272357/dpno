import argparse
import torch
import torch.nn as nn
from data.dataset import get_dataloader

from training.trainer import Trainer
from training.losses import *
from training.soft_dtw_cuda import SoftDTW

def main(args):
    # 1. Instantiate components
    if args.model_name.endswith('FNO'):
        from models.fno import get_model
    elif args.model_name.endswith('DeepONet'):
        from models.deeponet import get_model
    elif args.model_name.endswith('WNO'):
        from models.wno import get_model
    elif args.model_name=='MLP':
        from models.mlp import get_model
    else:
        raise NotImplementedError('Model does not exist')
    # elif args.model_name=='Transolver':
    #     from models.transolver import get_model
    model = get_model(args.pde_name,args.model_name,fno_block=4)
    train_loader, val_loader = get_dataloader(pde_name=args.pde_name,batch_size=args.batch_size)
    
    # 2. Choose standard components
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    
    if args.loss=='relative_l2':
        criterion = LpLoss(size_average=True)
    elif args.loss=='softdtw':
        criterion = SoftDTW(use_cuda=True, gamma=0.1, normalize=True)
    else:
        criterion = wLpLoss(size_average=True)
    
    # 3. Instantiate and run the Trainer
    trainer = Trainer(model, train_loader, val_loader, 
                      optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                      model_name=args.model_name,pde_name=args.pde_name,loss=args.loss,
                      epochs=args.epochs)
    
    # Start training (you can pass the path to a previous checkpoint here)
    trainer.run()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Set parameters")
    parser.add_argument("--pde_name", type=str, default="Izhikevich", help="PDE name")
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--model_name",type=str,default="FNO")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--loss', type=str, default='softdtw')
    args= parser.parse_args()
    main(args)