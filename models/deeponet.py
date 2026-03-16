# coding=utf-8
import torch
import torch.nn as nn
from .model_utils import _get_act, _get_initializer,MLP

def get_model(pde_name,model_name,**args):
    if model_name=='DPDeepONet':
        if pde_name=='Burgers' or pde_name.endswith('Darcy'):
            model = DPDON1D(size=256,
                        query_dim= 2)
        elif pde_name=='Darcy_Flow':
            model = DPDON2D(size=128,
                        query_dim= 2)
        elif pde_name=='Navier_Stokes_2D' or pde_name=='Irregular_NS':
            model = DPDON3D(size=64,
                        query_dim= 3,
                        time_step=10)
    elif model_name=='DeepONet':
        if pde_name=='Burgers' :
            model = DeepONet1D(size=256,
                        query_dim= 2)
        elif pde_name=='HH':
            model = DeepONet1D(size=1001,
                               query_dim=1)
        elif pde_name=='Darcy_Flow':
            model = DeepONet2D(size=128,
                        query_dim= 2)
        elif pde_name=='Navier_Stokes_2D':
            model = DeepONet3D(size=64,
                        query_dim= 3,
                        time_step=10)        
    return model

class DeepONet1D(nn.Module):
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size :int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()
        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[64]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0].unsqueeze(-1)
        # Branch net to encode the input function
        x = self.branch(x)
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))

        x=torch.mm(x,grid.transpose(0,1))
        #x = torch.einsum("bhi,rh->bri", x, grid)
        # Add bias
        x += self.b
        return x

class DeepONet2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=64*out_channel
        layer_sizes=[64]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))
        
        x = x.reshape([batchsize,self.out_channel,-1])
        grid = grid.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, grid)
        
        # Add bias
        x += self.b

        return x

class DeepONet3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=64*out_channel
        layer_sizes=[64]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))
        
        x = x.reshape([batchsize,self.out_channel,-1])
        grid = grid.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, grid)
        # Add bias
        x += self.b

        return x

class DenseDON1D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_layers: int = 3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        self.query_dim=query_dim
        self.out_channel = out_channel
        self.num_layers = num_layers

        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        
        self.trunk_list = torch.nn.ModuleList()

        self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_layers):
            self.trunk_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0]  
        num_points=grid.shape[0]
        grid=grid.reshape([num_points,self.query_dim])#(num_point, query_dim)
        batchsize=x.shape[0]

        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

            
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x     

        
class DenseDON2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_layers: int = 3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        self.out_channel = out_channel
        self.query_dim=query_dim
        layer_sizes = [128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.num_layers = num_layers
        self.trunk_list = torch.nn.ModuleList()

        self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_layers):
            self.trunk_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
            

        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x


class DenseDON3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        num_layers:int =3,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

        self.num_layers = num_layers
        self.branch_list = torch.nn.ModuleList()
        self.trunk_list = torch.nn.ModuleList()

        for i in range(num_layers):
            self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        
            if i>0:
                self.branch_list.append(MLP(in_channel_branch*i, out_channel_branch, layer_sizes, activation_branch, kernel_initializer) )
            else:
                self.branch_list.append(MLP(in_channel_branch, out_channel_branch, layer_sizes, activation_branch, kernel_initializer) )
            

        
       
    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x

class DPDON1D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_res: int = 4,
        num_dense: int=3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        self.query_dim=query_dim
        self.out_channel = out_channel
        self.num_res = num_res
        self.num_dense = num_dense

        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=64*out_channel
        layer_sizes=[64]*3
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.param_list=torch.nn.ModuleList()
        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()

        for i in range(num_res):
            if i==0:
                self.res_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            else:
                self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            self.param_list.append(nn.Conv2d(out_channel_trunk,out_channel_trunk,1))

        self.dense_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_dense):
            self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0]  
        num_points=grid.shape[0]
        #grid=grid.reshape([num_points,self.query_dim])#(num_point, query_dim)
        batchsize=x.shape[0]

        x = self.branch(x.reshape([batchsize,-1]))


        
        basis= self.dense_list[0](grid)
        for i in range(1,self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=2)

        basis1=self.res_list[0](grid)
        for i in range(1,self.num_res):
            new_basis = self.res_list[i](basis1)
            new_basis = new_basis.permute(2,0,1)
            new_basis=self.param_list[i](new_basis)
            new_basis = new_basis.permute(1,2,0)
            basis1=basis1+new_basis
            
        
        basis=torch.cat((basis,basis1),dim=2)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        if len(grid.shape)==2:
            basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x     

        
class DPDON2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_res: int = 4,
        num_dense: int=3 ,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=64*out_channel
        self.num_res = num_res
        self.num_dense=num_dense
        self.out_channel = out_channel
        self.query_dim=query_dim
        layer_sizes = [64]*3
        branch_layer=[100]*5#[64]*5
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        
        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()
        #self.param_list = torch.nn.ModuleList()
        
        for i in range(num_res):
            #self.param_list.append(nn.BatchNorm1d(64*out_channel))
            if i==0:
                self.res_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            else:
                self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            
        for i in range(num_dense):
            #self.res_bn_list.append(nn.BatchNorm1d(64*out_channel))
            if i==0:
                self.dense_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            else:
                self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
                out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, branch_layer, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        #grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))

        
        basis= self.dense_list[0](grid)
        for i in range(1,self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=2)

        basis1=self.res_list[0](grid)
        for i in range(1,self.num_res):
            new_basis = self.res_list[i](basis1)
            #new_basis = self.res_bn_list[i](new_basis)
            basis1=basis1+new_basis
        
        basis=torch.cat((basis,basis1),dim=2)
        
        #x = x.reshape([batchsize,self.out_channel,-1])
        #basis = basis.reshape([num_points,self.out_channel,-1])
        x=torch.einsum("bi,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x


class DPDON3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        num_res: int = 4,
        num_dense: int=3,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=64*out_channel
        layer_sizes=[64]*3
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.num_res = num_res
        self.num_dense=num_dense
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()
        #self.param_list=torch.nn.ModuleList()
        
        self.trunk=MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) 
        for i in range(num_res):
            self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            #self.param_list.append(nn.Conv1d(out_channel_trunk,out_channel_trunk,1))

        for i in range(num_dense):
            self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2

        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        
       
    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        basis= self.activation_trunk(self.trunk(grid))
        basis1=basis.clone()

        for i in range(self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

        
        for i in range(self.num_res):
            new_basis = self.res_list[i](basis1)
            basis1=basis1+new_basis
        
        basis=torch.cat((basis,basis1),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x 

class DualDON2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_res: int = 4,
        num_dense: int=3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        self.out_channel = out_channel
        self.query_dim=query_dim
        layer_sizes=[128]*3
        self.num_res=num_res
        self.num_dense=num_dense
        self.res_branch_list = torch.nn.ModuleList()
        self.res_trunk_list = torch.nn.ModuleList()
        self.dense_branch_list = torch.nn.ModuleList()
        self.dense_trunk_list = torch.nn.ModuleList()
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)
        for i in range(num_res):
            self.res_branch_list.append(MLP(in_channel_branch,out_channel_trunk,layer_sizes, activation_branch, kernel_initializer))
            self.res_trunk_list.append(MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer))
        for i in range(num_dense):
            self.dense_branch_list.append(MLP(in_channel_branch,out_channel_trunk,layer_sizes, activation_branch, kernel_initializer))
            self.dense_trunk_list.append(MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer))

        self.post=MLP(2**num_dense+1,out_channel,layer_sizes, self.activation_trunk, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        basis = self.activation_trunk(self.trunk(grid))

        x = torch.mm(x,basis.T)

        x_res = x.clone()
        x_dense=x.clone()
        x_dense=x_dense.unsqueeze(dim=1)

        for i in range(self.num_res):
            param=self.res_branch_list[i](x_res)
            basis1=self.res_trunk_list[i](grid)
            x2 = torch.mm(param,basis1.T)
            x_res = x_res + x2

        for i in range(self.num_dense):
            param=self.dense_branch_list[i](x_dense)
            basis2=self.dense_trunk_list[i](grid)
            x2 = torch.matmul(param,basis2.T)
            x_dense =torch.cat((x_dense,x2),dim=1)

        x_res=x_res.unsqueeze(dim=1)

        x=torch.cat((x_res,x_dense),dim=1)
        x=x.permute(0,2,1)
        x=self.post(x)
        # Add bias
        x += self.b

        return x
