import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(pde_name,model_name,**args):
    if model_name=='FNO':
        if pde_name=='Darcy_Flow':
            model=FNO2d(num_blocks=args['fno_block'])
        elif pde_name.endswith('HH'):
            model=FNO1d(num_blocks=args['fno_block'])
    elif model_name=='DPNO':
        if pde_name=='Darcy_Flow':
            model=DPFNO2d()
        elif pde_name=='Navier_Stokes_2D1':
            model = DPFNO3d()
    return model

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))#66, 34
        #self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 258, 130, dtype=torch.cfloat))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        #out_ft=self.compl_mul2d(x_ft,self.weights)
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

# class SpectralConv3dhigh(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
#         super(SpectralConv3d, self).__init__()

#         """
#         3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2
#         self.modes3 = modes3

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
#         self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul3d(self, input, weights):
#         # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
#         return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, self.modes1:self.modes1+8, self.modes2:self.modes2+8, :self.modes3], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1-8:-self.modes1, self.modes2:self.modes2+8, :self.modes3], self.weights2)
#         out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, self.modes1:self.modes1+8, -self.modes2-8:-self.modes2, :self.modes3], self.weights3)
#         out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1-8:-self.modes1, -self.modes2-8:-self.modes2, :self.modes3], self.weights4)

#         #Return to physical space
#         x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
#         return x

class FNOBlock1d(nn.Module):
    def __init__(self, modes=12, width=128):
        super(FNOBlock1d,self).__init__()
        self.modes = modes
        self.width = width
        
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)
        
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x

# class FNO1d(nn.Module):
#     def __init__(self, in_channels=2,out_channels=101,out_grid=1,modes=12, width=20, lift_dim=200, num_blocks=4):
#         super(FNO1d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes = modes
#         self.width = width
#         self.out_grid=out_grid
#         self.padding = 2 # pad the domain if input is non-periodic
#         self.num_blocks=num_blocks
#         self.fc0 = nn.Linear(in_channels, self.width) # input channel is 2: (a(x), x)

#         self.blocks = nn.ModuleList([FNOBlock1d(self.modes,self.width) for _ in range(self.num_blocks)])  
        
        
#         self.fc1 = nn.Linear(self.width, lift_dim)
#         self.fc2 = nn.Linear(lift_dim, out_channels)

#     def forward(self, x, grid):
#         # x dim = [b, x1, t*v]
#         x = torch.stack((x, grid),dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 2, 1)
        
#         x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

#         for block in self.blocks:
#             x=block(x)
        
#         x = x[..., :-self.padding]
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

class FNO1d(nn.Module):
    def __init__(self, in_channels=2,out_channels=1,out_grid=1,modes=16, width=64, lift_dim=128, num_blocks=4):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes = modes
        self.width = width
        self.out_grid=out_grid
        self.padding = 2 # pad the domain if input is non-periodic
        self.num_blocks=num_blocks
        self.fc0 = nn.Sequential(nn.Linear(in_channels, 32), # input channel is 2: (a(x), x)
                                nn.GELU(),
                                nn.Linear(32,self.width)
                                )
        self.blocks = nn.ModuleList([FNOBlock1d(self.modes,self.width) for _ in range(self.num_blocks)])  
        
        
        self.fc1 = nn.Sequential(nn.Linear(self.width, lift_dim),
                                 nn.GELU(),
                                nn.Linear(lift_dim, self.width),
                                nn.GELU(),
                                nn.Linear(self.width,out_channels))
        
        #self.fc3=nn.Linear(101,out_grid)

    def forward(self, x, grid):
        # x dim = [b, x1, t*v]
        x = torch.stack((x, grid),dim=-1)
        # mean = x.mean(dim=0, keepdim=True)
        # std = x.std(dim=0, keepdim=True)
        # x= (x - mean) / (std + 1e-8)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        for block in self.blocks:
            x=block(x)
        
        x = x[..., :-self.padding]
        # if self.out_grid!=1:
        #     x=self.fc3(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x



class FNOBlock2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=20):
        super(FNOBlock2d,self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        #self.conv = nn.Conv2d(self.width,self.width,3,padding=1)
        self.w = nn.Conv2d(self.width, self.width, 1)
        self.bc=nn.BatchNorm2d(width)
        
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1+x2
        x = self.bc(x)
        x = F.gelu(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=1,modes1=12, modes2=12, width=20, lift_dim=64,num_blocks=4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.blocks = nn.ModuleList([FNOBlock2d(self.modes1,self.modes2,width) for _ in range(num_blocks)])  
        
        self.fc1 = nn.Linear(width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            x=block(x)

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)
    
class FNOBlock3d(nn.Module):
    def __init__(self, modes1=8, modes2=8, modes3=8,  width=10):
        super(FNOBlock3d,self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        self.conv = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        #self.conv2 = SpectralConv3dhigh(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w = nn.Conv3d(self.width, self.width, 1)
        
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, in_channels=4,out_channels=1, modes1=8, modes2=8, modes3=8, width=10, num_blocks=12):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        self.blocks = nn.ModuleList([FNOBlock3d(self.modes1,self.modes2,self.modes3,width) for _ in range(num_blocks)])  
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, x3, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        for block in self.blocks:
            x=block(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class DenseFNO1d(nn.Module):
    def __init__(self, in_channels=2,out_channels=1, modes=12, width=20, lift_dim=200, num_blocks=3):
        super(DenseFNO1d, self).__init__()


        self.modes = modes
        self.width = width
        self.num_blocks=num_blocks
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        layer=[]
        for i in range(self.num_blocks):
            layer.append(FNOBlock1d(self.modes, width))
            width=width*2
        
        self.blocks = nn.ModuleList(layer)  
        
        self.fc1 = nn.Linear(width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    def forward(self, x, grid):
        x = torch.stack((x, grid),dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding])

        for block in self.blocks:
            y=block(x)
            x=torch.cat((x,y),dim=1)

        x = x[..., :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.unsqueeze(-2)


class DenseFNO2d(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=1, modes1=12, modes2=12, width=20, lift_dim=200, num_blocks=3):
        super(DenseFNO2d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_blocks=num_blocks
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        layer=[]
        for i in range(self.num_blocks):
            layer.append(FNOBlock2d(self.modes1,self.modes2,width))
            width=width*2
        
        self.blocks = nn.ModuleList(layer)  
        
        self.fc1 = nn.Linear(width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            y=block(x)
            x=torch.cat((x,y),dim=1)


        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)

class ResFNO3d(nn.Module):
    def __init__(self,  in_channels=4,out_channels=1, modes1=8, modes2=8, modes3=8, width=20, num_blocks=2,num_res=4):
        super(ResFNO3d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_blocks=num_blocks
        self.num_res=num_res
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        
        for i in range(self.num_res):
            layer=[]
            for _ in range(self.num_blocks):
                layer.append(FNOBlock3d(self.modes1,self.modes2,self.modes3,width))
                layer.append(nn.BatchNorm3d(self.width))
            
            exec(f'self.blocks{i+1} = nn.ModuleList(layer)')  

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)# input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        

    def forward(self, x, grid):
        # x dim = [b, x1, x2, x3, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        for i in range(self.num_res):
            y = x.clone()
            exec(f'module=self.blocks{i+1}')
            for block in self.layer:
                y=block(y)
            x=x+y

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
        

class DPFNO1d(nn.Module):
    def __init__(self,in_channels=2,out_channels=1,out_grid=1, modes=12, width=20, lift_dim=200,num_res=4, num_dense=3):
        super(DPFNO1d, self).__init__()


        self.modes = modes
        self.width = width
        self.out_grid=out_grid

        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        self.res_path=nn.ModuleList()
        self.dense_path=nn.ModuleList()

        for i in range(num_res):
            self.res_path.append(FNOBlock1d(self.modes, width))

        for i in range(num_dense):
            self.dense_path.append(FNOBlock1d(self.modes, width))
            width=width*2
        
         
        
        self.fc1 = nn.Linear(width+self.width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    def forward(self, x, grid):
        x = torch.stack((x, grid),dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding])

        x1=x.clone()
        for net in self.res_path:
            y=net(x)
            x=x+y
        
        for net in self.dense_path: 
            y=net(x1)
            x1=torch.cat((x1,y),dim=1)

        x=torch.cat((x,x1),dim=1)

        x = x[..., :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

class DPFNO1d_Darcy(nn.Module):
    def __init__(self,in_channels=2,out_channels=1,out_grid=1, modes=12, width=20, lift_dim=200,num_res=4, num_dense=3):
        super(DPFNO1d, self).__init__()


        self.modes = modes
        self.width = width
        self.out_grid=out_grid

        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        self.res_path=nn.ModuleList()
        self.dense_path=nn.ModuleList()

        for i in range(num_res):
            self.res_path.append(FNOBlock1d(self.modes, width))

        for i in range(num_dense):
            self.dense_path.append(FNOBlock1d(self.modes, width))
            width=width*2
        
         
        
        self.fc1 = nn.Linear(width+self.width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)
        self.fc3=nn.Linear(101,out_grid)

    def forward(self, x, grid):
        x = torch.stack((x, grid),dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding])

        x1=x.clone()
        for net in self.res_path:
            y=net(x)
            x=x+y
        
        for net in self.dense_path: 
            y=net(x1)
            x1=torch.cat((x1,y),dim=1)

        x=torch.cat((x,x1),dim=1)

        x = x[..., :-self.padding] # Unpad the tensor
        if self.out_grid!=1:
            x=self.fc3(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

class DPFNO2d(nn.Module):
    def __init__(self, in_channels=3 ,out_channels=1, modes1=12, modes2=12, width=20, lift_dim=200,num_res=4,num_dense=3):
        super(DPFNO2d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        self.res_path=nn.ModuleList()
        self.dense_path=nn.ModuleList()

        for i in range(num_res):
            self.res_path.append(FNOBlock2d(self.modes1,self.modes2,width))
        
        for i in range(num_dense):
            self.dense_path.append(FNOBlock2d(self.modes1,self.modes2,width))
            width=width*2
        
        
        self.fc1 = nn.Linear(width+self.width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1=x.clone()
        for net in self.res_path:
            y=net(x)
            x=x+y

        for net in self.dense_path:
            y=net(x1)
            x1=torch.cat((x1,y),dim=1)
        
        x=torch.cat((x,x1),dim=1)
            
        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x      


class DPFNO3d(nn.Module):
    def __init__(self,  in_channels=4,out_channels=1, modes1=8, modes2=8, modes3=8, width=20, num_res=4,num_dense=3):
        super(DPFNO3d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_res=num_res
        self.num_dense=num_dense
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, self.width)
        
        self.res_path=nn.ModuleList()
        self.dense_path=nn.ModuleList()

        for i in range(num_res):
            self.res_path.append(FNOBlock3d(self.modes1,self.modes2,self.modes3,width))
        
        for i in range(num_dense):
            self.dense_path.append(FNOBlock3d(self.modes1,self.modes2,self.modes3,width))
            width=width*2
        
            
        self.fc1 = nn.Linear(width+self.width, 200)
        self.fc2 = nn.Linear(200, out_channels)# input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        
        

    def forward(self, x, grid):
        # x dim = [b, x1, x2, x3, t*v]
        x=x.unsqueeze(-1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        
        x1=x.clone()
        for net in self.res_path:
            y=net(x)
            x=x+y

        for net in self.dense_path:
            y=net(x1)
            x1=torch.cat((x1,y),dim=1)
        
        x=torch.cat((x,x1),dim=1)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
        


         
class DNFNO2d(nn.Module):
    # activation: str = "gelu",
    #     kernel_initializer: str = "Glorot normal",
    #     size=64,timestep=10,
    def __init__(self,in_channels=12 ,out_channels=1,
        modes1=12, modes2=12, 
        width=20, lift_dim=256,num_blocks=4):
        super(DNFNO2d, self).__init__()

        # self.branch=MLP(size**2*timestep,lift_dim,layer_sizes=[256]*3,activation=activation,kernel_initializer=kernel_initializer)
        # self.trunk=MLP(2,lift_dim,layer_sizes=[64]*3+[128],activation=activation,kernel_initializer=kernel_initializer)

        # self.b = torch.nn.parameter.Parameter(torch.zeros(out_channels,dtype=torch.float32))
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = []  
        for _ in range(num_blocks):
            self.blocks.append(FNOBlock2d(self.modes1,self.modes2,width))
            #self.blocks.append(nn.BatchNorm2d(width))
        self.blocks=nn.ModuleList(self.blocks)
        self.fc1 = nn.Linear(width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)
        
    def forward(self, x, grid_y):
        # batchsize=x.shape[0]
        # basis = grid_y[0]
        # x = self.branch(x.reshape([batchsize,-1]))
        # # Trunk net to encode the domain of the output function
        # basis = self.trunk(basis)
        
        # x = torch.einsum("ni,bci->nbc", x, basis)
        # x += self.b
        
        # x=x.unsqueeze(-1)

        x = torch.cat((x, grid_y), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x = F.pad(x, [0, self.padding, 0, self.padding]) # pad the domain if input is non-periodic
        for block in self.blocks:
            x=block(x)
            # y=block(x)
            # x=x+y

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x