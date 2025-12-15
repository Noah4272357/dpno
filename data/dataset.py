import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import TensorDataset,DataLoader,random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_dataloader(pde_name,ntrain=900,ntest=100,batch_size=64,seed=42,noise='None'):
# noise_name:'Gaussian' or 'Laplace'
# pde_name:'Darcy_Flow' 'Navier_Stokes_2D' 'Irregular_NS''Irregular_Darcy'
# 'Circle_Darcy'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pde_name=='Darcy_Flow':
        r=2
        s=128
        filepath='/data/wyc/data/new_darcyflow.mat'
        data = scio.loadmat(filepath)
        features=data['a']
        label=data['u']
        scaler=StandardScaler()
        features=scaler.fit_transform(features.reshape(-1,1)).reshape(features.shape)
        features = features[:ntrain+ntest, ::r, ::r]
        features = torch.from_numpy(features).float()
        labels = label[:ntrain+ntest, ::r, ::r]
        labels = torch.from_numpy(labels).float()
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        x, y = np.meshgrid(x, y)
        grid = np.stack((x,y),axis=-1)
        grid = torch.tensor(grid, dtype=torch.float)

        grids= grid.repeat(ntrain+ntest, 1, 1, 1)
    
    elif pde_name=='KS':
        filepath='/data/wyc/data/kuramoto_s.mat'
        r=1
        xstep=128
        tstep=101
        data = scio.loadmat(filepath)
        features=data['input']
        labels=data['output']
        features=np.repeat(np.expand_dims(features,axis=1),label.shape[1],axis=1)
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels)
        features = features[:, ::r, ::r]
        labels = labels[:, ::r, ::r]

        grid_x = np.linspace(0, 10, xstep)
        grid_t = np.linspace(0, 10, tstep)
        grid_x,grid_t=np.meshgrid(grid_x,grid_t)
        grid=np.stack([grid_x,grid_t],axis=2)
        grid = torch.tensor(grid, dtype=torch.float)

        grids = grid.repeat(ntrain+ntest, 1, 1, 1)
        

    elif pde_name=='Navier_Stokes_2D':
        filepath='/data/wyc/data/ns_2d'
        features=np.load(filepath+'/in_f.npy')
        label=np.load(filepath+'/out_f.npy')
        grid=np.load(filepath+'/grid.npy')
        grid = torch.tensor(grid, dtype=torch.float)


        x_train = features[:ntrain]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain]
        y_train = torch.from_numpy(y_train)

        x_test = features[-ntest:]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:]
        y_test = torch.from_numpy(y_test)

        grid_train = grid.repeat(ntrain, 1, 1, 1, 1)
        grid_test = grid.repeat(ntest, 1, 1, 1, 1)
    
    elif pde_name=='Navier_Stokes_2D1':
        T_in=10
        T=10
        filepath='/data/wyc/data/NavierStokes_V1e-5_N1200_T20.mat'
        data = scio.loadmat(filepath)
        features=data['u'][:,:,:,:T_in]
        label=data['u'][:,:,:,T_in:T_in+T]
        h = features.shape[1]

        x_train = features[:ntrain]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain]
        y_train = torch.from_numpy(y_train)

        x_test = features[-ntest:]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:]
        y_test = torch.from_numpy(y_test)

        
        x = np.linspace(0, 1, h,endpoint=False)
        y = np.linspace(0, 1, h,endpoint=False)
        #t_out = np.linspace(10, 20, 10,endpoint=False)
        # X, Y, t_in = np.meshgrid(x, y, t_in)
        # grid_x = np.stack((X,Y,t_in),axis=-1)
        # grid_x = torch.tensor(grid_x, dtype=torch.float)
        #X, Y, t_out = np.meshgrid(x, y, t_out)
        X, Y = np.meshgrid(x, y)
        grid = np.stack((X,Y),axis=-1)
        grid = torch.tensor(grid, dtype=torch.float)
        grid_train = grid.repeat(ntrain, 1, 1, 1)
        grid_test = grid.repeat(ntest, 1, 1, 1)
        # grid_train_x = grid_x.repeat(ntrain, 1, 1, 1, 1)
        # grid_test_x = grid_x.repeat(ntest, 1, 1, 1, 1)
        # grid_train_y = grid_y.repeat(ntrain, 1, 1, 1, 1)
        # grid_test_y = grid_y.repeat(ntest, 1, 1, 1, 1)
    
    elif pde_name.endswith('HH'):
        if pde_name.endswith('single_pulse_HH'):
            filepath='../data_folder/HH_model_data.npz'
        # features=data['I_store']
        # labels=data['X_store'][:,0,:]
        elif pde_name.endswith('sin_pulse_HH'):
            filepath='../data_folder/sin_HH_model_data.npz'
        elif pde_name.endswith('long_pulse_HH'):
            filepath='../data_folder/long_HH_model_data.npz'
        else:
            filepath='../data_folder/multi_HH_model_data.npz'
        data=np.load(filepath)
        if pde_name.startswith('inverse'):
            labels=data['Iapp']
            features=data['V']
        else:
            features=data['Iapp']
            labels=data['V']
        grid=np.linspace(0,100,features.shape[1]).reshape(1,-1)
        grids=np.repeat(grid,features.shape[0],axis=0)
        features,labels,grids=torch.from_numpy(features).float(),torch.from_numpy(labels).float(),torch.from_numpy(grids).float()     

    elif pde_name=='Burgers':
        filepath='/data/wyc/data/new_burgers.mat'
        #filepath2='/data/wyc/data/analytic_burgers.mat'
        # burgers parameters
        r=1
        xstep=256
        tstep=101
        data = scio.loadmat(filepath)
        #data1 = scio.loadmat(filepath2)
        features=data['input']
        labels=data['output']
        # features1=data1['input']
        # label1=data1['output']
        features=np.repeat(np.expand_dims(features,axis=1),labels.shape[1],axis=1)
        #features1=np.repeat(np.expand_dims(features1,axis=1),label.shape[1],axis=1)

        grid_x = np.linspace(0, 1, xstep)
        grid_t = np.linspace(0, 1, tstep)
        grid_x,grid_t=np.meshgrid(grid_x,grid_t)
        grid = np.stack([grid_x,grid_t],axis=2)
        grids=grid.repeat(ntrain+ntest, 1, 1, 1)
        features,labels,grids=torch.from_numpy(features).float(),torch.from_numpy(labels).float(),torch.from_numpy(grids).float()
    elif pde_name=='Schrodinger':
        filepath='/data/wyc/data/schrodinger.mat'
        r=1
        xstep=256
        tstep=101
        data = scio.loadmat(filepath)
        features=data['input']
        label=data['output']
        label=torch.from_numpy(label).float()
        labels=label.permute(0,2,1)
        features=np.repeat(np.expand_dims(features,axis=1),label.shape[1],axis=1)
        features = torch.from_numpy(features).float()
        
        
        grid_x = np.linspace(0, 1, xstep)
        grid_t = np.linspace(0, 1, tstep)
        grid_x,grid_t=np.meshgrid(grid_x,grid_t)
        grid=np.stack([grid_x,grid_t],axis=2)
        grid = torch.tensor(grid, dtype=torch.float)

        grids = grid.repeat(ntrain+ntest, 1, 1, 1)

    elif pde_name.endswith('Darcy'):
        if pde_name=='Irregular_Darcy':
            filepath='/data/neural/darcy_irregular/datasets/Darcy_Triangular.mat'
        else:
            filepath='/data/neural/darcy_circle/Darcy_Circle.mat'
        data = scio.loadmat(filepath)
        features=data['f_bc']
        label=data['u_field']
        grid=data['x_bc']
        grid=torch.from_numpy(grid).float()
        x_train = features[:ntrain, :]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain, :]
        y_train = torch.from_numpy(y_train)
        grid_train=grid.repeat(ntrain, 1)
        

        x_test = features[-ntest:, :]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:, :]
        y_test = torch.from_numpy(y_test)
        grid_test=grid.repeat(ntest, 1)
        
    elif pde_name=='weather':
        ntrain, nvalid, ntest = 1825, 500, 1325
        d = np.load("/data/wyc/data/weather_dataset.npz")
        x = d["U_train"]
        y = d["S_train"] / 1000.0
        x=x.reshape(-1,72,72)
        y=y.reshape(-1,72,72)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        gridx = np.linspace(0, 1, 72) 
        gridy = np.linspace(0, 1, 72) 
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*[gridx, gridy])]).T
        grid = grid.reshape(72, 72, 2)
        grid = torch.tensor(grid, dtype=torch.float)
        x_train = x[:ntrain, ...]
        x_valid = x[ntrain : ntrain + nvalid, ...]
        x_test = x[ntrain + nvalid : ntrain + nvalid + ntest, ...]
        y_train = y[:ntrain, ...]
        y_valid = y[ntrain : ntrain + nvalid, ...]
        y_test = y[ntrain + nvalid : ntrain + nvalid + ntest, ...]
        grid_train = grid.repeat(ntrain, 1, 1, 1)
        grid_test = grid.repeat(ntest, 1, 1, 1)

    elif pde_name=='Poisson':
        filepath='/data/wyc/data/poisson'
        features=np.load(filepath+'/in_f.npy')
        labels=np.load(filepath+'/out_f.npy')
        grid=np.load(filepath+'/grid.npy')
        grid = torch.tensor(grid, dtype=torch.float)


        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).float()

        grids = grid.repeat(ntrain+ntest, 1, 1, 1)


    if noise=='Laplace':
        delta=np.random.laplace(0,1,size=features.shape)
        delta=torch.from_numpy(delta).float()
        features+=0.01*torch.max(torch.abs(features))*delta

    elif noise=='Gaussian':
        delta=torch.randn_like(features)
        features+=0.01*torch.max(torch.abs(features))*delta

    # train_loader = DataLoader(TensorDataset(x_train, y_train, grid_train),
    #                                         batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(TensorDataset(x_test, y_test,grid_test),
    #                                         batch_size=batch_size, shuffle=False)
    dataset=TensorDataset(features,labels,grids)
    train_dataset, test_dataset = random_split(
            dataset, 
            [ntrain, ntest],
            generator=torch.Generator().manual_seed(seed)  # 设置随机种子保证可重复性
        )
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader

