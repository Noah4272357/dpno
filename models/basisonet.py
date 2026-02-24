import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.neighbors import KernelDensity
from torch.nn import Module, Parameter


def _parralleled_inner_product(f1, f2, h):
    prod = f1 * f2
    return torch.matmul((prod[:, :, :-1] + prod[:, :, 1:]), h) / 2


def trapezoidal_2d_parralleled(f, h):
    assert len(h) == 2
    _, _, l1, l2 = f.size()
    # l1, l2 = f.shape[2], f.shape[3]
    c = torch.ones((l1, l2), device=f.device)
    c[[0, -1], :] = 1 / 2
    c[:, [0, -1]] = 1 / 2
    c[[0, 0, -1, -1], [0, -1, 0, -1]] = 1 / 4
    return h[0] * h[1] * torch.sum(torch.mul(c, f), dim=(-2, -1))


def _parralleled_inner_product_2d(f1, f2, h):
    prod = f1 * f2
    return trapezoidal_2d_parralleled(prod, h)


class Basic_Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def load_params_from_file(self, filename, optimizer=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print(
            "==> Loading parameters from checkpoint %s to %s"
            % (filename, "CPU" if to_cpu else "GPU")
        )
        loc_type = torch.device("cpu")  # if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint["model_state"]
        if optimizer:
            optimizer_state_disk = checkpoint["optimizer_state"]
            optimizer.load_state_dict(optimizer_state_disk)
            # print("loaded optimizer")
        else:
            # print("optimizer is not loaded")
            pass

        update_model_state = {}
        for key, val in model_state_disk.items():
            if (
                key in self.state_dict()
                and self.state_dict()[key].shape == model_state_disk[key].shape
            ):
                update_model_state[key] = val
                print("Update weight %s: %s" % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                print("Not updated weight %s: %s" % (key, str(state_dict[key].shape)))
                pass

        print(
            "==> Done (loaded %d/%d)"
            % (len(update_model_state), len(self.state_dict()))
        )

    def freeze_basis(self, lr=1e-4, weight_decay=0, mode=None):
        total = 0
        freezed = 0
        for name, param in self.named_parameters():
            if name[:5] in mode:
                param.requires_grad = False
                freezed += 1
                print("Freeze " + name)
            total += 1
        print("==> Freezed (loaded %d/%d)" % (freezed, total))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer

    def check_orthogonality_in(self, path=None):
        T = self.t_in
        # evaluate the current basis nodes at time grid
        self.bases_in = self.BL_in(T)  # (J, n_base)
        orth_matrix = torch.ones((self.bases_in.shape[1], self.bases_in.shape[1])).to(
            self.device
        )
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = (
                    torch.einsum("s,s->", self.bases_in[:, i], self.bases_in[:, j])
                    / self.bases_in.shape[0]
                )
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix

    def check_orthogonality_out(self, path=None):
        T = self.t_out
        # evaluate the current basis nodes at time grid
        self.bases_out = self.BL_out(T)  # (J, n_base)
        orth_matrix = torch.ones((self.bases_out.shape[1], self.bases_out.shape[1])).to(
            self.device
        )
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = (
                    torch.einsum(
                        "s,s->",
                        self.bases_out[:, i] / self.density_out.squeeze(1),
                        self.bases_out[:, j],
                    )
                    / self.bases_out.shape[0]
                )
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix


class FNN(nn.Module):
    def __init__(self, hidden_layer=[64, 64], dim_in=-1, dim_out=-1, activation=None):
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden_layer + [dim_out]
        self.layers = nn.ModuleList(
            [nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))]
        )
        self.lns = nn.ModuleList([nn.LayerNorm(dim[i]) for i in range(1, len(dim) - 1)])

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.lns[i](x)
            x = self.sigma(x)
        # linear activation at the last layer
        return self.layers[-1](x)


class NeuralBasis(nn.Module):
    def __init__(self, dim_in=1, hidden=[4, 4, 4], n_base=4, activation=None):
        super().__init__()
        self.n_base = n_base
        self.sigma = activation
        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList(
            [nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))]
        )
        # self.lns = nn.ModuleList([nn.LayerNorm(dim[i]) for i in range(1, len(dim) - 1)])

    def forward(self, t):
        for i in range(len(self.layers) - 1):
            t = self.layers[i](t)
            # t = self.lns[i](t)
            t = self.sigma(t)
        # linear activation at the last layer
        x = self.layers[-1](t)
        # shape = x.shape
        # q, _ = torch.linalg.qr(x.reshape(-1, self.n_base))
        return x  # q.reshape(shape)


class BasisONet_1d1d(Basic_Model):
    def __init__(
        self,
        n_base_in=9,
        base_in_hidden=[64, 64, 64],
        middle_hidden=[64, 64, 64],
        n_base_out=9,
        base_out_hidden=[64, 64, 64],
        grid_in=None,
        grid_out=None,
        device="cuda",
        activation=F.gelu,
    ):
        super().__init__()
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.device = device
        assert grid_in.shape[-1] != 1 and grid_out.shape[-1] != 1
        self.h_in = torch.tensor(grid_in[1:] - grid_in[:-1]).to(device).float()
        self.h_out = torch.tensor(grid_out[1:] - grid_out[:-1]).to(device).float()
        self.t_in = torch.tensor(grid_in).to(device).float().reshape(-1, 1)
        self.t_out = torch.tensor(grid_out).to(device).float().reshape(-1, 1)
        self.BL_in = NeuralBasis(
            1, hidden=base_in_hidden, n_base=n_base_in, activation=activation
        )
        self.Middle = FNN(
            hidden_layer=middle_hidden,
            dim_in=n_base_in,
            dim_out=n_base_out,
            activation=activation,
        )
        self.BL_out = NeuralBasis(
            1, hidden=base_out_hidden, n_base=n_base_out, activation=activation
        )

    def forward(self, x, y):
        B_in, J1_in = x.size()
        B_out, J1_out = y.size()
        T_in, T_out = self.t_in, self.t_out
        self.bases_in = self.BL_in(T_in)  # (J1_in, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out, n_base_out)
        score_in = _parralleled_inner_product(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1)),
            self.bases_in.transpose(-1, -2).unsqueeze(0).repeat((B_in, 1, 1)),
            self.h_in,
        )  # (B_in, n_base_in)
        score = self.Middle(score_in)  # (B, n_basis_out)
        out = torch.einsum("bn,sn->bs", score, self.bases_out)  # (B, J1_out)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        score_out_temp = _parralleled_inner_product(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1)),
            self.bases_out.transpose(-1, -2).unsqueeze(0).repeat((B_out, 1, 1)),
            self.h_out,
        )  # (B_out, n_base_out)
        autoencoder_out = torch.einsum("bn,sn->bs", score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out

    def forward_in(self, x):
        B_in, J1_in = x.size()
        x = x.reshape(B_in, -1)
        T_in = self.t_in
        self.bases_in = self.BL_in(T_in)  # (J1_in, n_base_in)
        score_in = _parralleled_inner_product(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1)),
            self.bases_in.transpose(-1, -2).unsqueeze(0).repeat((B_in, 1, 1)),
            self.h_in,
        )  # (B_in, n_base_in)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        return autoencoder_in

    def forward_out(self, y):
        B_out, J1_out = y.size()
        y = y.reshape(B_out, -1)
        T_out = self.t_out
        self.bases_out = self.BL_out(T_out)  # (J1_out, n_base_out)
        score_out = _parralleled_inner_product(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1)),
            self.bases_out.transpose(-1, -2).unsqueeze(0).repeat((B_out, 1, 1)),
            self.h_out,
        )
        autoencoder_out = torch.einsum("bn,sn->bs", score_out, self.bases_out)
        return autoencoder_out


class BasisONet_2d2d(Basic_Model):
    def __init__(
        self,
        n_base_in=9,
        base_in_hidden=[64, 64, 64],
        middle_hidden=[64, 64, 64],
        sigma=[0.1, 0.1],
        n_base_out=9,
        base_out_hidden=[64, 64, 64],
        grid_in=None,
        grid_out=None,
        device="cuda",
        activation=F.gelu,
    ):
        super().__init__()
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.device = device
        assert grid_in.shape[-1] == 2 and grid_out.shape[-1] == 2
        self.h_in = (
            torch.tensor(
                [
                    grid_in[0, 1, 0] - grid_in[0, 0, 0],
                    grid_in[1, 0, 1] - grid_in[0, 0, 1],
                ]
            )
            .to(device)
            .float()
        )
        print(self.h_in, self.h_in.shape)
        self.h_out = (
            torch.tensor(
                [
                    grid_out[0, 1, 0] - grid_out[0, 0, 0],
                    grid_out[1, 0, 1] - grid_out[0, 0, 1],
                ]
            )
            .to(device)
            .float()
        )
        self.t_in = torch.tensor(grid_in).to(device).float().reshape(-1, 2)
        self.t_out = torch.tensor(grid_out).to(device).float().reshape(-1, 2)
        self.BL_in = NeuralBasis(
            2, hidden=base_in_hidden, n_base=n_base_in, activation=activation
        )
        self.Middle = FNN(
            hidden_layer=middle_hidden,
            dim_in=n_base_in,
            dim_out=n_base_out,
            activation=activation,
        )
        self.BL_out = NeuralBasis(
            2, hidden=base_out_hidden, n_base=n_base_out, activation=activation
        )

    def forward(self, x, y):
        B_in, J1_in, J2_in = x.size()
        B_out, J1_out, J2_out = y.size()
        T_in, T_out = self.t_in, self.t_out
        self.bases_in = self.BL_in(T_in)  # (J1_in*J2_in, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out*J2_out, n_base_out)
        score_in = _parralleled_inner_product_2d(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1, 1)),
            self.bases_in.transpose(-1, -2)
            .unsqueeze(0)
            .repeat((B_in, 1, 1))
            .reshape(B_in, self.n_base_in, J1_in, J2_in),
            self.h_in,
        )  # (B_in, n_base_in)
        score = self.Middle(score_in)  # (B, n_basis_out)
        out = torch.einsum("bn,sn->bs", score, self.bases_out)  # (B, J1_out*J2_out)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        score_out_temp = _parralleled_inner_product_2d(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)),
            self.bases_out.transpose(-1, -2)
            .unsqueeze(0)
            .repeat((B_out, 1, 1))
            .reshape(B_out, self.n_base_out, J1_out, J2_out),
            self.h_out,
        )  # (B_out, n_base_out)
        autoencoder_out = torch.einsum("bn,sn->bs", score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out

    def forward_in(self, x):
        B_in, J1_in, J2_in = x.size()
        T_in = self.t_in
        self.bases_in = self.BL_in(T_in)  # (J1_in*J2_in, n_base_in)
        score_in = _parralleled_inner_product_2d(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1, 1)),
            self.bases_in.transpose(-1, -2)
            .unsqueeze(0)
            .repeat((B_in, 1, 1))
            .reshape(B_in, self.n_base_in, J1_in, J2_in),
            self.h_in,
        )  # (B_in, n_base_in)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        return autoencoder_in

    def forward_out(self, y):
        B_out, J1_out, J2_out = y.size()
        T_out = self.t_out
        self.bases_out = self.BL_out(T_out)  # (J1_out, n_base_out)
        score_out = _parralleled_inner_product_2d(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)),
            self.bases_out.transpose(-1, -2)
            .unsqueeze(0)
            .repeat((B_out, 1, 1))
            .reshape(B_out, self.n_base_out, J1_out, J2_out),
            self.h_out,
        )  # (B_out, n_base_out)
        autoencoder_out = torch.einsum("bn,sn->bs", score_out, self.bases_out)
        return autoencoder_out
