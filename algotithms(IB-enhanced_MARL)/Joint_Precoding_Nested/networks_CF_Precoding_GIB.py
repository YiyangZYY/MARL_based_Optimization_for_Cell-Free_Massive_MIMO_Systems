import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions import Gumbel
from torch_geometric.utils import degree, softmax


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))


def reparameterize_diagonal(model, input, mode="diag"):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if mode.startswith("diagg"):
        if isinstance(mean_logit, tuple):
            mean = mean_logit[0]
        else:
            mean = mean_logit
        std = torch.ones(mean.shape).to(mean.device)
        dist = Normal(mean, std)
        return dist, (mean, std)
    elif mode.startswith("diag"):
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def scatter_sample(src, index, temperature, num_nodes=None):
    gumbel = Gumbel(torch.tensor([0.]).to(src.device), torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src + 1e-16)
    logit = (log_prob + gumbel) / temperature
    return softmax(logit, index, num_nodes=num_nodes)


def uniform_prior(index):
    deg = degree(index)
    deg = deg[index]
    return 1. / deg.unsqueeze(1)


def neighbor_sampling(x_i, x_j, att, BATCH_SIZE, device):
    # Vertex / Edge-1D: x_i, x_j: Batch_Size * Num_edge * Input_dim
    # Edge-2D: x_i_a, x_j_b: Batch_Size * Num_edge * Input_dim
    # Edge-3D: x_i_a_x, x_j_b_y: Batch_Size * Num_edge * Input_dim
    negative_slope = 0.2
    temperature = 0.1
    sample_neighbor_size = 0.5
    # Compute attention coefficients - 均值 [Batch_size, N_edge]
    alpha = torch.clamp(torch.sigmoid(F.leaky_relu((torch.cat([x_i, x_j], dim=-1) @ att).sum(dim=-1), negative_slope)), 0.01, 0.99).to(device)
    prior = (torch.ones_like(alpha) * sample_neighbor_size).to(device)
    Bernoulli_alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.Tensor([temperature]).to(device), probs=alpha).rsample()
    return x_j * Bernoulli_alpha.reshape(BATCH_SIZE, -1, 1), alpha, prior


class Layer_EIB_1DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_EIB_1DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparam_mode = 'diag'
        self.prior_mode = 'Gaussian'
        self.sample_size = sample_size
        self.device = device

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim * 2)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim * 2, 1])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim * 2, 1])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)

        A_reshape = A.transpose(1, 2).reshape(BATCH_SIZE * permutation_size1, self.output_dim * 2)
        # dist: [BATCH_SIZE * nbrOfRealizations * K * L, E]
        dist, _ = reparameterize_diagonal(model=None, input=A_reshape, mode=self.reparam_mode)
        Z = sample(dist, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations *L, E]
        feature_prior = Normal(loc=torch.zeros(A_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_reshape.size(0), self.output_dim).to(self.device))

        Z_logit = dist.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations *L]
        prior_logit = feature_prior.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations *L]
        # upper bound of I(X; Z)
        I_XZ_reshape = (Z_logit - prior_logit).mean(0)  # [BATCH_SIZE * nbrOfRealizations * L]
        I_XZ = I_XZ_reshape.view(BATCH_SIZE, permutation_size1).sum(-1)

        A_output = A[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, L]

        return A_output, I_XZ


class EIB_GNN1D_L_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN1D_L_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_1DPE_Nested(self.nested_dim , self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_1DPE_Nested(self.nested_dim , self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        I_XZ = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                Hhat_updated, I_XZ[i, :] = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                Hhat_updated, _ = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * K * U].view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(2, 3).transpose(1, 2)
        Fhat_u2 = Hhat_updated[:, :2 * K * U].view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(2, 3).transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * K * U:].view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot


class Layer_EIB_2DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_EIB_2DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparam_mode = 'diag'
        self.prior_mode = 'Gaussian'
        self.sample_size = sample_size
        self.device = device

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim * 2)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, permutation_size2])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, permutation_size2])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1, permutation_size2])

        A3_final = torch.matmul(self.P3[0], aggr_func(A, -1)[:, :, 0].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim * 2, 1, 1])
        for i in range(1, self.nested_dim):
            A3_updated = torch.matmul(self.P3[i], aggr_func(A, -1)[:, :, i].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim * 2, 1, 1])
            A3_final = torch.cat([A3_final, A3_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1, 1])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final + 0.1 * A3_final
        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)

        A_reshape = A.transpose(1, 3).reshape(BATCH_SIZE * permutation_size2 * permutation_size1, self.output_dim * 2)
        # dist: [BATCH_SIZE * nbrOfRealizations * K * L, E]
        dist, _ = reparameterize_diagonal(model=None, input=A_reshape, mode=self.reparam_mode)
        Z = sample(dist, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K * L, E]
        feature_prior = Normal(loc=torch.zeros(A_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_reshape.size(0), self.output_dim).to(self.device))

        Z_logit = dist.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K * L]
        prior_logit = feature_prior.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K * L]
        # upper bound of I(X; Z)
        I_XZ_reshape = (Z_logit - prior_logit).mean(0)  # [BATCH_SIZE * nbrOfRealizations * L]
        I_XZ = I_XZ_reshape.view(BATCH_SIZE, permutation_size2, permutation_size1).sum(-1).sum(-1)

        A_output = A[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, L, K]

        return A_output, I_XZ


class EIB_GNN2D_L_U_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN2D_L_U_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K, L, U).float()

        I_XZ = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                Hhat_updated, I_XZ[i, :] = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                Hhat_updated, _ = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * K].view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 0].transpose(1, 2)
        Fhat_u2 = Hhat_updated[:, :2 * K].view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 1].transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * K:].view([BATCH_SIZE, nbrOfRealizations, K, L, U]).transpose(2, 3)
        Phat_u2 = torch.mean(Phat_u1, dim=4)
        Phat_u = Phat_u2 / torch.linalg.norm(Phat_u2, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot


class EIB_GNN2D_L_K_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN2D_L_K_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        I_XZ = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                Hhat_updated, I_XZ[i, :] = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                Hhat_updated, _ = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * U].view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = Hhat_updated[:, :2 * U].view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * U:].view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot


class Layer_EIB_3DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_EIB_3DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparam_mode = 'diag'
        self.prior_mode = 'Gaussian'
        self.sample_size = sample_size
        self.device = device

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim * 2)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([nested_dim, output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2, permutation_size3])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2, permutation_size3])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1, permutation_size2, permutation_size3])

        A2_final = torch.matmul(self.P2[0], aggr_func(A, -1)[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2, 1])
        for i in range(1, self.nested_dim):
            A2_updated = torch.matmul(self.P2[i], aggr_func(A, -1)[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, permutation_size2, 1])
            A2_final = torch.cat([A2_final, A2_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1, permutation_size2, 1])

        A3_final = torch.matmul(self.P3[0], aggr_func(A, -2)[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, 1, permutation_size3])
        for i in range(1, self.nested_dim):
            A3_updated = torch.matmul(self.P3[i], aggr_func(A, -2)[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, 1, 1, permutation_size3])
            A3_final = torch.cat([A3_final, A3_updated], dim=2).view([BATCH_SIZE, self.output_dim * 2, i + 1, 1, permutation_size3])

        # A = A1 + A2 + A3 + A4
        # if aggr_func = torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final + 0.1 * A2_final + 0.1 * A3_final

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)

        A_reshape = A.transpose(1, 4).transpose(2, 3).reshape(BATCH_SIZE * permutation_size3 * permutation_size2 * permutation_size1, self.output_dim * 2)
        # dist: [BATCH_SIZE * nbrOfRealizations * U * K * L, E]
        dist, _ = reparameterize_diagonal(model=None, input=A_reshape, mode=self.reparam_mode)
        Z = sample(dist, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * U * K * L, E]
        feature_prior = Normal(loc=torch.zeros(A_reshape.size(0), self.output_dim).to(self.device), scale=torch.ones(A_reshape.size(0), self.output_dim).to(self.device))
        Z_logit = dist.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * U * K * L]
        prior_logit = feature_prior.log_prob(Z).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * U * K * L]
        # upper bound of I(X; Z)
        I_XZ_reshape = (Z_logit - prior_logit).mean(0)  # [BATCH_SIZE * nbrOfRealizations * U * K * L]
        I_XZ = I_XZ_reshape.view(BATCH_SIZE, permutation_size3, permutation_size2, permutation_size1).sum(-1).sum(-1).sum(-1)

        A_output = A[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, L, K, U]

        return A_output, I_XZ


class EIB_GNN3D_L_K_U_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN3D_L_K_U_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_3DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_3DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).float()

        I_XZ = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                Hhat_updated, I_XZ[i, :] = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                Hhat_updated, _ = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2].view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated[:, :2].view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3

        Phat_u1 = Hhat_updated[:, 2:].view([BATCH_SIZE, nbrOfRealizations, L, K, U]).transpose(2, 3)
        # Phat_u2 = torch.mean(Phat_u1, dim=4)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=(3, 4)).view([BATCH_SIZE, nbrOfRealizations, K, 1, 1]) * math.sqrt(L * U)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.transpose(2, 3).view([BATCH_SIZE, nbrOfRealizations, L, K, U])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot
