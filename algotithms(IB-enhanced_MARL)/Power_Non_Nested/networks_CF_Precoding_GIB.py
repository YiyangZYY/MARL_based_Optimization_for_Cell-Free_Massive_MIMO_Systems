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


class Layer_VIB(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_VIB, self).__init__()

        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        self.reparam_mode = 'diag'
        self.prior_mode = 'Gaussian'
        self.sample_size = sample_size
        self.device = device

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim * 2)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim_1]))
        self.Q1_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q1_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim_2]))
        self.P1_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P1_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_2], requires_grad=True) * 2 * ini - ini)

    def forward(self, A_AP, A_UE, H, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A_AP_1 = torch.matmul(self.Q1_AP, A_AP.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim * 2, permutation_size1])
        A_AP_2 = torch.matmul(self.Q2_AP, aggr_func(A_UE, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim * 2, 1])
        A_AP_3 = torch.matmul(self.P1_AP, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim * 2, permutation_size1])

        A_UE_1 = torch.matmul(self.Q1_UE, A_UE.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim * 2, permutation_size2])
        A_UE_2 = torch.matmul(self.Q2_UE, aggr_func(A_AP, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim * 2, 1])
        A_UE_3 = torch.matmul(self.P1_UE, aggr_func(H, -2).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim * 2, permutation_size2])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A_AP = 2 * A_AP_1 + 2 * A_AP_2 + 0.1 * A_AP_3
        A_UE = 2 * A_UE_1 + 2 * A_UE_2 + 0.1 * A_UE_3

        if self.is_transfer:
            A_AP = self.activation(A_AP)
            A_UE = self.activation(A_UE)
        if self.is_BN:
            A_AP = self.batch_norms(A_AP)
            A_UE = self.batch_norms(A_UE)
        A_AP_reshape = A_AP.transpose(1, 2).reshape(BATCH_SIZE * permutation_size1, self.output_dim * 2)
        A_UE_reshape = A_UE.transpose(1, 2).reshape(BATCH_SIZE * permutation_size2, self.output_dim * 2)

        # dist_AP: [BATCH_SIZE * nbrOfRealizations * L, E]
        dist_AP, _ = reparameterize_diagonal(model=None, input=A_AP_reshape, mode=self.reparam_mode)
        Z_AP = sample(dist_AP, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L, E]
        feature_prior_AP = Normal(loc=torch.zeros(A_AP_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_AP_reshape.size(0), self.output_dim).to(self.device))
        Z_logit_AP = dist_AP.log_prob(Z_AP).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L]
        prior_logit_AP = feature_prior_AP.log_prob(Z_AP).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L]
        # upper bound of I(X; Z) - AP
        I_XZ_AP_reshape = (Z_logit_AP - prior_logit_AP).mean(0)  # [BATCH_SIZE * nbrOfRealizations * L]
        I_XZ_AP = I_XZ_AP_reshape.view(BATCH_SIZE, permutation_size1).sum(-1)

        # dist_UE: [BATCH_SIZE * nbrOfRealizations * K, E]
        dist_UE, _ = reparameterize_diagonal(model=None, input=A_UE_reshape, mode=self.reparam_mode)
        Z_UE = sample(dist_UE, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K, E]
        feature_prior_UE = Normal(loc=torch.zeros(A_UE_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_UE_reshape.size(0), self.output_dim).to(self.device))
        Z_logit_UE = dist_UE.log_prob(Z_UE).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K]
        prior_logit_UE = feature_prior_UE.log_prob(Z_UE).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K]
        # upper bound of I(X; Z) - UE
        I_XZ_UE_reshape = (Z_logit_UE - prior_logit_UE).mean(0)  # [BATCH_SIZE * nbrOfRealizations * K]
        I_XZ_UE = I_XZ_UE_reshape.view(BATCH_SIZE, permutation_size2).sum(-1)

        A_AP_output = A_AP[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, L]
        A_UE_output = A_UE[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, K]

        return A_AP_output, A_UE_output, I_XZ_AP, I_XZ_UE


class VIB_GNN_Power(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(VIB_GNN_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.sample_size = sample_size
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_VIB(input_dim_1, input_dim_2, input_dim_1, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_VIB(input_dim_1, input_dim_2, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L).float().to(self.device)
        V_UE_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L).float().to(self.device)

        I_XZ_AP = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)
        I_XZ_UE = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated, V_UE_updated, I_XZ_AP[i, :], I_XZ_UE[i, :] = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output, _, _, _ = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = V_AP_output.view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_AP_tot = torch.sum(I_XZ_AP, dim=0)
        I_XZ_UE_tot = torch.sum(I_XZ_UE, dim=0)
        I_XZ_tot = I_XZ_AP_tot + I_XZ_UE_tot

        return Fhat_u, I_XZ_tot


class Layer_VGIB(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_VGIB, self).__init__()

        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        self.reparam_mode = 'diag'
        self.prior_mode = 'Gaussian'
        # 'categorical' or 'Bernoulli'
        self.struct_dropout_mode = 'bernoulli'
        self.neighbor_sampling_mode = 'bernoulli'
        self.sample_size = sample_size
        self.device = device

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim * 2)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim_1]))
        self.Q1_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q1_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_1], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / 2 / input_dim_2]))
        self.P1_AP = nn.Parameter(torch.rand([output_dim * 2, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P1_UE = nn.Parameter(torch.rand([output_dim * 2, input_dim_2], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / input_dim_1 / 2]))
        self.Att_AP = nn.Parameter(torch.rand([input_dim_1 * 2, 1], requires_grad=True) * 2 * ini - ini)
        self.Att_UE = nn.Parameter(torch.rand([input_dim_1 * 2, 1], requires_grad=True) * 2 * ini - ini)

    def forward(self, A_AP, A_UE, H, Graph_AP_reshape, GFA_AP, Graph_UE_reshape, GFA_UE, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        # lower bound of I(Y; Z) - AP / UE - edge_index_i, x_i, x_j
        edge_index_i_AP, edge_index_i_UE = Graph_AP_reshape[:, 1].to(torch.int64), Graph_UE_reshape[:, 1].to(torch.int64)

        # Message - AP / UE - edge_index_i, x_i, x_j: Batch_size * node_edge * self.input_dim_1
        edge_x_i_AP_tot, edge_x_j_AP_tot = A_AP[:, :, edge_index_i_AP].transpose(1, 2).diagonal().transpose(0, 2), A_AP[:, :, Graph_AP_reshape[:, 0].to(torch.int64)].transpose(1, 2).diagonal().transpose(0, 2)
        edge_x_i_UE_tot, edge_x_j_UE_tot = A_UE[:, :, edge_index_i_UE].transpose(1, 2).diagonal().transpose(0, 2), A_UE[:, :, Graph_UE_reshape[:, 0].to(torch.int64)].transpose(1, 2).diagonal().transpose(0, 2)
        edge_x_j_AP_tot, alpha_AP, prior_AP = neighbor_sampling(edge_x_i_AP_tot, edge_x_j_AP_tot, self.Att_AP, BATCH_SIZE, self.device)
        edge_x_j_UE_tot, alpha_UE, prior_UE = neighbor_sampling(edge_x_i_UE_tot, edge_x_j_UE_tot, self.Att_UE, BATCH_SIZE, self.device)

        A_AP, A_UE = torch.sum(edge_x_j_AP_tot[:, GFA_AP.to(torch.int64), :].diagonal(), dim=1).transpose(0, 2), torch.sum(edge_x_j_UE_tot[:, GFA_UE.to(torch.int64), :].diagonal(), dim=1).transpose(0, 2)

        if 'categorical' in self.neighbor_sampling_mode:
            # AIB - KL_Divergence - Categorical
            I_AZ_AP = torch.sum(alpha_AP * torch.log((alpha_AP + 1e-16) / prior_AP), dim=1)
            I_AZ_UE = torch.sum(alpha_UE * torch.log((alpha_UE + 1e-16) / prior_UE), dim=1)
        else:
            # AIB - KL_Divergence - Bernoulli
            I_AZ_AP = kl_divergence(Bernoulli(alpha_AP), Bernoulli(prior_AP)).sum(-1)
            I_AZ_UE = kl_divergence(Bernoulli(alpha_UE), Bernoulli(prior_UE)).sum(-1)

        # Propagate - AP / UE
        A_AP_1 = torch.matmul(self.Q1_AP, A_AP.reshape([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim * 2, permutation_size1])
        A_AP_2 = torch.matmul(self.Q2_AP, aggr_func(A_UE, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim * 2, 1])
        A_AP_3 = torch.matmul(self.P1_AP, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim * 2, permutation_size1])

        A_UE_1 = torch.matmul(self.Q1_UE, A_UE.reshape([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim * 2, permutation_size2])
        A_UE_2 = torch.matmul(self.Q2_UE, aggr_func(A_AP, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim * 2, 1])
        A_UE_3 = torch.matmul(self.P1_UE, aggr_func(H, -2).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim * 2, permutation_size2])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A_AP = 2 * A_AP_1 + 2 * A_AP_2 + 0.1 * A_AP_3
        A_UE = 2 * A_UE_1 + 2 * A_UE_2 + 0.1 * A_UE_3

        if self.is_transfer:
            A_AP = self.activation(A_AP)
            A_UE = self.activation(A_UE)
        if self.is_BN:
            A_AP = self.batch_norms(A_AP)
            A_UE = self.batch_norms(A_UE)

        A_AP_reshape = A_AP.transpose(1, 2).reshape(BATCH_SIZE * permutation_size1, self.output_dim * 2)
        A_UE_reshape = A_UE.transpose(1, 2).reshape(BATCH_SIZE * permutation_size2, self.output_dim * 2)

        # upper bound of I(X; Z) - AP / UE
        dist_AP, _ = reparameterize_diagonal(model=None, input=A_AP_reshape, mode=self.reparam_mode)  # [BATCH_SIZE * nbrOfRealizations * L, E]
        Z_AP = sample(dist_AP, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L, E]
        feature_prior_AP = Normal(loc=torch.zeros(A_AP_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_AP_reshape.size(0), self.output_dim).to(self.device))
        Z_logit_AP = dist_AP.log_prob(Z_AP).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L]
        prior_logit_AP = feature_prior_AP.log_prob(Z_AP).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * L]
        I_XZ_AP_reshape = (Z_logit_AP - prior_logit_AP).mean(0)  # [BATCH_SIZE * nbrOfRealizations * L]
        I_XZ_AP = I_XZ_AP_reshape.view(BATCH_SIZE, permutation_size1).sum(-1)

        dist_UE, _ = reparameterize_diagonal(model=None, input=A_UE_reshape, mode=self.reparam_mode)  # [BATCH_SIZE * nbrOfRealizations * K, E]
        Z_UE = sample(dist_UE, self.sample_size)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K, E]
        feature_prior_UE = Normal(loc=torch.zeros(A_UE_reshape.size(0), self.output_dim).to(self.device),
                                  scale=torch.ones(A_UE_reshape.size(0), self.output_dim).to(self.device))
        Z_logit_UE = dist_UE.log_prob(Z_UE).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K]
        prior_logit_UE = feature_prior_UE.log_prob(Z_UE).sum(-1)  # [sample_size, BATCH_SIZE * nbrOfRealizations * K]
        I_XZ_UE_reshape = (Z_logit_UE - prior_logit_UE).mean(0)  # [BATCH_SIZE * nbrOfRealizations * K]
        I_XZ_UE = I_XZ_UE_reshape.view(BATCH_SIZE, permutation_size2).sum(-1)

        A_AP_output = A_AP[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, L]
        A_UE_output = A_UE[:, :self.output_dim]  # [BATCH_SIZE * nbrOfRealizations, E, K]

        return A_AP_output, A_UE_output, I_XZ_AP, I_XZ_UE, I_AZ_AP, I_AZ_UE


class VGIB_GNN_Power(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(VGIB_GNN_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.sample_size = sample_size
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_VGIB(input_dim_1, input_dim_2, input_dim_1, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_VGIB(input_dim_1, input_dim_2, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, Ahat_AP_reshape, GFA_AP, Ahat_UE_reshape, GFA_UE, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L).float().to(self.device)
        V_UE_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L).float().to(self.device)

        I_XZ_AP = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)
        I_XZ_UE = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)
        I_AZ_AP = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)
        I_AZ_UE = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated, V_UE_updated, I_XZ_AP[i], I_XZ_UE[i], I_AZ_AP[i], I_AZ_UE[i] = \
                    self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, Ahat_AP_reshape, GFA_AP, Ahat_UE_reshape, GFA_UE, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output, _, _, _, _, _ = \
                    self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, Ahat_AP_reshape, GFA_AP, Ahat_UE_reshape, GFA_UE, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = V_AP_output.view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ_AP, dim=0) + torch.sum(I_XZ_UE, dim=0)
        I_AZ_tot = torch.sum(I_AZ_AP, dim=0) + torch.sum(I_AZ_UE, dim=0)

        return Fhat_u, I_XZ_tot, I_AZ_tot


class Layer_EIB_1DPE(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_EIB_1DPE, self).__init__()

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
        self.P1 = nn.Parameter(torch.rand([output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A)
        A2 = torch.matmul(self.P2, aggr_func(A, -1, keepdim=True))

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A2

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


class EIB_GNN1D_L_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN1D_L_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_1DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_1DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
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
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot


class EIB_GNN1D_K_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN1D_K_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_1DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_1DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, U, K)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, U, K)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * U, K).float()

        I_XZ = torch.zeros(len(self.dim) - 2, BATCH_SIZE * nbrOfRealizations).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                Hhat_updated, I_XZ[i, :] = self.layers[i](Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                Hhat_updated, _ = self.layers[i](Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot


class Layer_EIB_2DPE(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_EIB_2DPE, self).__init__()

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
        self.P1 = nn.Parameter(torch.rand([output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim * 2, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim * 2, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim * 2, permutation_size1, 1)
        A4 = torch.matmul(self.P4, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim * 2, 1, permutation_size2)

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A3 + 0.1 * A4

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


class EIB_GNN2D_L_K_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sample_size, device, BATCH_SIZE):
        super(EIB_GNN2D_L_K_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.sample_size = sample_size
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_EIB_2DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_EIB_2DPE(self.dim[i], self.dim[i + 1], sample_size, device, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
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
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        # to compute upper bound of I(X;Z)
        I_XZ_tot = torch.sum(I_XZ, dim=0)

        return Fhat_u, I_XZ_tot
