import math
import numpy
import numpy as np
import torch
import torch.nn as nn


class Layer_MPNN_vertexPE(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_MPNN_vertexPE, self).__init__()

        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_1]))
        self.Q1_AP = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_AP = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q1_UE = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.Q2_UE = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_2]))
        self.P1_AP = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P1_UE = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)

    def forward(self, A_AP, A_UE, H, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A_AP_1 = torch.matmul(self.Q1_AP, A_AP.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1])
        A_AP_2 = torch.matmul(self.Q2_AP, aggr_func(A_UE, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim, 1])
        A_AP_3 = torch.matmul(self.P1_AP, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim, permutation_size1])

        A_UE_1 = torch.matmul(self.Q1_UE, A_UE.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim, permutation_size2])
        A_UE_2 = torch.matmul(self.Q2_UE, aggr_func(A_AP, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim, 1])
        A_UE_3 = torch.matmul(self.P1_UE, aggr_func(H, -2).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim, permutation_size2])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A_AP = 2 * A_AP_1 + 2 * A_AP_2 + 0.1 * A_AP_3
        A_UE = 2 * A_UE_1 + 2 * A_UE_2 + 0.1 * A_UE_3

        if self.is_transfer:
            A_AP = self.activation(A_AP)
            A_UE = self.activation(A_UE)
        if self.is_BN:
            A_AP = self.batch_norms(A_AP)
            A_UE = self.batch_norms(A_UE)
        return A_AP, A_UE


class MPNN_vertex_GNN_Power(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_vertex_GNN_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_vertexPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_vertexPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L).float().to(self.device)
        V_UE_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated, V_UE_updated = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output, _ = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = V_AP_output.view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class Layer_1DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_1DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A)
        A2 = torch.matmul(self.P2, aggr_func(A, -1, keepdim=True))

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A2
        # A = A1 + A2

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN1D_L_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_L_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class GNN1D_K_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_K_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * U, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class Layer_2DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1)
        A4 = torch.matmul(self.P4, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, permutation_size2)

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A3 + 0.1 * A4
        # A = A1 + A3 + A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN2D_L_K_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_K_Power, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, Vhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Phat_u1 = Hhat_updated.view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Vhat.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u
