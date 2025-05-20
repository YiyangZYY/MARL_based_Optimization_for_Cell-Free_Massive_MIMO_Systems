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


class MPNN_vertex_GNN(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_vertex_GNN, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L).float().to(self.device)
        V_UE_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L).float().to(self.device)
        V_UE_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, K).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated, V_UE_updated = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output, V_UE_output = self.layers[i](V_AP_updated, V_UE_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, 1, U, L])[:, 0].transpose(1, 3).transpose(2, 3)
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, 1, U, L])[:, 1].transpose(1, 3).transpose(2, 3)
        Fhat_u1 = V_UE_output.view([BATCH_SIZE * nbrOfRealizations, 2, 1, U, K])[:, 0].transpose(2, 3) + Fhat_u1
        Fhat_u2 = V_UE_output.view([BATCH_SIZE * nbrOfRealizations, 2, 1, U, K])[:, 1].transpose(2, 3) + Fhat_u2
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_MPNN_3DPE(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_MPNN_3DPE, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim = output_dim
        self.is_BN = is_BN
        self.is_transfer = is_transfer
        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_1]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_2]))
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P6 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P7 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, H, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, 1)
        A5 = torch.matmul(self.P5, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim_1, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1, permutation_size3)
        A6 = torch.matmul(self.P6, aggr_func(H, -2).view(BATCH_SIZE, self.input_dim_2, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1, permutation_size3)
        A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim_1, -1)).view(BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3)
        A7 = torch.matmul(self.P7, aggr_func(H, -3).view(BATCH_SIZE, self.input_dim_2, -1)).view(BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3)

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = 2 * A1 + 2 * A2 + 2 * A3 + 2 * A4 + 0.1 * A5 + 0.1 * A6 + 0.1 * A7
        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class MPNN_GNN3D_L_K_U(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_GNN3D_L_K_U, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_3DPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_3DPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1)
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L, K, U).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L, K, U).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_MPNN_2DPE(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_MPNN_2DPE, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.output_dim = output_dim
        self.is_BN = is_BN
        self.is_transfer = is_transfer
        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_1]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_2]))
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, H, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim, permutation_size1, 1])
        A4 = torch.matmul(self.P4, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim, permutation_size1, 1])
        A3 = torch.matmul(self.P2, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim, 1, permutation_size2])
        A5 = torch.matmul(self.P4, aggr_func(H, -2).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim, 1, permutation_size2])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = 2 * A1 + 2 * A2 + 2 * A3 + 0.1 * A4 + 0.1 * A5
        # A = A1 + A2 + A3 + A4 + A5
        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class MPNN_GNN2D_L_K(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_GNN2D_L_K, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_2DPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_2DPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L, K).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_MPNN_1DPE(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_MPNN_1DPE, self).__init__()
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
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim_1], requires_grad=True) * 2 * ini - ini)
        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim_2]))
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim_2], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, H, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim_1, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim_1, -1)).view([BATCH_SIZE, self.output_dim, 1])
        A3 = torch.matmul(self.P3, aggr_func(H, -1).view(BATCH_SIZE, self.input_dim_2, -1)).view([BATCH_SIZE, self.output_dim, 1])
        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = 2 * A1 + 2 * A2 + 0.1 * A3
        # A = A1 + A2 + A3

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class MPNN_GNN1D_L(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_GNN1D_L, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, L).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, L).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(1, 3).transpose(2, 3)
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(1, 3).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class MPNN_GNN1D_K(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_GNN1D_K, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * U, K).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, K).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, K).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, U, K])[:, 0].transpose(2, 3)
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, U, K])[:, 1].transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class MPNN_GNN1D_U(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, output_dim, device, BATCH_SIZE):
        super(MPNN_GNN1D_U, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.embedding_dim = input_dim_1
        self.output_dim = output_dim
        self.dim = [input_dim_1] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, input_dim_1, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_MPNN_1DPE(input_dim_1, input_dim_2, output_dim, transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * K, U).float()
        V_AP_updated = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.embedding_dim, U).float().to(self.device)
        V_AP_output = torch.zeros(BATCH_SIZE * nbrOfRealizations, self.output_dim, U).float().to(self.device)

        for i in range(len(self.dim)-1):
            if i != len(self.dim) - 2:
                V_AP_updated = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
            else:
                V_AP_output = self.layers[i](V_AP_updated, Hhat_updated, permutation_size1=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)
        # to satisfy constraints
        Fhat_u1 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = V_AP_output.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_3DPE_attention(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE_attention, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

        ini = torch.sqrt(torch.FloatTensor([1.0 / output_dim / input_dim]))
        self.q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.k = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    # input A_feature: BATCH_SIZE, 2 (C1), K, Nt, Ns
    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1, permutation_size3)

        AT = A.transpose(1, 3)
        Q = torch.matmul(self.q, AT).transpose(1, 3).reshape(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2 * permutation_size3)
        K = torch.matmul(self.k, AT).transpose(1, 3).reshape(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2 * permutation_size3).transpose(2, 3)
        Alpha = torch.matmul(Q, K)
        Alpha_norm = abs(Alpha) / torch.linalg.norm(Alpha, dim=3).view([BATCH_SIZE, self.output_dim, permutation_size1, 1])
        V = torch.matmul(self.P4, AT).transpose(1, 3)
        Y = torch.matmul(Alpha_norm, V.reshape(BATCH_SIZE, self.output_dim, permutation_size1, -1)).reshape(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3)
        A = A1 + 0.1 * A2 + 0.1 * A3 + 0.1 * Y

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNNA3D_L_K_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNNA3D_L_K_U, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_3DPE_attention(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_3DPE_attention(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_2DPE_attention(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE_attention, self).__init__()

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

        ini = torch.sqrt(torch.FloatTensor([1.0 / output_dim / input_dim]))
        self.q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.k = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1)

        AT = A.transpose(1, 2)
        Q = torch.matmul(self.q, AT).transpose(1, 2)
        K = torch.matmul(self.k, AT)
        KT = K.permute(0, 2, 3, 1)
        Alpha = torch.matmul(Q, KT)
        Alpha_norm = abs(Alpha) / torch.linalg.norm(Alpha, dim=3).view([BATCH_SIZE, self.output_dim, permutation_size1, 1])
        V = torch.matmul(self.P4, AT).transpose(1, 2)
        Y = torch.matmul(Alpha_norm, V)
        A = A1 + 0.1 * A3 + 0.1 * Y

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNNA2D_L_K(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNNA2D_L_K, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_attention(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_attention(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, U, L, K)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, U, L, K)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_1DPE_attention(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_1DPE_attention, self).__init__()

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

        ini = torch.sqrt(torch.FloatTensor([1.0 / output_dim / input_dim]))
        self.q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.k = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A)

        Q = torch.matmul(self.q, A).reshape(BATCH_SIZE, self.output_dim, permutation_size1, 1)
        K = torch.matmul(self.k, A).reshape(BATCH_SIZE, self.output_dim, 1, permutation_size1)
        Alpha = torch.matmul(Q, K)
        Alpha_norm = abs(Alpha) / torch.linalg.norm(Alpha, dim=3).view([BATCH_SIZE, self.output_dim, permutation_size1, 1])
        V = torch.matmul(self.P2, A)
        Y = torch.matmul(Alpha_norm, V.view([BATCH_SIZE, self.output_dim, permutation_size1, 1])).view([BATCH_SIZE, self.output_dim, permutation_size1])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + V + 0.1 * Y
        # A = A1 + A2

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNNA1D_L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNNA1D_L, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE_attention(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE_attention(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(2, 3).transpose(1, 2)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(2, 3).transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
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


class GNN1D_L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_L, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(2, 3).transpose(1, 2)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(2, 3).transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class GNN1D_K(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_K, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, U, K)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, U, K)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * U, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, U, K])[:, 0].transpose(2, 3)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, U, K])[:, 1].transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class GNN1D_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_U, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).reshape(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L * K, U).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_1DPE_Distributed(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_1DPE_Distributed, self).__init__()

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
        # A2 = torch.matmul(self.P2, aggr_func(A, -1, keepdim=True))

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        # A = A1 + 0.1 * A2
        A = A1

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN1D_L_Distributed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_L_Distributed, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE_Distributed(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE_Distributed(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4).reshape(BATCH_SIZE * nbrOfRealizations, 1, K, U, L)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(2, 3).transpose(1, 2)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(2, 3).transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
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


class GNN2D_L_K(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_K, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class GNN2D_K_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_K_U, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * L, K, U).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=K, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class GNN2D_L_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_U, self).__init__()
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

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K, L, U).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 0].transpose(1, 2)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 1].transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_2DPE_Distributed(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE_Distributed, self).__init__()

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

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1)

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        # A = A1 + 0.1 * A3 + 0.1 * A4
        A = A1 + 0.1 * A3

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN2D_L_K_Distributed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_K_Distributed, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class GNN2D_L_U_Distributed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_U_Distributed, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K, L, U).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 0].transpose(1, 2)
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 1].transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_3DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    # input A_feature: BATCH_SIZE, 2 (C1), K, Nt, Ns
    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1, permutation_size3)
        A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3)

        # A = A1 + A2 + A3 + A4
        # if aggr_func = torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A2 + 0.1 * A3 + 0.1 * A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN3D_L_K_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN3D_L_K_U, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                # 中间层
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                # 输出层
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u


class Layer_3DPE_Distributed(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE_Distributed, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        # self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    # input A_feature: BATCH_SIZE, 2 (C1), K, Nt, Ns
    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, permutation_size1, 1, permutation_size3)
        # A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3)

        # A = A1 + A2 + A3 + A4
        # if aggr_func = torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1 * A2 + 0.1 * A3

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN3D_L_K_U_Distributed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN3D_L_K_U_Distributed, self).__init__()
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_3DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_3DPE_Distributed(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated.view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u