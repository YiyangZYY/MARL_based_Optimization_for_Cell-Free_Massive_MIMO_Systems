import math
import numpy
import numpy as np
import torch
import torch.nn as nn


class Layer_1DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_1DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim, 1])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim, 1])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN1D_L_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D_L_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE_Nested(self.nested_dim, self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE_Nested(self.nested_dim, self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(3, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K * U, L).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * K * U].view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 0].transpose(2, 3).transpose(1, 2)
        Fhat_u2 = Hhat_updated[:, :2 * K * U].view([BATCH_SIZE * nbrOfRealizations, 2, K, U, L])[:, 1].transpose(2, 3).transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * K * U:].view([BATCH_SIZE, nbrOfRealizations, K, L]).transpose(2, 3)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class Layer_2DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, permutation_size2])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, permutation_size2])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1, permutation_size2])

        A3_final = torch.matmul(self.P3[0], aggr_func(A, -1)[:, :, 0].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim, 1, 1])
        for i in range(1, self.nested_dim):
            A3_updated = torch.matmul(self.P3[i], aggr_func(A, -1)[:, :, i].view([BATCH_SIZE, self.input_dim, 1])).view([BATCH_SIZE, self.output_dim, 1, 1])
            A3_final = torch.cat([A3_final, A3_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1, 1])

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final + 0.1 * A3_final
        # A = A1 + A3 + A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN2D_L_K_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_K_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3).transpose(2, 4)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * U, L, K).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * U].view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 0].transpose(1, 2).transpose(2, 3)
        Fhat_u2 = Hhat_updated[:, :2 * U].view([BATCH_SIZE * nbrOfRealizations, 2, U, L, K])[:, 1].transpose(1, 2).transpose(2, 3)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * U:].view([BATCH_SIZE, nbrOfRealizations, L, K])
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class GNN2D_L_U_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D_L_U_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U).transpose(2, 3)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).view(BATCH_SIZE * nbrOfRealizations, 2 * K, L, U).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2 * K].view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 0].transpose(1, 2)
        Fhat_u2 = Hhat_updated[:, :2 * K].view([BATCH_SIZE * nbrOfRealizations, 2, K, L, U])[:, 1].transpose(1, 2)
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3
        Phat_u1 = Hhat_updated[:, 2 * K:].view([BATCH_SIZE, nbrOfRealizations, K, L, U]).transpose(2, 3)
        Phat_u2 = torch.mean(Phat_u1, dim=4)
        Phat_u = Phat_u2 / torch.linalg.norm(Phat_u2, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])
        return Fhat_u


class Layer_3DPE_Nested(nn.Module):
    def __init__(self, nested_dim, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE_Nested, self).__init__()
        self.nested_dim = nested_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([nested_dim, output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    # input A_feature: BATCH_SIZE, 2 (C1), K, Nt, Ns
    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1_final = torch.matmul(self.P1[0], A[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3])
        for i in range(1, self.nested_dim):
            A1_updated = torch.matmul(self.P1[i], A[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2, permutation_size3])
            A1_final = torch.cat([A1_final, A1_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1, permutation_size2, permutation_size3])

        A2_final = torch.matmul(self.P2[0], aggr_func(A, -1)[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2, 1])
        for i in range(1, self.nested_dim):
            A2_updated = torch.matmul(self.P2[i], aggr_func(A, -1)[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, permutation_size2, 1])
            A2_final = torch.cat([A2_final, A2_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1, permutation_size2, 1])

        A3_final = torch.matmul(self.P3[0], aggr_func(A, -2)[:, :, 0].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, 1, permutation_size3])
        for i in range(1, self.nested_dim):
            A3_updated = torch.matmul(self.P3[i], aggr_func(A, -2)[:, :, i].view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, 1, 1, permutation_size3])
            A3_final = torch.cat([A3_final, A3_updated], dim=2).view([BATCH_SIZE, self.output_dim, i + 1, 1, permutation_size3])

        # A = A1 + A2 + A3 + A4
        # if aggr_func = torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1_final + 0.1 * A2_final + 0.1 * A3_final

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN3D_L_K_U_Nested_Power(nn.Module):
    def __init__(self, nested_dim, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN3D_L_K_U_Nested_Power, self).__init__()
        self.nested_dim = nested_dim
        self.out = nn.Sigmoid()
        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                # 中间层
                self.layers.append(Layer_3DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                # 输出层
                self.layers.append(Layer_3DPE_Nested(self.nested_dim, self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, Hhat, nbrOfRealizations, L, K, U):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        Hhat1 = Hhat.real.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat2 = Hhat.imag.view(BATCH_SIZE * nbrOfRealizations, 1, L, K, U)
        Hhat_updated = torch.cat([Hhat1, Hhat2], dim=1).float()

        for i in range(len(self.dim)-1):
            Hhat_updated = self.layers[i](Hhat_updated, permutation_size1=L, permutation_size2=K, permutation_size3=U, BATCH_SIZE=BATCH_SIZE * nbrOfRealizations)

        # to satisfy constraints
        Fhat_u1 = Hhat_updated[:, :2].view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 0]
        Fhat_u2 = Hhat_updated[:, :2].view([BATCH_SIZE * nbrOfRealizations, 2, L, K, U])[:, 1]
        Fhat_u3 = torch.tile(torch.sqrt(torch.sum(Fhat_u1 ** 2, dim=3) + torch.sum(Fhat_u2 ** 2, dim=3)).view(BATCH_SIZE * nbrOfRealizations, L, K, 1), (1, 1, 1, U))
        Fhat_u = (Fhat_u1 + 1j * Fhat_u2) / Fhat_u3

        # Phat_u1 = Hhat_updated[:, 2:].view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        # Phat_u2 = torch.mean(Phat_u1, dim=4)
        # Phat_u = Phat_u2 / torch.linalg.norm(Phat_u2, dim=2).view([BATCH_SIZE, nbrOfRealizations, 1, K]) * math.sqrt(L)
        # Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, 1])

        Phat_u1 = Hhat_updated[:, 2:].view([BATCH_SIZE, nbrOfRealizations, L, K, U]).transpose(2, 3)
        # Phat_u2 = torch.mean(Phat_u1, dim=4)
        Phat_u = Phat_u1 / torch.linalg.norm(Phat_u1, dim=(3, 4)).view([BATCH_SIZE, nbrOfRealizations, K, 1, 1]) * math.sqrt(L * U)
        Fhat_u = Fhat_u.view([BATCH_SIZE, nbrOfRealizations, L, K, U]) * Phat_u.transpose(2, 3).view([BATCH_SIZE, nbrOfRealizations, L, K, U])
        return Fhat_u
