import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import time


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def cal_laplacian(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)

class ChebGraphConvForBatch(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, dnGso):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('bhi,btij->bthj', dnGso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('bhi,btij->bthj', dnGso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('bhi,btij->bthj', 2 * dnGso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cuda'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()

class LUpdator(nn.Module):
    def __init__(self, T, N, k_spatial=60, k_temporal=12):
        super().__init__()
        self.T = T
        self.N = N

        u_spatial = torch.randn((N, k_spatial), requires_grad=True)
        self.u_spatial = torch.nn.Parameter(u_spatial)
        self.register_parameter("u_spatial", self.u_spatial)

        u_temporal = torch.randn((T, k_temporal), requires_grad=True)
        self.u_temporal = torch.nn.Parameter(u_temporal)
        self.register_parameter("u_temporal", self.u_temporal)

        self.conv2d_1 = torch.nn.Conv2d(3, 1, kernel_size=(1, 1))
        self.norm = nn.LayerNorm(normalized_shape=[N, N])
        self.relu = nn.ReLU()

    def forward(self, x, spatialL, geo_mask):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3)
        u_spatial = torch.einsum("ik,kj->ij", [self.u_spatial, torch.transpose(self.u_spatial, 0, 1)])
        u_temporal = torch.einsum("ik,kj->ij", [self.u_temporal, torch.transpose(self.u_temporal, 0, 1)])
        sn_lowrank = torch.einsum("bitk,ij->bjtk", [x, u_spatial])
        sn_lowrank = torch.einsum("bitk,tj->bijk",  [sn_lowrank, u_temporal])
        sn_exception = x - sn_lowrank

        sn_lowrank = sn_lowrank.reshape(B, N, T * D)
        sn_exception = sn_exception.reshape(B, N, T * D)
        sn_lowrank = sn_lowrank - torch.mean(sn_lowrank, dim=2, keepdim=True)
        sn_exception = sn_exception - torch.mean(sn_exception, dim=2, keepdim=True)

        x_ee = torch.einsum("bik,bkj->bij", [sn_exception, torch.transpose(sn_exception, 1, 2)])
        x_ee = self.norm(x_ee)
        x_le = torch.einsum("bik,bkj->bij", [sn_lowrank, torch.transpose(sn_exception, 1, 2)])
        x_le = self.norm(x_le)
        x_el = torch.einsum("bik,bkj->bij", [sn_exception, torch.transpose(sn_lowrank, 1, 2)])
        x_el = self.norm(x_el)

        covList = []
        covList.append(x_ee)
        covList.append(x_le)
        covList.append(x_el)
        covTensor = torch.stack(covList, dim=-1).permute(0, 3, 1, 2)
        B_conv = self.conv2d_1(covTensor)
        B_conv = self.relu(B_conv)

        B_conv = B_conv.reshape(B, N, N)

        BQ = torch.einsum("bik,bkj->bij", [B_conv, spatialL])
        taylorTerm = 3
        forIterm = -torch.einsum("bik,bkj->bij", [spatialL, BQ])
        spatialL_update = forIterm.clone()
        for i in range(taylorTerm):
            forIterm = -torch.einsum("bik,bkj->bij", [forIterm, BQ])
            spatialL_update = spatialL_update + forIterm

        spatialL = spatialL + spatialL_update
        spatialL.masked_fill_(geo_mask, 0)

        spatialL = torch.clamp(spatialL, -1, 0)
        spatialL_degree = torch.abs(torch.sum(spatialL, dim=2, keepdims=True))
        spatialL = torch.eye(int(N), device=torch.device("cuda")) + spatialL / torch.clamp(spatialL_degree, 0.00000001, 10000000)
        return spatialL

class DSTFormer(nn.Module):
    def __init__(self, c_in, c_out, qkv_bias=False, attn_drop=0., proj_drop=0., device=torch.device('cuda')):
        super().__init__()
        self.sem_q_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(c_in, c_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def assemblingMask(self, geo_mask):
        N = geo_mask.shape[0]
        true_mask = torch.ones([N, N], device=torch.device("cuda"))
        uneye_mask = torch.ones([N, N], device=torch.device("cuda")) - torch.eye(int(N), device=torch.device("cuda"))
        true_mask = true_mask.bool()
        uneye_mask = uneye_mask.bool()

        L_row1 = torch.concat([geo_mask, uneye_mask, true_mask], 0)
        L_row2 = torch.concat([uneye_mask, geo_mask, uneye_mask], 0)
        L_row3 = torch.concat([true_mask, uneye_mask, geo_mask], 0)
        return torch.concat([L_row1, L_row2, L_row3], 1)

    def forward(self, x, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape
        x = torch.concat([x[:, 0, :, :].reshape(B,1,N,D), x, x[:, T-1, :, :].reshape(B,1,N,D)], 1)

        stX = x.unfold(1,3,1)
        stX = stX.permute(0, 3, 1, 4, 2)
        stX = stX.reshape(stX.shape[0], stX.shape[1], stX.shape[2], stX.shape[3]*stX.shape[4])

        sem_q = self.sem_q_conv(stX).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(stX).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(stX).permute(0, 2, 3, 1)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1))

        geo_mask = self.assemblingMask(geo_mask)
        if geo_mask is not None:
            sem_attn.masked_fill_(geo_mask, float('-inf'))


        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, 3*N, -1)
        sem_x = sem_x.reshape(B, T, 3, N, -1)
        sem_x = sem_x[:, :, 1, :, :].reshape(B, T, N, -1)
        x = self.proj_drop(sem_x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, c_in=22, c_out=22, qkv_bias=False, attn_drop=0., proj_drop=0., device=torch.device('cuda')):
        super().__init__()
        self.sem_q_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(c_in, c_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, D, T, N = x.shape

        # x作为Q，y作为K和V
        sem_q = self.sem_q_conv(x).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(y).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(y).permute(0, 2, 3, 1)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)

        sem_x = (sem_attn @ sem_v)
        z = self.proj_drop(sem_x)
        return z

class AFF(nn.Module):
    def __init__(self, channels=22, inter_channels=6):
        super(AFF, self).__init__()

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class DSTGCN(nn.Module):
    def __init__(self, c_in, c_out, ks, N, T=12, bias=None):
        super().__init__()
        self.cheb_graph_conv1 = ChebGraphConvForBatch(c_in, c_out, ks, bias)
        self.cheb_graph_conv2 = ChebGraphConvForBatch(c_out, c_out, ks, bias)

        p_t12 = torch.randn(N, requires_grad=True)
        self.p_t12 = torch.nn.Parameter(p_t12)
        self.register_parameter("p_t12", self.p_t12)

        p_t21 = torch.randn(N, requires_grad=True)
        self.p_t21 = torch.nn.Parameter(p_t21)
        self.register_parameter("p_t21", self.p_t21)

        p_t23 = torch.randn(N, requires_grad=True)
        self.p_t23 = torch.nn.Parameter(p_t23)
        self.register_parameter("p_t23", self.p_t23)

        p_t32 = torch.randn(N, requires_grad=True)
        self.p_t32 = torch.nn.Parameter(p_t32)
        self.register_parameter("p_t32", self.p_t32)

        self.norm = nn.LayerNorm(c_out)

    def assemblingL(self, p_t12, p_t21, p_t23, p_t32, gso, B):
        N = p_t12.shape[0]
        p_t12 = torch.diag_embed(p_t12).reshape(1, N, N).repeat(B, 1, 1)
        p_t21 = torch.diag_embed(p_t21).reshape(1, N, N).repeat(B, 1, 1)
        p_t23 = torch.diag_embed(p_t23).reshape(1, N, N).repeat(B, 1, 1)
        p_t32 = torch.diag_embed(p_t32).reshape(1, N, N).repeat(B, 1, 1)

        zeroN = torch.zeros([B, N, N], device=torch.device("cuda"))
        L_row1 = torch.concat([gso, p_t12, zeroN], 1)
        L_row2 = torch.concat([p_t21, gso, p_t23], 1)
        L_row3 = torch.concat([zeroN, p_t32, gso], 1)
        return torch.concat([L_row1, L_row2, L_row3], 2)

    def forward(self, x, st_gso):
        B, T, N, D = x.shape

        st_gso = self.assemblingL(self.p_t12, self.p_t21, self.p_t23, self.p_t32, st_gso, B)

        x = torch.concat([x[:, 0, :, :].reshape(B,1,N,D), x, x[:, T - 1, :, :].reshape(B,1,N,D)], 1)

        stX = x.unfold(1,3,1)
        stX = stX.permute(0, 3, 1, 4, 2)
        stX = stX.reshape(stX.shape[0], stX.shape[1], stX.shape[2], stX.shape[3]*stX.shape[4])

        h = self.cheb_graph_conv1(stX, st_gso).permute(0, 3, 1, 2)
        h = h.reshape(h.shape[0], h.shape[1], h.shape[2], 3, -1)
        h = h[:, :, :, 1, :].reshape(h.shape[0], h.shape[1], h.shape[2], -1)
        h = self.norm(torch.permute(h, (0, 2, 3, 1)))
        return h

class LSTGCN(nn.Module):

    def __init__(self, c_in, c_out, ks, gso, bias=None):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.gso = gso
        self.bias = bias

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.cheb_graph_conv = ChebGraphConv(c_in, c_out, ks, gso, bias)

        self.conv2d = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(12, 1))
        self.sig_conv2d = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(12, 1))

    def forward(self, x):
        B, T, N, D = x.shape
        x = torch.permute(x, (0, 3, 1, 2))
        x=self.cheb_graph_conv(x).permute(0, 3, 1, 2)
        x = torch.mul(self.tanh(self.conv2d(x)), self.sigmoid(self.sig_conv2d(x)))
        x = x.permute(0, 2, 3, 1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class STEncoderBlock(nn.Module):
    def __init__(
        self, c_in, c_out, ks, gso, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cuda'), output_dim=1):
        super().__init__()
        self.norm1 = norm_layer(c_in)
        self.dstformer = DSTFormer(c_in, c_out, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, device=device)
        self.dstgcn = DSTGCN(c_in, c_out, ks, int(gso.shape[0]), bias=None)
        self.lstgcn = LSTGCN(c_in, c_out, ks, gso, bias=None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c_in)
        self.mlp = Mlp(in_features=c_in, hidden_features=c_in, act_layer=act_layer, drop=drop)

        self.aff = AFF()
        self.cross_atten = CrossAttention()
        self.device = torch.device('cuda')

    def forward(self, x, st_gso, geo_mask=None, sem_mask=None):
        norm_x = self.norm1(x)
        torch.cuda.synchronize(self.device)
        x_dg = self.dstgcn(norm_x, st_gso).permute(0, 3, 1, 2)
        torch.cuda.synchronize(self.device)

        x_lg = self.lstgcn(norm_x).repeat(1, 12, 1, 1).permute(0, 3, 1, 2)
        torch.cuda.synchronize(self.device)

        x_df = self.dstformer(norm_x, geo_mask=geo_mask, sem_mask=sem_mask).permute(0, 3, 1, 2)
        torch.cuda.synchronize(self.device)

        x_dst = self.aff(x_df, x_lg)
        x_st = self.cross_atten(x_dg, x_dst)

        x_st = torch.cat([x_dg.permute(0, 2, 3, 1), x_dst.permute(0, 2, 3, 1), x_st], dim=-1)
        x = x + self.drop_path(x_st)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class DMSTG(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)

        self.dtw_matrix = self.data_feature.get('dtw_matrix')

        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger()
        self.dataset = config.get('dataset')

        self.c_in = config.get('c_in', 66)
        self.c_out = config.get('c_out', 22)
        self.ks = config.get('ks', 3)

        self.embed_dim = 66
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        enc_depth = config.get("enc_depth", 2)
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cuda'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.dtw_delta = config.get('dtw_delta', 5)

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:

            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask[i]] = 0
            self.sem_mask = self.sem_mask.bool()


        st_gso = cal_laplacian(self.dtw_matrix)
        self.st_gso = torch.from_numpy(st_gso).float().to(self.device)

        gso = cal_laplacian(self.adj_mx)
        self.gso = torch.from_numpy(gso).float().to(self.device)

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]

        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                c_in=self.c_in, c_out=self.c_out, ks=self.ks, gso=self.gso, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=enc_dpr[i], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, output_dim=self.output_dim
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )
        self.lupdator = LUpdator(12, int(self.num_nodes))

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        enc = self.enc_embed_layer(x, lap_mx)
        skip = 0
        N = int(self.st_gso.shape[0])
        B = int(enc.shape[0])
        st_gso = self.st_gso.reshape(1, N, N)
        st_gso = st_gso.repeat(B, 1, 1)

        st_gso = self.lupdator(enc, st_gso, self.geo_mask)


        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, st_gso, self.geo_mask, self.sem_mask)

            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))

        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))

        return skip.permute(0, 3, 2, 1)

    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - 1, self.task_level))
                self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:
                return lf(y_predicted[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :])
            else:
                return lf(y_predicted, y_true)
        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)
