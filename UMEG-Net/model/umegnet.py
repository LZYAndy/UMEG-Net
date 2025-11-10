import copy as cp
import torch
import torch.nn as nn
import torchvision
import timm
import math
import numpy as np
import hashlib
import os
import os.path as osp
import requests
from einops import rearrange, repeat
from mmengine.runner import load_checkpoint
from mmcv.cnn import build_norm_layer, build_activation_layer

EPS = 1e-4

def download_file(url, filename=None):
    """
    Download file from a given URL.
    """
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)

def cache_checkpoint(filename, cache_dir='.cache'):
    """
    Download and cache remote checkpoint file.
    Returns the local filename.
    """
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = osp.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not osp.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename

def conv_init(conv):
    """
    Kaiming initialization for Conv2d layer weights.
    """
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """
    Initialize BatchNorm weights and bias.
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_blockgcn(nn.Module):
    """
    Block Graph Convolutional Network (GCN) unit with multi-head, hop-based attention, and position encoding.
    """
    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super(unit_blockgcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1

        # Learnable adjacency matrices for each head (and hop)
        self.fc1 = nn.Parameter(
            torch.stack([
                torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0)
                for _ in range(3)
            ], dim=0), requires_grad=True
        )
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])

        # Downsample if input/output channels do not match
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Layer initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        # Compute k-hop matrices and hop distances
        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(A.shape[-1])]
        h[0] = np.eye(A.shape[-1])
        h[1] = h1
        self.hops = 0 * h[0]
        for i in range(2, A.shape[-1]):
            h[i] = h[i - 1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1
        for i in range(A.shape[-1] - 1, 0, -1):
            if torch.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i * np.array(h[i])
            else:
                continue
        self.hops = torch.tensor(self.hops).long()

        # Learnable relative positional encoding for hops
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, self.hops.max() + 1,)))
        self.in_channels = in_channels
        self.hidden_channels = in_channels if in_channels > 3 else 64

        # Optional learnable scaling
        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

        # Redundant double initialization to be robust
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def L2_norm(self, weight):
        """
        Normalize adjacency matrices by L2 norm.
        """
        weight_norm = torch.norm(weight, 2, dim=-2, keepdim=True) + EPS
        return weight_norm

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        pos_emb = self.rpe[:, :, self.hops]  # [3, num_heads, V, V]
        for i in range(3):
            weight_norm = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i] / weight_norm
            # Add positional encoding
            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z = torch.einsum("nhctv, hvw->nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

def edge2mat(link, num_node):
    """
    Convert edge list to adjacency matrix.
    """
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A, dim=0):
    """
    Normalize a directed graph adjacency matrix.
    """
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_hop_distance(num_node, edge, max_hop=1):
    """
    Compute hop distance between every pair of nodes.
    """
    A = np.eye(num_node)
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['openpose', 'nturgb+d', 'coco', 'handmp', 'sports_individual', 'sports_racket', 'sports_soccer']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout == 'sports_individual':
            self.num_node = 12
            self.inward = [
                # keypoints
                (10, 8), (8, 6), (11, 9), (9, 7), (6, 0), (7, 1),
                (4, 2), (2, 0), (5, 3), (3, 1), (7, 6), (1, 0)
            ]
            self.center = 0
        elif layout == 'sports_racket':
            self.num_node = 29
            self.inward = [
                # keypoints
                (10, 8), (8, 6), (11, 9), (9, 7), (6, 0), (7, 1),
                (4, 2), (2, 0), (5, 3), (3, 1), (7, 6), (1, 0),
                (22, 20), (20, 18), (23, 21), (21, 19), (18, 12), (19, 13),
                (16, 14), (14, 12), (17, 15), (15, 13), (19, 18), (13, 12),

                # court-player
                (25, 24), (26, 24), (27, 26), (27, 25),
                (24, 10), (25, 10), (26, 10), (27, 10),
                (24, 11), (25, 11), (26, 11), (27, 11),
                (24, 22), (25, 22), (26, 22), (27, 22),
                (24, 23), (25, 23), (26, 23), (27, 23),

                # ball-player
                (28, 4), (28, 5), (28, 16), (28, 17)
            ]
            self.center = 0
        elif layout == 'sports_soccer':
            self.num_node = 265
            self.inward = [
                # keypoints
                (10, 8), (8, 6), (11, 9), (9, 7), (6, 0), (7, 1),
                (4, 2), (2, 0), (5, 3), (3, 1), (7, 6), (1, 0),

                (22, 20), (20, 18), (23, 21), (21, 19), (18, 12), (19, 13),
                (16, 14), (14, 12), (17, 15), (15, 13), (19, 18), (13, 12),

                (34, 32), (32, 30), (35, 33), (33, 31), (30, 24), (31, 25),
                (28, 26), (26, 24), (29, 27), (27, 25), (31, 30), (25, 24),

                (46, 44), (44, 42), (47, 45), (45, 43), (42, 36), (43, 37),
                (40, 38), (38, 36), (41, 39), (39, 37), (43, 42), (37, 36),

                (58, 56), (56, 54), (59, 57), (57, 55), (54, 48), (55, 49),
                (52, 50), (50, 48), (53, 51), (51, 49), (55, 54), (49, 48),

                (70, 68), (68, 66), (71, 69), (69, 67), (66, 60), (67, 61),
                (64, 62), (62, 60), (65, 63), (63, 61), (67, 66), (61, 60),

                (82, 80), (80, 78), (83, 81), (81, 79), (78, 72), (79, 73),
                (76, 74), (74, 72), (77, 75), (75, 73), (79, 78), (73, 72),

                (94, 92), (92, 90), (95, 93), (93, 91), (90, 84), (91, 85),
                (88, 86), (86, 84), (89, 87), (87, 85), (91, 90), (85, 84),

                (106, 104), (104, 102), (107, 105), (105, 103), (102, 96), (103, 97),
                (100, 98), (98, 96), (101, 99), (99, 97), (103, 102), (97, 96),

                (118, 116), (116, 114), (119, 117), (117, 115), (114, 108), (115, 109),
                (112, 110), (110, 108), (113, 111), (111, 109), (115, 114), (109, 108),

                (130, 128), (128, 126), (131, 129), (129, 127), (126, 120), (127, 121),
                (124, 122), (122, 120), (125, 123), (123, 121), (127, 126), (121, 120),

                (142, 140), (140, 138), (143, 141), (141, 139), (138, 132), (139, 133),
                (136, 134), (134, 132), (137, 135), (135, 133), (139, 138), (133, 132),

                (154, 152), (152, 150), (155, 153), (153, 151), (150, 144), (151, 145),
                (148, 146), (146, 144), (149, 147), (147, 145), (151, 150), (145, 144),

                (166, 164), (164, 162), (167, 165), (165, 163), (162, 156), (163, 157),
                (160, 158), (158, 156), (161, 159), (159, 157), (163, 162), (157, 156),

                (178, 176), (176, 174), (179, 177), (177, 175), (174, 168), (175, 169),
                (172, 170), (170, 168), (173, 171), (171, 169), (175, 174), (169, 168),

                (190, 188), (188, 186), (191, 189), (189, 187), (186, 180), (187, 181),
                (184, 182), (182, 180), (185, 183), (183, 181), (187, 186), (181, 180),

                (202, 200), (200, 198), (203, 201), (201, 199), (198, 192), (199, 193),
                (196, 194), (194, 192), (197, 195), (195, 193), (199, 198), (193, 192),

                (214, 212), (212, 210), (215, 213), (213, 211), (210, 204), (211, 205),
                (208, 206), (206, 204), (209, 207), (207, 205), (211, 210), (205, 204),

                (226, 224), (224, 222), (227, 225), (225, 223), (222, 216), (223, 217),
                (220, 218), (218, 216), (221, 219), (219, 217), (223, 222), (217, 216),

                (238, 236), (236, 234), (239, 237), (237, 235), (234, 228), (235, 229),
                (232, 230), (230, 228), (233, 231), (231, 229), (235, 234), (229, 228),

                (250, 248), (248, 246), (251, 249), (249, 247), (246, 240), (247, 241),
                (244, 242), (242, 240), (245, 243), (243, 241), (247, 246), (241, 240),

                (262, 260), (260, 258), (263, 261), (261, 259), (258, 252), (259, 253),
                (256, 254), (254, 252), (257, 255), (255, 253), (259, 258), (253, 252),

                # ball-player
                (264, 0), (264, 1), (264, 10), (264, 11),
                (264, 12), (264, 13), (264, 22), (264, 23),
                (264, 24), (264, 25), (264, 34), (264, 35),
                (264, 36), (264, 37), (264, 46), (264, 47),
                (264, 48), (264, 49), (264, 58), (264, 59),
                (264, 60), (264, 61), (264, 70), (264, 71),
                (264, 72), (264, 73), (264, 82), (264, 83),
                (264, 84), (264, 85), (264, 94), (264, 95),
                (264, 96), (264, 97), (264, 106), (264, 107),
                (264, 108), (264, 109), (264, 118), (264, 119),
                (264, 120), (264, 121), (264, 130), (264, 131),
                (264, 132), (264, 133), (264, 142), (264, 143),
                (264, 144), (264, 145), (264, 154), (264, 155),
                (264, 156), (264, 157), (264, 166), (264, 167),
                (264, 168), (264, 169), (264, 178), (264, 179),
                (264, 180), (264, 181), (264, 190), (264, 191),
                (264, 192), (264, 193), (264, 202), (264, 203),
                (264, 204), (264, 205), (264, 214), (264, 215),
                (264, 216), (264, 217), (264, 226), (264, 227),
                (264, 228), (264, 229), (264, 238), (264, 239),
                (264, 240), (264, 241), (264, 250), (264, 251),
                (264, 252), (264, 253), (264, 262), (264, 263)
            ]
            self.center = 0
        elif layout == 'handmp':
            self.num_node = 21
            self.inward = [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13),
                (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)
            ]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off


class unit_tcn(nn.Module):
    """
    Temporal Convolutional Network (TCN) block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0), stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)

def shift(x, fold_div=8, step=1):
    """
    Channel shift along time for temporal modeling.
    """
    n, c, t, v = x.size()
    fold = c // fold_div
    if fold == 0:
        return x
    out = torch.zeros_like(x)
    # shift left
    out[:, :fold, :-step, :] = x[:, :fold, step:, :]
    # shift right
    out[:, fold:2*fold, step:, :] = x[:, fold:2*fold, :-step, :]
    # keep the rest
    out[:, 2*fold:, :, :] = x[:, 2*fold:, :, :]
    return out

class UMEGBlock(nn.Module):
    """
    UMEGBlock: Stacks a Block-GCN, temporal shifts, and channel transforms.
    """
    def __init__(self, in_channels, out_channels, A, stride=1, steps=[1, 2, 4], residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_blockgcn(in_channels, out_channels, A)
        self.steps = steps
        branch_c = out_channels // len(steps)
        tin_channels = branch_c * len(steps)
        self.down = nn.Conv2d(out_channels, out_channels // len(steps), kernel_size=1)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels),
            nn.ReLU(),
            nn.Conv2d(tin_channels, out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU()
        # Residual connection logic
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        out = []
        for step in self.steps:
            feat = self.down(self.gcn(shift(x, step=step)))
            out.append(feat)
        out = torch.cat(out, dim=1)
        out = self.transform(out) + res
        return out


class UMEG(nn.Module):
    """
    Unified Multi-Entity Graph (UMEG) Backbone for skeleton/keypoint-based feature extraction in sports analytics.

    This model stacks multiple UMEGBlocks, with each block composed of a spatial GCN and temporal modules.
    The underlying graph is chosen based on the number of persons/entities (e.g., racket sports, soccer, individual sports).

    Args:
        in_channels (int): Number of input channels (e.g., 3 for (x,y,conf) or 2 for (x,y)).
        base_channels (int): Base number of channels for the first stage.
        data_bn_type (str): Type of batch normalization. Supports 'MVC' (multi-person), 'VC' (single person), or 'None'.
        ch_ratio (float): Channel width multiplier for each stage.
        num_person (int): Number of persons/entities (for graph layout selection).
        num_stages (int): Number of UMEGBlocks to stack.
        inflate_stages, down_stages: Stages at which to increase width or downsample.
        steps: List of temporal shift steps for each block.
        pretrained: Optional pretrained weights path.
    """

    def __init__(self,
                 graph_cfg=None,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='MVC',
                 ch_ratio=2,
                 num_person=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 steps=[1, 2, 4],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        # 1. Choose the graph topology based on number of persons
        if num_person > 2:
            self.graph = Graph(layout='sports_soccer', mode='spatial')
        elif num_person == 2:
            self.graph = Graph(layout='sports_racket', mode='spatial')
        else:
            self.graph = Graph(layout='sports_individual', mode='spatial')

        # 2. Prepare adjacency matrix
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        # 3. Data batch normalization (along VC or MVC axes)
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        # 4. Expand block-specific kwargs per stage, supporting tuple overrides
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)  # Remove dropout for first stage

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        self.num_person = num_person

        # 5. Build the stack of UMEGBlocks
        modules = []
        # First block handles in_channels → base_channels
        if self.in_channels != self.base_channels:
            modules.append(
                UMEGBlock(in_channels, base_channels, A.clone(), 1,
                          residual=False, steps=steps, **lw_kwargs[0])
            )
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1  # Optionally: + (i in down_stages) for downsampling
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(
                UMEGBlock(in_channels, out_channels, A.clone(), stride,
                          steps=steps, **lw_kwargs[i - 1])
            )
        # If in_channels == base_channels, we have one less stage
        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        """
        Load pretrained weights if provided.
        """
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x_sk, sp):
        """
        Forward pass.
        Args:
            x_sk: Skeleton/keypoint input tensor [N, M, T, V, C]
            sp:   Supplementary features (e.g., court/ball/others) [N, T, ...]
        Returns:
            Feature tensor: [N, T, C_out]
        """
        # Slice and format input: only use selected persons and remove first 5 joints
        x_sk = x_sk[:, :self.num_person, :, 5:, :]
        N, M, T, V, C = x_sk.size()
        # Reshape and merge persons/joints
        x_sk = x_sk.transpose(1, 2)  # [N, T, M, V, C]
        x_sk = x_sk.reshape(N, T, M * V, C)
        sp = sp.view(N, T, -1, 2)
        # Concatenate supplementary features as additional nodes
        if M < 2:
            x = x_sk
        elif M > 2:
            x = torch.cat((x_sk, sp[:, :, 4:, :]), dim=2)
        else:
            x = torch.cat((x_sk, sp), dim=2)
        # Handle NaNs and permute to [N, C, T, V]
        x = torch.nan_to_num(x, nan=0.0).permute(0, 3, 1, 2)

        # Pass through GCN backbone
        N, C, T, V = x.shape
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.view(N, -1, T, V)
        # Average over joints, then transpose to [N, T, C]
        x = x.mean(-1).transpose(1, 2)
        return x
