import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .common import SingleStageTCN
from .impl.asformer import MyTransformer
from .impl.gtad import GCNeXt
from .impl.actionformer import ConvTransformerBackbone, FPN1D, PtTransformerClsHead

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class VideoClassifier(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self._fc = nn.Linear(feat_dim, feat_dim)
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self._dropout = nn.Dropout()
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._dropout(self._relu(self._fc(x)))
        return self._fc_out(x)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class GRU(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, hidden_dim)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)
    
    
class GCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim=256, num_layers=2):
        super().__init__()

        self.idx_list = []
        self.fc_in = nn.Linear(feat_dim, hidden_dim)
        gcn_layers = [GCNeXt(hidden_dim, hidden_dim, k=3, groups=32,
                             idx=self.idx_list) for _ in range(num_layers)]
        self.backbone = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=4),
            nn.ReLU(inplace=True),
            *gcn_layers
        )
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        del self.idx_list[:]
        batch_size, clip_len, _ = x.shape
        x = self.fc_in(x.view(batch_size * clip_len, -1))
        x = F.relu(x).view(batch_size, clip_len, -1)
        x = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        return self.fc(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class ActionFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, kernal_size=3, d_model=256, n_head=4, max_len=128):
        super().__init__()

        self.backbone = ConvTransformerBackbone(feat_dim, d_model, n_head, 3, max_len)
        self.neck = FPN1D([d_model], d_model)
        self.cls_head = PtTransformerClsHead(d_model, d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, C, T = x.size()
        batch_masks = torch.ones((B, 1, T), device=x.device, dtype=torch.bool)
        feats, masks = self.backbone(x, batch_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        return out_cls_logits[0].transpose(1, 2)
        

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.2,
                 maxlen: int = 200):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        # pos_embedding = torch.zeros((maxlen, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size), requires_grad=False)

        self.pos_embedding[:, :, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, :, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)].detach())


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Sequence Encoder
class Encoder(nn.Module):
    def __init__(self, num_classes, d_model, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu',
                 device=None):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.gelu = nn.GELU()
        self.tok_emb = nn.Linear(num_classes, d_model)
        self.generator = nn.Linear(d_model, num_classes)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        src = self.tok_emb(src)
        # print(src.shape)
        src = self.positional_encoding(src)
        src = self.encoder(src, mask, src_key_padding_mask, is_causal)
        src = self.gelu(src)
        # print(src.shape)
        return self.generator(src)


class EDSGPMIXERLayers(nn.Module):
    def __init__(self, feat_dim, clip_len, num_layers=1, ks=3, k=2, k_factor=2, concat=True):
        super().__init__()
        self.num_layers = num_layers
        self.tot_layers = num_layers * 2 + 1
        self._sgp = nn.ModuleList(
            SGPBlock(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1) for _ in range(self.tot_layers))
        self._pooling = nn.ModuleList(
            nn.AdaptiveMaxPool1d(output_size=math.ceil(clip_len / (k_factor ** (i + 1)))) for i in range(num_layers))
        # self._upsample = nn.ModuleList(nn.Upsample(size = math.ceil(clip_len / (k_factor**i)), mode = 'linear', align_corners = True) for i in range(num_layers))
        self._sgpMixer = nn.ModuleList(SGPMixer(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1,
                                                t_size=math.ceil(clip_len / (k_factor ** i)), concat=concat) for i in
                                       range(num_layers))

    def forward(self, x):
        store_x = []  # Store the intermediate outputs
        # Downsample
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x = self._sgp[i](x)
            store_x.append(x)
            x = self._pooling[i](x)

        # Intermediate
        x = self._sgp[self.num_layers](x)

        # Upsample
        for i in range(self.num_layers):
            x = self._sgpMixer[- (i + 1)](x=x, z=store_x[- (i + 1)])
            x = self._sgp[self.num_layers + i + 1](x)
        x = x.permute(0, 2, 1)

        return x


class SGPBlock(nn.Module):

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            mode='normal'
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        self.act = act_layer()
        self.sigm = nn.Sigmoid()
        self.reset_params(init_conv_vars=init_conv_vars)

        self.mode = mode

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x):
        # X shape: B, C, T
        B, C, T = x.shape

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        if self.mode == 'normal':
            out = fc * phi + (
                        convw + convkw) * psi + out  # fc * phi instant level / (convw + convkw) * psi window level
        elif self.mode == 'sigm1':
            out = fc * phi + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm2':
            out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm3':
            out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out
        # out = fc * phi + out #only instant level
        # out = (convw + convkw) * psi + out #only window level
        # out = fc * phi + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level
        # out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level + up branch instant-level
        # out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out # sigmoid up branch window-level + down branch instant-level

        out = x + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out


class SGPMixer(nn.Module):

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            t_size=0,
            concat=True
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.concat = concat

        if n_out is None:
            n_out = n_embd

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.psi2 = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                              groups=n_embd)
        self.convw1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw1 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.convw2 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw2 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        self.fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        self.fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        self.upsample = nn.Upsample(size=t_size, mode='linear', align_corners=True)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        if self.concat:
            self.concat_fc = nn.Conv1d(n_embd * 6, n_embd, 1, groups=group)

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.psi2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc2.weight, 0, init_conv_vars)

        torch.nn.init.constant_(self.psi1.bias, 0)
        torch.nn.init.constant_(self.psi2.bias, 0)
        torch.nn.init.constant_(self.convw1.bias, 0)
        torch.nn.init.constant_(self.convkw1.bias, 0)
        torch.nn.init.constant_(self.convw2.bias, 0)
        torch.nn.init.constant_(self.convkw2.bias, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.global_fc1.bias, 0)
        torch.nn.init.constant_(self.global_fc2.bias, 0)

        if self.concat:
            torch.nn.init.normal_(self.concat_fc.weight, 0, init_conv_vars)
            torch.nn.init.constant_(self.concat_fc.bias, 0)

    def forward(self, x, z):
        # X shape: B, C, T
        B, C, T = x.shape
        z = self.ln1(z)
        x = self.ln2(x)
        x = self.upsample(x)
        # x = self.ln2(x) # modified to have upsample inside sgp-mixer module (which seems more elegant)
        psi1 = self.psi1(z)
        psi2 = self.psi2(x)
        convw1 = self.convw1(z)
        convkw1 = self.convkw1(z)
        convw2 = self.convw2(x)
        convkw2 = self.convkw2(x)
        # Instant level branches
        fc1 = self.fc1(z)
        fc2 = self.fc2(x)
        phi1 = torch.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        phi2 = torch.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))

        out1 = (convw1 + convkw1) * psi1
        out2 = (convw2 + convkw2) * psi2
        out3 = fc1 * phi1
        out4 = fc2 * phi2

        if self.concat:
            out = torch.cat((out1, out2, out3, out4, z, x), dim=1)
            out = self.act(self.concat_fc(out))

        else:
            out = out1 + out2 + out3 + out4 + z + x

        # out = z + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(self.dropout(x).reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class FC2Layers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc1 = FCLayers(feat_dim, num_classes[0])
        self._fc2 = FCLayers(feat_dim, num_classes[1])

    def forward(self, x):
        x = torch.cat([self._fc1(x), self._fc2(x)], dim=2)
        return x

    