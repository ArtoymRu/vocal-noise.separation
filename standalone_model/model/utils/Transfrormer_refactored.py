import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from typing import Optional

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from ..utils.fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from ..utils.normalization import LayerNorm, CLayerNorm, ScaleNorm
from ..utils.conv_module import ConvModule

def exists(val):
    return val is not None

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

def default(val, d):
    return val if exists(val) else d

class FFConvM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mdl(x)

class Gated_FSMN_Block_Dilated(nn.Module):
    def __init__(self, dim, inner_channels=256, lorder=20):
        super().__init__()
        norm_klass = ScaleNorm
        self.conv1 = nn.Sequential(nn.Conv1d(dim, inner_channels, kernel_size=1), nn.GELU())
        self.norm1 = CLayerNorm(inner_channels)
        self.fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder, inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        x = self.conv1(input.transpose(2, 1))
        x = self.norm1(x)
        x = self.fsmn(x.transpose(2, 1))
        x = self.norm2(x.transpose(2, 1))
        x = self.conv2(x)
        return x.transpose(2, 1) + input

class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        dropout=0.0,
        activation=nn.GELU,
        normalize_before=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FLASHTransformer_DualA_FSMN(
                dim=d_model,
                depth=num_layers,
                group_size=256,
                query_key_dim=128,
                expansion_factor=4.,
                causal=False,
                attn_dropout=0.1,
                norm_type='scalenorm',
                shift_tokens=True
            ) for _ in range(num_layers)
        ])
        self.norm = (LayerNorm(d_model) if normalize_before else nn.Identity())

    def forward(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None):
        x = src
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
