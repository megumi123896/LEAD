import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
from einops import rearrange
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(1, in_channels // ratio)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class Dynamic_Hierarchical_Selection(torch.nn.Module):
    """
    Intra- and Inter-Selection on the object query
    """
    def __init__(self, levels=3, num_heads=8, hidden_dim=256, tokens=21,
                 fixed_token_weight=None):
        super().__init__()
        self.num_heads = num_heads
        self.level = levels
        self.tokens = tokens
        self.fixed_token_weight = fixed_token_weight

        self.level_gating = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        if fixed_token_weight is None:
            self.token_gates = nn.ModuleList([
                ChannelAttention(in_channels=tokens, ratio=4) for _ in range(levels)
            ])
        else:
            self.token_gates = None


    def forward(self, attn_maps: List[Tensor], outputs: List[Tensor], kernels: List[Tensor], weights: List[float]):
        # attn_maps: baseline [B, n_h, N, H, W] or direct [B, N, H, W]
        # kernels: List[kn=[B, N, 2]]
        # output: List[out=[B, N, C]]
        # return: List[mask=[B, 2, H, W]]

        semantic_features = [out.detach().mean(dim=1) for out in outputs]    # l * [B C]
        scores = self.level_gating(torch.stack(semantic_features))    # [l, B, 1]

        outputs_seg_masks = []
        prev_attn_mask = None
        cur_attn_mask = None

        for i_attn, attn in enumerate(attn_maps):
            if attn.dim() == 5:
                # Original LEAD path: merge its response-map heads.
                attn = attn.sum(dim=1) / self.num_heads
            elif attn.dim() != 4:
                raise ValueError(
                    f"Expected a 4-D or 5-D response map, got {attn.dim()}-D"
                )

            if self.fixed_token_weight is None:
                token_score = self.token_gates[i_attn](attn)
                attn = attn * token_score.view(-1, self.tokens, 1, 1)
            else:
                # Latent path: keep all three maps and assign each a fixed 1/3.
                if attn.size(1) != self.tokens:
                    raise ValueError(
                        f"Expected {self.tokens} response maps, got {attn.size(1)}"
                    )
                attn = attn * self.fixed_token_weight

            attn = attn * scores[i_attn].view(-1, 1, 1, 1)             # dynamic level weight

            prev_attn_mask = attn
            if cur_attn_mask is None:
                size = attn_maps[i_attn+1].size()[-2:]
                cur_attn_mask = F.interpolate(prev_attn_mask, size=size, mode='bilinear', align_corners=False)
                outputs_seg_masks.append(prev_attn_mask)
            else:
                cur_attn_mask =  weights[i_attn - 1] * cur_attn_mask + prev_attn_mask
                outputs_seg_masks.append(cur_attn_mask)
                if i_attn < len(attn_maps) - 1:
                    size = attn_maps[i_attn+1].size()[-2:]
                    cur_attn_mask = F.interpolate(cur_attn_mask, size=size, mode="bilinear", align_corners=False)

        prediction_masks = []
        for kn, pred_mask in zip(kernels, outputs_seg_masks):
            # Keep the final dynamic convolution in FP32. Otherwise global
            # autocast may cast the stabilized response maps back to FP16.
            with torch.cuda.amp.autocast(enabled=False):
                mask = torch.einsum(
                    "bqa, bqhw -> bahw",
                    kn.float(),
                    pred_mask.float(),
                )
            prediction_masks.append(mask)
        return prediction_masks
