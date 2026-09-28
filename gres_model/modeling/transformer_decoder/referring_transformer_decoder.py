import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional, List
import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F
from einops import rearrange, repeat

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear
from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .query_generation import Query_Embedding_Padding_Simple, Query_Updating_Block
from .latent_query_generation import LatentQueryEmbedding

from .aoc import AdaptiveObjectCounting
from .dha import Dynamic_Hierarchical_Selection
from .hsd import Semantic_Decoder
from .latent_hierarchical_encoder import (
    DirectSemanticResponseMap,
    LatentEncoderLayer,
    SubjectDistributor,
    VisualConceptInjector,
    select_level_gate_tokens,
    select_semantic_cls,
)

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class NT_MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        h = [64, 128, 64]
        self.num_layers = len(h) + 1
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedReferringDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        rla_weight: float = 0.1,
        aux_loss: bool,
        weights: List[float],
        latent_enabled: bool = False,
        latent_num_tokens: List[int] = (6, 12),
        latent_drop_probs: List[float] = (0.2, 0.15),
        latent_temperature: float = 0.07,
        latent_encoder_ffn_dim: int = 1024,
        latent_encoder_dropout: float = 0.1,
        latent_pool_size: int = 30,
        latent_vc_tokens: int = 100,
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.aux_loss = aux_loss
        self.num_queries = num_queries
        self.weights = weights
        self.latent_enabled = latent_enabled
        self.num_visual_special_tokens = 1 if latent_enabled else 0
        self.original_text_tokens = 21
        self.latent_num_tokens = tuple(latent_num_tokens)

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Hierarchical Semantic Decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = 3
        self.vis_hws = [30*30, 60*60, 120*120]
        self.query_update_block = nn.ModuleList()
        if latent_enabled:
            # Exactly three EncoderLayers: one independently supplied visual
            # hierarchy per layer, while text and the subject cascade forward.
            self.decoders = nn.ModuleList(
                LatentEncoderLayer(
                    dim=hidden_dim,
                    num_heads=nheads,
                    ffn_dim=latent_encoder_ffn_dim,
                    dropout=latent_encoder_dropout,
                    original_text_tokens=self.original_text_tokens,
                    latent_tokens=self.latent_num_tokens,
                    pool_size=latent_pool_size,
                )
                for _ in range(self.num_feature_levels)
            )
            self.subject_distributor = SubjectDistributor(
                original_text_tokens=self.original_text_tokens,
                latent_tokens=self.latent_num_tokens,
            )
            # Latent-VG reuses one orthogonal concept bank throughout layers.
            self.visual_concept_injector = VisualConceptInjector(
                dim=hidden_dim,
                num_concepts=latent_vc_tokens,
                original_text_tokens=self.original_text_tokens,
                latent_tokens=self.latent_num_tokens,
            )
            self.response_maps = nn.ModuleList(
                DirectSemanticResponseMap(dim=hidden_dim)
                for _ in range(self.num_feature_levels)
            )
            selection_tokens = 1 + len(self.latent_num_tokens)
            counting_tokens = self.original_text_tokens + sum(
                length - 2 for length in self.latent_num_tokens
            )
        else:
            hierarchical_semantic_decoder = nn.ModuleList()
            for idx in range(self.num_feature_levels):
                vis_hw = self.vis_hws[idx]
                semantic_decoder = Semantic_Decoder(
                    vis_hw=vis_hw,
                    num_q=num_queries,
                    dim=hidden_dim,
                    num_visual_special_tokens=0,
                )
                hierarchical_semantic_decoder.append(semantic_decoder)
                if idx < 2:
                    self.query_update_block.append(
                        Query_Updating_Block(query_dim=hidden_dim)
                    )
            self.decoders = hierarchical_semantic_decoder
            selection_tokens = num_queries
            counting_tokens = num_queries

        ### define intra-and inter-level selection
        self.dha = Dynamic_Hierarchical_Selection(
            num_heads=nheads,
            hidden_dim=hidden_dim,
            tokens=selection_tokens,
            fixed_token_weight=(1.0 / selection_tokens) if latent_enabled else None,
        )
       
        ### define counting head
        self.counting_head = AdaptiveObjectCounting(
            input_len=counting_tokens, input_dim=hidden_dim,
            hidden_dim=hidden_dim*2, num_classes=num_classes, num_layer=3)

        if latent_enabled:
            expected_queries = 21 + sum(latent_num_tokens)
            if num_queries != expected_queries:
                raise ValueError(
                    f"Latent-LEAD expects {expected_queries} queries, got {num_queries}"
                )
            self.query_feat = LatentQueryEmbedding(
                output_dim=hidden_dim,
                num_tokens=latent_num_tokens,
                drop_probs=latent_drop_probs,
                temperature=latent_temperature,
            )
        else:
            self.query_feat = Query_Embedding_Padding_Simple(
                out_query_num=num_queries,
                output_dim=hidden_dim,
                pre_merge=True,
                pos_type='fixed',
            )

        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.nt_embed = NT_MLP(num_classes, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["aux_loss"] = cfg.MODEL.SEM_SEG_HEAD.AUX_LOSS
        ret["weights"] = cfg.MODEL.SEM_SEG_HEAD.WEIGHTS
        ret["latent_enabled"] = cfg.MODEL.MASK_FORMER.LATENT_ENABLED
        ret["latent_num_tokens"] = cfg.MODEL.MASK_FORMER.LATENT_NUM_TOKENS
        ret["latent_drop_probs"] = cfg.MODEL.MASK_FORMER.LATENT_DROP_PROBS
        ret["latent_temperature"] = cfg.MODEL.MASK_FORMER.LATENT_TEMPERATURE
        ret["latent_encoder_ffn_dim"] = cfg.MODEL.MASK_FORMER.LATENT_ENCODER_FFN_DIM
        ret["latent_encoder_dropout"] = cfg.MODEL.MASK_FORMER.LATENT_ENCODER_DROPOUT
        ret["latent_pool_size"] = cfg.MODEL.MASK_FORMER.LATENT_POOL_SIZE
        ret["latent_vc_tokens"] = cfg.MODEL.MASK_FORMER.LATENT_VC_TOKENS
        return ret

    def forward(self, x, mask_features, lang_feat, lang_sent=None, lang_mask=None):
        # x is a list of multi-scale feature
        # lang_feat: [B d l]
        # lang_sent: [B D]
        assert len(x) == self.num_feature_levels
        query_result = self.query_feat(lang_feat, lang_sent, lang_mask)
        if self.latent_enabled:
            query, visual_subject = query_result
        else:
            query = query_result
            visual_subject = None

        src = []
        size_list = []
        feature_inputs = list(x) + [mask_features]
        for i in range(self.num_feature_levels):
            size_list.append(feature_inputs[i+1].shape[-2:])
            visual_tokens = (
                self.input_proj[i](feature_inputs[i+1]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            src.append(visual_tokens)

        size_list.append(mask_features.shape[-2:])

        query_embed = None

        outputs = []
        semantic_outputs = []
        level_gate_outputs = []
        attns = []
        if self.latent_enabled:
            current_text = query
            current_subject = visual_subject
            for idx, encoder_layer in enumerate(self.decoders):
                h, w = size_list[idx]
                current_text, current_subject, restored_visual = encoder_layer(
                    src[idx].transpose(1, 2),
                    current_subject,
                    current_text,
                    (h, w),
                )
                if not torch.isfinite(current_text).all() or not torch.isfinite(
                    current_subject
                ).all():
                    raise FloatingPointError(
                        f"Non-finite values after LatentEncoderLayer stage {idx}"
                    )
                current_text = self.subject_distributor(
                    current_text, current_subject
                )
                current_text = self.visual_concept_injector(
                    restored_visual, current_text, current_subject
                )
                if not torch.isfinite(current_text).all():
                    raise FloatingPointError(
                        f"Non-finite values after VisualConceptInjector stage {idx}"
                    )

                semantic_cls = select_semantic_cls(
                    current_text,
                    original_text_tokens=self.original_text_tokens,
                    latent_tokens=self.latent_num_tokens,
                )
                level_gate_tokens = select_level_gate_tokens(
                    current_text,
                    original_text_tokens=self.original_text_tokens,
                    latent_tokens=self.latent_num_tokens,
                )
                # restored_visual contains spatial positions only; the subject
                # never enters the direct 256-D response-map multiplication.
                attn = self.response_maps[idx](restored_visual, semantic_cls)
                if not torch.isfinite(attn).all():
                    raise FloatingPointError(
                        f"Non-finite semantic response map at stage {idx}"
                    )
                attn = rearrange(
                    attn, "b q (h w) -> b q h w", h=h, w=w
                )
                attns.append(attn)
                outputs.append(current_text)
                semantic_outputs.append(semantic_cls)
                level_gate_outputs.append(level_gate_tokens)
        else:
            vis_pre = None
            output = None
            for idx, decoder in enumerate(self.decoders):
                h, w = size_list[idx]
                output, attn, count, vis_pre = decoder(
                    query,
                    src[idx],
                    query_pos=query_embed,
                    vis_pre=vis_pre,
                )
                if idx < 2:
                    output = self.query_update_block[idx](output, query)
                    query = output

                attn = rearrange(
                    attn, "b n q (h w) -> b n q h w", h=h, w=w
                )
                attns.append(attn)
                outputs.append(output)
        
        prediction_masks = []

        pred_outputs = torch.stack(outputs)
        counting_inputs = (
            torch.stack(level_gate_outputs)
            if self.latent_enabled
            else pred_outputs
        )
        # Small output heads are inexpensive in FP32 and are shared by all
        # losses, so keeping them out of autocast prevents a single overflow
        # from contaminating mask, no-target, and count predictions together.
        with torch.cuda.amp.autocast(enabled=False):
            counting_pred, _ = self.counting_head(
                counting_inputs.float(), None, aggregate=True
            )
        dha_outputs = level_gate_outputs if self.latent_enabled else outputs
        mask_inputs = (
            torch.stack(semantic_outputs)
            if self.latent_enabled
            else pred_outputs
        )
        with torch.cuda.amp.autocast(enabled=False):
            semantic_region_embeds = self.mask_embed(mask_inputs.float())
            class_embed = self.class_embed(semantic_region_embeds) #l b q a=2
        prediction_masks = self.dha(
            attns, dha_outputs, class_embed, weights=self.weights
        )
        layer_outputs = []
        for pm in prediction_masks:
            layer_output = {
                "pred_masks": pm,
            }
            layer_outputs.append(layer_output)
        out = layer_outputs[-1]
        out["pred_count"] = counting_pred
        nt_feat = counting_pred.detach()
        with torch.cuda.amp.autocast(enabled=False):
            nt_pred = self.nt_embed(nt_feat.float())
        out['nt_label'] = nt_pred
        if self.aux_loss:
            out["aux_outputs"] = layer_outputs[:-1]
        return out

    def semantic_inference(self, output, attns):
        """
        output: [b nq c]
        attns: [b nq h w]
        """
        #[b c h w]
        output = F.softmax(output, dim=-1)
        mask_pred = attns.sigmoid()
        mask_features = torch.einsum("bqc, bqhw -> bchw", output, mask_pred)
        return mask_features
    
