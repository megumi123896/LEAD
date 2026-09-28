"""Latent-VG-style hierarchical encoder used by the Latent-LEAD path.

The spatial feature of every hierarchy is supplied independently by LEAD's
deformable pixel decoder.  Self-attention always operates on a 30 x 30 visual
grid, while its learned residual is restored to the native hierarchy before
the Subject Distributor, Visual Concept Injector, and response-map branch.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """The full visual/original-text FFN used by the Latent encoder layer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.activation_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        return self.output_dropout(x)


class LatentEncoderLayer(nn.Module):
    """One 256-D shared-MSA, group-specific-FFN Latent encoder layer."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        original_text_tokens: int = 21,
        latent_tokens: Sequence[int] = (6, 12),
        pool_size: int = 30,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.original_text_tokens = original_text_tokens
        self.latent_tokens = tuple(latent_tokens)
        self.text_tokens = original_text_tokens + sum(self.latent_tokens)
        self.pool_size = pool_size
        self.visual_tokens = pool_size * pool_size + 1

        # Latent-VG applies independent pre-norms to visual, original-text,
        # and every latent-expression group, followed by one shared MSA.
        group_count = 2 + len(self.latent_tokens)
        self.attention_norms = nn.ModuleList(
            nn.LayerNorm(dim) for _ in range(group_count)
        )
        self.self_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norms = nn.ModuleList(
            nn.LayerNorm(dim) for _ in range(group_count)
        )
        self.visual_ffn = FeedForwardNetwork(dim, ffn_dim, dropout)
        self.original_text_ffn = FeedForwardNetwork(dim, ffn_dim, dropout)
        # The released Latent-VG multiway wrapper uses this lighter branch
        # (Linear -> LayerNorm -> GELU) for each latent expression.
        self.latent_ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            for _ in self.latent_tokens
        )

    def _split_groups(
        self, x: torch.Tensor, visual_length: int
    ) -> List[torch.Tensor]:
        expected = visual_length + self.text_tokens
        if x.size(1) != expected:
            raise ValueError(f"Expected {expected} joint tokens, got {x.size(1)}")

        groups = [x[:, :visual_length, :]]
        start = visual_length
        groups.append(x[:, start:start + self.original_text_tokens, :])
        start += self.original_text_tokens
        for length in self.latent_tokens:
            groups.append(x[:, start:start + length, :])
            start += length
        return groups

    def _group_norm(
        self,
        x: torch.Tensor,
        norms: nn.ModuleList,
        visual_length: int,
    ) -> torch.Tensor:
        groups = self._split_groups(x, visual_length)
        return torch.cat(
            [norm(group) for norm, group in zip(norms, groups)], dim=1
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        visual_subject: torch.Tensor,
        text_tokens: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_tokens: native deformable feature, [B, H*W, D].
            visual_subject: propagated subject token, [B, 1, D].
            text_tokens: cascaded [original, latent-1, latent-2], [B, 39, D].
            spatial_size: native (H, W) for this hierarchy.

        Returns:
            updated_text: [B, 39, D].
            updated_subject: [B, 1, D].
            restored_visual: native-resolution [B, H*W, D].
        """
        batch_size, native_length, dim = visual_tokens.shape
        height, width = spatial_size
        if native_length != height * width:
            raise ValueError(
                f"Visual length {native_length} does not match {height}x{width}"
            )
        if dim != self.dim:
            raise ValueError(f"Expected visual dim {self.dim}, got {dim}")
        if visual_subject.shape != (batch_size, 1, dim):
            raise ValueError(
                "visual_subject must have shape "
                f"{(batch_size, 1, dim)}, got {tuple(visual_subject.shape)}"
            )
        if text_tokens.size(1) != self.text_tokens:
            raise ValueError(
                f"Expected {self.text_tokens} text tokens, got {text_tokens.size(1)}"
            )

        native_map = visual_tokens.transpose(1, 2).reshape(
            batch_size, dim, height, width
        )
        pooled_map = F.adaptive_avg_pool2d(
            native_map, (self.pool_size, self.pool_size)
        )
        pooled_tokens = pooled_map.flatten(2).transpose(1, 2)

        visual_length = 1 + self.pool_size * self.pool_size
        joint_tokens = torch.cat(
            [visual_subject, pooled_tokens, text_tokens], dim=1
        )

        residual = joint_tokens
        normalized = self._group_norm(
            joint_tokens, self.attention_norms, visual_length
        )
        attention_output, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        joint_tokens = residual + self.attention_dropout(attention_output)

        residual = joint_tokens
        normalized = self._group_norm(
            joint_tokens, self.ffn_norms, visual_length
        )
        groups = self._split_groups(normalized, visual_length)
        ffn_groups = [
            self.visual_ffn(groups[0]),
            self.original_text_ffn(groups[1]),
        ]
        ffn_groups.extend(
            ffn(group) for ffn, group in zip(self.latent_ffns, groups[2:])
        )
        joint_tokens = residual + torch.cat(ffn_groups, dim=1)

        updated_subject = joint_tokens[:, :1, :]
        updated_pooled = joint_tokens[:, 1:visual_length, :]
        updated_text = joint_tokens[:, visual_length:, :]

        updated_pooled_map = updated_pooled.transpose(1, 2).reshape(
            batch_size, dim, self.pool_size, self.pool_size
        )
        # Preserve native deformable detail and upsample only the change learned
        # by the 30x30 Latent encoder. At 30x30 this is exactly updated_pooled.
        attention_delta = updated_pooled_map - pooled_map
        if (height, width) != (self.pool_size, self.pool_size):
            attention_delta = F.interpolate(
                attention_delta,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        restored_map = native_map + attention_delta
        restored_visual = restored_map.flatten(2).transpose(1, 2)
        return updated_text, updated_subject, restored_visual


class SubjectDistributor(nn.Module):
    """Copy the current visual subject into every latent expression."""

    def __init__(
        self,
        original_text_tokens: int = 21,
        latent_tokens: Sequence[int] = (6, 12),
    ) -> None:
        super().__init__()
        self.original_text_tokens = original_text_tokens
        self.latent_tokens = tuple(latent_tokens)

    def forward(
        self, text_tokens: torch.Tensor, visual_subject: torch.Tensor
    ) -> torch.Tensor:
        original = text_tokens[:, :self.original_text_tokens, :]
        start = self.original_text_tokens
        distributed = []
        for length in self.latent_tokens:
            expression = text_tokens[:, start:start + length, :]
            distributed.append(
                torch.cat(
                    [expression[:, :1, :], visual_subject, expression[:, 2:, :]],
                    dim=1,
                )
            )
            start += length
        if start != text_tokens.size(1):
            raise ValueError(
                f"Unexpected text length {text_tokens.size(1)}; parsed {start}"
            )
        return torch.cat([original, *distributed], dim=1)


class VisualConceptInjector(nn.Module):
    """Released Latent-VG visual concept competition, adapted to 256-D."""

    def __init__(
        self,
        dim: int = 256,
        num_concepts: int = 100,
        original_text_tokens: int = 21,
        latent_tokens: Sequence[int] = (6, 12),
    ) -> None:
        super().__init__()
        if num_concepts > dim:
            raise ValueError(
                f"Orthogonal concept count {num_concepts} cannot exceed dim {dim}"
            )
        concepts, _ = torch.linalg.qr(
            torch.randn(dim, num_concepts), mode="reduced"
        )
        self.concept_tokens = nn.Parameter(concepts.transpose(0, 1))
        self.original_text_tokens = original_text_tokens
        self.latent_tokens = tuple(latent_tokens)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_subject: torch.Tensor,
    ) -> torch.Tensor:
        original = text_tokens[:, :self.original_text_tokens, :]
        original_cls = original[:, :1, :]

        latent_groups = []
        latent_attributes = []
        start = self.original_text_tokens
        for length in self.latent_tokens:
            expression = text_tokens[:, start:start + length, :]
            latent_groups.append(expression)
            latent_attributes.append(expression[:, 2:, :])
            start += length
        if start != text_tokens.size(1):
            raise ValueError(
                f"Unexpected text length {text_tokens.size(1)}; parsed {start}"
            )

        attributes = torch.cat(latent_attributes, dim=1)

        # Normalization eps=1e-12 and masked softmax are numerically fragile in
        # FP16. Run only the VCI competition in FP32 and return to the model's
        # original dtype afterwards.
        with torch.cuda.amp.autocast(enabled=False):
            visual_fp32 = visual_tokens.float()
            original_cls_fp32 = original_cls.float()
            visual_norm = F.normalize(
                visual_fp32, p=2, dim=-1, eps=1e-6
            )
            text_cls_norm = F.normalize(
                original_cls_fp32, p=2, dim=-1, eps=1e-6
            )
            concept_region_map = torch.einsum(
                "bnd,bmd->bnm", visual_norm, text_cls_norm
            )
            mean_similarity = concept_region_map.mean(dim=1, keepdim=True)
            target_region = concept_region_map >= mean_similarity

            concepts = self.concept_tokens.float().unsqueeze(0).expand(
                visual_tokens.size(0), -1, -1
            )
            concept_norm = F.normalize(
                concepts, p=2, dim=-1, eps=1e-6
            )
            concept_scores = torch.einsum(
                "btd,bfd->btf", concept_norm, visual_norm
            )
            # A large finite negative value is safer than -inf for AMP
            # forward/backward while remaining effectively zero after softmax.
            concept_scores = concept_scores.masked_fill(
                ~target_region.transpose(1, 2), -1e4
            )
            concept_attention = F.softmax(
                concept_scores, dim=-1, dtype=torch.float32
            )
            updated_concepts = torch.einsum(
                "btf,bfd->btd", concept_attention, visual_fp32
            )

            attribute_norm = F.normalize(
                attributes.float(), p=2, dim=-1, eps=1e-6
            )
            updated_concept_norm = F.normalize(
                updated_concepts, p=2, dim=-1, eps=1e-6
            )
            attribute_scores = torch.einsum(
                "btd,bfd->btf", attribute_norm, updated_concept_norm
            )
            # Keep the released implementation's competition across all
            # latent attribute slots for each visual concept.
            attribute_attention = F.softmax(
                attribute_scores, dim=1, dtype=torch.float32
            )
            updated_attributes = torch.einsum(
                "btf,bfd->btd", attribute_attention, updated_concepts
            ).to(dtype=text_tokens.dtype)

        rebuilt_groups = []
        attribute_start = 0
        for length, expression in zip(self.latent_tokens, latent_groups):
            attribute_end = attribute_start + length - 2
            rebuilt_groups.append(
                torch.cat(
                    [
                        expression[:, :1, :],
                        visual_subject,
                        updated_attributes[:, attribute_start:attribute_end, :],
                    ],
                    dim=1,
                )
            )
            attribute_start = attribute_end
        return torch.cat([original, *rebuilt_groups], dim=1)


class DirectSemanticResponseMap(nn.Module):
    """Direct 256-D similarity between restored image tokens and three CLSs."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

    def forward(
        self, visual_tokens: torch.Tensor, semantic_cls: torch.Tensor
    ) -> torch.Tensor:
        if visual_tokens.size(-1) != self.dim or semantic_cls.size(-1) != self.dim:
            raise ValueError(
                f"Direct response map expects {self.dim}-D image and CLS tokens"
            )
        # [B, 3, D] @ [B, D, HW] -> [B, 3, HW]. There are no additional
        # key/value projections and no second multi-head split at this stage.
        # The unnormalized residual features can overflow an FP16 dot product.
        # Keep response maps in FP32; the final dynamic mask branch also
        # consumes them in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            return (
                semantic_cls.float() @ visual_tokens.float().transpose(-2, -1)
            ) * self.scale


def select_semantic_cls(
    text_tokens: torch.Tensor,
    original_text_tokens: int = 21,
    latent_tokens: Sequence[int] = (6, 12),
) -> torch.Tensor:
    """Select pooler CLS plus the two raw-BERT latent-expression CLS tokens."""
    cls_tokens = [text_tokens[:, :1, :]]
    start = original_text_tokens
    for length in latent_tokens:
        cls_tokens.append(text_tokens[:, start:start + 1, :])
        start += length
    if start != text_tokens.size(1):
        raise ValueError(f"Unexpected text length {text_tokens.size(1)}; parsed {start}")
    return torch.cat(cls_tokens, dim=1)


def select_level_gate_tokens(
    text_tokens: torch.Tensor,
    original_text_tokens: int = 21,
    latent_tokens: Sequence[int] = (6, 12),
) -> torch.Tensor:
    """Select 21 original tokens plus the 4/10 generated latent attributes."""
    gate_tokens = [text_tokens[:, :original_text_tokens, :]]
    start = original_text_tokens
    for length in latent_tokens:
        # Exclude the copied CLS and distributed subject from each expression.
        gate_tokens.append(text_tokens[:, start + 2:start + length, :])
        start += length
    if start != text_tokens.size(1):
        raise ValueError(f"Unexpected text length {text_tokens.size(1)}; parsed {start}")
    return torch.cat(gate_tokens, dim=1)
