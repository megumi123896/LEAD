import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .query_generation import FixedAbsolutePositionEmbedding


class LatentExpressionGenerator(nn.Module):
    """Generate Latent-VG-style expressions from BERT token features."""

    def __init__(
        self,
        input_len=20,
        hidden_dim=768,
        num_tokens=(6, 12),
        drop_probs=(0.2, 0.15),
        temperature=0.07,
    ):
        super().__init__()
        if len(num_tokens) != len(drop_probs):
            raise ValueError("num_tokens and drop_probs must have the same length")
        if any(token_num < 3 for token_num in num_tokens):
            raise ValueError("Each latent expression needs CLS, subject, and attribute tokens")

        self.input_len = input_len
        self.num_tokens = tuple(num_tokens)
        self.drop_probs = tuple(drop_probs)
        self.length_transform_linear_set = nn.ModuleList(
            nn.Linear(input_len - 1, token_num - 2)
            for token_num in self.num_tokens
        )
        self.subject_selector = nn.Linear(hidden_dim, 1)
        self.latent_norm = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in self.num_tokens
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def _select_subject(self, text_tokens, valid_mask):
        """Use soft Gumbel selection during training and deterministic selection at test time."""
        invalid_mask = ~valid_mask.unsqueeze(-1)
        output_dtype = text_tokens.dtype

        # Gumbel's logarithms and the low-temperature softmax are unsafe in
        # FP16: 1e-20 underflows to zero and can introduce inf/NaN. Keep this
        # small selection branch in FP32 while preserving its gradients.
        with torch.cuda.amp.autocast(enabled=False):
            text_tokens_fp32 = text_tokens.float()
            subject_logits = F.linear(
                text_tokens_fp32,
                self.subject_selector.weight.float(),
                None
                if self.subject_selector.bias is None
                else self.subject_selector.bias.float(),
            )
            subject_logits = subject_logits.masked_fill(invalid_mask, -1e4)

            if self.training:
                uniform = torch.rand(
                    subject_logits.shape,
                    dtype=torch.float32,
                    device=subject_logits.device,
                ).clamp_(min=1e-6, max=1.0 - 1e-6)
                gumbel_noise = -torch.log(-torch.log(uniform))
                subject_logits = subject_logits + gumbel_noise

            # Keep the released learnable temperature, but prevent division
            # by zero if optimization drives it too close to zero.
            safe_temperature = self.temperature.float().abs().clamp_min(1e-3)
            subject_logits = subject_logits / safe_temperature
            subject_logits = subject_logits.masked_fill(invalid_mask, -1e4)
            subject_probabilities = F.softmax(
                subject_logits, dim=1, dtype=torch.float32
            )
            subject = torch.sum(
                subject_probabilities * text_tokens_fp32,
                dim=1,
                keepdim=True,
            )

        return subject.to(dtype=output_dtype)

    def forward(self, lang_feat, lang_mask):
        """
        Args:
            lang_feat: BERT last hidden state, [B, 20, 768].
            lang_mask: BERT attention mask, [B, 20, 1] or [B, 20].

        Returns:
            latent_expressions: groups with lengths 6 and 12 by default.
            subject: selected subject token, [B, 1, 768].
        """
        if lang_feat.size(1) != self.input_len:
            raise ValueError(
                f"Expected {self.input_len} BERT tokens, got {lang_feat.size(1)}"
            )

        if lang_mask.dim() == 3:
            lang_mask = lang_mask.squeeze(-1)
        valid_mask = lang_mask.to(torch.bool)[:, 1:]
        text_tokens = lang_feat[:, 1:, :]
        text_cls = lang_feat[:, :1, :]

        subject = self._select_subject(text_tokens, valid_mask)
        selected_semantics = [
            text_tokens[batch_idx, valid_mask[batch_idx]]
            for batch_idx in range(text_tokens.size(0))
        ]

        latent_expressions = []
        max_semantic_len = self.input_len - 1
        for transform, drop_prob, norm in zip(
            self.length_transform_linear_set, self.drop_probs, self.latent_norm
        ):
            dropped_semantics = [
                F.dropout(sequence, p=drop_prob, training=self.training)
                for sequence in selected_semantics
            ]
            padded_semantics = [
                F.pad(
                    sequence[:max_semantic_len],
                    (0, 0, 0, max(0, max_semantic_len - sequence.size(0))),
                    "constant",
                    0,
                )
                for sequence in dropped_semantics
            ]
            padded_semantics = torch.stack(padded_semantics, dim=0)
            latent_semantics = transform(
                padded_semantics.transpose(1, 2)
            ).transpose(1, 2)
            latent_expression = torch.cat(
                [text_cls, subject, latent_semantics], dim=1
            )
            latent_expressions.append(norm(latent_expression))

        return latent_expressions, subject


class LatentQueryEmbedding(nn.Module):
    """Build 21 original LEAD queries plus two Latent-VG query groups."""

    def __init__(
        self,
        input_len=20,
        hidden_dim=768,
        output_dim=256,
        num_tokens=(6, 12),
        drop_probs=(0.2, 0.15),
        temperature=0.07,
        drop=0.0,
    ):
        super().__init__()
        self.input_len = input_len
        self.num_tokens = tuple(num_tokens)
        self.out_query_num = input_len + 1 + sum(self.num_tokens)
        self.latent_generator = LatentExpressionGenerator(
            input_len=input_len,
            hidden_dim=hidden_dim,
            num_tokens=self.num_tokens,
            drop_probs=drop_probs,
            temperature=temperature,
        )
        self.position_embed = FixedAbsolutePositionEmbedding(hidden_size=hidden_dim)
        # This is the same 768 -> 256 projection used by original LEAD queries.
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.out_drop = nn.Dropout(drop)

    @staticmethod
    def _replace_padding_with_sentence(lang_feat, lang_sent, lang_mask):
        if lang_mask.dim() == 3:
            lang_mask = lang_mask.squeeze(-1)
        valid_mask = lang_mask.to(torch.bool).unsqueeze(-1)
        return torch.where(valid_mask, lang_feat, lang_sent.unsqueeze(1))

    def forward(self, lang_feat, lang_sent, lang_mask):
        """
        Args:
            lang_feat: [B, 768, 20].
            lang_sent: original LEAD BERT pooler output, [B, 768].
            lang_mask: [B, 20, 1].

        Returns:
            queries: [B, 39, 256] with the default latent lengths.
            visual_subject: [B, 1, 256], without textual position encoding.
        """
        lang_feat = rearrange(lang_feat, "b d l -> b l d")
        latent_expressions, subject = self.latent_generator(lang_feat, lang_mask)
        original_tokens = self._replace_padding_with_sentence(
            lang_feat, lang_sent, lang_mask
        )

        all_text_queries = torch.cat(
            [lang_sent.unsqueeze(1), original_tokens, *latent_expressions], dim=1
        )
        if all_text_queries.size(1) != self.out_query_num:
            raise RuntimeError(
                f"Expected {self.out_query_num} queries, got {all_text_queries.size(1)}"
            )

        queries = self.position_embed(all_text_queries)
        queries = self.out_drop(self.proj(queries))
        # Reuse the same projection, but do not add a textual position embedding.
        visual_subject = self.proj(subject)
        return queries, visual_subject
