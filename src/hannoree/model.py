"""
The HanNoReE model: a pre-trained MuSViT vision transformer reads a stave image, a small adapter (projection +
BiLSTM) turns its patch features into a sequence, and a Zeus Bahdanau-attention LSTM decoder transcribes it into
LMX tokens. The backbone is part of the model, so it can be frozen or fine-tuned.
"""

from __future__ import annotations

import inspect
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ViTConfig, ViTModel
except ImportError:
    ViTConfig = ViTModel = None


def lengths_to_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool mask on `device`, True at valid timesteps."""
    lengths = lengths.to(device, non_blocking=True)
    return torch.arange(max_len, device=device)[None, :] < lengths[:, None]


# ==============================================================================
# Zeus Bahdanau Attention LSTM Decoder
# ==============================================================================

class BahdanauAttention(nn.Module):
    """
    Additive Bahdanau attention as in Zeus (rnn_cells_with_attention.py): scores the encoder outputs
    against the previous decoder state.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=True)
        self.decoder_proj = nn.Linear(decoder_dim * 2, attention_dim, bias=True)  # concatenated (h, c)
        self.score_proj = nn.Linear(attention_dim, 1, bias=True)

    def project_encoder(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Projects encoder outputs once per sequence, instead of at every decoding step."""
        return self.encoder_proj(encoder_outputs)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_proj: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state: (h, c) each of shape (B, decoder_dim)
        prev_state = torch.cat(state, dim=-1)  # (B, 2 * decoder_dim)
        dec_proj = self.decoder_proj(prev_state).unsqueeze(1)  # (B, 1, attention_dim)

        # Energy scores: (B, T, 1) -> (B, T)
        scores = self.score_proj(torch.tanh(encoder_proj + dec_proj)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)

        return context, attn_weights


class ZeusDecoder(nn.Module):
    """
    Autoregressive single-layer LSTM decoder with Bahdanau attention, as in Zeus.
    Zeus decoder has no dropout by default; all model dropout is in the encoder.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        bos_idx: int = 0,
        eos_idx: int = 1,
        pad_idx: int = 2,
        max_length: int = 600,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_dim=dim, decoder_dim=dim, attention_dim=dim)
        self.lstm_cell = nn.LSTMCell(input_size=dim * 2, hidden_size=dim)  # input: [embedded_token; context]
        self.fc_out = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        context: torch.Tensor,
        target_seq: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training, as Zeus decoder_training.
        Args:
            context: (B, T, dim) encoder output sequence
            target_seq: (B, L) input token IDs (starting with BOS)
            context_mask: optional (B, T) mask, True at valid timesteps
        Returns:
            logits: (B, L, vocab_size)
        """
        batch_size, seq_len = target_seq.shape
        embedded = self.dropout(self.embedding(target_seq))  # (B, L, dim)
        encoder_proj = self.attention.project_encoder(context)

        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (B, dim)
            ctx_t, _ = self.attention(context, encoder_proj, (h, c), mask=context_mask)  # (B, dim)
            lstm_input = torch.cat([x_t, ctx_t], dim=-1)  # (B, 2*dim)
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs.append(h)

        stacked_h = torch.stack(outputs, dim=1)  # (B, L, dim)
        return self.fc_out(self.dropout(stacked_h))  # (B, L, vocab_size)

    @torch.inference_mode()
    def generate(
        self,
        context: torch.Tensor,
        max_length: int | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns the (B, max_len) predicted token IDs."""
        max_len = max_length or self.max_length
        batch_size = context.shape[0]
        encoder_proj = self.attention.project_encoder(context)

        curr_token = torch.full((batch_size,), self.bos_idx, dtype=torch.long, device=context.device)
        h = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)
        c = torch.zeros(batch_size, self.dim, device=context.device, dtype=context.dtype)

        preds = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=context.device)

        for step in range(max_len):
            x_t = self.embedding(curr_token)  # (B, dim)
            ctx_t, _ = self.attention(context, encoder_proj, (h, c), mask=context_mask)
            lstm_input = torch.cat([x_t, ctx_t], dim=-1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.fc_out(h)  # (B, vocab_size)
            next_token = torch.argmax(logits, dim=-1)  # (B,)

            # If sequence was already finished, record pad_idx
            token_to_record = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)
            preds.append(token_to_record)

            is_finished = is_finished | (next_token == self.eos_idx)
            # Checking on the CPU waits for the GPU, so do it only every few steps; extra steps only add padding
            if step % 8 == 7 and is_finished.all():
                break
            curr_token = torch.where(is_finished, torch.full_like(next_token, self.pad_idx), next_token)

        if not preds:
            return torch.empty((batch_size, 0), dtype=torch.long, device=context.device)

        return torch.stack(preds, dim=1)  # (B, pred_len)


class BiLSTMSumBlock(nn.Module):
    """
    Bidirectional LSTM with merge_mode='sum' and an optional residual connection, as in Zeus
    (keras_model.py:105-111).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, residual: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = residual

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            out, _ = self.lstm(x)  # (B, T, 2 * out_dim)
        else:
            # Packing keeps the backward direction from reading the batch padding first.
            # It needs the lengths on the CPU; keeping them there avoids waiting for the GPU.
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        fwd, bwd = out.chunk(2, dim=-1)
        summed = self.dropout(fwd + bwd)  # (B, T, out_dim)
        if self.residual:
            summed = summed + x
        return summed


# ==============================================================================
# MuSViT Backbone
# ==============================================================================

def load_musvit(model_name: str = "PRAIG/musvit", device: str | None = None, token: str | None = None):
    """
    Loads a MuSViT encoder (e.g. 'PRAIG/musvit' or 'PRAIG/musvit-light') as a plain ViTModel, as its model card
    prescribes: AutoModel would give ViT-MAE, which randomly masks 70% of the patches.
    """
    if ViTModel is None:
        raise ImportError("Please install transformers: pip install transformers")

    print(f"Loading pre-trained MuSViT model '{model_name}'...")
    hf_token = token or os.environ.get("HF_TOKEN")
    # The MAE checkpoint has no pooler, and the pooler is unused here
    model, loading_info = ViTModel.from_pretrained(
        model_name, token=hf_token, add_pooling_layer=False, output_loading_info=True
    )

    # Anything missing would be randomly initialized
    missing = [k for k in loading_info["missing_keys"] if not k.startswith("pooler.")]
    if missing or loading_info["mismatched_keys"]:
        raise RuntimeError(
            f"'{model_name}' did not load cleanly into ViTModel. "
            f"Missing weights: {missing}. Mismatched weights: {loading_info['mismatched_keys']}."
        )

    if device is not None:
        model.to(device)
    model.eval()
    return model


def build_vit(config: dict):
    """
    A ViTModel of the given configuration (a ViTConfig.to_dict()) with untrained weights, for weights that come
    from a checkpoint. Nothing is downloaded.
    """
    if ViTModel is None:
        raise ImportError("Please install transformers: pip install transformers")
    return ViTModel(ViTConfig.from_dict(config), add_pooling_layer=False).eval()


def stave_patch_grid(last_hidden_state: torch.Tensor, patch_rows: int, patch_columns: int) -> torch.Tensor:
    """
    The ViT output as a patch grid.

    Args:
        last_hidden_state: (B, 1 + N, D) with the [CLS] token first and N patches in row-major order
        patch_rows: number of patch rows of the ViT input
        patch_columns: number of patch columns of the ViT input
    Returns:
        (B, patch_rows, patch_columns, D) patch features of the stave
    """
    patches = last_hidden_state[:, 1:, :]  # drop [CLS]
    B, N, D = patches.shape
    if N % patch_columns != 0 or N // patch_columns < patch_rows:
        raise ValueError(f"Expected at least {patch_rows} rows of {patch_columns} patch tokens, got {N} tokens.")
    return patches.reshape(B, N // patch_columns, patch_columns, D)[:, :patch_rows]


def anchor_position_embeddings(embeddings: nn.Module, reference_width: int):
    """
    Replaces the position-embedding interpolation of a Hugging Face ViTEmbeddings module. Hugging Face stretches the
    pre-trained grid over the whole (padded) input, so a patch column's embedding would depend on the widest stave of
    its batch. Here the grid is stretched over `reference_width` pixels and cut at the input's width instead: a patch
    column always gets the same embedding, and every input is on the same horizontal scale. Vertically, the grid is
    stretched over the input's height, as before. For inputs `reference_width` wide, this is Hugging Face's result.
    """
    patch_size = embeddings.patch_size
    reference_columns = reference_width // patch_size

    def interpolate_pos_encoding(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        rows, columns = height // patch_size, width // patch_size
        if columns > reference_columns:
            raise ValueError(f"Input {width} px wide exceeds the position-embedding reference width of {reference_width} px.")
        position_embeddings = embeddings.position_embeddings  # (1, 1 + side^2, D), [CLS] first
        cls_position, patch_positions = position_embeddings[:, :1], position_embeddings[:, 1:]
        side = math.isqrt(patch_positions.shape[1])
        grid = patch_positions.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(rows, reference_columns), mode="bicubic", align_corners=False)[..., :columns]
        return torch.cat((cls_position, grid.permute(0, 2, 3, 1).reshape(1, rows * columns, -1)), dim=1)

    # An instance attribute shadows the method, which ViTEmbeddings.forward calls with the input's height and width
    embeddings.interpolate_pos_encoding = interpolate_pos_encoding


class MusvitBackbone(nn.Module):
    """
    Pre-trained MuSViT (or another ViT on the Hugging Face Hub) applied to stave bands, trainable or frozen.

    The band is processed alone, with interpolated position embeddings, as the MuSViT model card recommends for
    fine-tuning. The bands may have any width up to `max_width`; the pre-trained position grid is stretched over
    `max_width` pixels and cut at the band's width (see anchor_position_embeddings), and the patches of the white
    padding to the widest band of a batch are masked out of the attention, so a band's features do not depend on its
    batch.

    While frozen, it runs without gradients and stays in eval mode even when the model is trained.

    With `vit_config` (a ViTConfig.to_dict()), the ViT is built with untrained weights instead of loading the
    pre-trained `model_name`, for loading a trained model whose checkpoint holds all weights.
    """
    def __init__(
        self,
        model_name: str = "PRAIG/musvit",
        stave_height: int = 96,
        token: str | None = None,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        max_width: int = 1024,
        vit_config: dict | None = None,
    ):
        super().__init__()
        self.vit = build_vit(vit_config) if vit_config is not None else load_musvit(model_name, token=token)
        self.model_name = model_name
        self.bf16 = bf16
        self.patch_size = self.vit.config.patch_size
        self.hidden_size = self.vit.config.hidden_size

        if stave_height % self.patch_size != 0 or stave_height <= 0:
            raise ValueError(f"The stave height must be a positive multiple of the patch size ({self.patch_size}), got {stave_height}.")
        if max_width % self.patch_size != 0 or max_width <= 0:
            raise ValueError(f"The maximum stave width must be a positive multiple of the patch size ({self.patch_size}), got {max_width}.")
        self.stave_height = stave_height
        self.patch_rows = stave_height // self.patch_size
        self.max_width = max_width
        anchor_position_embeddings(self.vit.embeddings, max_width)

        if gradient_checkpointing:
            # Takes effect only while the ViT is in training mode, i.e. while fine-tuning
            self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.trainable = False
        self.set_trainable(False)

    def set_trainable(self, trainable: bool):
        """Freezes or unfreezes the ViT weights."""
        self.trainable = trainable
        self.vit.requires_grad_(trainable)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.vit.eval()
        return self

    def padding_mask(self, widths: torch.Tensor, height: int, width: int) -> torch.Tensor | None:
        """
        (B, 1 + rows * columns) attention mask of the ViT tokens ([CLS] first, patches in row-major order), False at
        the patches of bands padded beyond their `widths`; None without padding.
        """
        columns = widths.to(self.vit.device, non_blocking=True) // self.patch_size
        total_columns = width // self.patch_size
        if (columns >= total_columns).all():
            return None
        if "attention_mask" not in inspect.signature(self.vit.forward).parameters:
            raise RuntimeError(
                "Masking the padding of variable-width staves needs a ViTModel that takes an attention_mask "
                "(transformers 5); upgrade transformers, or use a fixed --stave-width."
            )
        valid = torch.arange(total_columns, device=columns.device)[None, :] < columns[:, None]  # (B, columns)
        patches = valid[:, None, :].expand(-1, height // self.patch_size, -1).flatten(1)
        return torch.cat((patches.new_ones(patches.shape[0], 1), patches), dim=1)

    def forward(self, bands: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            bands: (B, 3, stave_height, width) stave bands in [0, 1], without normalization (as documented);
                the width must be a multiple of the patch size up to `max_width`
            widths: optional (B,) unpadded band widths in pixels; the patches beyond them are masked out of the
                attention
        Returns:
            (B, patch_rows, width / patch_size, hidden_size) patch grid of the staves, in FP32
        """
        height, width = bands.shape[-2:]
        if height != self.stave_height or width % self.patch_size != 0 or width > self.max_width:
            raise ValueError(
                f"Expected stave bands of height {self.stave_height} and a width divisible by {self.patch_size} "
                f"of at most {self.max_width}, got {height}x{width}."
            )

        with (
            torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()),
            torch.autocast(bands.device.type, dtype=torch.bfloat16, enabled=self.bf16),
        ):
            mask = self.padding_mask(widths, height, width) if widths is not None else None
            kwargs = {"attention_mask": mask} if mask is not None else {}
            hidden = self.vit(bands, interpolate_pos_encoding=True, **kwargs).last_hidden_state
        return stave_patch_grid(hidden.float(), self.patch_rows, width // self.patch_size)


# ==============================================================================
# Encoder: MuSViT Backbone + Adapter
# ==============================================================================

class MusvitEncoder(nn.Module):
    """
    The MuSViT backbone followed by an adapter that turns its patch grid into the sequence the decoder attends to:
    one timestep per patch column, its rows concatenated (like the Zeus encoder flattening H x C per column, this
    keeps the vertical, i.e. pitch, position), then a linear projection to the model dimension, a BiLSTM (Zeus's
    merge_mode='sum') and a residual LayerNorm.
    """
    def __init__(self, backbone: MusvitBackbone, dim: int = 256, dropout: float = 0.2):
        super().__init__()
        musvit_dim = backbone.hidden_size * backbone.patch_rows
        self.dim = dim
        self.musvit_dim = musvit_dim

        # Projection from the MuSViT embedding space to the model dimension
        self.proj = nn.Sequential(
            nn.Linear(musvit_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        # Contextualizer BiLSTM, as Zeus's merge_mode='sum' sequence modeling
        self.lstm = BiLSTMSumBlock(in_dim=dim, out_dim=dim, dropout=dropout, residual=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.backbone = backbone

    def forward(self, bands: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            bands: (B, 3, stave_height, width) stave bands in [0, 1], padded with white to the widest of the batch
            lengths: optional (B,) unpadded band widths in pixels (multiples of the patch size), preferably on the CPU;
                the patches of the padding are then masked out
        Returns:
            the (B, T, dim) context for the decoder, T being the number of patch columns, and its (B, T) mask of
            valid timesteps (None without `lengths`)
        """
        grid = self.backbone(bands, lengths)  # (B, rows, columns, D)
        features = grid.transpose(1, 2).flatten(2)  # (B, columns, rows * D)
        columns = lengths.cpu() // self.backbone.patch_size if lengths is not None else None

        projected = self.proj(features)
        context = self.lstm(projected, columns)
        context = self.layer_norm(context + projected)  # Residual connection
        mask = lengths_to_mask(columns, context.shape[1], context.device) if columns is not None else None
        return context, mask


# ==============================================================================
# The Model
# ==============================================================================

def model_arguments(config: dict) -> dict:
    """The HanNoReEModel arguments of a saved model configuration. Other keys are ignored."""
    parameters = inspect.signature(HanNoReEModel.__init__).parameters
    return {k: v for k, v in config.items() if k in parameters}


class HanNoReEModel(nn.Module):
    """MuSViT encoder (backbone + adapter) and Zeus decoder: stave bands in, LMX token IDs out."""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        bos_idx: int = 0,
        eos_idx: int = 1,
        pad_idx: int = 2,
        max_length: int = 600,
        dropout: float = 0.2,
        musvit_model: str = "PRAIG/musvit",
        stave_height: int = 96,
        max_stave_width: int = 1024,
        vit_config: dict | None = None,
        hf_token: str | None = None,
        musvit_bf16: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        backbone = MusvitBackbone(
            model_name=musvit_model,
            stave_height=stave_height,
            token=hf_token,
            bf16=musvit_bf16,
            gradient_checkpointing=gradient_checkpointing,
            max_width=max_stave_width,
            vit_config=vit_config,
        )
        self.encoder = MusvitEncoder(backbone, dim=dim, dropout=dropout)
        # As in Zeus, all model dropout is in the encoder; the decoder has none
        self.decoder = ZeusDecoder(
            vocab_size=vocab_size,
            dim=dim,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            max_length=max_length,
            dropout=0.0,
        )

    @property
    def backbone(self) -> MusvitBackbone:
        return self.encoder.backbone

    def forward(self, x: torch.Tensor, target_seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x, lengths)
        return self.decoder(context, target_seq, context_mask=ctx_mask)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, lengths: torch.Tensor | None = None, max_length: int | None = None) -> torch.Tensor:
        context, ctx_mask = self.encoder(x, lengths)
        return self.decoder.generate(context, max_length=max_length, context_mask=ctx_mask)
