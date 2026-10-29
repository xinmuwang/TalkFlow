
import torch
import torch.nn as nn

try:
    from .condition_encoder_2stream import (
        ModalityEncoder,
        BranchGatedFusion,
        LocalTextEncoder,
        GlobalTextEncoder,
        BranchTextHead,
        AudioTextFusion,
        TemporalContextEncoder,
        TemporalConditionDownsampler,
    )
except ImportError:
    from condition_encoder_2stream import (
        ModalityEncoder,
        BranchGatedFusion,
        LocalTextEncoder,
        GlobalTextEncoder,
        BranchTextHead,
        AudioTextFusion,
        TemporalContextEncoder,
        TemporalConditionDownsampler,
    )


# =========================================================
# 1. Three-stream audio encoder
# =========================================================

class AudioEncoder3Stream(nn.Module):
    """
    Three-branch audio encoder for upper/lower/expression latents.

    Compared with the original 2-stream version:
        audio_pose -> audio_upper + audio_lower
        audio_expr -> audio_expr

    Inputs:
        hubert : [B, T, 1024]
        librosa: [B, T, 35]
        valid_mask: [B, T] or None

    Outputs:
        {
            "hubert_feat":  [B, T, D],
            "librosa_feat": [B, T, D],
            "audio_upper":  [B, T, D],
            "audio_lower":  [B, T, D],
            "audio_expr":   [B, T, D],
        }

    Gate meaning:
        gate[..., 0, :] = HuBERT contribution
        gate[..., 1, :] = Librosa contribution
    """
    def __init__(
        self,
        hubert_dim=1024,
        librosa_dim=35,
        model_dim=256,
        hidden_dim=256,
        hubert_blocks=2,
        librosa_blocks=2,
        fusion_hidden_dim=256,
        dropout=0.1,

        # Branch-specific HuBERT/Librosa preferences.
        # upper: gesture/prosody, usually rhythm + phonetic mixed.
        upper_hubert_bias=0.0,
        upper_librosa_bias=0.4,

        # lower: posture/stability, weakly speech driven.
        lower_hubert_bias=-0.4,
        lower_librosa_bias=0.2,

        # expression: articulation/facial expression, HuBERT dominant.
        expr_hubert_bias=0.6,
        expr_librosa_bias=-0.2,

        hubert_kernel_size=3,
        librosa_kernel_size=5,
        use_librosa_temporal_context=True,
        librosa_temporal_dilations=(1, 2, 4, 8),
        return_audio_gates=False,
    ):
        super().__init__()

        self.return_audio_gates = return_audio_gates

        self.hubert_encoder = ModalityEncoder(
            in_dim=hubert_dim,
            out_dim=model_dim,
            hidden_dim=hidden_dim,
            num_blocks=hubert_blocks,
            kernel_size=hubert_kernel_size,
            dropout=dropout,
            use_temporal_context=False,
        )

        self.librosa_encoder = ModalityEncoder(
            in_dim=librosa_dim,
            out_dim=model_dim,
            hidden_dim=hidden_dim,
            num_blocks=librosa_blocks,
            kernel_size=librosa_kernel_size,
            dropout=dropout,
            use_temporal_context=use_librosa_temporal_context,
            temporal_dilations=librosa_temporal_dilations,
        )

        self.upper_fusion = BranchGatedFusion(
            dim=model_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            hubert_bias=upper_hubert_bias,
            librosa_bias=upper_librosa_bias,
        )

        self.lower_fusion = BranchGatedFusion(
            dim=model_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            hubert_bias=lower_hubert_bias,
            librosa_bias=lower_librosa_bias,
        )

        self.expr_fusion = BranchGatedFusion(
            dim=model_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            hubert_bias=expr_hubert_bias,
            librosa_bias=expr_librosa_bias,
        )

    def forward(self, hubert, librosa, valid_mask=None):
        h_hubert = self.hubert_encoder(hubert, mask=valid_mask)
        h_librosa = self.librosa_encoder(librosa, mask=valid_mask)

        if self.return_audio_gates:
            audio_upper, upper_gate = self.upper_fusion(
                h_hubert,
                h_librosa,
                mask=valid_mask,
                return_gate=True,
            )
            audio_lower, lower_gate = self.lower_fusion(
                h_hubert,
                h_librosa,
                mask=valid_mask,
                return_gate=True,
            )
            audio_expr, expr_gate = self.expr_fusion(
                h_hubert,
                h_librosa,
                mask=valid_mask,
                return_gate=True,
            )
        else:
            audio_upper = self.upper_fusion(h_hubert, h_librosa, mask=valid_mask)
            audio_lower = self.lower_fusion(h_hubert, h_librosa, mask=valid_mask)
            audio_expr = self.expr_fusion(h_hubert, h_librosa, mask=valid_mask)
            upper_gate = lower_gate = expr_gate = None

        out = {
            "hubert_feat": h_hubert,
            "librosa_feat": h_librosa,
            "audio_upper": audio_upper,
            "audio_lower": audio_lower,
            "audio_expr": audio_expr,
        }

        if self.return_audio_gates:
            out["upper_audio_gate"] = upper_gate
            out["lower_audio_gate"] = lower_gate
            out["expr_audio_gate"] = expr_gate

        return out


# =========================================================
# 2. Three-stream text encoder
# =========================================================

class TextEncoder3Stream(nn.Module):
    """
    Text encoder with branch-specific heads:
        text_upper, text_lower, text_expr
    """
    def __init__(
        self,
        local_in_dim=300,
        global_in_dim=384,
        model_dim=256,
        hidden_dim=256,
        dropout=0.1,
        use_local_temporal_refine=True,
    ):
        super().__init__()

        self.local_encoder = LocalTextEncoder(
            in_dim=local_in_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_temporal_refine=use_local_temporal_refine,
        )

        self.global_encoder = GlobalTextEncoder(
            in_dim=global_in_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.upper_head = BranchTextHead(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.lower_head = BranchTextHead(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.expr_head = BranchTextHead(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        text_local_raw,
        text_global_raw,
        text_local_mask=None,
        valid_mask=None,
    ):
        text_local_feat = self.local_encoder(
            text_local_raw,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        text_global_feat = self.global_encoder(text_global_raw)

        text_upper = self.upper_head(
            text_local_feat,
            text_global_feat,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        text_lower = self.lower_head(
            text_local_feat,
            text_global_feat,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        text_expr = self.expr_head(
            text_local_feat,
            text_global_feat,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        return {
            "text_local_feat": text_local_feat,
            "text_global_feat": text_global_feat,
            "text_upper": text_upper,
            "text_lower": text_lower,
            "text_expr": text_expr,
        }


# =========================================================
# 3. Full three-stream condition encoder
# =========================================================

class ConditionEncoder3Stream(nn.Module):
    """
    Multimodal condition encoder for 3-stream latent generation.

    It outputs:
        cond_upper: [B, ceil(T / factor), D]
        cond_lower: [B, ceil(T / factor), D]
        cond_expr : [B, ceil(T / factor), D]
        cond_mask : [B, ceil(T / factor)]

    This matches AE latents:
        z_upper, z_lower, z_expr

    Recommended default modality bias:
        upper:
            moderate HuBERT + stronger Librosa
            for gesture/prosody/rhythm.
        lower:
            weak HuBERT + weak/moderate Librosa
            for posture/stability.
        expression:
            stronger HuBERT + weaker Librosa
            for articulation/facial expression.
    """
    def __init__(
        self,
        hubert_dim=1024,
        librosa_dim=35,
        text_local_dim=300,
        text_global_dim=384,
        model_dim=256,
        hidden_dim=256,
        dropout=0.1,
        hubert_blocks=2,
        librosa_blocks=2,
        fusion_hidden_dim=256,

        # Text residual strengths per stream.
        upper_alpha_init=0.10,
        lower_alpha_init=0.03,
        expr_alpha_init=0.10,

        downsample_factor=4,
        downsample_kernel_size=3,
        use_local_text_temporal_refine=True,
        use_condition_temporal_context=True,

        # Branch-specific HuBERT / Librosa audio preferences.
        upper_hubert_bias=0.0,
        upper_librosa_bias=0.4,
        lower_hubert_bias=-0.4,
        lower_librosa_bias=0.2,
        expr_hubert_bias=0.6,
        expr_librosa_bias=-0.2,

        hubert_kernel_size=3,
        librosa_kernel_size=5,
        use_librosa_temporal_context=True,
        librosa_temporal_dilations=(1, 2, 4, 8),

        # Branch-specific temporal context after audio-text fusion.
        upper_condition_kernel_size=5,
        upper_condition_dilations=(1, 2, 4, 8),
        lower_condition_kernel_size=5,
        lower_condition_dilations=(1, 2, 4),
        expr_condition_kernel_size=3,
        expr_condition_dilations=(1, 2, 4),

        return_audio_gates=False,
        return_intermediate=False,
    ):
        super().__init__()

        assert downsample_factor in [1, 2, 4, 8], (
            "downsample_factor must be one of [1, 2, 4, 8]."
        )

        self.downsample_factor = downsample_factor
        self.use_condition_temporal_context = use_condition_temporal_context
        self.return_intermediate = return_intermediate

        self.audio_encoder = AudioEncoder3Stream(
            hubert_dim=hubert_dim,
            librosa_dim=librosa_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            hubert_blocks=hubert_blocks,
            librosa_blocks=librosa_blocks,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            upper_hubert_bias=upper_hubert_bias,
            upper_librosa_bias=upper_librosa_bias,
            lower_hubert_bias=lower_hubert_bias,
            lower_librosa_bias=lower_librosa_bias,
            expr_hubert_bias=expr_hubert_bias,
            expr_librosa_bias=expr_librosa_bias,
            hubert_kernel_size=hubert_kernel_size,
            librosa_kernel_size=librosa_kernel_size,
            use_librosa_temporal_context=use_librosa_temporal_context,
            librosa_temporal_dilations=librosa_temporal_dilations,
            return_audio_gates=return_audio_gates,
        )

        self.text_encoder = TextEncoder3Stream(
            local_in_dim=text_local_dim,
            global_in_dim=text_global_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_local_temporal_refine=use_local_text_temporal_refine,
        )

        self.upper_fusion = AudioTextFusion(
            audio_dim=model_dim,
            text_local_dim=model_dim,
            text_global_dim=model_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha_init=upper_alpha_init,
        )

        self.lower_fusion = AudioTextFusion(
            audio_dim=model_dim,
            text_local_dim=model_dim,
            text_global_dim=model_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha_init=lower_alpha_init,
        )

        self.expr_fusion = AudioTextFusion(
            audio_dim=model_dim,
            text_local_dim=model_dim,
            text_global_dim=model_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha_init=expr_alpha_init,
        )

        if use_condition_temporal_context:
            self.upper_temporal = TemporalContextEncoder(
                dim=model_dim,
                hidden_dim=hidden_dim,
                kernel_size=upper_condition_kernel_size,
                dilations=upper_condition_dilations,
                dropout=dropout,
            )

            self.lower_temporal = TemporalContextEncoder(
                dim=model_dim,
                hidden_dim=hidden_dim,
                kernel_size=lower_condition_kernel_size,
                dilations=lower_condition_dilations,
                dropout=dropout,
            )

            self.expr_temporal = TemporalContextEncoder(
                dim=model_dim,
                hidden_dim=hidden_dim,
                kernel_size=expr_condition_kernel_size,
                dilations=expr_condition_dilations,
                dropout=dropout,
            )
        else:
            self.upper_temporal = nn.Identity()
            self.lower_temporal = nn.Identity()
            self.expr_temporal = nn.Identity()

        self.upper_downsampler = TemporalConditionDownsampler(
            dim=model_dim,
            downsample_factor=downsample_factor,
            kernel_size=downsample_kernel_size,
            dropout=dropout,
        )

        self.lower_downsampler = TemporalConditionDownsampler(
            dim=model_dim,
            downsample_factor=downsample_factor,
            kernel_size=downsample_kernel_size,
            dropout=dropout,
        )

        self.expr_downsampler = TemporalConditionDownsampler(
            dim=model_dim,
            downsample_factor=downsample_factor,
            kernel_size=downsample_kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        hubert,
        librosa,
        text_local_raw,
        text_global_raw,
        valid_mask=None,
        text_local_mask=None,
    ):
        B, T, _ = hubert.shape

        if valid_mask is None:
            valid_mask = torch.ones(
                B,
                T,
                dtype=torch.bool,
                device=hubert.device,
            )

        audio_out = self.audio_encoder(
            hubert=hubert,
            librosa=librosa,
            valid_mask=valid_mask,
        )

        text_out = self.text_encoder(
            text_local_raw=text_local_raw,
            text_global_raw=text_global_raw,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        cond_upper_full = self.upper_fusion(
            audio_feat=audio_out["audio_upper"],
            text_local=text_out["text_upper"],
            text_global=text_out["text_global_feat"],
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        cond_lower_full = self.lower_fusion(
            audio_feat=audio_out["audio_lower"],
            text_local=text_out["text_lower"],
            text_global=text_out["text_global_feat"],
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        cond_expr_full = self.expr_fusion(
            audio_feat=audio_out["audio_expr"],
            text_local=text_out["text_expr"],
            text_global=text_out["text_global_feat"],
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        if self.use_condition_temporal_context:
            cond_upper_full = self.upper_temporal(cond_upper_full, mask=valid_mask)
            cond_lower_full = self.lower_temporal(cond_lower_full, mask=valid_mask)
            cond_expr_full = self.expr_temporal(cond_expr_full, mask=valid_mask)
        else:
            cond_upper_full = self.upper_temporal(cond_upper_full)
            cond_lower_full = self.lower_temporal(cond_lower_full)
            cond_expr_full = self.expr_temporal(cond_expr_full)

        cond_upper_full = cond_upper_full * valid_mask.unsqueeze(-1).float()
        cond_lower_full = cond_lower_full * valid_mask.unsqueeze(-1).float()
        cond_expr_full = cond_expr_full * valid_mask.unsqueeze(-1).float()

        cond_upper, cond_mask_upper = self.upper_downsampler(
            cond_upper_full,
            valid_mask,
        )

        cond_lower, cond_mask_lower = self.lower_downsampler(
            cond_lower_full,
            valid_mask,
        )

        cond_expr, cond_mask_expr = self.expr_downsampler(
            cond_expr_full,
            valid_mask,
        )

        cond_masks = [
            m for m in [cond_mask_upper, cond_mask_lower, cond_mask_expr]
            if m is not None
        ]

        if len(cond_masks) > 0:
            cond_mask = cond_masks[0]
            for m in cond_masks[1:]:
                cond_mask = cond_mask & m
        else:
            cond_mask = None

        if cond_mask is not None:
            cond_upper = cond_upper * cond_mask.unsqueeze(-1).float()
            cond_lower = cond_lower * cond_mask.unsqueeze(-1).float()
            cond_expr = cond_expr * cond_mask.unsqueeze(-1).float()

        out = {
            "cond_upper": cond_upper,
            "cond_lower": cond_lower,
            "cond_expr": cond_expr,
            "cond_mask": cond_mask,
        }

        if self.return_intermediate:
            out.update(audio_out)
            out.update(text_out)
            out["cond_upper_full"] = cond_upper_full
            out["cond_lower_full"] = cond_lower_full
            out["cond_expr_full"] = cond_expr_full

        return out


# Backward-compatible aliases for new 3-stream usage.
ConditionEncoder = ConditionEncoder3Stream
ConditionEncoder3 = ConditionEncoder3Stream


CONDITION_ENCODER_CONFIGS = {
    "ds1": dict(downsample_factor=1),
    "ds2": dict(downsample_factor=2),
    "ds4": dict(downsample_factor=4),
    "ds8": dict(downsample_factor=8),
}


def build_condition_encoder_for_downsample(
    downsample_factor,
    **kwargs,
):
    assert downsample_factor in [1, 2, 4, 8]
    return ConditionEncoder3Stream(
        downsample_factor=downsample_factor,
        **kwargs,
    )


if __name__ == "__main__":
    B, T = 2, 240
    hubert = torch.randn(B, T, 1024)
    librosa = torch.randn(B, T, 35)
    text_local = torch.randn(B, T, 300)
    text_global = torch.randn(B, 384)
    valid_mask = torch.ones(B, T, dtype=torch.bool)

    model = ConditionEncoder3Stream(
        model_dim=512,
        hidden_dim=512,
        dropout=0.01,
        downsample_factor=4,
        return_audio_gates=True,
        return_intermediate=True,
    )

    out = model(
        hubert=hubert,
        librosa=librosa,
        text_local_raw=text_local,
        text_global_raw=text_global,
        valid_mask=valid_mask,
    )

    print("cond_upper:", out["cond_upper"].shape)
    print("cond_lower:", out["cond_lower"].shape)
    print("cond_expr :", out["cond_expr"].shape)
    print("cond_mask :", out["cond_mask"].shape)
