import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Basic temporal blocks
# =========================================================

class TemporalConvBlock(nn.Module):
    """
    Lightweight temporal conv block for [B, T, C].
    Local temporal refinement.
    """
    def __init__(self, dim, hidden_dim=None, kernel_size=3, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        assert kernel_size % 2 == 1

        padding = (kernel_size - 1) // 2

        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size, padding=padding)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size, padding=padding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x:    [B, T, C]
        mask: [B, T], True means valid
        """
        residual = x

        h = self.norm1(x)
        h = h.transpose(1, 2)
        h = self.conv1(h)
        h = h.transpose(1, 2)

        h = F.gelu(self.norm2(h))
        h = self.dropout(h)

        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.dropout(h)

        out = residual + h

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class DilatedTemporalBlock(nn.Module):
    """
    Dilated residual TCN block for longer temporal context.
    Input/output: [B, T, C]
    """
    def __init__(self, dim, hidden_dim=None, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        assert kernel_size % 2 == 1

        padding = dilation * (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        residual = x

        h = self.norm(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)

        out = self.out_norm(residual + h)

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class TemporalContextEncoder(nn.Module):
    """
    Branch-specific temporal context modeling before latent alignment.

    For pose/body, use larger kernels or larger dilation range.
    For expression, keep shorter kernels so articulation remains responsive.
    """
    def __init__(
        self,
        dim,
        hidden_dim=None,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.blocks = nn.ModuleList([
            DilatedTemporalBlock(
                dim=dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask=mask)

        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        return x


# =========================================================
# 2. Audio encoder
# =========================================================

class ModalityEncoder(nn.Module):
    """
    Encode one modality into temporal feature space.

    Input:
        x: [B, T, in_dim]

    Output:
        h: [B, T, out_dim]
    """
    def __init__(
        self,
        in_dim,
        out_dim=256,
        hidden_dim=256,
        num_blocks=2,
        kernel_size=3,
        dropout=0.1,
        use_temporal_context=False,
        temporal_dilations=(1, 2, 4),
    ):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, out_dim)
        self.in_norm = nn.LayerNorm(out_dim)

        self.blocks = nn.ModuleList([
            TemporalConvBlock(
                dim=out_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.use_temporal_context = use_temporal_context
        if use_temporal_context:
            self.temporal_context = TemporalContextEncoder(
                dim=out_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilations=temporal_dilations,
                dropout=dropout,
            )
        else:
            self.temporal_context = None

        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        x = self.in_norm(x)
        x = F.gelu(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        for blk in self.blocks:
            x = blk(x, mask=mask)

        if self.temporal_context is not None:
            x = self.temporal_context(x, mask=mask)

        x = self.out_norm(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        return x


class BranchGatedFusion(nn.Module):
    """
    Stable branch-specific fusion between HuBERT and Librosa.

    Uses softmax competition between modalities.

    Larger librosa_bias makes a branch more rhythm/prosody driven.
    Larger hubert_bias makes a branch more articulation/phonetic driven.
    """
    def __init__(
        self,
        dim,
        hidden_dim=256,
        dropout=0.1,
        hubert_bias=0.0,
        librosa_bias=0.0,
    ):
        super().__init__()

        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_gate = nn.Linear(hidden_dim, dim * 2)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.hubert_bias = float(hubert_bias)
        self.librosa_bias = float(librosa_bias)

    def forward(self, h_hubert, h_librosa, mask=None, return_gate=False):
        B, T, D = h_hubert.shape

        x = torch.cat([h_hubert, h_librosa], dim=-1)
        h = self.fuse_mlp(x)

        logits = self.to_gate(h).view(B, T, 2, D)
        logits[:, :, 0, :] = logits[:, :, 0, :] + self.hubert_bias
        logits[:, :, 1, :] = logits[:, :, 1, :] + self.librosa_bias

        gate = torch.softmax(logits, dim=2)

        fused = gate[:, :, 0, :] * h_hubert + gate[:, :, 1, :] * h_librosa
        fused = self.to_out(fused)

        if mask is not None:
            fused = fused * mask.unsqueeze(-1).float()

        if return_gate:
            return fused, gate
        return fused


class AudioEncoder(nn.Module):
    """
    Two-branch audio encoder.

    Inputs:
        hubert : [B, T, 1024]
        librosa: [B, T, 35]
        valid_mask: [B, T] or None

    Outputs:
        {
            "hubert_feat":  [B, T, D],
            "librosa_feat": [B, T, D],
            "audio_pose":   [B, T, D],
            "audio_expr":   [B, T, D],
            "pose_audio_gate": optional [B, T, 2, D],
            "expr_audio_gate": optional [B, T, 2, D],
        }
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
        pose_hubert_bias=-0.1,
        pose_librosa_bias=0.5,
        expr_hubert_bias=0.5,
        expr_librosa_bias=-0.1,
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

        self.pose_fusion = BranchGatedFusion(
            dim=model_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            hubert_bias=pose_hubert_bias,
            librosa_bias=pose_librosa_bias,
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
            audio_pose, pose_gate = self.pose_fusion(
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
            audio_pose = self.pose_fusion(h_hubert, h_librosa, mask=valid_mask)
            audio_expr = self.expr_fusion(h_hubert, h_librosa, mask=valid_mask)
            pose_gate = None
            expr_gate = None

        out = {
            "hubert_feat": h_hubert,
            "librosa_feat": h_librosa,
            "audio_pose": audio_pose,
            "audio_expr": audio_expr,
        }

        if self.return_audio_gates:
            out["pose_audio_gate"] = pose_gate
            out["expr_audio_gate"] = expr_gate

        return out


# =========================================================
# 3. Text encoder
# =========================================================

class LocalTextEncoder(nn.Module):
    def __init__(
        self,
        in_dim=300,
        model_dim=256,
        hidden_dim=256,
        dropout=0.1,
        use_temporal_refine=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

        self.refine = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

        self.use_temporal_refine = use_temporal_refine
        if use_temporal_refine:
            self.temporal = TemporalContextEncoder(
                dim=model_dim,
                hidden_dim=hidden_dim,
                kernel_size=3,
                dilations=(1, 2, 4),
                dropout=dropout,
            )
        else:
            self.temporal = None

        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, x, text_local_mask=None, valid_mask=None):
        h = self.proj(x)
        h = self.norm(h)
        h = F.gelu(h)

        h = h + self.refine(h)

        if self.temporal is not None:
            h = self.temporal(h, mask=valid_mask)

        h = self.out_norm(h)

        if text_local_mask is not None:
            h = h * text_local_mask.unsqueeze(-1).float()

        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1).float()

        return h


class GlobalTextEncoder(nn.Module):
    def __init__(self, in_dim=384, model_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x):
        return self.net(x)


class BranchTextHead(nn.Module):
    def __init__(self, model_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, local_feat, global_feat, text_local_mask=None, valid_mask=None):
        B, T, D = local_feat.shape
        g = global_feat.unsqueeze(1).expand(B, T, D)
        x = torch.cat([local_feat, g], dim=-1)
        out = self.fuse(x)

        if text_local_mask is not None:
            out = out * text_local_mask.unsqueeze(-1).float()

        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).float()

        return out


class TextEncoder(nn.Module):
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

        self.pose_head = BranchTextHead(
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

        text_pose = self.pose_head(
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
            "text_pose": text_pose,
            "text_expr": text_expr,
        }


# =========================================================
# 4. Audio-text fusion
# =========================================================

class GlobalFiLM(nn.Module):
    """
    Residual global text FiLM modulation for audio features.
    """
    def __init__(self, feat_dim, global_text_dim, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.global_proj = nn.Sequential(
            nn.Linear(global_text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_gamma = nn.Linear(hidden_dim, feat_dim)
        self.to_beta = nn.Linear(hidden_dim, feat_dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, audio_feat, text_global):
        h = self.global_proj(text_global)
        gamma = self.to_gamma(h).unsqueeze(1)
        beta = self.to_beta(h).unsqueeze(1)
        return audio_feat * (1.0 + gamma) + beta


class LocalResidualGate(nn.Module):
    """
    Local text residual gating.
    """
    def __init__(
        self,
        feat_dim,
        local_text_dim,
        hidden_dim=256,
        dropout=0.1,
        alpha_init=0.1,
    ):
        super().__init__()

        self.local_proj = nn.Sequential(
            nn.Linear(local_text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_delta = nn.Linear(hidden_dim, feat_dim)
        self.to_gate = nn.Linear(hidden_dim, feat_dim)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.out_norm = nn.LayerNorm(feat_dim)

        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    def forward(self, x, text_local, text_local_mask=None, valid_mask=None):
        h = self.local_proj(text_local)

        delta = self.to_delta(h)
        gate = torch.sigmoid(self.to_gate(h))

        if text_local_mask is not None:
            gate = gate * text_local_mask.unsqueeze(-1).float()
            delta = delta * text_local_mask.unsqueeze(-1).float()

        out = x + self.alpha * gate * delta
        out = self.out_norm(out)

        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).float()

        return out


class AudioTextFusion(nn.Module):
    """
    Audio-main, text-modulated fusion.
    """
    def __init__(
        self,
        audio_dim,
        text_local_dim,
        text_global_dim,
        model_dim=256,
        hidden_dim=256,
        dropout=0.1,
        alpha_init=0.1,
    ):
        super().__init__()

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

        self.global_film = GlobalFiLM(
            feat_dim=model_dim,
            global_text_dim=text_global_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.local_gate = LocalResidualGate(
            feat_dim=model_dim,
            local_text_dim=text_local_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha_init=alpha_init,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        audio_feat,
        text_local,
        text_global,
        text_local_mask=None,
        valid_mask=None,
    ):
        h = self.audio_proj(audio_feat)

        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1).float()

        h = self.global_film(h, text_global)

        h = self.local_gate(
            h,
            text_local,
            text_local_mask=text_local_mask,
            valid_mask=valid_mask,
        )

        h = self.out_proj(h)

        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1).float()

        return h


# =========================================================
# 5. Temporal downsampling for latent alignment
# =========================================================

class ConditionDownsampleBlock(nn.Module):
    """
    Residual-style temporal downsample block.

    Input:
        x: [B, T, D]

    Output:
        out: [B, ceil(T/2), D]
    """
    def __init__(self, dim, kernel_size=3, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1

        padding = kernel_size // 2

        self.main = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        self.skip = nn.Conv1d(
            dim,
            dim,
            kernel_size=1,
            stride=2,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.main(x) + self.skip(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        return out


class TemporalConditionDownsampler(nn.Module):
    """
    Downsample condition features along time axis.

    Supports:
        downsample_factor = 1, 2, 4, 8

    Input:
        x:    [B, T, D]
        mask: [B, T] or None, True means valid position

    Output:
        x_down:    [B, ceil(T / factor), D]
        mask_down: [B, ceil(T / factor)] or None
    """
    def __init__(
        self,
        dim,
        downsample_factor=4,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        assert downsample_factor in [1, 2, 4, 8], (
            "downsample_factor must be one of [1, 2, 4, 8]."
        )
        self.downsample_factor = downsample_factor

        num_blocks = {
            1: 0,
            2: 1,
            4: 2,
            8: 3,
        }[downsample_factor]

        self.blocks = nn.ModuleList([
            ConditionDownsampleBlock(
                dim=dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    @staticmethod
    def downsample_mask_once(mask):
        """
        mask: [B, T], True means valid.
        Use max-pooling with ceil_mode=True to match stride-2 conv length.
        """
        if mask is None:
            return None

        m = mask.float().unsqueeze(1)
        m = F.max_pool1d(
            m,
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )
        return m.squeeze(1).bool()

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x)
            mask = self.downsample_mask_once(mask)

            if mask is not None:
                x = x * mask.unsqueeze(-1).float()

        return x, mask


# =========================================================
# 6. Full 2-stream condition encoder
# =========================================================

class ConditionEncoder2Stream(nn.Module):
    """
    Multimodal condition encoder for a 2-stream latent Flow backbone.

    It outputs only pose and expression conditions by default:
        cond_pose: [B, ceil(T / factor), D]
        cond_expr: [B, ceil(T / factor), D]
        cond_mask: [B, ceil(T / factor)]

    No cond_shared is produced. This matches an AE with two latents:
        z_pose, z_expr

    Set return_intermediate=True for diagnostics or visualization.
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
        pose_alpha_init=0.10,
        expr_alpha_init=0.10,
        downsample_factor=4,
        downsample_kernel_size=3,
        use_local_text_temporal_refine=True,
        use_condition_temporal_context=True,
        pose_hubert_bias=-0.1,
        pose_librosa_bias=0.5,
        expr_hubert_bias=0.5,
        expr_librosa_bias=-0.1,
        hubert_kernel_size=3,
        librosa_kernel_size=5,
        use_librosa_temporal_context=True,
        librosa_temporal_dilations=(1, 2, 4, 8),
        pose_condition_kernel_size=5,
        pose_condition_dilations=(1, 2, 4, 8),
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

        self.audio_encoder = AudioEncoder(
            hubert_dim=hubert_dim,
            librosa_dim=librosa_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            hubert_blocks=hubert_blocks,
            librosa_blocks=librosa_blocks,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            pose_hubert_bias=pose_hubert_bias,
            pose_librosa_bias=pose_librosa_bias,
            expr_hubert_bias=expr_hubert_bias,
            expr_librosa_bias=expr_librosa_bias,
            hubert_kernel_size=hubert_kernel_size,
            librosa_kernel_size=librosa_kernel_size,
            use_librosa_temporal_context=use_librosa_temporal_context,
            librosa_temporal_dilations=librosa_temporal_dilations,
            return_audio_gates=return_audio_gates,
        )

        self.text_encoder = TextEncoder(
            local_in_dim=text_local_dim,
            global_in_dim=text_global_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_local_temporal_refine=use_local_text_temporal_refine,
        )

        self.pose_fusion = AudioTextFusion(
            audio_dim=model_dim,
            text_local_dim=model_dim,
            text_global_dim=model_dim,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha_init=pose_alpha_init,
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
            self.pose_temporal = TemporalContextEncoder(
                dim=model_dim,
                hidden_dim=hidden_dim,
                kernel_size=pose_condition_kernel_size,
                dilations=pose_condition_dilations,
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
            self.pose_temporal = nn.Identity()
            self.expr_temporal = nn.Identity()

        self.pose_downsampler = TemporalConditionDownsampler(
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

        cond_pose_full = self.pose_fusion(
            audio_feat=audio_out["audio_pose"],
            text_local=text_out["text_pose"],
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
            cond_pose_full = self.pose_temporal(cond_pose_full, mask=valid_mask)
            cond_expr_full = self.expr_temporal(cond_expr_full, mask=valid_mask)
        else:
            cond_pose_full = self.pose_temporal(cond_pose_full)
            cond_expr_full = self.expr_temporal(cond_expr_full)

        cond_pose_full = cond_pose_full * valid_mask.unsqueeze(-1).float()
        cond_expr_full = cond_expr_full * valid_mask.unsqueeze(-1).float()

        cond_pose, cond_mask_pose = self.pose_downsampler(
            cond_pose_full,
            valid_mask,
        )

        cond_expr, cond_mask_expr = self.expr_downsampler(
            cond_expr_full,
            valid_mask,
        )

        if cond_mask_pose is not None and cond_mask_expr is not None:
            cond_mask = cond_mask_pose & cond_mask_expr
        else:
            cond_mask = cond_mask_pose if cond_mask_pose is not None else cond_mask_expr

        if cond_mask is not None:
            cond_pose = cond_pose * cond_mask.unsqueeze(-1).float()
            cond_expr = cond_expr * cond_mask.unsqueeze(-1).float()

        out = {
            "cond_pose": cond_pose,
            "cond_expr": cond_expr,
            "cond_mask": cond_mask,
        }

        if self.return_intermediate:
            out.update(audio_out)
            out.update(text_out)
            out["cond_pose_full"] = cond_pose_full
            out["cond_expr_full"] = cond_expr_full

        return out


# Backward-compatible alias.
ConditionEncoder = ConditionEncoder2Stream


# =========================================================
# 7. Convenience config examples
# =========================================================

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
    """
    Convenience constructor.

    Example:
        cond_encoder = build_condition_encoder_for_downsample(
            4,
            model_dim=512,
            hidden_dim=512,
            dropout=0.01,
        )
    """
    assert downsample_factor in [1, 2, 4, 8]
    return ConditionEncoder2Stream(
        downsample_factor=downsample_factor,
        **kwargs,
    )
