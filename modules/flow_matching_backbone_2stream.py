import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Time embedding
# =========================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Continuous scalar time embedding.

    Args:
        dim: embedding dimension.
        time_scale: scale applied to t before sinusoidal projection.

    Input:
        t: [B], values in [0, 1]

    Output:
        emb: [B, dim]
    """

    def __init__(self, dim: int, time_scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.view(-1)

        half_dim = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, device=device).float()
            / max(half_dim - 1, 1)
        )

        args = t[:, None] * self.time_scale * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimeMLP(nn.Module):
    """
    Sinusoidal time embedding followed by a two-layer MLP.
    """

    def __init__(self, time_dim: int, model_dim: int, time_scale: float = 1000.0):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(
            dim=time_dim,
            time_scale=time_scale,
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.time_embed(t))


# =========================================================
# 2. Position embedding
# =========================================================


class LearnablePositionEmbedding(nn.Module):
    """
    Learnable temporal positional embedding.

    Input/output:
        x: [B, T, D]
    """

    def __init__(self, max_seq_len: int, model_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len in the model config."
            )
        return x + self.pos_emb[:, :T, :]


# =========================================================
# 3. FiLM / AdaLN-style modulation
# =========================================================


class ResidualFiLM(nn.Module):
    """
    Safe residual FiLM:
        out = x * (1 + gamma) + beta

    gamma/beta are zero initialized, so this module starts as identity.

    Input:
        x:    [B, T, D]
        cond: [B, D] or [B, T, D]
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)

        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return x * (1.0 + gamma) + beta


# =========================================================
# 4. Temporal conv / refinement
# =========================================================


class DepthwiseTemporalConv(nn.Module):
    """
    Lightweight local temporal bias.

    Input/output:
        x: [B, T, D]
    """

    def __init__(self, model_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(model_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=model_dim,
            ),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        out = residual + h

        if seq_mask is not None:
            out = out * seq_mask.unsqueeze(-1).float()
        return out


class TemporalRefineBlock(nn.Module):
    """
    Final branch-specific temporal refinement before velocity head.
    Useful for local pose smoothness.
    """

    def __init__(self, model_dim: int, kernel_size: int = 7, dropout: float = 0.05):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(model_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=model_dim,
            ),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        out = residual + h

        if seq_mask is not None:
            out = out * seq_mask.unsqueeze(-1).float()
        return out


# =========================================================
# 5. Condition injection
# =========================================================


class ConditionInjection(nn.Module):
    """
    Per-layer condition injection:
        1. residual FiLM
        2. optional gated cross-attention

    Input:
        x:    [B, T, D]
        cond: [B, T, D]
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        cross_attn_gate_init: float = 1.0,
        learnable_cross_attn_gate: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.cond_norm = nn.LayerNorm(model_dim)
        self.x_norm = nn.LayerNorm(model_dim)

        self.film = ResidualFiLM(
            cond_dim=model_dim,
            feat_dim=model_dim,
        )

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            if learnable_cross_attn_gate:
                self.cross_attn_gate = nn.Parameter(
                    torch.tensor(float(cross_attn_gate_init))
                )
            else:
                self.register_buffer(
                    "cross_attn_gate",
                    torch.tensor(float(cross_attn_gate_init)),
                )
        else:
            self.cross_attn = None
            self.register_buffer("cross_attn_gate", torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.cond_norm(cond)
        x = self.film(x, cond)

        if self.use_cross_attn:
            q = self.x_norm(x)
            attn_out, _ = self.cross_attn(
                query=q,
                key=cond,
                value=cond,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + self.cross_attn_gate * attn_out

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1).float()
        return x


# =========================================================
# 6. Stream Transformer block
# =========================================================


class StreamTransformerBlock(nn.Module):
    """
    One temporal transformer block for a single stream.

    Structure:
        self-attention + optional temporal conv + condition injection + FFN
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_condition_cross_attn: bool = True,
        use_local_temporal_conv: bool = True,
        temporal_kernel_size: int = 3,
        condition_cross_attn_gate_init: float = 1.0,
        learnable_condition_cross_attn_gate: bool = True,
    ):
        super().__init__()
        self.use_local_temporal_conv = use_local_temporal_conv

        self.norm1 = nn.LayerNorm(model_dim)
        self.time_film1 = ResidualFiLM(model_dim, model_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_local_temporal_conv:
            self.temporal_conv = DepthwiseTemporalConv(
                model_dim=model_dim,
                kernel_size=temporal_kernel_size,
                dropout=dropout,
            )
        else:
            self.temporal_conv = None

        self.cond_inject = ConditionInjection(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_cross_attn=use_condition_cross_attn,
            cross_attn_gate_init=condition_cross_attn_gate_init,
            learnable_cross_attn_gate=learnable_condition_cross_attn_gate,
        )

        self.norm2 = nn.LayerNorm(model_dim)
        self.time_film2 = ResidualFiLM(model_dim, model_dim)

        hidden_dim = int(model_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.time_film1(h, t_emb)

        attn_out, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1).float()

        if self.temporal_conv is not None:
            x = self.temporal_conv(x, seq_mask=seq_mask)

        x = self.cond_inject(
            x=x,
            cond=cond,
            key_padding_mask=key_padding_mask,
            seq_mask=seq_mask,
        )

        h = self.norm2(x)
        h = self.time_film2(h, t_emb)
        x = x + self.ffn(h)

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1).float()

        return x


# =========================================================
# 7. Two-stream pose-expression coordination
# =========================================================


class GatedPoseExprCrossBlock(nn.Module):
    """
    Lightweight bidirectional pose-expression coordination.

    Replaces the previous 3-stream shared block:
        old: pose <- shared, expr <- shared, shared <- pose+expr
        new: pose <- expr,   expr <- pose
    """

    def __init__(self, model_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.pose_norm = nn.LayerNorm(model_dim)
        self.expr_norm = nn.LayerNorm(model_dim)

        self.pose_from_expr = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.expr_from_pose = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pose_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid(),
        )
        self.expr_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_pose: torch.Tensor,
        h_expr: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        p = self.pose_norm(h_pose)
        e = self.expr_norm(h_expr)

        p_delta, _ = self.pose_from_expr(
            query=p,
            key=e,
            value=e,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        e_delta, _ = self.expr_from_pose(
            query=e,
            key=p,
            value=p,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        gp = self.pose_gate(torch.cat([h_pose, p_delta], dim=-1))
        ge = self.expr_gate(torch.cat([h_expr, e_delta], dim=-1))

        h_pose = h_pose + self.dropout(gp * p_delta)
        h_expr = h_expr + self.dropout(ge * e_delta)

        if seq_mask is not None:
            m = seq_mask.unsqueeze(-1).float()
            h_pose = h_pose * m
            h_expr = h_expr * m

        return h_pose, h_expr


# =========================================================
# 8. Velocity head
# =========================================================


class LatentVelocityHead(nn.Module):
    """
    Predict local velocity v_theta(x_t, t, condition).

    In Flow Matching, this velocity is used both for training target matching
    and for ODE sampling:
        x_{t+dt} = x_t + dt * v_theta(x_t, t, c)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# 9. Flow Matching Transformer Backbone, 2-stream version
# =========================================================


class FlowMatchingTransformerBackbone2Stream(nn.Module):
    """
    Two-stream Transformer backbone for conditional Flow Matching.

    Streams:
        1. pose stream
        2. expression stream

    This version removes the explicit shared stream from the previous 3-stream
    backbone. Coordination is performed by lightweight bidirectional
    pose-expression cross-attention.

    Forward input:
        x_pose_t:  [B, T, D_pose]
        x_expr_t:  [B, T, D_expr]
        cond_pose: [B, T, D_cond]
        cond_expr: [B, T, D_cond]
        t:         [B], scalar interpolation time in [0, 1]
        seq_mask:  [B, T], optional bool mask. True means valid frame.

    Returns:
        {
            "hidden_pose": [B, T, model_dim],
            "hidden_expr": [B, T, model_dim],
            "v_pose":      [B, T, D_pose],
            "v_expr":      [B, T, D_expr],
        }
    """

    def __init__(
        self,
        pose_latent_dim: int = 512,
        expr_latent_dim: int = 384,
        cond_dim: int = 256,
        model_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        time_scale: float = 1000.0,
        use_condition_cross_attn: bool = True,
        use_local_temporal_conv: bool = True,
        cross_stream_every: int = 2,
        zero_init_head: bool = True,
        # Branch-specific temporal and condition settings.
        pose_temporal_kernel_size: int = 7,
        expr_temporal_kernel_size: int = 3,
        pose_condition_cross_attn_gate_init: float = 0.1,
        expr_condition_cross_attn_gate_init: float = 1.0,
        learnable_condition_cross_attn_gate: bool = True,
        # State shortcut control.
        pose_state_shortcut_scale: float = 0.3,
        expr_state_shortcut_scale: float = 1.0,
        use_pose_state_gate: bool = True,
        use_expr_state_gate: bool = False,
        # Final temporal refinement.
        use_pose_temporal_refine: bool = True,
        use_expr_temporal_refine: bool = False,
        pose_refine_kernel_size: int = 7,
        expr_refine_kernel_size: int = 3,
    ):
        super().__init__()

        self.pose_latent_dim = pose_latent_dim
        self.expr_latent_dim = expr_latent_dim
        self.cond_dim = cond_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.cross_stream_every = cross_stream_every

        self.pose_state_shortcut_scale = float(pose_state_shortcut_scale)
        self.expr_state_shortcut_scale = float(expr_state_shortcut_scale)
        self.use_pose_state_gate = use_pose_state_gate
        self.use_expr_state_gate = use_expr_state_gate
        self.use_pose_temporal_refine = use_pose_temporal_refine
        self.use_expr_temporal_refine = use_expr_temporal_refine

        # State projections.
        self.pose_state_proj = nn.Sequential(
            nn.Linear(pose_latent_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.expr_state_proj = nn.Sequential(
            nn.Linear(expr_latent_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Condition projections.
        self.pose_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.expr_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.pos_emb = LearnablePositionEmbedding(
            max_seq_len=max_seq_len,
            model_dim=model_dim,
        )

        self.time_mlp = TimeMLP(
            time_dim=time_dim,
            model_dim=model_dim,
            time_scale=time_scale,
        )

        self.pose_blocks = nn.ModuleList([
            StreamTransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_condition_cross_attn=use_condition_cross_attn,
                use_local_temporal_conv=use_local_temporal_conv,
                temporal_kernel_size=pose_temporal_kernel_size,
                condition_cross_attn_gate_init=pose_condition_cross_attn_gate_init,
                learnable_condition_cross_attn_gate=learnable_condition_cross_attn_gate,
            )
            for _ in range(num_layers)
        ])

        self.expr_blocks = nn.ModuleList([
            StreamTransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_condition_cross_attn=use_condition_cross_attn,
                use_local_temporal_conv=use_local_temporal_conv,
                temporal_kernel_size=expr_temporal_kernel_size,
                condition_cross_attn_gate_init=expr_condition_cross_attn_gate_init,
                learnable_condition_cross_attn_gate=learnable_condition_cross_attn_gate,
            )
            for _ in range(num_layers)
        ])

        self.cross_blocks = nn.ModuleList([
            GatedPoseExprCrossBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.pose_out_norm = nn.LayerNorm(model_dim)
        self.expr_out_norm = nn.LayerNorm(model_dim)

        if use_pose_state_gate:
            self.pose_state_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.pose_state_gate = None

        if use_expr_state_gate:
            self.expr_state_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.expr_state_gate = None

        # 2-stream refine uses the other branch instead of h_shared:
        # pose: [h_pose, h_expr, p_state_short, p_cond]
        # expr: [h_expr, h_pose, e_state_short, e_cond]
        self.pose_refine = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.expr_refine = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        if use_pose_temporal_refine:
            self.pose_temporal_refine = TemporalRefineBlock(
                model_dim=model_dim,
                kernel_size=pose_refine_kernel_size,
                dropout=dropout,
            )
        else:
            self.pose_temporal_refine = nn.Identity()

        if use_expr_temporal_refine:
            self.expr_temporal_refine = TemporalRefineBlock(
                model_dim=model_dim,
                kernel_size=expr_refine_kernel_size,
                dropout=dropout,
            )
        else:
            self.expr_temporal_refine = nn.Identity()

        self.pose_head = LatentVelocityHead(
            in_dim=model_dim,
            out_dim=pose_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_init=zero_init_head,
        )
        self.expr_head = LatentVelocityHead(
            in_dim=model_dim,
            out_dim=expr_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_init=zero_init_head,
        )

    @staticmethod
    def _apply_state_shortcut_gate(
        state: torch.Tensor,
        cond: torch.Tensor,
        gate_module: Optional[nn.Module],
        scale: float,
    ) -> torch.Tensor:
        if gate_module is not None:
            gate = gate_module(torch.cat([state, cond], dim=-1))
            state = gate * state
        return scale * state

    def forward(
        self,
        x_pose_t: torch.Tensor,
        x_expr_t: torch.Tensor,
        cond_pose: torch.Tensor,
        cond_expr: torch.Tensor,
        t: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x_pose_t.shape

        if x_expr_t.shape[0] != B or x_expr_t.shape[1] != T:
            raise ValueError(
                "x_pose_t and x_expr_t must have the same batch and temporal dimensions."
            )
        if cond_pose.shape[0] != B or cond_pose.shape[1] != T:
            raise ValueError(
                "cond_pose must have shape [B, T, cond_dim]."
            )
        if cond_expr.shape[0] != B or cond_expr.shape[1] != T:
            raise ValueError(
                "cond_expr must have shape [B, T, cond_dim]."
            )

        if seq_mask is None:
            seq_mask = torch.ones(
                B,
                T,
                dtype=torch.bool,
                device=x_pose_t.device,
            )
        else:
            seq_mask = seq_mask.to(device=x_pose_t.device, dtype=torch.bool)

        key_padding_mask = ~seq_mask
        m = seq_mask.unsqueeze(-1).float()

        p_state = self.pose_state_proj(x_pose_t)
        e_state = self.expr_state_proj(x_expr_t)

        p_cond = self.pose_cond_proj(cond_pose)
        e_cond = self.expr_cond_proj(cond_expr)

        h_pose = self.pos_emb(p_state)
        h_expr = self.pos_emb(e_state)

        h_pose = h_pose * m
        h_expr = h_expr * m
        p_state = p_state * m
        e_state = e_state * m
        p_cond = p_cond * m
        e_cond = e_cond * m

        if t.dim() != 1:
            t = t.view(B)
        t_emb = self.time_mlp(t)

        for layer_idx in range(self.num_layers):
            h_pose = self.pose_blocks[layer_idx](
                x=h_pose,
                cond=p_cond,
                t_emb=t_emb,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

            h_expr = self.expr_blocks[layer_idx](
                x=h_expr,
                cond=e_cond,
                t_emb=t_emb,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

            if self.cross_stream_every > 0:
                if (layer_idx + 1) % self.cross_stream_every == 0:
                    h_pose, h_expr = self.cross_blocks[layer_idx](
                        h_pose=h_pose,
                        h_expr=h_expr,
                        key_padding_mask=key_padding_mask,
                        seq_mask=seq_mask,
                    )

        h_pose = self.pose_out_norm(h_pose) * m
        h_expr = self.expr_out_norm(h_expr) * m

        p_state_short = self._apply_state_shortcut_gate(
            state=p_state,
            cond=p_cond,
            gate_module=self.pose_state_gate,
            scale=self.pose_state_shortcut_scale,
        )
        e_state_short = self._apply_state_shortcut_gate(
            state=e_state,
            cond=e_cond,
            gate_module=self.expr_state_gate,
            scale=self.expr_state_shortcut_scale,
        )

        h_pose_out = self.pose_refine(
            torch.cat([h_pose, h_expr, p_state_short, p_cond], dim=-1)
        )
        h_expr_out = self.expr_refine(
            torch.cat([h_expr, h_pose, e_state_short, e_cond], dim=-1)
        )

        h_pose_out = h_pose_out * m
        h_expr_out = h_expr_out * m

        if self.use_pose_temporal_refine:
            h_pose_out = self.pose_temporal_refine(h_pose_out, seq_mask=seq_mask)
        else:
            h_pose_out = self.pose_temporal_refine(h_pose_out)

        if self.use_expr_temporal_refine:
            h_expr_out = self.expr_temporal_refine(h_expr_out, seq_mask=seq_mask)
        else:
            h_expr_out = self.expr_temporal_refine(h_expr_out)

        h_pose_out = h_pose_out * m
        h_expr_out = h_expr_out * m

        v_pose = self.pose_head(h_pose_out) * m
        v_expr = self.expr_head(h_expr_out) * m

        return {
            "hidden_pose": h_pose,
            "hidden_expr": h_expr,
            "v_pose": v_pose,
            "v_expr": v_expr,
        }


# =========================================================
# 10. Optional quick test
# =========================================================


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T = 60
    D_pose = 512
    D_expr = 384
    D_cond = 1024

    model = FlowMatchingTransformerBackbone2Stream(
        pose_latent_dim=D_pose,
        expr_latent_dim=D_expr,
        cond_dim=D_cond,
        model_dim=512,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        mlp_ratio=3.0,
        dropout=0.05,
        max_seq_len=128,
        cross_stream_every=2,
    ).to(device)

    x_pose_t = torch.randn(B, T, D_pose, device=device)
    x_expr_t = torch.randn(B, T, D_expr, device=device)
    cond_pose = torch.randn(B, T, D_cond, device=device)
    cond_expr = torch.randn(B, T, D_cond, device=device)
    t = torch.rand(B, device=device)
    seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    out = model(
        x_pose_t=x_pose_t,
        x_expr_t=x_expr_t,
        cond_pose=cond_pose,
        cond_expr=cond_expr,
        t=t,
        seq_mask=seq_mask,
    )

    print("v_pose:", out["v_pose"].shape)
    print("v_expr:", out["v_expr"].shape)
