
import torch
import torch.nn as nn
from typing import Optional, Dict

try:
    from .flow_matching_backbone_2stream import (
        LearnablePositionEmbedding,
        TimeMLP,
        ResidualFiLM,
        StreamTransformerBlock,
        TemporalRefineBlock,
        LatentVelocityHead,
    )
except ImportError:
    from flow_matching_backbone_2stream import (
        LearnablePositionEmbedding,
        TimeMLP,
        ResidualFiLM,
        StreamTransformerBlock,
        TemporalRefineBlock,
        LatentVelocityHead,
    )


# =========================================================
# 1. Pairwise stream coordination
# =========================================================

class GatedPairwiseCrossBlock(nn.Module):
    """
    Lightweight bidirectional cross-stream coordination.

    Input:
        h_a: [B, T, D]
        h_b: [B, T, D]

    Output:
        h_a, h_b with gated residual cross-attention updates.

    The residual scales are learnable and initialized small to keep early
    training stable.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        a_to_b_scale_init: float = 0.1,
        b_to_a_scale_init: float = 0.1,
    ):
        super().__init__()

        self.a_norm = nn.LayerNorm(model_dim)
        self.b_norm = nn.LayerNorm(model_dim)

        # a receives b
        self.a_from_b = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # b receives a
        self.b_from_a = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.a_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid(),
        )

        self.b_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid(),
        )

        self.a_scale = nn.Parameter(torch.tensor(float(b_to_a_scale_init)))
        self.b_scale = nn.Parameter(torch.tensor(float(a_to_b_scale_init)))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        a = self.a_norm(h_a)
        b = self.b_norm(h_b)

        a_delta, _ = self.a_from_b(
            query=a,
            key=b,
            value=b,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        b_delta, _ = self.b_from_a(
            query=b,
            key=a,
            value=a,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        ga = self.a_gate(torch.cat([h_a, a_delta], dim=-1))
        gb = self.b_gate(torch.cat([h_b, b_delta], dim=-1))

        h_a = h_a + self.a_scale * self.dropout(ga * a_delta)
        h_b = h_b + self.b_scale * self.dropout(gb * b_delta)

        if seq_mask is not None:
            m = seq_mask.unsqueeze(-1).float()
            h_a = h_a * m
            h_b = h_b * m

        return h_a, h_b


class GatedTriStreamCrossBlock(nn.Module):
    """
    3-stream coordination block.

    Streams:
        upper <-> lower
        upper <-> expr
        lower <-> expr

    Each pair can be enabled/disabled independently.

    Recommended default:
        - upper-lower: enabled
        - upper-expr : enabled
        - lower-expr : enabled but weak, or disabled if lower-body generation jitters
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_upper_lower: bool = True,
        use_upper_expr: bool = True,
        use_lower_expr: bool = True,
        upper_lower_scale_init: float = 0.1,
        upper_expr_scale_init: float = 0.1,
        lower_expr_scale_init: float = 0.03,
    ):
        super().__init__()

        self.use_upper_lower = use_upper_lower
        self.use_upper_expr = use_upper_expr
        self.use_lower_expr = use_lower_expr

        if use_upper_lower:
            self.upper_lower = GatedPairwiseCrossBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                a_to_b_scale_init=upper_lower_scale_init,
                b_to_a_scale_init=upper_lower_scale_init,
            )
        else:
            self.upper_lower = None

        if use_upper_expr:
            self.upper_expr = GatedPairwiseCrossBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                a_to_b_scale_init=upper_expr_scale_init,
                b_to_a_scale_init=upper_expr_scale_init,
            )
        else:
            self.upper_expr = None

        if use_lower_expr:
            self.lower_expr = GatedPairwiseCrossBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                a_to_b_scale_init=lower_expr_scale_init,
                b_to_a_scale_init=lower_expr_scale_init,
            )
        else:
            self.lower_expr = None

    def forward(
        self,
        h_upper: torch.Tensor,
        h_lower: torch.Tensor,
        h_expr: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        # Order follows the AE design:
        # body coordination first, then upper-expression, then lower-expression.
        if self.upper_lower is not None:
            h_upper, h_lower = self.upper_lower(
                h_upper,
                h_lower,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

        if self.upper_expr is not None:
            h_upper, h_expr = self.upper_expr(
                h_upper,
                h_expr,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

        if self.lower_expr is not None:
            h_lower, h_expr = self.lower_expr(
                h_lower,
                h_expr,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

        return h_upper, h_lower, h_expr


# =========================================================
# 2. Flow Matching Transformer Backbone, 3-stream version
# =========================================================

class FlowMatchingTransformerBackbone3Stream(nn.Module):
    """
    Three-stream Transformer backbone for conditional Flow Matching.

    Streams:
        1. upper-body pose stream
        2. lower-body pose stream
        3. expression stream

    Forward input:
        x_upper_t:  [B, T, D_upper]
        x_lower_t:  [B, T, D_lower]
        x_expr_t:   [B, T, D_expr]

        cond_upper: [B, T, D_cond]
        cond_lower: [B, T, D_cond]
        cond_expr:  [B, T, D_cond]

        t:          [B], scalar interpolation time in [0, 1]
        seq_mask:   [B, T], optional bool mask. True means valid frame.

    Returns:
        {
            "hidden_upper": [B, T, model_dim],
            "hidden_lower": [B, T, model_dim],
            "hidden_expr":  [B, T, model_dim],
            "v_upper":      [B, T, D_upper],
            "v_lower":      [B, T, D_lower],
            "v_expr":       [B, T, D_expr],
        }
    """

    def __init__(
        self,
        upper_latent_dim: int = 512,
        lower_latent_dim: int = 128,
        expr_latent_dim: int = 512,
        cond_dim: int = 512,
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

        # Branch-specific temporal kernels.
        upper_temporal_kernel_size: int = 7,
        lower_temporal_kernel_size: int = 7,
        expr_temporal_kernel_size: int = 3,

        # Branch-specific condition cross-attn gate.
        upper_condition_cross_attn_gate_init: float = 0.5,
        lower_condition_cross_attn_gate_init: float = 0.1,
        expr_condition_cross_attn_gate_init: float = 1.0,
        learnable_condition_cross_attn_gate: bool = True,

        # State shortcut control.
        upper_state_shortcut_scale: float = 0.5,
        lower_state_shortcut_scale: float = 0.3,
        expr_state_shortcut_scale: float = 1.0,
        use_upper_state_gate: bool = True,
        use_lower_state_gate: bool = True,
        use_expr_state_gate: bool = False,

        # Cross-stream controls.
        use_upper_lower_cross: bool = True,
        use_upper_expr_cross: bool = True,
        use_lower_expr_cross: bool = True,
        upper_lower_cross_scale_init: float = 0.1,
        upper_expr_cross_scale_init: float = 0.1,
        lower_expr_cross_scale_init: float = 0.03,

        # Final temporal refinement.
        use_upper_temporal_refine: bool = True,
        use_lower_temporal_refine: bool = True,
        use_expr_temporal_refine: bool = False,
        upper_refine_kernel_size: int = 7,
        lower_refine_kernel_size: int = 7,
        expr_refine_kernel_size: int = 3,
    ):
        super().__init__()

        self.upper_latent_dim = upper_latent_dim
        self.lower_latent_dim = lower_latent_dim
        self.expr_latent_dim = expr_latent_dim
        self.cond_dim = cond_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.cross_stream_every = cross_stream_every

        self.upper_state_shortcut_scale = float(upper_state_shortcut_scale)
        self.lower_state_shortcut_scale = float(lower_state_shortcut_scale)
        self.expr_state_shortcut_scale = float(expr_state_shortcut_scale)

        self.use_upper_state_gate = use_upper_state_gate
        self.use_lower_state_gate = use_lower_state_gate
        self.use_expr_state_gate = use_expr_state_gate

        self.use_upper_temporal_refine = use_upper_temporal_refine
        self.use_lower_temporal_refine = use_lower_temporal_refine
        self.use_expr_temporal_refine = use_expr_temporal_refine

        # -------------------------------------------------
        # State projections
        # -------------------------------------------------
        self.upper_state_proj = nn.Sequential(
            nn.Linear(upper_latent_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.lower_state_proj = nn.Sequential(
            nn.Linear(lower_latent_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.expr_state_proj = nn.Sequential(
            nn.Linear(expr_latent_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # -------------------------------------------------
        # Condition projections
        # -------------------------------------------------
        self.upper_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.lower_cond_proj = nn.Sequential(
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

        # -------------------------------------------------
        # Per-stream transformer blocks
        # -------------------------------------------------
        self.upper_blocks = nn.ModuleList([
            StreamTransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_condition_cross_attn=use_condition_cross_attn,
                use_local_temporal_conv=use_local_temporal_conv,
                temporal_kernel_size=upper_temporal_kernel_size,
                condition_cross_attn_gate_init=upper_condition_cross_attn_gate_init,
                learnable_condition_cross_attn_gate=learnable_condition_cross_attn_gate,
            )
            for _ in range(num_layers)
        ])

        self.lower_blocks = nn.ModuleList([
            StreamTransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_condition_cross_attn=use_condition_cross_attn,
                use_local_temporal_conv=use_local_temporal_conv,
                temporal_kernel_size=lower_temporal_kernel_size,
                condition_cross_attn_gate_init=lower_condition_cross_attn_gate_init,
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
            GatedTriStreamCrossBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_upper_lower=use_upper_lower_cross,
                use_upper_expr=use_upper_expr_cross,
                use_lower_expr=use_lower_expr_cross,
                upper_lower_scale_init=upper_lower_cross_scale_init,
                upper_expr_scale_init=upper_expr_cross_scale_init,
                lower_expr_scale_init=lower_expr_cross_scale_init,
            )
            for _ in range(num_layers)
        ])

        self.upper_out_norm = nn.LayerNorm(model_dim)
        self.lower_out_norm = nn.LayerNorm(model_dim)
        self.expr_out_norm = nn.LayerNorm(model_dim)

        # -------------------------------------------------
        # Optional state gates
        # -------------------------------------------------
        if use_upper_state_gate:
            self.upper_state_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.upper_state_gate = None

        if use_lower_state_gate:
            self.lower_state_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.lower_state_gate = None

        if use_expr_state_gate:
            self.expr_state_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.expr_state_gate = None

        # -------------------------------------------------
        # Final stream-specific refinement
        # -------------------------------------------------
        # upper: [h_upper, h_lower, h_expr, u_state_short, u_cond]
        # lower: [h_lower, h_upper, h_expr, l_state_short, l_cond]
        # expr : [h_expr, h_upper, h_lower, e_state_short, e_cond]
        refine_in_dim = model_dim * 5

        self.upper_refine = nn.Sequential(
            nn.Linear(refine_in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.lower_refine = nn.Sequential(
            nn.Linear(refine_in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.expr_refine = nn.Sequential(
            nn.Linear(refine_in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        if use_upper_temporal_refine:
            self.upper_temporal_refine = TemporalRefineBlock(
                model_dim=model_dim,
                kernel_size=upper_refine_kernel_size,
                dropout=dropout,
            )
        else:
            self.upper_temporal_refine = nn.Identity()

        if use_lower_temporal_refine:
            self.lower_temporal_refine = TemporalRefineBlock(
                model_dim=model_dim,
                kernel_size=lower_refine_kernel_size,
                dropout=dropout,
            )
        else:
            self.lower_temporal_refine = nn.Identity()

        if use_expr_temporal_refine:
            self.expr_temporal_refine = TemporalRefineBlock(
                model_dim=model_dim,
                kernel_size=expr_refine_kernel_size,
                dropout=dropout,
            )
        else:
            self.expr_temporal_refine = nn.Identity()

        # -------------------------------------------------
        # Velocity heads
        # -------------------------------------------------
        self.upper_head = LatentVelocityHead(
            in_dim=model_dim,
            out_dim=upper_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_init=zero_init_head,
        )

        self.lower_head = LatentVelocityHead(
            in_dim=model_dim,
            out_dim=lower_latent_dim,
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
        x_upper_t: torch.Tensor,
        x_lower_t: torch.Tensor,
        x_expr_t: torch.Tensor,
        cond_upper: torch.Tensor,
        cond_lower: torch.Tensor,
        cond_expr: torch.Tensor,
        t: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x_upper_t.shape

        if x_lower_t.shape[0] != B or x_lower_t.shape[1] != T:
            raise ValueError("x_upper_t and x_lower_t must have the same batch/time shape.")
        if x_expr_t.shape[0] != B or x_expr_t.shape[1] != T:
            raise ValueError("x_upper_t and x_expr_t must have the same batch/time shape.")

        if cond_upper.shape[0] != B or cond_upper.shape[1] != T:
            raise ValueError("cond_upper must have shape [B, T, cond_dim].")
        if cond_lower.shape[0] != B or cond_lower.shape[1] != T:
            raise ValueError("cond_lower must have shape [B, T, cond_dim].")
        if cond_expr.shape[0] != B or cond_expr.shape[1] != T:
            raise ValueError("cond_expr must have shape [B, T, cond_dim].")

        if seq_mask is None:
            seq_mask = torch.ones(
                B,
                T,
                dtype=torch.bool,
                device=x_upper_t.device,
            )
        else:
            seq_mask = seq_mask.to(device=x_upper_t.device, dtype=torch.bool)

        key_padding_mask = ~seq_mask
        m = seq_mask.unsqueeze(-1).float()

        u_state = self.upper_state_proj(x_upper_t)
        l_state = self.lower_state_proj(x_lower_t)
        e_state = self.expr_state_proj(x_expr_t)

        u_cond = self.upper_cond_proj(cond_upper)
        l_cond = self.lower_cond_proj(cond_lower)
        e_cond = self.expr_cond_proj(cond_expr)

        h_upper = self.pos_emb(u_state)
        h_lower = self.pos_emb(l_state)
        h_expr = self.pos_emb(e_state)

        h_upper = h_upper * m
        h_lower = h_lower * m
        h_expr = h_expr * m

        u_state = u_state * m
        l_state = l_state * m
        e_state = e_state * m

        u_cond = u_cond * m
        l_cond = l_cond * m
        e_cond = e_cond * m

        if t.dim() != 1:
            t = t.view(B)
        t_emb = self.time_mlp(t)

        for layer_idx in range(self.num_layers):
            h_upper = self.upper_blocks[layer_idx](
                x=h_upper,
                cond=u_cond,
                t_emb=t_emb,
                key_padding_mask=key_padding_mask,
                seq_mask=seq_mask,
            )

            h_lower = self.lower_blocks[layer_idx](
                x=h_lower,
                cond=l_cond,
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
                    h_upper, h_lower, h_expr = self.cross_blocks[layer_idx](
                        h_upper=h_upper,
                        h_lower=h_lower,
                        h_expr=h_expr,
                        key_padding_mask=key_padding_mask,
                        seq_mask=seq_mask,
                    )

        h_upper = self.upper_out_norm(h_upper) * m
        h_lower = self.lower_out_norm(h_lower) * m
        h_expr = self.expr_out_norm(h_expr) * m

        u_state_short = self._apply_state_shortcut_gate(
            state=u_state,
            cond=u_cond,
            gate_module=self.upper_state_gate,
            scale=self.upper_state_shortcut_scale,
        )

        l_state_short = self._apply_state_shortcut_gate(
            state=l_state,
            cond=l_cond,
            gate_module=self.lower_state_gate,
            scale=self.lower_state_shortcut_scale,
        )

        e_state_short = self._apply_state_shortcut_gate(
            state=e_state,
            cond=e_cond,
            gate_module=self.expr_state_gate,
            scale=self.expr_state_shortcut_scale,
        )

        h_upper_out = self.upper_refine(
            torch.cat([h_upper, h_lower, h_expr, u_state_short, u_cond], dim=-1)
        )

        h_lower_out = self.lower_refine(
            torch.cat([h_lower, h_upper, h_expr, l_state_short, l_cond], dim=-1)
        )

        h_expr_out = self.expr_refine(
            torch.cat([h_expr, h_upper, h_lower, e_state_short, e_cond], dim=-1)
        )

        h_upper_out = h_upper_out * m
        h_lower_out = h_lower_out * m
        h_expr_out = h_expr_out * m

        if self.use_upper_temporal_refine:
            h_upper_out = self.upper_temporal_refine(h_upper_out, seq_mask=seq_mask)
        else:
            h_upper_out = self.upper_temporal_refine(h_upper_out)

        if self.use_lower_temporal_refine:
            h_lower_out = self.lower_temporal_refine(h_lower_out, seq_mask=seq_mask)
        else:
            h_lower_out = self.lower_temporal_refine(h_lower_out)

        if self.use_expr_temporal_refine:
            h_expr_out = self.expr_temporal_refine(h_expr_out, seq_mask=seq_mask)
        else:
            h_expr_out = self.expr_temporal_refine(h_expr_out)

        h_upper_out = h_upper_out * m
        h_lower_out = h_lower_out * m
        h_expr_out = h_expr_out * m

        v_upper = self.upper_head(h_upper_out) * m
        v_lower = self.lower_head(h_lower_out) * m
        v_expr = self.expr_head(h_expr_out) * m

        return {
            "hidden_upper": h_upper,
            "hidden_lower": h_lower,
            "hidden_expr": h_expr,

            "hidden_upper_out": h_upper_out,
            "hidden_lower_out": h_lower_out,
            "hidden_expr_out": h_expr_out,

            "v_upper": v_upper,
            "v_lower": v_lower,
            "v_expr": v_expr,
        }


# Backward-compatible shorter alias.
FlowMatchingBackbone3Stream = FlowMatchingTransformerBackbone3Stream


# =========================================================
# 3. Optional quick test
# =========================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T = 60
    D_upper = 512
    D_lower = 128
    D_expr = 512
    D_cond = 512

    model = FlowMatchingTransformerBackbone3Stream(
        upper_latent_dim=D_upper,
        lower_latent_dim=D_lower,
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

    x_upper_t = torch.randn(B, T, D_upper, device=device)
    x_lower_t = torch.randn(B, T, D_lower, device=device)
    x_expr_t = torch.randn(B, T, D_expr, device=device)

    cond_upper = torch.randn(B, T, D_cond, device=device)
    cond_lower = torch.randn(B, T, D_cond, device=device)
    cond_expr = torch.randn(B, T, D_cond, device=device)

    t = torch.rand(B, device=device)
    seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    out = model(
        x_upper_t=x_upper_t,
        x_lower_t=x_lower_t,
        x_expr_t=x_expr_t,
        cond_upper=cond_upper,
        cond_lower=cond_lower,
        cond_expr=cond_expr,
        t=t,
        seq_mask=seq_mask,
    )

    print("v_upper:", out["v_upper"].shape)
    print("v_lower:", out["v_lower"].shape)
    print("v_expr :", out["v_expr"].shape)
