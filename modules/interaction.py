import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGatedFiLM(nn.Module):
    """
    Lightweight gated FiLM:
        x_out = x + alpha * gate(cond) * (gamma(cond) * x + beta(cond))

    Supports:
        x: [B, T, D] or [B, T, J, D]
        cond: [B, T, C]
    """
    def __init__(
        self,
        feat_dim,
        cond_dim,
        hidden_dim=None,
        dropout=0.01,
        alpha_init=0.1,
        out_norm=False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or cond_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_gamma = nn.Linear(hidden_dim, feat_dim)
        self.to_beta = nn.Linear(hidden_dim, feat_dim)
        self.to_gate = nn.Linear(hidden_dim, feat_dim)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.use_out_norm = out_norm
        if out_norm:
            self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, cond):
        h = self.cond_proj(cond)                  # [B, T, H]

        gamma = self.to_gamma(h)                 # [B, T, D]
        beta = self.to_beta(h)                   # [B, T, D]
        gate = torch.sigmoid(self.to_gate(h))    # [B, T, D]

        if x.dim() == 4:
            gamma = gamma.unsqueeze(2)           # [B, T, 1, D]
            beta = beta.unsqueeze(2)
            gate = gate.unsqueeze(2)

        mod = gamma * x + beta
        out = x + self.alpha * gate * mod

        if self.use_out_norm:
            out = self.out_norm(out)

        return out


class ResidualGate(nn.Module):
    """
    Light residual gating:
        x_out = x + alpha * gate(cond) * delta(cond)

    Supports:
        x: [B, T, D] or [B, T, J, D]
        cond: [B, T, C]
    """
    def __init__(
        self,
        feat_dim,
        cond_dim,
        hidden_dim=None,
        dropout=0.01,
        alpha_init=0.1,
        out_norm=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or cond_dim
        self.out_norm_enabled = out_norm

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_delta = nn.Linear(hidden_dim, feat_dim)
        self.to_gate = nn.Linear(hidden_dim, feat_dim)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if out_norm:
            self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, cond):
        h = self.cond_proj(cond)
        delta = self.to_delta(h)                 # [B, T, D]
        gate = torch.sigmoid(self.to_gate(h))    # [B, T, D]

        if x.dim() == 4:
            delta = delta.unsqueeze(2)           # [B, T, 1, D]
            gate = gate.unsqueeze(2)

        out = x + self.alpha * gate * delta

        if self.out_norm_enabled:
            out = self.out_norm(out)

        return out


class AsymmetricPoseExprInteraction(nn.Module):
    """
    Asymmetric bidirectional interaction with different pose/expr dims.

    Main direction:
        expr -> pose_node

    Weak reverse:
        pose_global -> expr

    Inputs:
        pose_node_feat   : [B, T, J, Cp]
        pose_global_feat : [B, T, Cp]
        expr_feat        : [B, T, Ce]

    Outputs:
        pose_node_feat_out   : [B, T, J, Cp]
        pose_global_feat_out : [B, T, Cp]
        expr_feat_out        : [B, T, Ce]
    """
    def __init__(
        self,
        pose_dim,
        expr_dim,
        pose_hidden_dim=None,
        expr_hidden_dim=None,
        dropout=0.01,
        update_pose_global=True,
        use_film_for_pose=True,
        pose_alpha_init=0.1,
        expr_alpha_init=0.1,
        global_alpha_init=0.5,
        pose_out_norm=False,
        expr_out_norm=True,
    ):
        super().__init__()

        pose_hidden_dim = pose_hidden_dim or max(pose_dim, expr_dim)
        expr_hidden_dim = expr_hidden_dim or max(pose_dim, expr_dim)

        self.pose_dim = pose_dim
        self.expr_dim = expr_dim
        self.update_pose_global = update_pose_global
        self.use_film_for_pose = use_film_for_pose

        # expr -> pose_node
        if use_film_for_pose:
            self.expr_to_pose = LightGatedFiLM(
                feat_dim=pose_dim,
                cond_dim=expr_dim,
                hidden_dim=pose_hidden_dim,
                dropout=dropout,
                alpha_init=pose_alpha_init,
                out_norm=pose_out_norm,
            )
        else:
            self.expr_to_pose = ResidualGate(
                feat_dim=pose_dim,
                cond_dim=expr_dim,
                hidden_dim=pose_hidden_dim,
                dropout=dropout,
                alpha_init=pose_alpha_init,
                out_norm=pose_out_norm,
            )

        # pose_global -> expr
        self.pose_to_expr = ResidualGate(
            feat_dim=expr_dim,
            cond_dim=pose_dim,
            hidden_dim=expr_hidden_dim,
            dropout=dropout,
            alpha_init=expr_alpha_init,
            out_norm=expr_out_norm,
        )

        # blend old/new pose global instead of hard replace
        self.global_alpha = nn.Parameter(torch.tensor(float(global_alpha_init)))
        self.pose_global_norm = nn.LayerNorm(pose_dim)

    def forward(self, pose_node_feat, pose_global_feat, expr_feat):
        assert pose_node_feat.size(-1) == self.pose_dim
        assert pose_global_feat.size(-1) == self.pose_dim
        assert expr_feat.size(-1) == self.expr_dim

        # expr -> pose_node
        pose_node_feat_out = self.expr_to_pose(
            pose_node_feat,
            expr_feat,
        )

        # update pose global
        if self.update_pose_global:
            pose_global_new = pose_node_feat_out.mean(dim=2)
            pose_global_feat_out = pose_global_feat + self.global_alpha * (
                pose_global_new - pose_global_feat
            )
            pose_global_feat_out = self.pose_global_norm(pose_global_feat_out)
        else:
            pose_global_feat_out = pose_global_feat

        # pose_global -> expr
        expr_feat_out = self.pose_to_expr(
            expr_feat,
            pose_global_feat_out,
        )

        return pose_node_feat_out, pose_global_feat_out, expr_feat_out
    

class BottleneckPoseExprInteraction(nn.Module):
    """
    Lightweight latent-level interaction for bottleneck latents.

    Inputs:
        z_pose: [B, T_latent_pose, Dp]
        z_expr: [B, T_latent_expr, De]

    Note:
        This assumes pose/expr latent time length is aligned.
        In practice, use the same downsample factor for both branches
        when enabling bottleneck interaction.
    """
    def __init__(
        self,
        pose_dim,
        expr_dim,
        hidden_dim=256,
        dropout=0.01,
        pose_alpha_init=0.1,
        expr_alpha_init=0.1,
        pose_out_norm=True,
        expr_out_norm=True,
    ):
        super().__init__()

        self.expr_to_pose = nn.Sequential(
            nn.Linear(expr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pose_dim),
        )
        self.pose_to_expr = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expr_dim),
        )

        self.pose_gate = nn.Linear(expr_dim, pose_dim)
        self.expr_gate = nn.Linear(pose_dim, expr_dim)

        self.pose_alpha = nn.Parameter(torch.tensor(float(pose_alpha_init)))
        self.expr_alpha = nn.Parameter(torch.tensor(float(expr_alpha_init)))

        self.pose_out_norm_enabled = pose_out_norm
        self.expr_out_norm_enabled = expr_out_norm

        if pose_out_norm:
            self.pose_norm = nn.LayerNorm(pose_dim)
        if expr_out_norm:
            self.expr_norm = nn.LayerNorm(expr_dim)

    def forward(self, z_pose, z_expr):
        if z_pose.shape[1] != z_expr.shape[1]:
            raise ValueError(
                f"Bottleneck interaction requires aligned temporal length, "
                f"got z_pose.shape={tuple(z_pose.shape)}, z_expr.shape={tuple(z_expr.shape)}"
            )

        delta_pose = self.expr_to_pose(z_expr)
        gate_pose = torch.sigmoid(self.pose_gate(z_expr))
        z_pose = z_pose + self.pose_alpha * gate_pose * delta_pose
        if self.pose_out_norm_enabled:
            z_pose = self.pose_norm(z_pose)

        delta_expr = self.pose_to_expr(z_pose)
        gate_expr = torch.sigmoid(self.expr_gate(z_pose))
        z_expr = z_expr + self.expr_alpha * gate_expr * delta_expr
        if self.expr_out_norm_enabled:
            z_expr = self.expr_norm(z_expr)

        return z_pose, z_expr