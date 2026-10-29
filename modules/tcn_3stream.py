
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Refine-style temporal blocks
# =========================================================

class ConvNormAct1d(nn.Module):
    """
    Conv1d block for [B, T, C].
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout=0.01,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd."

        padding = ((kernel_size - 1) // 2) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class TemporalResBlock(nn.Module):
    """
    Refine residual block for [B, T, C].
    Temporal length unchanged.
    """
    def __init__(
        self,
        dim,
        hidden_dim=None,
        kernel_size=3,
        dilation=1,
        dropout=0.01,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.block1 = ConvNormAct1d(
            in_dim=dim,
            out_dim=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            dropout=dropout,
        )
        self.block2 = ConvNormAct1d(
            in_dim=hidden_dim,
            out_dim=dim,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.out_norm(x + self.block2(self.block1(x)))


class TemporalDownsampleBlock(nn.Module):
    """
    Refine-style downsample block.
    [B, T, C] -> [B, T/2, C_out]
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()

        self.down = ConvNormAct1d(
            in_dim=in_dim,
            out_dim=out_dim,
            kernel_size=kernel_size,
            stride=2,
            dilation=1,
            dropout=dropout,
        )

        self.refine = TemporalResBlock(
            dim=out_dim,
            hidden_dim=out_dim,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.down(x)
        x = self.refine(x)
        return x


class TemporalUpsampleBlock(nn.Module):
    """
    Refine-style upsample block.
    [B, T, C] -> [B, 2T, C_out]
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        dropout=0.01,
        mode="nearest",
    ):
        super().__init__()

        self.mode = mode
        self.proj = nn.Linear(in_dim, out_dim)
        self.proj_norm = nn.LayerNorm(out_dim)

        self.refine = TemporalResBlock(
            dim=out_dim,
            hidden_dim=out_dim,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        if self.mode == "linear":
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="linear",
                align_corners=False,
            )
        else:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode=self.mode,
            )

        x = x.transpose(1, 2)

        x = self.proj(x)
        x = self.proj_norm(x)
        x = F.gelu(x)
        x = self.refine(x)
        return x


class TemporalEncoder(nn.Module):
    """
    Refine-style temporal encoder.

    Input:
        x: [B, T, in_dim]

    Output:
        z: [B, T/downsample_factor, latent_dim]
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=512,
        latent_dim=512,
        num_pre_blocks=1,
        num_post_blocks=1,
        kernel_size=3,
        pre_dilations=(1, 2),
        post_dilations=(1, 2),
        downsample_factor=4,
        dropout=0.01,
    ):
        super().__init__()
        assert downsample_factor in [1, 2, 4]

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.pre_blocks = nn.ModuleList([
            TemporalResBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=pre_dilations[i % len(pre_dilations)],
                dropout=dropout,
            )
            for i in range(num_pre_blocks)
        ])

        self.down_blocks = nn.ModuleList()
        num_down = 0
        if downsample_factor >= 2:
            num_down += 1
        if downsample_factor >= 4:
            num_down += 1

        for _ in range(num_down):
            self.down_blocks.append(
                TemporalDownsampleBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        self.post_blocks = nn.ModuleList([
            TemporalResBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=post_dilations[i % len(post_dilations)],
                dropout=dropout,
            )
            for i in range(num_post_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        for blk in self.pre_blocks:
            x = blk(x)

        for blk in self.down_blocks:
            x = blk(x)

        for blk in self.post_blocks:
            x = blk(x)

        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


class TemporalDecoder(nn.Module):
    """
    Refine-style temporal decoder.

    Input:
        z: [B, T_latent, latent_dim]

    Output:
        x: [B, T, out_dim]
    """
    def __init__(
        self,
        latent_dim,
        hidden_dim=512,
        out_dim=512,
        num_pre_blocks=1,
        num_post_blocks=1,
        kernel_size=3,
        pre_dilations=(1, 2),
        post_dilations=(1, 2),
        upsample_factor=4,
        dropout=0.01,
        upsample_mode="nearest",
        final_norm=True,
    ):
        super().__init__()
        assert upsample_factor in [1, 2, 4]

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.pre_blocks = nn.ModuleList([
            TemporalResBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=pre_dilations[i % len(pre_dilations)],
                dropout=dropout,
            )
            for i in range(num_pre_blocks)
        ])

        self.up_blocks = nn.ModuleList()
        num_up = 0
        if upsample_factor >= 2:
            num_up += 1
        if upsample_factor >= 4:
            num_up += 1

        for _ in range(num_up):
            self.up_blocks.append(
                TemporalUpsampleBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    mode=upsample_mode,
                )
            )

        self.post_blocks = nn.ModuleList([
            TemporalResBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=post_dilations[i % len(post_dilations)],
                dropout=dropout,
            )
            for i in range(num_post_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.output_norm = nn.LayerNorm(out_dim) if final_norm else nn.Identity()

    def forward(self, z):
        x = self.input_proj(z)
        x = self.input_norm(x)
        x = F.gelu(x)

        for blk in self.pre_blocks:
            x = blk(x)

        for blk in self.up_blocks:
            x = blk(x)

        for blk in self.post_blocks:
            x = blk(x)

        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


# =========================================================
# 2. Simple residual-style temporal blocks
# =========================================================

class ResidualDownsampleBlock1D(nn.Module):
    """
    Residual-style temporal downsample.

    Input:
        x: [B, T, D]

    Output:
        out: [B, T/2, D]
    """
    def __init__(
        self,
        dim,
        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        padding = (kernel_size - 1) // 2

        self.main_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
        )
        self.main_norm = nn.LayerNorm(dim)
        self.main_dropout = nn.Dropout(dropout)

        self.short_proj = nn.Linear(dim, dim)
        self.short_norm = nn.LayerNorm(dim)

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        main = x.transpose(1, 2)
        main = self.main_conv(main)
        main = main.transpose(1, 2)
        main = self.main_norm(main)
        main = F.gelu(main)
        main = self.main_dropout(main)

        short = x.transpose(1, 2)
        short = F.avg_pool1d(short, kernel_size=2, stride=2)
        short = short.transpose(1, 2)
        short = self.short_proj(short)
        short = self.short_norm(short)

        out = self.out_norm(main + short)
        return out


class ResidualUpsampleBlock1D(nn.Module):
    """
    Residual-style temporal upsample.

    Input:
        x: [B, T, D]

    Output:
        out: [B, 2T, D]
    """
    def __init__(
        self,
        dim,
        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()
        assert kernel_size == 3, (
            "ResidualUpsampleBlock1D currently assumes kernel_size=3 "
            "for exact 2x temporal upsampling."
        )

        self.main_up = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.main_norm = nn.LayerNorm(dim)
        self.main_dropout = nn.Dropout(dropout)

        self.short_proj = nn.Linear(dim, dim)
        self.short_norm = nn.LayerNorm(dim)

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        main = x.transpose(1, 2)
        main = self.main_up(main)
        main = main.transpose(1, 2)
        main = self.main_norm(main)
        main = F.gelu(main)
        main = self.main_dropout(main)

        short = x.transpose(1, 2)
        short = F.interpolate(short, scale_factor=2, mode="nearest")
        short = short.transpose(1, 2)
        short = self.short_proj(short)
        short = self.short_norm(short)

        out = self.out_norm(main + short)
        return out


class ResidualTemporalEncoder(nn.Module):
    """
    Simple residual-style temporal encoder.

    Input:
        x: [B, T, D]

    Output:
        z: [B, T/downsample_factor, D]
    """
    def __init__(
        self,
        dim,
        downsample_factor=4,
        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()
        assert downsample_factor in [1, 2, 4]

        self.down_blocks = nn.ModuleList()

        if downsample_factor >= 2:
            self.down_blocks.append(
                ResidualDownsampleBlock1D(
                    dim=dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        if downsample_factor >= 4:
            self.down_blocks.append(
                ResidualDownsampleBlock1D(
                    dim=dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        for blk in self.down_blocks:
            x = blk(x)
        return x


class ResidualTemporalDecoder(nn.Module):
    """
    Simple residual-style temporal decoder.

    Input:
        z: [B, T_latent, D]

    Output:
        x: [B, T, D]
    """
    def __init__(
        self,
        dim,
        upsample_factor=4,
        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()
        assert upsample_factor in [1, 2, 4]

        self.up_blocks = nn.ModuleList()

        if upsample_factor >= 2:
            self.up_blocks.append(
                ResidualUpsampleBlock1D(
                    dim=dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        if upsample_factor >= 4:
            self.up_blocks.append(
                ResidualUpsampleBlock1D(
                    dim=dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

    def forward(self, z):
        for blk in self.up_blocks:
            z = blk(z)
        return z


# =========================================================
# 3. Projection wrappers for simple residual temporal path
# =========================================================

class SimpleTemporalEncoder(nn.Module):
    """
    Simple temporal encoder with optional channel projection.

    Input:
        x: [B, T, in_dim]

    Output:
        z: [B, T/downsample_factor, latent_dim]
    """
    def __init__(
        self,
        in_dim,
        latent_dim,
        downsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        use_input_norm=True,
    ):
        super().__init__()

        self.need_proj = in_dim != latent_dim

        if self.need_proj:
            self.in_proj = nn.Linear(in_dim, latent_dim)
            self.in_norm = nn.LayerNorm(latent_dim) if use_input_norm else nn.Identity()
        else:
            self.in_proj = nn.Identity()
            self.in_norm = nn.Identity()

        self.temporal = ResidualTemporalEncoder(
            dim=latent_dim,
            downsample_factor=downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.in_norm(x)

        if self.need_proj:
            x = F.gelu(x)

        z = self.temporal(x)
        return z


class SimpleTemporalDecoder(nn.Module):
    """
    Simple temporal decoder with optional channel projection.

    Input:
        z: [B, T_latent, latent_dim]

    Output:
        x: [B, T, out_dim]
    """
    def __init__(
        self,
        latent_dim,
        out_dim,
        upsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        final_norm=False,
    ):
        super().__init__()

        self.need_proj = latent_dim != out_dim

        self.temporal = ResidualTemporalDecoder(
            dim=latent_dim,
            upsample_factor=upsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        if self.need_proj:
            self.out_proj = nn.Linear(latent_dim, out_dim)
            self.out_norm = nn.LayerNorm(out_dim) if final_norm else nn.Identity()
        else:
            self.out_proj = nn.Identity()
            self.out_norm = nn.Identity()

    def forward(self, z):
        x = self.temporal(z)
        x = self.out_proj(x)
        x = self.out_norm(x)
        return x


# =========================================================
# 4. Pose node-wise temporal encoder / decoder
# =========================================================

class PoseNodeTemporalEncoder(nn.Module):
    """
    Pose node-wise temporal encoder.

    Input:
        pose_node_feat: [B, T, J, D]

    Output:
        z_pose: [B, T_latent, pose_latent_dim]
    """
    def __init__(
        self,
        num_nodes,
        node_dim,
        node_hidden_dim=64,
        node_latent_dim=64,
        pose_latent_dim=512,
        downsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        num_pre_blocks=1,
        num_post_blocks=1,
        temporal_type="refine",
    ):
        super().__init__()
        assert temporal_type in ["refine", "simple"]

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.node_latent_dim = node_latent_dim
        self.flat_dim = num_nodes * node_latent_dim

        if temporal_type == "refine":
            self.node_encoder = TemporalEncoder(
                in_dim=node_dim,
                hidden_dim=node_hidden_dim,
                latent_dim=node_latent_dim,
                num_pre_blocks=num_pre_blocks,
                num_post_blocks=num_post_blocks,
                downsample_factor=downsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.node_encoder = SimpleTemporalEncoder(
                in_dim=node_dim,
                latent_dim=node_latent_dim,
                downsample_factor=downsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        self.fuse_proj = nn.Linear(self.flat_dim, pose_latent_dim)
        self.fuse_norm = nn.LayerNorm(pose_latent_dim)

    def forward(self, pose_node_feat):
        B, T, J, D = pose_node_feat.shape
        assert J == self.num_nodes, f"Expected {self.num_nodes} nodes, got {J}"
        assert D == self.node_dim, f"Expected node_dim={self.node_dim}, got {D}"

        x = pose_node_feat.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * J, T, D)

        z_node = self.node_encoder(x)
        T_latent = z_node.shape[1]

        z_node = z_node.view(
            B,
            J,
            T_latent,
            self.node_latent_dim,
        ).permute(0, 2, 1, 3).contiguous()

        z_flat = z_node.view(B, T_latent, self.flat_dim)

        z_pose = self.fuse_proj(z_flat)
        z_pose = self.fuse_norm(z_pose)

        return z_pose


class PoseNodeTemporalDecoder(nn.Module):
    """
    Pose node-wise temporal decoder.

    Input:
        z_pose: [B, T_latent, pose_latent_dim]

    Output:
        pose_node_dec: [B, T, J, node_dim]
    """
    def __init__(
        self,
        latent_dim,
        num_nodes,
        node_dim,
        node_hidden_dim=64,
        node_latent_dim=64,
        upsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        num_pre_blocks=1,
        num_post_blocks=1,
        temporal_type="refine",
        final_norm=True,
    ):
        super().__init__()
        assert temporal_type in ["refine", "simple"]

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.node_latent_dim = node_latent_dim
        self.flat_dim = num_nodes * node_latent_dim

        self.expand_proj = nn.Linear(latent_dim, self.flat_dim)
        self.expand_norm = nn.LayerNorm(self.flat_dim)

        if temporal_type == "refine":
            self.node_decoder = TemporalDecoder(
                latent_dim=node_latent_dim,
                hidden_dim=node_hidden_dim,
                out_dim=node_dim,
                num_pre_blocks=num_pre_blocks,
                num_post_blocks=num_post_blocks,
                upsample_factor=upsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
                final_norm=final_norm,
            )
        else:
            self.node_decoder = SimpleTemporalDecoder(
                latent_dim=node_latent_dim,
                out_dim=node_dim,
                upsample_factor=upsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
                final_norm=final_norm,
            )

    def forward(self, z_pose):
        B, T_latent, _ = z_pose.shape

        x = self.expand_proj(z_pose)
        x = self.expand_norm(x)
        x = x.view(
            B,
            T_latent,
            self.num_nodes,
            self.node_latent_dim,
        )

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * self.num_nodes, T_latent, self.node_latent_dim)

        x = self.node_decoder(x)
        T_out = x.shape[1]

        x = x.view(
            B,
            self.num_nodes,
            T_out,
            self.node_dim,
        ).permute(0, 2, 1, 3).contiguous()

        return x


# =========================================================
# 5. Expression temporal encoder / decoder
# =========================================================

class ExpressionTemporalEncoder(nn.Module):
    """
    Expression temporal encoder.

    Input:
        expr_feat: [B, T, in_dim]

    Output:
        z_expr: [B, T_latent, latent_dim]
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=512,
        latent_dim=384,
        downsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        num_pre_blocks=0,
        num_post_blocks=0,
        temporal_type="simple",
    ):
        super().__init__()
        assert temporal_type in ["refine", "simple"]

        if temporal_type == "refine":
            self.encoder = TemporalEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_pre_blocks=num_pre_blocks,
                num_post_blocks=num_post_blocks,
                downsample_factor=downsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.encoder = SimpleTemporalEncoder(
                in_dim=in_dim,
                latent_dim=latent_dim,
                downsample_factor=downsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
            )

    def forward(self, x):
        return self.encoder(x)


class ExpressionTemporalDecoder(nn.Module):
    """
    Expression temporal decoder.

    Input:
        z_expr: [B, T_latent, latent_dim]

    Output:
        expr_temp_dec: [B, T, out_dim]
    """
    def __init__(
        self,
        latent_dim,
        out_dim,
        hidden_dim=512,
        upsample_factor=4,
        kernel_size=3,
        dropout=0.01,
        num_pre_blocks=0,
        num_post_blocks=0,
        temporal_type="simple",
        final_norm=False,
    ):
        super().__init__()
        assert temporal_type in ["refine", "simple"]

        if temporal_type == "refine":
            self.decoder = TemporalDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_pre_blocks=num_pre_blocks,
                num_post_blocks=num_post_blocks,
                upsample_factor=upsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
                final_norm=final_norm,
            )
        else:
            self.decoder = SimpleTemporalDecoder(
                latent_dim=latent_dim,
                out_dim=out_dim,
                upsample_factor=upsample_factor,
                kernel_size=kernel_size,
                dropout=dropout,
                final_norm=final_norm,
            )

    def forward(self, z):
        return self.decoder(z)


# =========================================================
# 6. Dual branch temporal AE
# =========================================================

class DualBranchTemporalAE(nn.Module):
    """
    Dual-branch temporal AE.

    This is kept for compatibility and ablation.

    Pose branch:
        pose_node_feat [B, T, J, Dp]
        -> z_pose [B, T_latent_pose, D_pose]
        -> pose_node_dec [B, T, J, Dp]

    Expr branch:
        expr_feat [B, T, De]
        -> z_expr [B, T_latent_expr, D_expr]
        -> expr_temp_dec [B, T, De]
    """
    def __init__(
        self,
        pose_num_nodes=55,
        pose_node_dim=64,
        expr_in_dim=384,

        pose_latent_dim=512,
        expr_latent_dim=384,

        pose_node_hidden_dim=64,
        pose_node_latent_dim=64,
        expr_hidden_dim=512,

        pose_decoder_node_hidden_dim=64,
        pose_decoder_node_latent_dim=64,
        expr_decoder_hidden_dim=512,

        pose_downsample_factor=4,
        expr_downsample_factor=4,

        pose_pre_blocks=1,
        pose_post_blocks=1,
        expr_pre_blocks=0,
        expr_post_blocks=0,

        pose_decoder_pre_blocks=1,
        pose_decoder_post_blocks=1,
        expr_decoder_pre_blocks=0,
        expr_decoder_post_blocks=0,

        pose_temporal_type="refine",
        expr_temporal_type="simple",

        kernel_size=3,
        dropout=0.01,

        pose_decoder_final_norm=True,
        expr_decoder_final_norm=False,
    ):
        super().__init__()

        assert pose_temporal_type in ["refine", "simple"]
        assert expr_temporal_type in ["refine", "simple"]

        self.pose_num_nodes = pose_num_nodes
        self.pose_node_dim = pose_node_dim
        self.expr_in_dim = expr_in_dim

        self.pose_latent_dim = pose_latent_dim
        self.expr_latent_dim = expr_latent_dim

        self.pose_downsample_factor = pose_downsample_factor
        self.expr_downsample_factor = expr_downsample_factor

        self.pose_temporal_type = pose_temporal_type
        self.expr_temporal_type = expr_temporal_type

        self.pose_encoder = PoseNodeTemporalEncoder(
            num_nodes=pose_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=pose_node_hidden_dim,
            node_latent_dim=pose_node_latent_dim,
            pose_latent_dim=pose_latent_dim,
            downsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=pose_pre_blocks,
            num_post_blocks=pose_post_blocks,
            temporal_type=pose_temporal_type,
        )

        self.expr_encoder = ExpressionTemporalEncoder(
            in_dim=expr_in_dim,
            hidden_dim=expr_hidden_dim,
            latent_dim=expr_latent_dim,
            downsample_factor=expr_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=expr_pre_blocks,
            num_post_blocks=expr_post_blocks,
            temporal_type=expr_temporal_type,
        )

        self.pose_decoder = PoseNodeTemporalDecoder(
            latent_dim=pose_latent_dim,
            num_nodes=pose_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=pose_decoder_node_hidden_dim,
            node_latent_dim=pose_decoder_node_latent_dim,
            upsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=pose_decoder_pre_blocks,
            num_post_blocks=pose_decoder_post_blocks,
            temporal_type=pose_temporal_type,
            final_norm=pose_decoder_final_norm,
        )

        self.expr_decoder = ExpressionTemporalDecoder(
            latent_dim=expr_latent_dim,
            out_dim=expr_in_dim,
            hidden_dim=expr_decoder_hidden_dim,
            upsample_factor=expr_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=expr_decoder_pre_blocks,
            num_post_blocks=expr_decoder_post_blocks,
            temporal_type=expr_temporal_type,
            final_norm=expr_decoder_final_norm,
        )

    def encode_to_latents(self, pose_node_feat, expr_feat):
        z_pose = self.pose_encoder(pose_node_feat)
        z_expr = self.expr_encoder(expr_feat)

        return {
            "z_pose": z_pose,
            "z_expr": z_expr,
        }

    def decode_from_latents(self, z_pose, z_expr):
        pose_node_dec = self.pose_decoder(z_pose)
        expr_temp_dec = self.expr_decoder(z_expr)

        return {
            "pose_node_dec": pose_node_dec,
            "expr_temp_dec": expr_temp_dec,
        }

    def forward(self, pose_node_feat, expr_feat):
        enc_out = self.encode_to_latents(
            pose_node_feat=pose_node_feat,
            expr_feat=expr_feat,
        )

        dec_out = self.decode_from_latents(
            z_pose=enc_out["z_pose"],
            z_expr=enc_out["z_expr"],
        )

        return {
            "z_pose": enc_out["z_pose"],
            "z_expr": enc_out["z_expr"],
            "pose_node_dec": dec_out["pose_node_dec"],
            "expr_temp_dec": dec_out["expr_temp_dec"],
        }


# =========================================================
# 7. Tri branch temporal AE
# =========================================================

class TriBranchTemporalAE(nn.Module):
    """
    Pure temporal 3-stream AE for the new ae.py.

    Important:
        This class does NOT implement any interaction.
        All interactions are handled in ae.py:
            - pre full pose-expression interaction
            - bottleneck upper/lower/expression interaction
            - post full pose-expression interaction

    Streams:
        pose_node_feat [B, T, 55, Dp]
            -> upper_ids -> z_upper
            -> lower_ids -> z_lower

        expr_feat [B, T, De]
            -> z_expr

    Outputs:
        z_upper      : [B, T_latent, upper_latent_dim]
        z_lower      : [B, T_latent, lower_latent_dim]
        z_expr       : [B, T_latent, expr_latent_dim]
        pose_node_dec: [B, T, 55, Dp]
        expr_temp_dec: [B, T, De]

    Compatibility:
        The constructor accepts interaction-related switches used by previous
        tcn_3stream.py versions, but ignores them. This keeps ae.py calls stable.
    """

    def __init__(
        self,
        pose_num_nodes=55,
        pose_node_dim=64,
        expr_in_dim=384,

        upper_ids=None,
        lower_ids=None,

        upper_latent_dim=512,
        lower_latent_dim=256,
        expr_latent_dim=384,

        upper_node_hidden_dim=64,
        lower_node_hidden_dim=64,
        upper_node_latent_dim=64,
        lower_node_latent_dim=64,
        expr_hidden_dim=512,

        upper_decoder_node_hidden_dim=64,
        lower_decoder_node_hidden_dim=64,
        upper_decoder_node_latent_dim=64,
        lower_decoder_node_latent_dim=64,
        expr_decoder_hidden_dim=512,

        pose_downsample_factor=4,
        expr_downsample_factor=4,

        upper_pre_blocks=1,
        upper_post_blocks=1,
        lower_pre_blocks=1,
        lower_post_blocks=1,
        expr_pre_blocks=0,
        expr_post_blocks=0,

        upper_decoder_pre_blocks=1,
        upper_decoder_post_blocks=1,
        lower_decoder_pre_blocks=1,
        lower_decoder_post_blocks=1,
        expr_decoder_pre_blocks=0,
        expr_decoder_post_blocks=0,

        pose_temporal_type="refine",
        expr_temporal_type="simple",

        kernel_size=3,
        dropout=0.01,

        pose_decoder_final_norm=True,
        expr_decoder_final_norm=False,

        lower_overwrites_upper=True,

        # Compatibility arguments. Ignored by design.
        use_early_full_pose_expr_interaction=False,
        use_latent_interaction=False,
        use_late_full_pose_expr_interaction=False,
        **unused_kwargs,
    ):
        super().__init__()

        assert pose_temporal_type in ["refine", "simple"]
        assert expr_temporal_type in ["refine", "simple"]
        assert pose_num_nodes == 55, "This tri-branch setup assumes 55 SMPL-X pose nodes."

        if len(unused_kwargs) > 0:
            # Keep silently compatible with older configs while making the behavior safe.
            pass

        # -------------------------------------------------
        # Default node split
        # -------------------------------------------------
        if lower_ids is None:
            lower_ids = [
                0,        # root / global_orient
                1, 2,    # hips
                4, 5,    # knees
                7, 8,    # ankles
                10, 11,  # feet
            ]

        if upper_ids is None:
            upper_ids = [
                0,        # root anchor
                3, 6, 9, 12, 15,          # spine / chest / neck / head
                13, 14, 16, 17, 18, 19, 20, 21,  # shoulders / arms / wrists
                22, 23, 24,               # jaw / eyes
                *range(25, 55),           # hands
            ]

        self.pose_num_nodes = pose_num_nodes
        self.pose_node_dim = pose_node_dim
        self.expr_in_dim = expr_in_dim

        self.upper_latent_dim = upper_latent_dim
        self.lower_latent_dim = lower_latent_dim
        self.expr_latent_dim = expr_latent_dim

        self.pose_downsample_factor = pose_downsample_factor
        self.expr_downsample_factor = expr_downsample_factor

        self.pose_temporal_type = pose_temporal_type
        self.expr_temporal_type = expr_temporal_type

        self.lower_overwrites_upper = lower_overwrites_upper

        self.register_buffer(
            "upper_ids",
            torch.tensor(upper_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "lower_ids",
            torch.tensor(lower_ids, dtype=torch.long),
            persistent=False,
        )

        self.upper_num_nodes = len(upper_ids)
        self.lower_num_nodes = len(lower_ids)

        # -------------------------------------------------
        # Upper pose branch
        # -------------------------------------------------
        self.upper_encoder = PoseNodeTemporalEncoder(
            num_nodes=self.upper_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=upper_node_hidden_dim,
            node_latent_dim=upper_node_latent_dim,
            pose_latent_dim=upper_latent_dim,
            downsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=upper_pre_blocks,
            num_post_blocks=upper_post_blocks,
            temporal_type=pose_temporal_type,
        )

        self.upper_decoder = PoseNodeTemporalDecoder(
            latent_dim=upper_latent_dim,
            num_nodes=self.upper_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=upper_decoder_node_hidden_dim,
            node_latent_dim=upper_decoder_node_latent_dim,
            upsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=upper_decoder_pre_blocks,
            num_post_blocks=upper_decoder_post_blocks,
            temporal_type=pose_temporal_type,
            final_norm=pose_decoder_final_norm,
        )

        # -------------------------------------------------
        # Lower pose branch
        # -------------------------------------------------
        self.lower_encoder = PoseNodeTemporalEncoder(
            num_nodes=self.lower_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=lower_node_hidden_dim,
            node_latent_dim=lower_node_latent_dim,
            pose_latent_dim=lower_latent_dim,
            downsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=lower_pre_blocks,
            num_post_blocks=lower_post_blocks,
            temporal_type=pose_temporal_type,
        )

        self.lower_decoder = PoseNodeTemporalDecoder(
            latent_dim=lower_latent_dim,
            num_nodes=self.lower_num_nodes,
            node_dim=pose_node_dim,
            node_hidden_dim=lower_decoder_node_hidden_dim,
            node_latent_dim=lower_decoder_node_latent_dim,
            upsample_factor=pose_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=lower_decoder_pre_blocks,
            num_post_blocks=lower_decoder_post_blocks,
            temporal_type=pose_temporal_type,
            final_norm=pose_decoder_final_norm,
        )

        # -------------------------------------------------
        # Expression branch
        # -------------------------------------------------
        self.expr_encoder = ExpressionTemporalEncoder(
            in_dim=expr_in_dim,
            hidden_dim=expr_hidden_dim,
            latent_dim=expr_latent_dim,
            downsample_factor=expr_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=expr_pre_blocks,
            num_post_blocks=expr_post_blocks,
            temporal_type=expr_temporal_type,
        )

        self.expr_decoder = ExpressionTemporalDecoder(
            latent_dim=expr_latent_dim,
            out_dim=expr_in_dim,
            hidden_dim=expr_decoder_hidden_dim,
            upsample_factor=expr_downsample_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            num_pre_blocks=expr_decoder_pre_blocks,
            num_post_blocks=expr_decoder_post_blocks,
            temporal_type=expr_temporal_type,
            final_norm=expr_decoder_final_norm,
        )

    @staticmethod
    def _match_length(x, target_len):
        """
        Crop or zero-pad temporal dimension to target_len.

        x: [B, T, ...]
        """
        if target_len is None:
            return x

        T = x.shape[1]

        if T == target_len:
            return x

        if T > target_len:
            return x[:, :target_len]

        pad_len = target_len - T
        pad_shape = list(x.shape)
        pad_shape[1] = pad_len
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=1)

    def split_pose_nodes(self, pose_node_feat):
        """
        pose_node_feat: [B, T, 55, D]
        """
        assert pose_node_feat.shape[2] == self.pose_num_nodes
        upper_feat = pose_node_feat[:, :, self.upper_ids, :]
        lower_feat = pose_node_feat[:, :, self.lower_ids, :]
        return upper_feat, lower_feat

    def merge_pose_nodes(self, upper_node_dec, lower_node_dec, target_len=None):
        """
        upper_node_dec: [B, T, J_upper, D]
        lower_node_dec: [B, T, J_lower, D]

        return:
            pose_node_dec: [B, T, 55, D]
        """
        upper_node_dec = self._match_length(upper_node_dec, target_len)
        lower_node_dec = self._match_length(lower_node_dec, target_len)

        B, T, _, D = upper_node_dec.shape

        pose_node_dec = upper_node_dec.new_zeros(
            B,
            T,
            self.pose_num_nodes,
            D,
        )

        if self.lower_overwrites_upper:
            # upper first, then lower overwrites duplicated anchor nodes such as root.
            pose_node_dec[:, :, self.upper_ids, :] = upper_node_dec
            pose_node_dec[:, :, self.lower_ids, :] = lower_node_dec
        else:
            # lower first, then upper overwrites duplicated anchor nodes.
            pose_node_dec[:, :, self.lower_ids, :] = lower_node_dec
            pose_node_dec[:, :, self.upper_ids, :] = upper_node_dec

        return pose_node_dec

    def encode_to_latents(
        self,
        pose_node_feat,
        expr_feat,
        apply_early_interaction=False,
        apply_latent_interaction=False,
        **unused_kwargs,
    ):
        """
        pose_node_feat: [B, T, 55, Dp]
        expr_feat     : [B, T, De]

        Interaction flags are accepted for compatibility but ignored.
        Interaction should be applied in ae.py.
        """
        upper_feat, lower_feat = self.split_pose_nodes(pose_node_feat)

        z_upper = self.upper_encoder(upper_feat)
        z_lower = self.lower_encoder(lower_feat)
        z_expr = self.expr_encoder(expr_feat)

        return {
            "z_upper": z_upper,
            "z_lower": z_lower,
            "z_expr": z_expr,

            # Raw aliases for compatibility/debugging.
            "z_upper_raw": z_upper,
            "z_lower_raw": z_lower,
            "z_expr_raw": z_expr,
        }

    def decode_from_latents(
        self,
        z_upper,
        z_lower,
        z_expr,
        target_pose_len=None,
        target_expr_len=None,
        apply_latent_interaction=False,
        apply_late_interaction=False,
        **unused_kwargs,
    ):
        """
        z_upper: [B, T_latent, upper_latent_dim]
        z_lower: [B, T_latent, lower_latent_dim]
        z_expr : [B, T_latent, expr_latent_dim]

        Interaction flags are accepted for compatibility but ignored.
        Interaction should be applied in ae.py.
        """
        upper_node_dec = self.upper_decoder(z_upper)
        lower_node_dec = self.lower_decoder(z_lower)

        pose_node_dec = self.merge_pose_nodes(
            upper_node_dec=upper_node_dec,
            lower_node_dec=lower_node_dec,
            target_len=target_pose_len,
        )

        expr_temp_dec = self.expr_decoder(z_expr)
        expr_temp_dec = self._match_length(expr_temp_dec, target_expr_len)

        return {
            "upper_node_dec": upper_node_dec,
            "lower_node_dec": lower_node_dec,
            "pose_node_dec": pose_node_dec,
            "expr_temp_dec": expr_temp_dec,
        }

    def forward(self, pose_node_feat, expr_feat):
        """
        pose_node_feat: [B, T, 55, Dp]
        expr_feat     : [B, T, De]
        """
        target_pose_len = pose_node_feat.shape[1]
        target_expr_len = expr_feat.shape[1]

        enc_out = self.encode_to_latents(
            pose_node_feat=pose_node_feat,
            expr_feat=expr_feat,
        )

        dec_out = self.decode_from_latents(
            z_upper=enc_out["z_upper"],
            z_lower=enc_out["z_lower"],
            z_expr=enc_out["z_expr"],
            target_pose_len=target_pose_len,
            target_expr_len=target_expr_len,
        )

        return {
            "z_upper": enc_out["z_upper"],
            "z_lower": enc_out["z_lower"],
            "z_expr": enc_out["z_expr"],

            "z_upper_raw": enc_out["z_upper_raw"],
            "z_lower_raw": enc_out["z_lower_raw"],
            "z_expr_raw": enc_out["z_expr_raw"],

            "upper_node_dec": dec_out["upper_node_dec"],
            "lower_node_dec": dec_out["lower_node_dec"],
            "pose_node_dec": dec_out["pose_node_dec"],
            "expr_temp_dec": dec_out["expr_temp_dec"],
        }


if __name__ == "__main__":
    # Minimal shape test for TriBranchTemporalAE.
    B, T = 2, 240
    pose_node_feat = torch.randn(B, T, 55, 64)
    expr_feat = torch.randn(B, T, 384)

    model = TriBranchTemporalAE(
        pose_num_nodes=55,
        pose_node_dim=64,
        expr_in_dim=384,
        upper_latent_dim=512,
        lower_latent_dim=256,
        expr_latent_dim=384,
        pose_downsample_factor=4,
        expr_downsample_factor=4,
        pose_temporal_type="refine",
        expr_temporal_type="simple",

        # Accepted but ignored.
        use_early_full_pose_expr_interaction=False,
        use_latent_interaction=False,
        use_late_full_pose_expr_interaction=False,
    )

    out = model(pose_node_feat, expr_feat)

    print("z_upper:", out["z_upper"].shape)
    print("z_lower:", out["z_lower"].shape)
    print("z_expr :", out["z_expr"].shape)
    print("pose_node_dec:", out["pose_node_dec"].shape)
    print("expr_temp_dec:", out["expr_temp_dec"].shape)
