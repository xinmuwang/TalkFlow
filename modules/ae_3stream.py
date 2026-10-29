
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose_gcn import HolisticRotGCNEncoder, HolisticRotGCNDecoder
from .expression_mlp import ExpressionFrameEncoder, ExpressionFrameDecoder
from .interaction import AsymmetricPoseExprInteraction, BottleneckPoseExprInteraction
from .tcn_3stream import TriBranchTemporalAE


class PoseExprVAE(nn.Module):
    """
    3-stream AE with interactions managed in ae.py.

    Compared with the old 2-stream version:
        old:
            pose -> z_pose
            expr -> z_expr

        new:
            pose -> z_upper, z_lower
            expr -> z_expr

    Pipeline:
        pose -> full GCN -> full pose node features
             -> optional pre full pose-expression interaction
             -> split upper/lower inside TriBranchTemporalAE
             -> z_upper, z_lower

        expr -> MLP -> expr frame features
             -> optional pre full pose-expression interaction
             -> z_expr

        optional bottleneck interactions:
             z_upper <-> z_lower
             z_upper <-> z_expr
             no z_lower <-> z_expr

        temporal decode:
             z_upper -> upper node features
             z_lower -> lower node features
             merge -> full pose node features
             z_expr  -> expr temporal features

        optional post full pose-expression interaction

        final:
             full pose node features -> GCN decoder -> pose reconstruction
             expr temporal features  -> MLP decoder -> expression reconstruction

    Notes:
        - pose_gcn.py is unchanged.
        - tcn.py only handles 3-stream temporal encode/decode.
        - interaction logic is kept here.
    """

    def __init__(
        self,
        pose_in_dim=6,
        expr_in_dim=100,
        pose_num_nodes=55,

        # frame-level feature dims
        pose_node_dim=32,
        expr_feat_dim=128,

        # spatial / frame encoders
        pose_gcn_hidden_dim=64,
        expr_mlp_hidden_dim=512,
        expr_num_blocks=0,

        # temporal bottleneck dims
        upper_latent_dim=256,
        lower_latent_dim=128,
        expr_latent_dim=256,

        # per-node temporal latent dims
        upper_node_latent_dim=64,
        lower_node_latent_dim=64,

        # temporal hidden dims
        upper_tcn_hidden_dim=64,
        lower_tcn_hidden_dim=64,
        expr_tcn_hidden_dim=512,

        upper_tcn_decoder_hidden_dim=64,
        lower_tcn_decoder_hidden_dim=64,
        expr_tcn_decoder_hidden_dim=512,

        # temporal compression
        pose_downsample_factor=4,
        expr_downsample_factor=4,

        # temporal block counts (encoder)
        upper_pre_blocks=1,
        upper_post_blocks=1,
        lower_pre_blocks=1,
        lower_post_blocks=1,
        expr_pre_blocks=1,
        expr_post_blocks=1,

        # temporal block counts (decoder)
        upper_decoder_pre_blocks=1,
        upper_decoder_post_blocks=1,
        lower_decoder_pre_blocks=1,
        lower_decoder_post_blocks=1,
        expr_decoder_pre_blocks=1,
        expr_decoder_post_blocks=1,

        pose_temporal_type="simple",
        expr_temporal_type="simple",

        # node split
        upper_ids=None,
        lower_ids=None,
        lower_overwrites_upper=True,

        # interaction switches
        use_pre_interaction=False,
        use_upper_lower_bottleneck_interaction=True,
        use_upper_expr_bottleneck_interaction=True,
        use_post_interaction=False,

        # pre interaction config
        pre_pose_interaction_hidden_dim=128,
        pre_expr_interaction_hidden_dim=256,
        pre_use_film_for_pose=True,
        pre_pose_alpha_init=0.05,
        pre_expr_alpha_init=0.05,
        pre_global_alpha_init=0.2,

        # bottleneck upper-lower interaction config
        upper_lower_interaction_hidden_dim=256,
        upper_lower_upper_alpha_init=0.03,
        upper_lower_lower_alpha_init=0.03,

        # bottleneck upper-expression interaction config
        upper_expr_interaction_hidden_dim=256,
        upper_expr_upper_alpha_init=0.05,
        upper_expr_expr_alpha_init=0.05,

        # post interaction config
        post_pose_interaction_hidden_dim=128,
        post_expr_interaction_hidden_dim=256,
        post_use_film_for_pose=True,
        post_pose_alpha_init=0.02,
        post_expr_alpha_init=0.02,
        post_global_alpha_init=0.1,

        kernel_size=3,
        dropout=0.01,
    ):
        super().__init__()

        assert pose_num_nodes == 55, "This 3-stream AE assumes 55 SMPL-X pose nodes."

        self.pose_in_dim = pose_in_dim
        self.expr_in_dim = expr_in_dim
        self.pose_num_nodes = pose_num_nodes

        self.pose_node_dim = pose_node_dim
        self.expr_feat_dim = expr_feat_dim

        self.upper_latent_dim = upper_latent_dim
        self.lower_latent_dim = lower_latent_dim
        self.expr_latent_dim = expr_latent_dim

        self.pose_downsample_factor = pose_downsample_factor
        self.expr_downsample_factor = expr_downsample_factor

        self.use_pre_interaction = use_pre_interaction
        self.use_upper_lower_bottleneck_interaction = use_upper_lower_bottleneck_interaction
        self.use_upper_expr_bottleneck_interaction = use_upper_expr_bottleneck_interaction
        self.use_post_interaction = use_post_interaction

        self.kernel_size = kernel_size
        self.dropout = dropout

        # -------------------------------------------------
        # 1) Pose spatial encoder / decoder
        # -------------------------------------------------
        self.pose_encoder = HolisticRotGCNEncoder(
            num_nodes=pose_num_nodes,
            in_dim=pose_in_dim,
            hidden_dim=pose_gcn_hidden_dim,
            out_dim=pose_node_dim,
            num_layers=3,
            cheb_k=3,
            dropout=dropout,
            pooling="mean",
        )

        self.pose_decoder = HolisticRotGCNDecoder(
            num_nodes=pose_num_nodes,
            latent_dim=pose_node_dim,
            hidden_dim=pose_gcn_hidden_dim,
            out_dim=pose_in_dim,
            num_layers=3,
            cheb_k=3,
            dropout=dropout,
            final_norm=False,
        )

        # -------------------------------------------------
        # 2) Expression frame encoder / decoder
        # -------------------------------------------------
        self.expr_encoder = ExpressionFrameEncoder(
            in_dim=expr_in_dim,
            hidden_dim=expr_mlp_hidden_dim,
            latent_dim=expr_feat_dim,
            num_blocks=expr_num_blocks,
            dropout=dropout,
        )

        self.expr_decoder = ExpressionFrameDecoder(
            latent_dim=expr_feat_dim,
            hidden_dim=expr_mlp_hidden_dim,
            out_dim=expr_in_dim,
            num_blocks=expr_num_blocks,
            dropout=dropout,
        )

        # -------------------------------------------------
        # 3) 3-stream Temporal AE
        # -------------------------------------------------
        # Interaction is intentionally disabled inside TriBranchTemporalAE.
        # All interaction logic is handled in this ae.py file.
        self.temporal_ae = TriBranchTemporalAE(
            pose_num_nodes=pose_num_nodes,
            pose_node_dim=pose_node_dim,
            expr_in_dim=expr_feat_dim,

            upper_ids=upper_ids,
            lower_ids=lower_ids,

            upper_latent_dim=upper_latent_dim,
            lower_latent_dim=lower_latent_dim,
            expr_latent_dim=expr_latent_dim,

            upper_node_hidden_dim=upper_tcn_hidden_dim,
            lower_node_hidden_dim=lower_tcn_hidden_dim,
            upper_node_latent_dim=upper_node_latent_dim,
            lower_node_latent_dim=lower_node_latent_dim,
            expr_hidden_dim=expr_tcn_hidden_dim,

            upper_decoder_node_hidden_dim=upper_tcn_decoder_hidden_dim,
            lower_decoder_node_hidden_dim=lower_tcn_decoder_hidden_dim,
            upper_decoder_node_latent_dim=upper_node_latent_dim,
            lower_decoder_node_latent_dim=lower_node_latent_dim,
            expr_decoder_hidden_dim=expr_tcn_decoder_hidden_dim,

            pose_downsample_factor=pose_downsample_factor,
            expr_downsample_factor=expr_downsample_factor,

            upper_pre_blocks=upper_pre_blocks,
            upper_post_blocks=upper_post_blocks,
            lower_pre_blocks=lower_pre_blocks,
            lower_post_blocks=lower_post_blocks,
            expr_pre_blocks=expr_pre_blocks,
            expr_post_blocks=expr_post_blocks,

            upper_decoder_pre_blocks=upper_decoder_pre_blocks,
            upper_decoder_post_blocks=upper_decoder_post_blocks,
            lower_decoder_pre_blocks=lower_decoder_pre_blocks,
            lower_decoder_post_blocks=lower_decoder_post_blocks,
            expr_decoder_pre_blocks=expr_decoder_pre_blocks,
            expr_decoder_post_blocks=expr_decoder_post_blocks,

            pose_temporal_type=pose_temporal_type,
            expr_temporal_type=expr_temporal_type,

            kernel_size=kernel_size,
            dropout=dropout,

            pose_decoder_final_norm=False,
            expr_decoder_final_norm=False,

            lower_overwrites_upper=lower_overwrites_upper,

            # Important: disable all interactions inside tcn.py.
            use_early_full_pose_expr_interaction=False,
            use_latent_interaction=False,
            use_late_full_pose_expr_interaction=False,
        )

        # -------------------------------------------------
        # 4) Optional pre full pose-expression interaction
        # -------------------------------------------------
        if self.use_pre_interaction:
            self.pre_interaction = AsymmetricPoseExprInteraction(
                pose_dim=pose_node_dim,
                expr_dim=expr_feat_dim,
                pose_hidden_dim=pre_pose_interaction_hidden_dim,
                expr_hidden_dim=pre_expr_interaction_hidden_dim,
                dropout=dropout,
                update_pose_global=True,
                use_film_for_pose=pre_use_film_for_pose,
                pose_alpha_init=pre_pose_alpha_init,
                expr_alpha_init=pre_expr_alpha_init,
                global_alpha_init=pre_global_alpha_init,
                pose_out_norm=False,
                expr_out_norm=True,
            )
        else:
            self.pre_interaction = None

        # -------------------------------------------------
        # 5) Optional bottleneck interactions
        # -------------------------------------------------
        # upper <-> lower
        if self.use_upper_lower_bottleneck_interaction:
            self.upper_lower_interaction = BottleneckPoseExprInteraction(
                pose_dim=upper_latent_dim,
                expr_dim=lower_latent_dim,
                hidden_dim=upper_lower_interaction_hidden_dim,
                dropout=dropout,
                pose_alpha_init=upper_lower_upper_alpha_init,
                expr_alpha_init=upper_lower_lower_alpha_init,
                pose_out_norm=True,
                expr_out_norm=True,
            )
        else:
            self.upper_lower_interaction = None

        # upper <-> expression
        if self.use_upper_expr_bottleneck_interaction:
            self.upper_expr_interaction = BottleneckPoseExprInteraction(
                pose_dim=upper_latent_dim,
                expr_dim=expr_latent_dim,
                hidden_dim=upper_expr_interaction_hidden_dim,
                dropout=dropout,
                pose_alpha_init=upper_expr_upper_alpha_init,
                expr_alpha_init=upper_expr_expr_alpha_init,
                pose_out_norm=True,
                expr_out_norm=True,
            )
        else:
            self.upper_expr_interaction = None

        # Important:
        # no lower-expression bottleneck interaction is defined.

        # -------------------------------------------------
        # 6) Optional post full pose-expression interaction
        # -------------------------------------------------
        if self.use_post_interaction:
            self.post_interaction = AsymmetricPoseExprInteraction(
                pose_dim=pose_node_dim,
                expr_dim=expr_feat_dim,
                pose_hidden_dim=post_pose_interaction_hidden_dim,
                expr_hidden_dim=post_expr_interaction_hidden_dim,
                dropout=dropout,
                update_pose_global=True,
                use_film_for_pose=post_use_film_for_pose,
                pose_alpha_init=post_pose_alpha_init,
                expr_alpha_init=post_expr_alpha_init,
                global_alpha_init=post_global_alpha_init,
                pose_out_norm=False,
                expr_out_norm=True,
            )
        else:
            self.post_interaction = None

    def apply_bottleneck_interaction(self, z_upper, z_lower, z_expr):
        """
        Apply bottleneck interactions:
            1) upper <-> lower
            2) upper <-> expression
            3) no lower <-> expression

        Order:
            upper-lower first, then upper-expression.

        This order reduces direct expression leakage into lower body.
        """
        if self.upper_lower_interaction is not None:
            z_upper, z_lower = self.upper_lower_interaction(z_upper, z_lower)

        if self.upper_expr_interaction is not None:
            z_upper, z_expr = self.upper_expr_interaction(z_upper, z_expr)

        return z_upper, z_lower, z_expr

    def encode_to_latents(self, pose, expr):
        """
        Args:
            pose: [B, T, 55, pose_in_dim]
            expr: [B, T, expr_in_dim]
        """
        pose_node_feat, pose_global_feat = self.pose_encoder(pose)
        expr_feat = self.expr_encoder(expr)

        # Pre full pose-expression interaction.
        # Here pose is still full pose node feature, so this is global coordination.
        if self.pre_interaction is not None:
            pose_node_feat, pose_global_feat, expr_feat = self.pre_interaction(
                pose_node_feat,
                pose_global_feat,
                expr_feat,
            )

        temporal_out = self.temporal_ae.encode_to_latents(
            pose_node_feat=pose_node_feat,
            expr_feat=expr_feat,

            # Make sure no interaction is applied inside tcn.py.
            apply_early_interaction=False,
            apply_latent_interaction=False,
        )

        z_upper = temporal_out["z_upper"]
        z_lower = temporal_out["z_lower"]
        z_expr = temporal_out["z_expr"]

        z_upper_raw = z_upper
        z_lower_raw = z_lower
        z_expr_raw = z_expr

        # Bottleneck interactions managed here.
        z_upper, z_lower, z_expr = self.apply_bottleneck_interaction(
            z_upper=z_upper,
            z_lower=z_lower,
            z_expr=z_expr,
        )

        return {
            "pose_node_feat": pose_node_feat,
            "pose_global_feat": pose_global_feat,
            "expr_feat": expr_feat,

            "z_upper": z_upper,
            "z_lower": z_lower,
            "z_expr": z_expr,

            "z_upper_raw": z_upper_raw,
            "z_lower_raw": z_lower_raw,
            "z_expr_raw": z_expr_raw,
        }

    def decode_from_latents(
        self,
        z_upper,
        z_lower,
        z_expr,
        target_pose_len=None,
        target_expr_len=None,
        apply_bottleneck_interaction=False,
    ):
        """
        Args:
            z_upper: [B, T_latent_pose, upper_latent_dim]
            z_lower: [B, T_latent_pose, lower_latent_dim]
            z_expr : [B, T_latent_expr, expr_latent_dim]

        apply_bottleneck_interaction:
            Usually False because encode_to_latents already applies interaction.
            It can be useful if external latents are fed directly during inference.
        """
        if apply_bottleneck_interaction:
            z_upper, z_lower, z_expr = self.apply_bottleneck_interaction(
                z_upper=z_upper,
                z_lower=z_lower,
                z_expr=z_expr,
            )

        temporal_dec = self.temporal_ae.decode_from_latents(
            z_upper=z_upper,
            z_lower=z_lower,
            z_expr=z_expr,
            target_pose_len=target_pose_len,
            target_expr_len=target_expr_len,

            # Make sure no interaction is applied inside tcn.py.
            apply_latent_interaction=False,
            apply_late_interaction=False,
        )

        pose_node_dec = temporal_dec["pose_node_dec"]   # [B, T, 55, pose_node_dim]
        expr_temp_dec = temporal_dec["expr_temp_dec"]   # [B, T, expr_feat_dim]

        pose_global_dec = pose_node_dec.mean(dim=2)

        # Post full pose-expression interaction.
        # This is optional and should usually be weaker than bottleneck interaction.
        if self.post_interaction is not None:
            pose_node_dec, pose_global_dec, expr_temp_dec = self.post_interaction(
                pose_node_dec,
                pose_global_dec,
                expr_temp_dec,
            )

        pose_recon = self.pose_decoder(pose_node_dec)
        expr_recon = self.expr_decoder(expr_temp_dec)

        return {
            "pose_recon": pose_recon,
            "expr_recon": expr_recon,

            "pose_node_dec": pose_node_dec,
            "pose_global_dec": pose_global_dec,
            "expr_temp_dec": expr_temp_dec,

            "upper_node_dec": temporal_dec.get("upper_node_dec", None),
            "lower_node_dec": temporal_dec.get("lower_node_dec", None),
        }

    def forward(self, pose, expr):
        target_pose_len = pose.shape[1]
        target_expr_len = expr.shape[1]

        enc_out = self.encode_to_latents(pose, expr)

        dec_out = self.decode_from_latents(
            z_upper=enc_out["z_upper"],
            z_lower=enc_out["z_lower"],
            z_expr=enc_out["z_expr"],
            target_pose_len=target_pose_len,
            target_expr_len=target_expr_len,
            apply_bottleneck_interaction=False,
        )

        return {
            "pose_recon": dec_out["pose_recon"],
            "expr_recon": dec_out["expr_recon"],

            "pose_node_feat": enc_out["pose_node_feat"],
            "pose_global_feat": enc_out["pose_global_feat"],
            "expr_feat": enc_out["expr_feat"],

            "z_upper": enc_out["z_upper"],
            "z_lower": enc_out["z_lower"],
            "z_expr": enc_out["z_expr"],

            "z_upper_raw": enc_out["z_upper_raw"],
            "z_lower_raw": enc_out["z_lower_raw"],
            "z_expr_raw": enc_out["z_expr_raw"],

            "pose_node_dec": dec_out["pose_node_dec"],
            "pose_global_dec": dec_out["pose_global_dec"],
            "expr_temp_dec": dec_out["expr_temp_dec"],

            "upper_node_dec": dec_out["upper_node_dec"],
            "lower_node_dec": dec_out["lower_node_dec"],
        }


if __name__ == "__main__":
    # Minimal usage example.
    B, T = 2, 240
    pose = torch.randn(B, T, 55, 6)
    expr = torch.randn(B, T, 100)

    model = PoseExprVAE(
        pose_node_dim=32,
        expr_feat_dim=128,

        upper_latent_dim=256,
        lower_latent_dim=128,
        expr_latent_dim=256,

        pose_temporal_type="simple",
        expr_temporal_type="simple",

        use_pre_interaction=False,
        use_upper_lower_bottleneck_interaction=True,
        use_upper_expr_bottleneck_interaction=True,
        use_post_interaction=False,
    )

    out = model(pose, expr)
    print("pose_recon:", out["pose_recon"].shape)
    print("expr_recon:", out["expr_recon"].shape)
    print("z_upper:", out["z_upper"].shape)
    print("z_lower:", out["z_lower"].shape)
    print("z_expr:", out["z_expr"].shape)
