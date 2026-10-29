import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import ChebConv
from torch_geometric.utils import add_self_loops


# =========================================================
# 1. Graph topology
# =========================================================
def build_holistic_edge_index(device=None):
    """
    Build a 55-node holistic graph for SMPL-X style pose.

    Node layout:
      0   : root (global_orient)
      1-21: body joints
      22  : jaw
      23  : left_eye
      24  : right_eye
      25-39: left hand (15)
      40-54: right hand (15)
    """

    # -----------------------------
    # Kinematic tree edges
    # -----------------------------
    base_edges = [
        # body trunk
        (0, 1), (0, 2), (0, 3),
        (1, 4), (4, 7), (7, 10),
        (2, 5), (5, 8), (8, 11),
        (3, 6), (6, 9), (9, 12),

        # upper body
        (12, 13), (12, 14), (12, 15),
        (13, 16), (16, 18), (18, 20),
        (14, 17), (17, 19), (19, 21),

        # head / face
        (15, 22), (15, 23), (15, 24),

        # left hand
        (20, 25), (25, 26), (26, 27),
        (20, 28), (28, 29), (29, 30),
        (20, 31), (31, 32), (32, 33),
        (20, 34), (34, 35), (35, 36),
        (20, 37), (37, 38), (38, 39),

        # right hand
        (21, 40), (40, 41), (41, 42),
        (21, 43), (43, 44), (44, 45),
        (21, 46), (46, 47), (47, 48),
        (21, 49), (49, 50), (50, 51),
        (21, 52), (52, 53), (53, 54),
    ]

    # -----------------------------
    # Body symmetry edges
    # -----------------------------
    sym_body_edges = [
        (1, 2),    # hips
        (4, 5),    # knees
        (7, 8),    # ankles
        (10, 11),  # feet
        (13, 14),  # collars
        (16, 17),  # shoulders
        (18, 19),  # elbows
        (20, 21),  # wrists
        (23, 24),  # eyes
    ]

    # -----------------------------
    # Hand same-level edges
    # left hand:
    #   index:  25,26,27
    #   middle: 28,29,30
    #   pinky:  31,32,33
    #   ring:   34,35,36
    #   thumb:  37,38,39
    # -----------------------------
    left_hand_same_level_edges = [
        (37, 25), (25, 28), (28, 34), (34, 31),  # level 1
        (38, 26), (26, 29), (29, 35), (35, 32),  # level 2
        (39, 27), (27, 30), (30, 36), (36, 33),  # level 3
    ]

    right_hand_same_level_edges = [
        (52, 40), (40, 43), (43, 49), (49, 46),  # level 1
        (53, 41), (41, 44), (44, 50), (50, 47),  # level 2
        (54, 42), (42, 45), (45, 51), (51, 48),  # level 3
    ]

    edges = (
        base_edges
        + sym_body_edges
        + left_hand_same_level_edges
        + right_hand_same_level_edges
    )

    # make graph undirected
    edges = edges + [(j, i) for (i, j) in edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    edge_index, _ = add_self_loops(edge_index, num_nodes=55)

    if device is not None:
        edge_index = edge_index.to(device)

    return edge_index


# =========================================================
# 2. Utility: batched edge index
# =========================================================
def build_batched_edge_index(edge_index, batch_size, num_nodes, device):
    """
    edge_index: [2, E] for one graph
    return:     [2, batch_size * E]
    """
    edge_index = edge_index.to(device)
    E = edge_index.shape[1]

    offsets = torch.arange(batch_size, device=device) * num_nodes   # [batch_size]
    offsets = offsets.view(batch_size, 1, 1)                        # [batch_size,1,1]

    edge_index = edge_index.view(1, 2, E) + offsets                 # [batch_size,2,E]
    edge_index = edge_index.permute(1, 0, 2).reshape(2, batch_size * E)
    return edge_index


# =========================================================
# 3. Residual GCN block
# =========================================================
class ResidualGCNBlock(nn.Module):
    """
    Residual ChebConv block.

    Input : [N, C_in]
    Output: [N, C_out]
    """
    def __init__(self, in_channels, out_channels, K=3, dropout=0.1):
        super().__init__()

        self.conv1 = ChebConv(in_channels, out_channels, K=K)
        self.norm1 = nn.LayerNorm(out_channels)

        self.conv2 = ChebConv(out_channels, out_channels, K=K)
        self.norm2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x, edge_index):
        identity = self.res_proj(x)

        out = self.conv1(x, edge_index)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out, edge_index)
        out = self.norm2(out)
        out = self.dropout(out)

        out = out + identity
        out = F.gelu(out)
        return out


# =========================================================
# 4. Encoder
# =========================================================
class HolisticRotGCNEncoder(nn.Module):
    """
    Spatial GCN encoder for pose rotations.

    Input:
        x: [B, T, 55, 6]

    Output:
        node_feat  : [B, T, 55, out_dim]
        global_feat: [B, T, out_dim]
    """
    def __init__(
        self,
        num_nodes=55,
        in_dim=6,
        hidden_dim=128,
        out_dim=128,
        num_layers=3,
        cheb_k=3,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        assert num_nodes == 55, "This implementation assumes 55 nodes."

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.pooling = pooling

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        blocks = []
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else out_dim
            out_ch = out_dim
            blocks.append(
                ResidualGCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    K=cheb_k,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.LayerNorm(out_dim)

        edge_index = build_holistic_edge_index()
        self.register_buffer("edge_index", edge_index, persistent=False)

    def forward(self, x):
        """
        x: [B, T, 55, 6]
        """
        B, T, J, C = x.shape
        assert J == self.num_nodes, f"Expected {self.num_nodes} nodes, got {J}"
        assert C == self.in_dim, f"Expected input dim {self.in_dim}, got {C}"

        # input projection
        x = self.input_proj(x)            # [B,T,55,H]
        x = self.input_norm(x)
        x = F.gelu(x)

        # flatten to graph batch
        x = x.reshape(B * T, J, self.hidden_dim)     # [B*T,55,H]
        x = x.reshape(B * T * J, self.hidden_dim)    # [B*T*55,H]

        edge_index = build_batched_edge_index(
            edge_index=self.edge_index,
            batch_size=B * T,
            num_nodes=J,
            device=x.device,
        )

        for block in self.blocks:
            x = block(x, edge_index)

        x = self.out_norm(x)
        x = x.reshape(B, T, J, self.out_dim)         # node_feat

        if self.pooling == "mean":
            global_feat = x.mean(dim=2)              # [B,T,D]
        elif self.pooling == "max":
            global_feat = x.max(dim=2).values
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return x, global_feat


# =========================================================
# 5. Decoder
# =========================================================
class HolisticRotGCNDecoder(nn.Module):
    """
    Spatial GCN decoder.

    Input:
        z_node: [B, T, 55, latent_dim]

    Output:
        rot_recon: [B, T, 55, 6]
    """
    def __init__(
        self,
        num_nodes=55,
        latent_dim=128,
        hidden_dim=128,
        out_dim=6,
        num_layers=3,
        cheb_k=3,
        dropout=0.1,
        final_norm=False,
    ):
        super().__init__()
        assert num_nodes == 55, "This implementation assumes 55 nodes."

        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.final_norm = final_norm

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualGCNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                K=cheb_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        if self.final_norm:
            self.out_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, out_dim)

        edge_index = build_holistic_edge_index()
        self.register_buffer("edge_index", edge_index, persistent=False)

    def forward(self, z_node):
        """
        z_node: [B, T, 55, latent_dim]
        """
        B, T, J, C = z_node.shape
        assert J == self.num_nodes, f"Expected {self.num_nodes} nodes, got {J}"
        assert C == self.latent_dim, f"Expected latent dim {self.latent_dim}, got {C}"

        x = self.input_proj(z_node)       # [B,T,55,H]
        x = self.input_norm(x)
        x = F.gelu(x)

        x = x.reshape(B * T, J, self.hidden_dim)
        x = x.reshape(B * T * J, self.hidden_dim)

        edge_index = build_batched_edge_index(
            edge_index=self.edge_index,
            batch_size=B * T,
            num_nodes=J,
            device=x.device,
        )

        for block in self.blocks:
            x = block(x, edge_index)

        x = x.reshape(B, T, J, self.hidden_dim)

        if self.final_norm:
            x = self.out_norm(x)

        rot = self.output_proj(x)         # [B,T,55,6]
        return rot


# =========================================================
# 6. Optional wrapper: Pose spatial autoencoder
# =========================================================
class PoseSpatialAutoEncoder(nn.Module):
    """
    A simple wrapper for:
        pose -> GCN encoder -> node_feat -> GCN decoder -> pose_recon
    """
    def __init__(
        self,
        num_nodes=55,
        in_dim=6,
        hidden_dim=128,
        latent_dim=128,
        num_layers=3,
        cheb_k=3,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()

        self.encoder = HolisticRotGCNEncoder(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            num_layers=num_layers,
            cheb_k=cheb_k,
            dropout=dropout,
            pooling=pooling,
        )

        self.decoder = HolisticRotGCNDecoder(
            num_nodes=num_nodes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=in_dim,
            num_layers=num_layers,
            cheb_k=cheb_k,
            dropout=dropout,
            final_norm=False,
        )

    def forward(self, pose):
        """
        pose: [B,T,55,6]
        """
        node_feat, global_feat = self.encoder(pose)
        pose_recon = self.decoder(node_feat)
        return {
            "node_feat": node_feat,       # [B,T,55,D]
            "global_feat": global_feat,   # [B,T,D]
            "pose_recon": pose_recon,     # [B,T,55,6]
        }
