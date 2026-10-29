import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """
    Lightweight channel reweighting for [..., D]
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class EnhancedGatedMLPBlock(nn.Module):
    """
    Stronger residual block for expression modeling.

    Structure:
        x
        -> LayerNorm
        -> gated MLP
        -> optional channel gate
        -> residual add

        then
        -> LayerNorm
        -> FFN refine
        -> residual add
    """
    def __init__(
        self,
        dim,
        hidden_dim,
        ffn_hidden_dim=None,
        dropout=0.1,
        use_channel_gate=True,
    ):
        super().__init__()
        ffn_hidden_dim = ffn_hidden_dim or hidden_dim
        self.use_channel_gate = use_channel_gate

        # gated branch
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout1 = nn.Dropout(dropout)

        if use_channel_gate:
            self.channel_gate = ChannelGate(dim)

        # refine branch
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # branch 1: gated mlp
        h = self.norm1(x)
        h = self.fc1(h)
        v, g = h.chunk(2, dim=-1)
        h = v * torch.sigmoid(g)
        h = self.fc2(h)
        h = self.dropout1(h)

        if self.use_channel_gate:
            h = self.channel_gate(h)

        x = x + h

        # branch 2: ffn refine
        x = x + self.ffn(self.norm2(x))
        return x


class ExpressionFrameEncoder(nn.Module):
    """
    expr: [B,T,in_dim] -> [B,T,latent_dim]
    """
    def __init__(
        self,
        in_dim=100,
        hidden_dim=512,
        latent_dim=128,
        num_blocks=3,
        dropout=0.1,
        use_channel_gate=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            EnhancedGatedMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=hidden_dim * 2,
                dropout=dropout,
                use_channel_gate=use_channel_gate,
            )
            for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)

        h = self.in_proj(x)
        h = self.in_norm(h)
        h = F.gelu(h)

        for blk in self.blocks:
            h = blk(h)

        z = self.out_proj(h)
        z = self.out_norm(z)
        z = z.reshape(B, T, self.latent_dim)
        return z


class ExpressionFrameDecoder(nn.Module):
    """
    z: [B,T,latent_dim] -> [B,T,out_dim]
    """
    def __init__(
        self,
        latent_dim=128,
        hidden_dim=512,
        out_dim=100,
        num_blocks=3,
        dropout=0.1,
        use_channel_gate=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            EnhancedGatedMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=hidden_dim * 2,
                dropout=dropout,
                use_channel_gate=use_channel_gate,
            )
            for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        B, T, C = z.shape
        z = z.reshape(B * T, C)

        h = self.in_proj(z)
        h = self.in_norm(h)
        h = F.gelu(h)

        for blk in self.blocks:
            h = blk(h)

        x = self.out_proj(h)
        x = x.reshape(B, T, self.out_dim)
        return x