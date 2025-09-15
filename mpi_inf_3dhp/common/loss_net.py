from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: mpjpe는 코드베이스에서 common.loss에 정의되어 있다고 가정
from common.loss import mpjpe


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def adj_from_parents(parents: list[int]) -> torch.Tensor:
    """
    Build an undirected adjacency matrix (J,J) from a parents list of length J.
    parent = -1 for root.
    """
    J = len(parents)
    A = torch.zeros(J, J, dtype=torch.float32)
    for j, p in enumerate(parents):
        if p >= 0:
            A[j, p] = 1.0
            A[p, j] = 1.0
    return A


def floyd_warshall(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute shortest-path hop distances for an unweighted graph.
    Args:
        adj: (N, N) 0/1 adjacency
    Returns:
        dist: (N, N) hop count (0 on diag), large if disconnected
    """
    assert adj.ndim == 2 and adj.size(0) == adj.size(1), "adj must be (N,N)"
    N = adj.size(0)
    device = adj.device
    INF = 1e6
    dist = torch.full((N, N), INF, device=device, dtype=torch.float32)
    dist[adj > 0] = 1.0
    idx = torch.arange(N, device=device)
    dist[idx, idx] = 0.0
    for k in range(N):
        dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    return dist


# -----------------------------------------------------------------------------
# Attention Bias Builder
# -----------------------------------------------------------------------------
class AttnBiasBuilder(nn.Module):
    """
    Build (B, H, N+1, N+1) attention bias from graph signals.

    Signals supported:
      - adj: adjacency (required)
      - edge_type_oh: (N,N,K) one-hot edge type (optional)
      - degree bias: (deg(i)+deg(j))/max_deg * learnable scale (optional)
      - shortest-path distance: -gamma * dist[i,j] (optional)
      - soft or hard masking for non-edges
      - VNode (CLS) special handling (replace/add/off)

    Learnable params:
      noedge_bias, edge_type_weight, deg_scale, gamma,
      vnode_to_node, node_to_vnode, vnode_self
    """
    def __init__(
        self,
        adj: torch.Tensor,
        edge_type_oh: Optional[torch.Tensor] = None,
        *,
        hard_mask: bool = False,
        use_degree: bool = True,
        learn_cls: bool = True,
        vnode_spd_mode: str = "replace",  # "replace" | "add" | "off"
    ):
        super().__init__()
        assert adj.ndim == 2 and adj.size(0) == adj.size(1), "adj must be (N,N)"
        N = adj.size(0)
        self.N = N
        self.hard_mask = hard_mask
        self.use_degree = use_degree
        self.vnode_spd_mode = vnode_spd_mode

        # Fixed buffers
        self.register_buffer("adj", (adj > 0).float())  # (N,N)
        if edge_type_oh is not None:
            assert edge_type_oh.shape[:2] == (N, N)
            self.register_buffer("edge_type_oh", edge_type_oh.float())  # (N,N,K)
            self.K = edge_type_oh.size(-1)
        else:
            self.edge_type_oh = None
            self.K = 0
        if use_degree:
            deg = self.adj.sum(-1)  # (N,)
            self.register_buffer("deg", deg)
        self.register_buffer("dist", None)  # set by set_distance()

        # Learnable scalars/vectors
        if self.K > 0:
            self.edge_type_weight = nn.Parameter(torch.zeros(self.K))  # (K,)
        else:
            self.edge_type_weight = nn.Parameter(torch.zeros(1))       # (1,)
        self.noedge_bias = nn.Parameter(torch.tensor(-2.0))           # soft bias
        if use_degree:
            self.deg_scale = nn.Parameter(torch.tensor(0.2))
        self.gamma = nn.Parameter(torch.tensor(0.0))  # distance scale (0=off)

        # VNode (CLS) special couplings
        if learn_cls:
            self.vnode_to_node = nn.Parameter(torch.zeros(1))   # CLS->node
            self.node_to_vnode = nn.Parameter(torch.zeros(1))   # node->CLS
            self.vnode_self    = nn.Parameter(torch.zeros(1))   # CLS->CLS
        else:
            self.register_parameter("vnode_to_node", None)
            self.register_parameter("node_to_vnode", None)
            self.register_parameter("vnode_self",    None)

    @torch.no_grad()
    def set_distance(self, dist: torch.Tensor) -> None:
        assert dist.shape == (self.N, self.N)
        self.register_buffer("dist", dist.float())

    def forward(self, B: int, H: int = 1) -> torch.Tensor:
        """
        Returns:
            attn_bias: (B, H, N+1, N+1)
        """
        N = self.N
        T = N + 1
        device = self.adj.device
        bias = torch.zeros(B, H, T, T, device=device, dtype=torch.float32)

        # ---- Node-Node block (1..N, 1..N) ----
        # (a) Edge-type term
        if self.edge_type_oh is not None:
            e = (self.edge_type_oh * self.edge_type_weight.view(1, 1, -1)).sum(-1)  # (N,N)
        else:
            e = torch.zeros_like(self.adj)
        # (b) Distance term
        if (self.dist is not None) and (float(self.gamma.item()) != 0.0):
            e = e + (-self.gamma) * self.dist
        # (c) Neighbor / non-neighbor handling
        if self.hard_mask:
            base = torch.full_like(self.adj, float('-inf'))
            e = torch.where(self.adj > 0, e, base)
        else:
            e = torch.where(self.adj > 0, e, e + self.noedge_bias)
        # (d) Degree bias
        if self.use_degree:
            deg_bias = (self.deg.view(N, 1) + self.deg.view(1, N)) / (self.deg.max() + 1e-6)
            e = e + self.deg_scale * deg_bias
        bias[:, :, 1:, 1:] = e.view(1, 1, N, N).expand(B, H, N, N)

        # ---- VNode (CLS) couplings ----
        if self.vnode_to_node is not None:
            # Base SPD term for CLS pairs (treat as dist=1)
            if (self.dist is not None) and (float(self.gamma.item()) != 0.0):
                v_spd = (-self.gamma).detach()  # scalar (-gamma*1)
            else:
                v_spd = torch.tensor(0.0, device=device)
            if self.vnode_spd_mode == "replace":
                bias[:, :, 0, 1:] = self.vnode_to_node
                bias[:, :, 1:, 0] = self.node_to_vnode
            elif self.vnode_spd_mode == "add":
                bias[:, :, 0, 1:] = v_spd + self.vnode_to_node
                bias[:, :, 1:, 0] = v_spd + self.node_to_vnode
            else:  # "off"
                pass
        if self.vnode_self is not None:
            if self.vnode_spd_mode == "add" and (self.dist is not None) and (float(self.gamma.item()) != 0.0):
                bias[:, :, 0, 0] = (-self.gamma).detach() + self.vnode_self
            elif self.vnode_spd_mode == "replace":
                bias[:, :, 0, 0] = self.vnode_self

        return bias


# -----------------------------------------------------------------------------
# Biased Multi-Head Self-Attention + Transformer Block
# -----------------------------------------------------------------------------
class BiasedMHSA(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dk = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D), attn_bias: (B, H, T, T) or None
        """
        B, T, D = x.shape
        H = self.h
        dk = self.dk
        q = self.q(x).view(B, T, H, dk).transpose(1, 2)  # (B,H,T,dk)
        k = self.k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.v(x).view(B, T, H, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # (B,H,T,T)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                   # (B,H,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BiasedMHSA(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# -----------------------------------------------------------------------------
# Continuous Graph LossNet (end-to-end)
# -----------------------------------------------------------------------------
class ContinuousGraphLossNet(nn.Module):
    """
    Graph-biased Transformer LossNet.

    Inputs
        x2d: (B, N, 2) or (B, T, N, 2)   # normalized to [-1,1]
        y3d: (B, N, 3) or (B, T, N, 3)   # root-centered / scale-normalized
    Returns
        energy: (B, 1)  scalar energy per item (lower is better)

    Notes
      * If inputs are 4D (B,T,N,C), T is flattened into the batch: (B*T,N,C)
        → per-frame energy, aggregated later by the training loop.
    """
    def __init__(
        self,
        adj: torch.Tensor,
        num_joints: int,
        *,
        edge_type_oh: Optional[torch.Tensor] = None,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        joint_emb_dim: int = 32,
        hard_mask: bool = False,
        use_degree_bias: bool = True,
        use_distance_bias: bool = True,
        distance_gamma_init: float = 0.8,
        vnode_spd_mode: str = "replace",
    ):
        super().__init__()
        self.N = num_joints
        self.d = d_model

        # Joint ID embedding
        self.joint_emb = nn.Embedding(num_joints, joint_emb_dim)

        # Node encoder: [x2d(2), y3d(3), joint_emb] -> d_model
        self.in_proj = nn.Linear(2 + 3 + joint_emb_dim, d_model)

        # Graph token (CLS)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Bias builder
        self.bias_builder = AttnBiasBuilder(
            adj=adj, edge_type_oh=edge_type_oh,
            hard_mask=hard_mask, use_degree=use_degree_bias, learn_cls=True,
            vnode_spd_mode=vnode_spd_mode,
        )
        if use_distance_bias:
            dist = floyd_warshall(adj.to(self.cls.device)) if adj.device != self.cls.device else floyd_warshall(adj)
            self.bias_builder.set_distance(dist)
            with torch.no_grad():
                self.bias_builder.gamma.copy_(torch.tensor(distance_gamma_init))

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Energy head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _ensure_3d(self, x2d: torch.Tensor, y3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert (B,T,N,C) to (B*T,N,C) if needed.
        """
        if x2d.dim() == 4:
            B, T, N, C = x2d.shape
            assert y3d.dim() == 4 and y3d.shape[:3] == (B, T, N), "x2d/y3d shape mismatch"
            x2d = x2d.reshape(B * T, N, C)
            y3d = y3d.reshape(B * T, N, -1)
        return x2d, y3d

    def forward(self, x2d: torch.Tensor, y3d: torch.Tensor) -> torch.Tensor:
        x2d, y3d = self._ensure_3d(x2d, y3d)  # (B,N,2/3)
        B, N, _ = x2d.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        device = x2d.device
        jid = torch.arange(N, device=device).view(1, N).expand(B, N)
        h0 = torch.cat([x2d, y3d, self.joint_emb(jid)], dim=-1)  # (B,N,2+3+E)
        z = self.in_proj(h0)                                     # (B,N,D)

        # prepend CLS
        cls = self.cls.to(device).expand(B, 1, -1)               # (B,1,D)
        z = torch.cat([cls, z], dim=1)                           # (B,N+1,D)

        # Build bias per batch & head
        attn_bias = self.bias_builder(B=B, H=self.blocks[0].attn.h).to(device)  # (B,H,N+1,N+1)

        # Pass through blocks
        for blk in self.blocks:
            z = blk(z, attn_bias=attn_bias)
        z = self.ln_f(z)

        # CLS → energy
        g = z[:, 0, :]                                           # (B,D)
        energy = self.head(g).squeeze(-1)                        # (B,)

        # Return shape (B,1) for compatibility with training loop
        return energy.view(-1, 1)


# -----------------------------------------------------------------------------
# Losses for SEAL (kept here for convenience)
# -----------------------------------------------------------------------------
class MarginBasedLoss:
    """
    Margin-based energy learning. Encourages E(x,y) + margin <= E(x, y_hat).
    margin ~ ratio * delta(y_hat, y) where delta=mpjpe|mse|l1
    """
    def __init__(self, margin_ratio: float = 1.0, loss_type: str = 'mse'):
        self.margin_ratio = margin_ratio
        self.loss_type = loss_type

    def __call__(self, y_hat: torch.Tensor, label: torch.Tensor,
                 energy_hat: torch.Tensor, energy_label: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            delta = F.mse_loss(y_hat, label, reduction='none').mean(dim=(-2, -1))
        elif self.loss_type == 'mpjpe':
            # mpjpe returns (B,) or (B*T,) – ensure shape (B,1) by unsqueeze
            delta = mpjpe(y_hat, label).view(-1, 1)
        elif self.loss_type == 'l1':
            delta = F.l1_loss(y_hat, label, reduction='none').mean(dim=(-2, -1))
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Broadcast-safe: energies are (B,1)
        zeros = torch.zeros_like(delta)
        if self.margin_ratio < 0:
            # pure ranking (no margin): minimize -E_hat + E_label
            margin_loss = -energy_hat + energy_label
        else:
            margin_loss = torch.max(self.margin_ratio * delta - energy_hat + energy_label, zeros)

        return margin_loss.mean()


class NCELoss:
    """
    Binary NCE on energies: y (pos) vs y_hat (neg).
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, y_hat: torch.Tensor, label: torch.Tensor,
                 energy_hat: torch.Tensor, energy_label: torch.Tensor) -> torch.Tensor:
        # Stabilize with max-trick
        max_e = torch.max(energy_hat, energy_label)
        num = torch.exp(torch.clamp((max_e - energy_label) / self.temperature, max=50))
        den = torch.exp(torch.clamp((max_e - energy_hat) / self.temperature,  max=50)) + num
        nce = -torch.log(1e-46 + num / den)
        return nce.mean()


# Backwards compatibility alias
ModelLoss = ContinuousGraphLossNet