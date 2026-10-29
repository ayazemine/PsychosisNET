"""PsychosisNET: a small connectivity-aware transformer for ROI-level fMRI.

The correlation matrix of a subject passes a learnable tangent-space layer; every brain region is
then one token whose feature vector is its row of the tangent matrix. Self-attention between
regions is biased by that matrix, a learned query pools the regions into one subject vector, and a
linear wide path on the connections is added to the logits.

The network is deliberately narrow (about 10^5 parameters) and uses GELU activations. Its weights
are kept small by AdamW weight decay (psikoz.train).

Components of earlier versions that did not improve the validation AUC (multi-view input, site
adversary, differential attention, temporal CNN branch, BrainNetCNN edge-to-edge filter, BNT
clustering readout, low-rank wide path) were removed; the earlier code is kept in archive/v10.
"""
import math

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: skips a whole residual branch for a random subset of samples."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty(x.shape[0], 1, 1).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class ConnectivityAttention(nn.Module):
    """Multi-head self-attention whose logits get a per-head connectivity prior.

    logits = q.k / sqrt(d) + gain_h * FC + offset_h * |FC|
    so each head can learn to attend along positive coupling, anticorrelation or
    connection strength, or to ignore the prior (gain = 0 at initialisation).
    """

    def __init__(self, dim, heads, attn_drop, proj_drop):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_gain = nn.Parameter(torch.zeros(heads, 2))
        self.attn_drop, self.proj_drop = nn.Dropout(attn_drop), nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, fc, key_mask=None):
        b, n, d = x.shape
        q, k, v = (t.view(b, n, self.heads, d // self.heads).transpose(1, 2) for t in self.qkv(x).view(b, n, 3, d).unbind(2))
        logits = (q @ k.transpose(-2, -1)) * self.scale
        prior = torch.stack([fc, fc.abs()], dim=-1)                       # (b, n, n, 2)
        logits = logits + torch.einsum("bijc,hc->bhij", prior, self.fc_gain)
        if key_mask is not None:                                          # dropped regions cannot be attended to
            logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = logits.softmax(-1)
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out)), attn


class SPDTangent(nn.Module):
    """Learnable tangent-space layer for correlation matrices (BiMap + ReEig + LogEig, after
    SPDNet, Huang & Van Gool 2017).

        C_s = (1 - a) C + a I                       learnable shrinkage, keeps C_s positive definite
        Y   = W' C_s W,   W = W0 (I + D)            congruence with a learnable matrix
        out = log(max(Y, eps))                      matrix logarithm with eigenvalue floor

    W0 is the inverse square root of the mean training matrix and D starts at zero, so the layer
    starts as the classical tangent-space embedding and can then move the reference point.
    """

    def __init__(self, n, shrink=0.3, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.shrink = nn.Parameter(torch.tensor(math.log(shrink / (1 - shrink))))
        self.delta = nn.Parameter(torch.zeros(n, n))
        self.register_buffer("w0", torch.eye(n, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(()))

    def _shrunk(self, c):
        a = self.shrink.sigmoid().double()
        return (1 - a) * c.double() + a * torch.eye(c.shape[-1], dtype=torch.float64, device=c.device)

    @torch.no_grad()
    def init_reference(self, c):
        """c (N, n, n): correlation matrices of the training subjects."""
        w, v = torch.linalg.eigh(self._shrunk(c).mean(0))
        self.w0.copy_((v * w.clamp(min=self.eps).rsqrt()) @ v.T)
        self.scale.fill_(1.0)
        self.scale.copy_(1 / self.forward(c).std())

    def forward(self, c):
        eye = torch.eye(c.shape[-1], dtype=torch.float64, device=c.device)
        w = self.w0 @ (eye + self.delta.double())
        y = w.T @ self._shrunk(c) @ w
        lam, v = torch.linalg.eigh((y + y.transpose(-1, -2)) / 2)
        out = (v * lam.clamp(min=self.eps).log().unsqueeze(-2)) @ v.transpose(-1, -2)
        return (out * self.scale).to(c.dtype)


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth on both branches."""

    def __init__(self, dim, heads, mlp_ratio, drop, attn_drop, drop_path, layer_scale=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = ConnectivityAttention(dim, heads, attn_drop, drop)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))
        self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale))
        self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, fc, key_mask=None):
        y, attn = self.attn(self.norm1(x), fc, key_mask)
        x = x + self.drop_path(self.gamma1 * y)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x, attn


class AttentionPool(nn.Module):
    """Learned-query pooling over regions; the weights double as a region-importance map."""

    def __init__(self, dim, drop):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.key = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.scale = dim ** -0.5

    def forward(self, x, key_mask=None):
        logits = (self.key(x) @ self.query) * self.scale
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask, float("-inf"))
        w = logits.softmax(-1)
        return (self.drop(w).unsqueeze(-1) * x).sum(1), w


def window_connectivity(ts, mask=None):
    """Fisher-z connectivity of a time window, ignoring censored / padded frames (pretraining input).

    ts (B, W, R), mask (B, W) bool -> (B, R, R)
    """
    m = torch.ones_like(ts[..., 0]) if mask is None else mask.to(ts.dtype)
    n = m.sum(1).clamp(min=2)[:, None, None]
    x = ts * m[..., None]
    x = (x - x.sum(1, keepdim=True) / n) * m[..., None]
    cov = x.transpose(1, 2) @ x / (n - 1)
    sd = cov.diagonal(dim1=1, dim2=2).clamp(min=1e-8).sqrt()
    r = cov / (sd[:, :, None] * sd[:, None, :])
    eye = torch.eye(r.shape[-1], dtype=torch.bool, device=r.device)
    return torch.atanh(r.clamp(-0.99, 0.99)).masked_fill(eye, 0.0)


class PsychosisNET(nn.Module):
    """
    Input   c: (B, R, R) correlation matrices (unit diagonal)
    Output  logits (B, n_classes)

      SPD tangent layer -> region tokens (row of the tangent matrix + region and network embeddings)
      -> depth x [connectivity-biased attention + GELU MLP] -> attention pooling -> GELU MLP head
      + linear wide path on the upper triangle of the tangent matrix (starts at zero)
      + optional linear age / sex head on the pooled vector (aux=True), used as an auxiliary loss

    Regularisation built into the architecture
      edge_drop    dropout on the input connectivity values
      token_drop   whole regions removed per sample during training
      drop         dropout after the embedding, in attention projections, MLPs and the head
      attn_drop    dropout on attention weights
      drop_path    stochastic depth, increasing linearly with depth
      skip_drop    dropout on the wide-path inputs
    Weight decay is applied by the optimiser (AdamW).
    """

    def __init__(self, n_roi, n_classes=2, networks=None, dim=64, depth=4, heads=4, mlp_ratio=2,
                 drop=0.3, attn_drop=0.2, drop_path=0.2, edge_drop=0.2, token_drop=0.15,
                 head_hidden=64, skip_drop=0.5, aux=False):
        super().__init__()
        self.spd = SPDTangent(n_roi)
        self.token_drop = token_drop
        self.edge_drop = nn.Dropout(edge_drop)
        self.embed = nn.Sequential(nn.Linear(n_roi, dim), nn.LayerNorm(dim))
        self.roi_embed = nn.Parameter(torch.zeros(n_roi, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))                  # used by masked pretraining
        # Functional-network identity of every region (e.g. the 7 Yeo networks).
        networks = torch.zeros(n_roi, dtype=torch.long) if networks is None else torch.as_tensor(networks)
        self.register_buffer("networks", networks)
        self.net_embed = nn.Embedding(int(networks.max()) + 1, dim)
        self.embed_drop = nn.Dropout(drop)
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, drop, attn_drop, r) for r in rates)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim, attn_drop)
        if head_hidden:
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, head_hidden), nn.GELU(),
                                      nn.LayerNorm(head_hidden), nn.Dropout(drop), nn.Linear(head_hidden, n_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, n_classes))
        self.aux_head = nn.Linear(dim, 2) if aux else None
        self.register_buffer("triu", torch.triu_indices(n_roi, n_roi, 1), persistent=False)
        self.skip = nn.Sequential(nn.Dropout(skip_drop), nn.Linear(self.triu.shape[1], n_classes))
        nn.init.zeros_(self.skip[1].weight)
        nn.init.trunc_normal_(self.roi_embed, std=0.02)
        nn.init.trunc_normal_(self.pool.query, std=0.02)

    def forward(self, c, return_attention=False, return_pooled=False):
        fc = self.spd(c)
        b, n, _ = fc.shape
        key_mask = None
        if self.training and self.token_drop > 0:
            key_mask = torch.rand(b, n, device=fc.device) >= self.token_drop
            key_mask[:, 0] |= ~key_mask.any(1)                             # never drop every region
        x, attns = self.encode(fc, key_mask)
        pooled, roi_weight = self.pool(x, key_mask)
        logits = self.head(pooled) + self.skip(fc[:, self.triu[0], self.triu[1]])
        if return_pooled:
            return logits, pooled
        return (logits, dict(attention=attns, roi_weight=roi_weight)) if return_attention else logits

    def encode(self, fc, key_mask=None, hidden=None):
        """Region tokens after the last block: (B, R, dim). fc is the output of the SPD layer.

        hidden (B, R) bool marks regions whose connectivity is withheld (masked pretraining):
        their rows and columns are zeroed and their token is replaced by a learned mask token.
        """
        if hidden is not None:
            visible = (~hidden).to(fc.dtype)
            fc = fc * visible[:, :, None] * visible[:, None, :]
        x = self.embed(self.edge_drop(fc))
        if hidden is not None:
            x = torch.where(hidden[..., None], self.mask_token.to(x.dtype), x)
        x = self.embed_drop(x + self.roi_embed + self.net_embed(self.networks))
        attns = []
        for blk in self.blocks:
            x, a = blk(x, fc, key_mask)
            attns.append(a)
        return self.norm(x), attns
