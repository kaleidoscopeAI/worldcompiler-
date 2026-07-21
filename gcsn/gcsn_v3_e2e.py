"""
GCSN-v3 — End-to-end differentiable, real-language-grounded, PointNet++
==========================================================================
Addresses, directly, the four gaps flagged in the previous round:

1. REAL LANGUAGE GROUNDING. Semantic vectors are no longer 10 hand-rolled
   random tensors. They are real sentence embeddings from a pretrained LM
   (sentence-transformers/all-MiniLM-L6-v2, 384-dim) over ~190 sentences
   built from 20 adjectives in systematic combination ("A rising, jagged
   motion.", "The path feels calm and expanding.", ...). The TEXT side is
   now real language, actually encoded by an actual model.

   What is still synthetic, stated plainly: the mapping from adjective ->
   motion parameters is a hand-written rule (see `ADJ` table), because no
   dataset of (English sentence, G-code) pairs exists to learn this from.
   That rule is what generates the *training targets* the decoder is
   taught to hit. This is the honest boundary of what's grounded here:
   real language -> real embeddings -> a rule-defined motion target. It is
   not "the network discovered on its own that 'jagged' means sharp
   turns" — I told it that, by construction. What IS learned is whether
   the network can (a) reproduce that mapping under a structured decoder
   and (b) generalize it to unseen adjective combinations, which is what
   the held-out split below actually tests.

2. HELD-OUT EVALUATION. Two held-out tiers, not one:
   - Tier A ("unseen combo"): specific adjective PAIRS withheld from
     training, both adjectives individually seen elsewhere.
   - Tier B ("unseen word"): two adjectives are entirely absent from every
     training sentence -- the model has never seen them in any combination.
     Tier B is the real test of generalization; Tier A is the easier one.

3. END-TO-END DIFFERENTIABILITY. No more two-stage training. The decoder
   is now NON-AUTOREGRESSIVE and STRUCTURED: it emits, per motion block, a
   fixed schema of categorical distributions (arc-or-linear, clockwise-or-
   not, and quantized X/Y/Z/I/J bins). Gumbel-softmax straight-through
   collapses the arc/clockwise choices to a hard decision with a usable
   gradient; coordinates use a differentiable soft-argmax expectation over
   their quantized bins. A fully differentiable G-code interpreter (pure
   tensor arithmetic -- no string parsing) turns those expectations into an
   actual 3D point cloud. The PointNet++ loss on that cloud backpropagates
   through the interpreter, through the soft coordinates, through the
   straight-through categorical choices, into the decoder -- one graph,
   one optimizer step. See self-critique for the real trade-off this
   required (non-autoregressive schema instead of free-form token
   sequences, G0 dropped).

4. REAL POINTNET++. Two hierarchical set-abstraction layers (farthest-
   point sampling + k-NN grouping + local per-point MLP + max-pool),
   96 points -> 32 centroids -> 8 centroids -> global max-pool -> 384-dim
   projection. This gives the encoder access to LOCAL neighborhood
   geometry at multiple scales, not just one global max over all points.
"""

import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

K_BLOCKS = 6
ARC_SAMPLES = 12
N_POINTS = K_BLOCKS * ARC_SAMPLES     # 72 points per generated shape
SEM_DIM = 384                          # MiniLM output dim

X_BINS = torch.arange(-20, 21).float()   # 41
Y_BINS = X_BINS.clone()
Z_BINS = X_BINS.clone()
IJ_BINS = torch.arange(-12, 13).float()  # 25

# ---------------------------------------------------------------------------
# 1. CORPUS: adjectives -> motion params (the disclosed rule-based part),
#    combined into real sentences, embedded by a real LM.
# ---------------------------------------------------------------------------

# word -> (vertical, curvature, radial, turbulence, size) deltas.
# `size` is a global scale multiplier on the whole generated shape (see
# synth_targets) -- added because none of the first four dims controls
# overall scale at all: "huge explosion" and "tiny explosion" generated
# identically-sized geometry before this. The 4 new pure-size anchors
# (huge/massive/tiny/miniature) are zero on the other 4 dims by design, so
# the size axis stays unconfounded with motion quality in the propagation
# kernel/regression -- a word can be purely big, purely turbulent, or both.
ADJ: Dict[str, Tuple[float, float, float, float, float]] = {
    "rising":      (1.0, 0.0, 0.0, 0.0, 0.0),
    "falling":     (-1.0, 0.0, 0.0, 0.0, 0.0),
    "flat":        (0.0, 0.0, 0.0, 0.0, 0.0),
    "smooth":      (0.0, 1.0, 0.0, -0.2, 0.0),
    "circular":    (0.0, 1.5, 0.0, -0.3, 0.0),
    "jagged":      (0.0, -1.2, 0.0, 1.0, 0.0),
    "blocky":      (0.0, -1.0, 0.0, 0.3, 0.0),
    "spiraling":   (0.3, 0.8, 0.6, 0.1, 0.0),
    "expanding":   (0.0, 0.0, 1.2, 0.0, 0.0),
    "contracting": (0.0, 0.0, -1.2, 0.0, 0.0),
    "constant":    (0.0, 0.0, 0.0, 0.0, 0.0),
    "calm":        (0.0, 0.3, 0.0, -1.0, 0.0),
    "chaotic":     (0.0, -0.3, 0.0, 1.4, 0.0),
    "stable":      (0.0, 0.2, 0.0, -0.8, 0.0),
    "explosive":   (0.0, -0.5, 1.0, 1.8, 0.0),
    "flowing":     (0.0, 1.2, 0.2, -0.3, 0.0),
    "radiating":   (0.0, 0.0, 1.3, 0.4, 0.0),
    "converging":  (0.0, 0.2, -1.3, 0.0, 0.0),
    "diverging":   (0.0, 0.2, 1.3, 0.0, 0.0),
    "oscillating": (0.2, -0.2, 0.0, 1.0, 0.0),
    "huge":        (0.0, 0.0, 0.0, 0.0, 1.5),
    "massive":     (0.0, 0.0, 0.0, 0.0, 1.8),
    "tiny":        (0.0, 0.0, 0.0, 0.0, -1.3),
    "miniature":   (0.0, 0.0, 0.0, 0.0, -1.0),
}

TEMPLATES = [
    "A {a1}, {a2} motion.",
    "The path feels {a1} and {a2}.",
    "An object moving in a {a1}, {a2} way.",
    "Something {a1}, almost {a2}, unfolding in space.",
]

# Tier B: entirely withheld words -- never appear in ANY training sentence.
UNSEEN_WORDS = {"radiating", "oscillating"}


@dataclass
class Example:
    text: str
    words: Tuple[str, str]
    params: Tuple[float, float, float, float]
    split: str   # "train" | "test_combo" | "test_word"


def build_corpus(test_combo_frac: float = 0.15) -> List[Example]:
    words = list(ADJ.keys())
    combos = list(itertools.combinations(words, 2))
    rng = random.Random(SEED)
    rng.shuffle(combos)

    examples = []
    remaining = []
    for (a1, a2) in combos:
        if a1 in UNSEEN_WORDS or a2 in UNSEEN_WORDS:
            split = "test_word"
        else:
            split = None  # decided below
        remaining.append((a1, a2, split))

    non_word_combos = [c for c in remaining if c[2] is None]
    n_test_combo = int(len(non_word_combos) * test_combo_frac)
    test_combo_set = set((a1, a2) for a1, a2, _ in non_word_combos[:n_test_combo])

    for a1, a2, forced_split in remaining:
        if forced_split == "test_word":
            split = "test_word"
        elif (a1, a2) in test_combo_set:
            split = "test_combo"
        else:
            split = "train"
        tmpl = rng.choice(TEMPLATES)
        text = tmpl.format(a1=a1, a2=a2)
        v1, v2 = ADJ[a1], ADJ[a2]
        params = tuple(x + y for x, y in zip(v1, v2))
        examples.append(Example(text=text, words=(a1, a2), params=params, split=split))
    return examples


def synth_targets(params: Tuple[float, float, float, float, float], seed: int) -> List[dict]:
    """The disclosed rule: continuous motion params -> K block targets
    (is_arc, clockwise, x, y, z, i, j). This is what the decoder is taught
    to reproduce -- it is NOT learned from data, it's the teacher signal.

    `size` scales the whole shape uniformly (radius, vertical extent, arc
    bulge) rather than adding a new independent motion -- it answers "how
    big is this," which vertical/curvature/radial/turbulence never did.
    Clamped so a "massive, explosive" combination degrades gracefully by
    saturating at the existing bin range (+-20 / +-12) instead of silently
    wrapping or needing a wider (and architecture-changing) bin range."""
    vert, curve, radial, turb, size = params
    size_mult = max(0.25, 1.0 + 0.6 * size)
    base_r = (4.0 + 2.0 * abs(radial)) * size_mult
    pts = [(0.0, 0.0, 0.0)]
    blocks = []
    for k in range(1, K_BLOCKS + 1):
        angle = (2 * math.pi / K_BLOCKS) * k + turb * 0.25 * ((-1) ** k)
        r = base_r * (1 + radial * 0.15 * k)
        z = vert * 1.5 * k * size_mult + turb * 0.4 * math.sin(k * 1.3)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        x0, y0, z0 = pts[-1]
        is_arc = curve > 0.15
        clockwise = (k % 2 == 0)
        i_off = j_off = 0.0
        if is_arc:
            mx, my = (x0 + x) / 2, (y0 + y) / 2
            dx, dy = x - x0, y - y0
            dist = math.hypot(dx, dy) + 1e-6
            perp = (-dy / dist, dx / dist)
            bulge = 0.3 * curve * dist
            cx, cy = mx + perp[0] * bulge, my + perp[1] * bulge
            i_off, j_off = cx - x0, cy - y0
        pts.append((x, y, z))
        blocks.append(dict(is_arc=is_arc, clockwise=clockwise,
                            x=max(-20, min(20, x)), y=max(-20, min(20, y)),
                            z=max(-20, min(20, z)),
                            i=max(-12, min(12, i_off)), j=max(-12, min(12, j_off))))
    return blocks


def quantize_targets(blocks: List[dict]) -> torch.Tensor:
    """Returns (K, 7) long tensor: [is_arc, clockwise, x_bin, y_bin, z_bin, i_bin, j_bin]
    where *_bin is an index into the corresponding BINS tensor."""
    out = torch.zeros(K_BLOCKS, 7, dtype=torch.long)
    for k, b in enumerate(blocks):
        out[k, 0] = int(b["is_arc"])
        out[k, 1] = int(b["clockwise"])
        out[k, 2] = int(round(b["x"])) + 20
        out[k, 3] = int(round(b["y"])) + 20
        out[k, 4] = int(round(b["z"])) + 20
        out[k, 5] = int(round(b["i"])) + 12
        out[k, 6] = int(round(b["j"])) + 12
    return out


# ---------------------------------------------------------------------------
# 2. NON-AUTOREGRESSIVE STRUCTURED DECODER
# ---------------------------------------------------------------------------

class StructuredGCodeDecoder(nn.Module):
    def __init__(self, sem_dim=SEM_DIM, d_model=128, n_heads=4, n_layers=3,
                 mem_slots=4, k_blocks=K_BLOCKS):
        super().__init__()
        self.k_blocks = k_blocks
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, d_model * mem_slots), nn.GELU(),
            nn.Linear(d_model * mem_slots, d_model * mem_slots),
        )
        self.mem_slots = mem_slots
        self.block_embed = nn.Embedding(k_blocks, d_model)

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4,
                                            batch_first=True, activation="gelu")
        self.blocks_decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        self.head_is_arc = nn.Linear(d_model, 2)
        self.head_cw = nn.Linear(d_model, 2)
        self.head_x = nn.Linear(d_model, 41)
        self.head_y = nn.Linear(d_model, 41)
        self.head_z = nn.Linear(d_model, 41)
        self.head_i = nn.Linear(d_model, 25)
        self.head_j = nn.Linear(d_model, 25)
        self.d_model = d_model

    def forward(self, sem_vec: torch.Tensor):
        b = sem_vec.size(0)
        memory = self.sem_proj(sem_vec).view(b, self.mem_slots, self.d_model)
        block_ids = torch.arange(self.k_blocks, device=sem_vec.device).unsqueeze(0).expand(b, -1)
        tgt = self.block_embed(block_ids)                          # (B, K, D) -- no mask: parallel/non-autoregressive
        h = self.blocks_decoder(tgt=tgt, memory=memory)             # (B, K, D)
        return dict(
            is_arc=self.head_is_arc(h), clockwise=self.head_cw(h),
            x=self.head_x(h), y=self.head_y(h), z=self.head_z(h),
            i=self.head_i(h), j=self.head_j(h),
        )


# ---------------------------------------------------------------------------
# 3. DIFFERENTIABLE G-CODE INTERPRETER
# ---------------------------------------------------------------------------

def soft_expect(logits: torch.Tensor, bin_vals: torch.Tensor) -> torch.Tensor:
    """(..., n_bins) logits -> (...,) differentiable expected value."""
    probs = F.softmax(logits, dim=-1)
    return probs @ bin_vals.to(logits.device)


def st_binary(logits: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Gumbel-softmax straight-through, returns prob-mass-of-class-1 with a
    hard 0/1 forward value and a soft gradient."""
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)[..., 1]


def differentiable_path(decoder_out: dict, device, tau: float = 0.7) -> torch.Tensor:
    """decoder_out values: (B, K, n_bins). Returns point cloud (B, K*ARC_SAMPLES, 3)."""
    B, Kb, _ = decoder_out["x"].shape
    t = torch.linspace(1.0 / ARC_SAMPLES, 1.0, ARC_SAMPLES, device=device)  # (S,)

    ex = soft_expect(decoder_out["x"], X_BINS)   # (B,K)
    ey = soft_expect(decoder_out["y"], Y_BINS)
    ez = soft_expect(decoder_out["z"], Z_BINS)
    ei = soft_expect(decoder_out["i"], IJ_BINS)
    ej = soft_expect(decoder_out["j"], IJ_BINS)
    is_arc_w = st_binary(decoder_out["is_arc"], tau)   # (B,K)
    cw_w = st_binary(decoder_out["clockwise"], tau)    # (B,K)

    cur = torch.zeros(B, 3, device=device)
    all_pts = []
    for k in range(Kb):
        x0, y0, z0 = cur[:, 0], cur[:, 1], cur[:, 2]
        x1, y1, z1 = ex[:, k], ey[:, k], ez[:, k]
        i_off, j_off = ei[:, k], ej[:, k]

        # linear branch: (B, S, 3)
        lin = torch.stack([
            x0.unsqueeze(1) + (x1 - x0).unsqueeze(1) * t,
            y0.unsqueeze(1) + (y1 - y0).unsqueeze(1) * t,
            z0.unsqueeze(1) + (z1 - z0).unsqueeze(1) * t,
        ], dim=-1)

        # arc branch (both sweep directions), center from I/J offset
        cx, cy = x0 + i_off, y0 + j_off
        r = torch.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2 + 1e-6)
        a0 = torch.atan2(y0 - cy, x0 - cx)
        a1 = torch.atan2(y1 - cy, x1 - cx)

        a1_cw = a1 + torch.where(a1 > a0, -2 * math.pi, torch.zeros_like(a1))
        a1_ccw = a1 + torch.where(a1 < a0, 2 * math.pi, torch.zeros_like(a1))

        ang_cw = a0.unsqueeze(1) + (a1_cw - a0).unsqueeze(1) * t
        ang_ccw = a0.unsqueeze(1) + (a1_ccw - a0).unsqueeze(1) * t
        z_ramp = z0.unsqueeze(1) + (z1 - z0).unsqueeze(1) * t

        arc_cw = torch.stack([cx.unsqueeze(1) + r.unsqueeze(1) * torch.cos(ang_cw),
                               cy.unsqueeze(1) + r.unsqueeze(1) * torch.sin(ang_cw), z_ramp], dim=-1)
        arc_ccw = torch.stack([cx.unsqueeze(1) + r.unsqueeze(1) * torch.cos(ang_ccw),
                                cy.unsqueeze(1) + r.unsqueeze(1) * torch.sin(ang_ccw), z_ramp], dim=-1)

        cw = cw_w[:, k].view(B, 1, 1)
        arc = cw * arc_cw + (1 - cw) * arc_ccw
        arcw = is_arc_w[:, k].view(B, 1, 1)
        block_pts = arcw * arc + (1 - arcw) * lin

        all_pts.append(block_pts)
        cur = torch.stack([x1, y1, z1], dim=-1)

    return torch.cat(all_pts, dim=1)   # (B, K*S, 3)


@torch.no_grad()
def render_gcode_text(decoder_out: dict, idx: int) -> str:
    """Argmax-decode one example's structured output into human-readable G-code (display only)."""
    lines = []
    cur = [0.0, 0.0, 0.0]
    Kb = decoder_out["x"].shape[1]
    for k in range(Kb):
        is_arc = bool(decoder_out["is_arc"][idx, k].argmax().item())
        cw = bool(decoder_out["clockwise"][idx, k].argmax().item())
        x = X_BINS[decoder_out["x"][idx, k].argmax().item()].item()
        y = Y_BINS[decoder_out["y"][idx, k].argmax().item()].item()
        z = Z_BINS[decoder_out["z"][idx, k].argmax().item()].item()
        if is_arc:
            i = IJ_BINS[decoder_out["i"][idx, k].argmax().item()].item()
            j = IJ_BINS[decoder_out["j"][idx, k].argmax().item()].item()
            cmd = "G2" if cw else "G3"
            lines.append(f"{cmd} X{x:g} Y{y:g} Z{z:g} I{i:g} J{j:g}")
        else:
            lines.append(f"G1 X{x:g} Y{y:g} Z{z:g}")
        cur = [x, y, z]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. POINTNET++ : hierarchical set abstraction
# ---------------------------------------------------------------------------

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, M: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    idx = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch_idx = torch.arange(B, device=xyz.device)
    for i in range(M):
        idx[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.min(dist, d)
        farthest = torch.max(dist, -1)[1]
    return idx


class SetAbstraction(nn.Module):
    def __init__(self, in_feat_dim: int, mlp_dims: List[int], n_centroids: int, n_neighbors: int):
        super().__init__()
        self.n_centroids = n_centroids
        self.n_neighbors = n_neighbors
        dims = [3 + in_feat_dim] + mlp_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = mlp_dims[-1]

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        # xyz: (B,N,3), feats: (B,N,C) or None
        B, N, _ = xyz.shape
        m = min(self.n_centroids, N)
        k = min(self.n_neighbors, N)
        idx = farthest_point_sample(xyz, m)
        centroids = index_points(xyz, idx)                      # (B,M,3)
        dists = torch.cdist(centroids, xyz)                      # (B,M,N)
        knn_idx = dists.topk(k, largest=False).indices            # (B,M,K)
        grouped_xyz = index_points(xyz, knn_idx)                  # (B,M,K,3)
        grouped_xyz_local = grouped_xyz - centroids.unsqueeze(2)
        if feats is not None:
            grouped_feats = index_points(feats, knn_idx)
            grouped = torch.cat([grouped_xyz_local, grouped_feats], dim=-1)
        else:
            grouped = grouped_xyz_local
        b, mm, kk, c = grouped.shape
        h = self.mlp(grouped.reshape(b * mm * kk, c)).reshape(b, mm, kk, -1)
        pooled, _ = h.max(dim=2)                                   # (B,M,C')
        return centroids, pooled


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, sem_dim: int = SEM_DIM):
        super().__init__()
        self.sa1 = SetAbstraction(in_feat_dim=0, mlp_dims=[64, 64, 128], n_centroids=32, n_neighbors=12)
        self.sa2 = SetAbstraction(in_feat_dim=128, mlp_dims=[128, 128, 256], n_centroids=8, n_neighbors=8)
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, sem_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = xyz - xyz.mean(dim=1, keepdim=True)     # shape-only, translation invariant
        c1, f1 = self.sa1(xyz, None)
        c2, f2 = self.sa2(c1, f1)
        global_feat, _ = f2.max(dim=1)                 # (B, 256)
        return self.head(global_feat)


# ---------------------------------------------------------------------------
# 5. JOINT END-TO-END TRAINING
# ---------------------------------------------------------------------------

def structured_ce_loss(decoder_out: dict, targets: torch.Tensor) -> torch.Tensor:
    # targets: (B, K, 7) -> [is_arc, cw, x_bin, y_bin, z_bin, i_bin, j_bin]
    is_arc_t, cw_t = targets[..., 0], targets[..., 1]
    x_t, y_t, z_t, i_t, j_t = targets[..., 2], targets[..., 3], targets[..., 4], targets[..., 5], targets[..., 6]
    arc_mask = is_arc_t.float()

    def ce(logits, tgt, mask=None):
        b, k, c = logits.shape
        l = F.cross_entropy(logits.reshape(b * k, c), tgt.reshape(b * k), reduction="none").reshape(b, k)
        if mask is not None:
            l = l * mask
            return l.sum() / mask.sum().clamp(min=1.0)
        return l.mean()

    loss = (ce(decoder_out["is_arc"], is_arc_t)
            + ce(decoder_out["clockwise"], cw_t, arc_mask)
            + ce(decoder_out["x"], x_t) + ce(decoder_out["y"], y_t) + ce(decoder_out["z"], z_t)
            + ce(decoder_out["i"], i_t, arc_mask) + ce(decoder_out["j"], j_t, arc_mask))
    return loss


def run_training(epochs=600, lr=2e-4, lambda_cycle=1.0, device="cpu"):
    print("Loading sentence-transformer (all-MiniLM-L6-v2) for real language grounding...")
    from sentence_transformers import SentenceTransformer
    lm = SentenceTransformer("all-MiniLM-L6-v2")

    corpus = build_corpus()
    train_ex = [e for e in corpus if e.split == "train"]
    test_combo_ex = [e for e in corpus if e.split == "test_combo"]
    test_word_ex = [e for e in corpus if e.split == "test_word"]
    print(f"Corpus: {len(corpus)} total | train {len(train_ex)} | "
          f"held-out combos {len(test_combo_ex)} | held-out words {len(test_word_ex)}")

    def make_batch(exs):
        texts = [e.text for e in exs]
        sem = torch.tensor(lm.encode(texts, show_progress_bar=False), dtype=torch.float32)
        targets = torch.stack([
            quantize_targets(synth_targets(e.params, seed=hash(e.words) % 100000)) for e in exs
        ])
        return sem.to(device), targets.to(device)

    sem_train, tgt_train = make_batch(train_ex)
    sem_test_combo, tgt_test_combo = make_batch(test_combo_ex)
    sem_test_word, tgt_test_word = make_batch(test_word_ex)

    decoder = StructuredGCodeDecoder().to(device)
    encoder = PointNetPlusPlusEncoder().to(device)
    params = list(decoder.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    print("\nJoint end-to-end training (structured CE + differentiable point-cloud cycle loss)...")
    batch_size = 32
    n = sem_train.size(0)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_ce, total_cyc, nb = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            bidx = perm[start:start + batch_size]
            sem_b, tgt_b = sem_train[bidx], tgt_train[bidx]

            opt.zero_grad()
            out = decoder(sem_b)
            ce_loss = structured_ce_loss(out, tgt_b)

            cloud = differentiable_path(out, device, tau=max(0.3, 0.9 * (0.995 ** ep)))
            grounded = encoder(cloud)
            mse = F.mse_loss(grounded, sem_b)
            cos = 1 - F.cosine_similarity(grounded, sem_b, dim=-1).mean()
            cyc_loss = mse + 0.5 * cos

            loss = ce_loss + lambda_cycle * cyc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_ce += ce_loss.item(); total_cyc += cyc_loss.item(); nb += 1

        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  ce {total_ce/nb:.4f}  cycle {total_cyc/nb:.4f}")

    return decoder, encoder, lm, corpus, (sem_train, tgt_train, train_ex), \
        (sem_test_combo, tgt_test_combo, test_combo_ex), (sem_test_word, tgt_test_word, test_word_ex)


# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def evaluate_split(decoder, encoder, sem, tgt, exs, name, device, show_examples=3):
    decoder.eval(); encoder.eval()
    out = decoder(sem)
    cloud = differentiable_path(out, device, tau=0.3)
    grounded = encoder(cloud)
    sims = cosine_rows(grounded, sem)
    print(f"\n--- {name}: n={len(exs)}  mean round-trip cosine sim = {sims.mean().item():.3f}"
          f"  (min {sims.min().item():.3f}, max {sims.max().item():.3f}) ---")
    for i in range(min(show_examples, len(exs))):
        print(f"  \"{exs[i].text}\"  words={exs[i].words}  sim={sims[i].item():.3f}")
        print(f"    generated gcode:\n" + "\n".join("    " + l for l in render_gcode_text(out, i).splitlines()))
    return sims.mean().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, encoder, lm, corpus, train_pack, test_combo_pack, test_word_pack = run_training(
        epochs=600, device=device
    )

    train_sim = evaluate_split(decoder, encoder, *train_pack, name="TRAIN", device=device)
    combo_sim = evaluate_split(decoder, encoder, *test_combo_pack, name="HELD-OUT COMBO (tier A)", device=device)
    word_sim = evaluate_split(decoder, encoder, *test_word_pack, name="HELD-OUT WORD (tier B, zero-shot)", device=device)

    print(f"\n=== Summary ===")
    print(f"train mean sim:            {train_sim:.3f}")
    print(f"held-out combo mean sim:   {combo_sim:.3f}   (gap vs train: {train_sim-combo_sim:+.3f})")
    print(f"held-out word mean sim:    {word_sim:.3f}   (gap vs train: {train_sim-word_sim:+.3f})")
