"""
GCSN-3D-Textured — point cloud + texture-image fusion
=========================================================
Adds the piece you asked for: an image library, encoded alongside the point
cloud, so the representation carries TEXTURE/VARIATION on top of pure shape.

Two distinct fusion paths, because "texture alongside a point cloud" means
two different things in real 3D graphics and both are worth having:

  1. PER-POINT COLOR (local, fine-grained): each concept's procedural
     texture image is UV-mapped onto its point cloud using the points' own
     (x,y) footprint as UV coordinates — exactly how planar texture mapping
     works in CG. Every point becomes (x, y, z, r, g, b) instead of just
     (x, y, z), so local texture variation across the SURFACE of the shape
     is available to the encoder, not just its outline.

  2. GLOBAL IMAGE EMBEDDING (holistic style): the same texture image is
     ALSO run through a small CNN independent of the point cloud and
     concatenated onto the pooled point-cloud feature before the final
     projection. This carries "what material/style is this" as a single
     signal, the way a glaze or material swatch is a property of the whole
     object, not of any one point on it.

HONEST SCOPE NOTE, stated up front: there is no real image dataset wired in
here (no internet image fetch was appropriate for a from-scratch generative
architecture demo). The "image library" is 10 procedurally generated 64x64
textures, one per concept, each using a different generator (stripes,
rings, swirl, noise, checkerboard, speckle...) chosen to be visually
distinct so the CNN has something non-trivial to encode. Swapping in a real
texture/photo library is a loader change, not an architecture change — see
self-critique.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 64

# ---------------------------------------------------------------------------
# 1. TOKENIZER  (unchanged from gcsn_3d.py)
# ---------------------------------------------------------------------------

COORD_RANGE = range(-20, 21)
IJK_RANGE = range(-12, 13)
FEED_LEVELS = [100, 300, 600, 1200, 2000]

class GCodeTokenizer:
    def __init__(self):
        specials = ["<PAD>", "<BOS>", "<EOS>", "<EOB>"]
        commands = ["G0", "G1", "G2", "G3"]
        axis_tokens = []
        for axis in "XYZ":
            axis_tokens += [f"{axis}{v}" for v in COORD_RANGE]
        offset_tokens = []
        for axis in "IJK":
            offset_tokens += [f"{axis}{v}" for v in IJK_RANGE]
        f_tokens = [f"F{v}" for v in FEED_LEVELS]
        vocab = specials + commands + axis_tokens + offset_tokens + f_tokens
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(vocab)
        self.PAD = self.token_to_id["<PAD>"]
        self.BOS = self.token_to_id["<BOS>"]
        self.EOS = self.token_to_id["<EOS>"]
        self.EOB = self.token_to_id["<EOB>"]

    def encode_block(self, block_tokens: List[str]) -> List[int]:
        for t in block_tokens:
            if t not in self.token_to_id:
                raise ValueError(f"Token '{t}' outside quantized vocab range")
        return [self.token_to_id[t] for t in block_tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[i] for i in ids if i not in (self.PAD,)]


# ---------------------------------------------------------------------------
# 2. G-CODE INTERPRETER -> DENSE 3D POINT CLOUD  (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    feed_at_vertex: List[int] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)

class GCodeInterpreter3D:
    ARC_SAMPLES = 16

    def run(self, tokens: List[str]) -> PathResult:
        result = PathResult()
        x = y = z = 0.0
        result.vertices.append((x, y, z))
        result.feed_at_vertex.append(0)
        cur_feed = FEED_LEVELS[0]
        block: List[str] = []
        for tok in tokens:
            if tok in ("<BOS>", "<PAD>"):
                continue
            if tok == "<EOS>":
                break
            if tok == "<EOB>":
                if block:
                    x, y, z, cur_feed = self._exec_block(block, x, y, z, cur_feed, result)
                block = []
                continue
            block.append(tok)
        if block:
            self._exec_block(block, x, y, z, cur_feed, result)
        return result

    def _exec_block(self, block, x, y, z, cur_feed, result: PathResult):
        cmd = None
        nx, ny, nz = x, y, z
        has_x = has_y = has_z = False
        i_off = j_off = None
        f_val = None
        for tok in block:
            if tok.startswith("G"):
                cmd = tok
            elif tok.startswith("X"):
                nx = float(tok[1:]); has_x = True
            elif tok.startswith("Y"):
                ny = float(tok[1:]); has_y = True
            elif tok.startswith("Z"):
                nz = float(tok[1:]); has_z = True
            elif tok.startswith("I"):
                i_off = float(tok[1:])
            elif tok.startswith("J"):
                j_off = float(tok[1:])
            elif tok.startswith("F"):
                f_val = int(tok[1:])
        if cmd is None:
            return x, y, z, cur_feed
        if f_val is not None:
            cur_feed = f_val
        result.raw_lines.append(" ".join(block))

        if cmd in ("G0", "G1"):
            if has_x or has_y or has_z:
                result.vertices.append((nx, ny, nz))
                result.feed_at_vertex.append(cur_feed if cmd == "G1" else 0)
            return nx, ny, nz, cur_feed

        if cmd in ("G2", "G3") and i_off is not None and j_off is not None and (has_x or has_y):
            cx, cy = x + i_off, y + j_off
            r = math.hypot(x - cx, y - cy)
            if r < 1e-6:
                result.vertices.append((nx, ny, nz))
                result.feed_at_vertex.append(cur_feed)
                return nx, ny, nz, cur_feed
            a0 = math.atan2(y - cy, x - cx)
            a1 = math.atan2(ny - cy, nx - cx)
            clockwise = (cmd == "G2")
            if clockwise:
                while a1 > a0:
                    a1 -= 2 * math.pi
            else:
                while a1 < a0:
                    a1 += 2 * math.pi
            for s in range(1, self.ARC_SAMPLES + 1):
                t = s / self.ARC_SAMPLES
                a = a0 + (a1 - a0) * t
                px = cx + r * math.cos(a)
                py = cy + r * math.sin(a)
                pz = z + (nz - z) * t
                result.vertices.append((px, py, pz))
                result.feed_at_vertex.append(cur_feed)
            return nx, ny, nz, cur_feed
        return x, y, z, cur_feed


def densify_to_point_cloud(path: PathResult, n_points: int = 256) -> torch.Tensor:
    verts = path.vertices
    if len(verts) < 2:
        v = verts[0] if verts else (0.0, 0.0, 0.0)
        return torch.tensor([v] * n_points, dtype=torch.float32)
    seg_lens, total = [], 0.0
    for k in range(1, len(verts)):
        x0, y0, z0 = verts[k - 1]
        x1, y1, z1 = verts[k]
        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        seg_lens.append(d); total += d
    if total < 1e-9:
        return torch.tensor([verts[0]] * n_points, dtype=torch.float32)
    targets = [total * k / (n_points - 1) for k in range(n_points)]
    pts, cum, seg_idx = [], 0.0, 0
    for t in targets:
        while seg_idx < len(seg_lens) - 1 and cum + seg_lens[seg_idx] < t:
            cum += seg_lens[seg_idx]; seg_idx += 1
        seg_len = seg_lens[seg_idx] if seg_lens[seg_idx] > 1e-9 else 1e-9
        local_t = min(max((t - cum) / seg_len, 0.0), 1.0)
        x0, y0, z0 = verts[seg_idx]
        x1, y1, z1 = verts[seg_idx + 1]
        pts.append((x0 + (x1 - x0) * local_t, y0 + (y1 - y0) * local_t, z0 + (z1 - z0) * local_t))
    return torch.tensor(pts, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. TEXTURE IMAGE LIBRARY (procedural — see module docstring for scope note)
# ---------------------------------------------------------------------------

def _grid(size: int):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    return xs, ys

def tex_stripes(size, freq=8.0):
    xs, ys = _grid(size)
    v = (torch.sin(xs * freq * math.pi) > 0).float()
    return v.unsqueeze(0).repeat(3, 1, 1) * 0.8 + 0.1

def tex_rings(size, freq=6.0):
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    v = (torch.sin(r * freq * math.pi) > 0).float()
    return v.unsqueeze(0).repeat(3, 1, 1) * 0.8 + 0.1

def tex_swirl(size):
    xs, ys = _grid(size)
    theta = torch.atan2(ys, xs)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    v = (torch.sin(theta * 6 + r * 10) + 1) / 2
    out = torch.stack([v, 1 - v, torch.full_like(v, 0.5)], dim=0)
    return out

def tex_checker(size, cells=6):
    xs, ys = _grid(size)
    cx = (xs * cells).floor()
    cy = (ys * cells).floor()
    v = ((cx + cy) % 2).float()
    return v.unsqueeze(0).repeat(3, 1, 1) * 0.8 + 0.1

def tex_noise(size, blur_passes=3):
    v = torch.rand(size, size)
    for _ in range(blur_passes):
        v = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v.unsqueeze(0).repeat(3, 1, 1)

def tex_speckle(size, density=0.05):
    base = torch.full((3, size, size), 0.15)
    mask = (torch.rand(size, size) < density).float()
    speck = torch.rand(3, size, size)
    return base * (1 - mask) + speck * mask

def tex_gradient(size, axis=0):
    xs, ys = _grid(size)
    v = (xs if axis == 0 else ys + 1) / 2
    v = (v + 1) / 2 if axis == 0 else v
    return torch.stack([v, torch.full_like(v, 0.3), 1 - v], dim=0)

def tex_rays(size, n_rays=8):
    xs, ys = _grid(size)
    theta = torch.atan2(ys, xs)
    v = (torch.sin(theta * n_rays) > 0).float()
    return torch.stack([v, v * 0.4, torch.full_like(v, 0.2)], dim=0)

def tex_solid_with_grain(size, base_color=(0.7, 0.7, 0.75)):
    base = torch.tensor(base_color).view(3, 1, 1).repeat(1, size, size)
    grain = (torch.rand(3, size, size) - 0.5) * 0.08
    return (base + grain).clamp(0, 1)

def tex_jagged_noise(size):
    v = torch.rand(size, size)
    v = (v > 0.5).float()
    return torch.stack([v, 1 - v, v * 0.3], dim=0)


def build_texture_library(size: int = IMG_SIZE) -> Dict[str, torch.Tensor]:
    """One visually-distinct procedural texture per concept -> (3, size, size), values in [0,1]."""
    return {
        "circle":      tex_rings(size, freq=5.0),
        "stability":   tex_solid_with_grain(size, base_color=(0.55, 0.6, 0.55)),
        "oscillation": tex_stripes(size, freq=10.0),
        "conflict":    tex_jagged_noise(size),
        "growth":      tex_gradient(size, axis=1),
        "convergence": tex_swirl(size),
        "divergence":  tex_rays(size, n_rays=10),
        "flow":        tex_stripes(size, freq=3.0),
        "boundary":    tex_checker(size, cells=6),
        "explosion":   tex_speckle(size, density=0.12),
    }


def sample_texture_colors(points_xyz: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
    """Planar UV-map: normalize each point's (x, y) into the texture's pixel
    grid using the point cloud's own bounding box, nearest-sample the color.
    This is literal texture mapping -- the same operation a renderer does
    when it wraps an image onto a mesh using UV coordinates."""
    xy = points_xyz[:, :2]
    mins = xy.min(dim=0).values
    maxs = xy.max(dim=0).values
    span = (maxs - mins).clamp(min=1e-6)
    uv = (xy - mins) / span                                    # in [0,1]
    size = texture.shape[-1]
    px = (uv[:, 0] * (size - 1)).round().long().clamp(0, size - 1)
    py = (uv[:, 1] * (size - 1)).round().long().clamp(0, size - 1)
    colors = texture[:, py, px].transpose(0, 1)                 # (N, 3)
    return colors


# ---------------------------------------------------------------------------
# 4. DECODER (unchanged architecture, semantic -> gcode tokens)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SemanticToMemory(nn.Module):
    def __init__(self, sem_dim, d_model, mem_slots=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.proj = nn.Sequential(nn.Linear(sem_dim, d_model * mem_slots), nn.GELU(),
                                   nn.Linear(d_model * mem_slots, d_model * mem_slots))

    def forward(self, sem_vec):
        b = sem_vec.size(0)
        return self.proj(sem_vec).view(b, self.mem_slots, -1)


class GCodeDecoder(nn.Module):
    def __init__(self, vocab_size, sem_dim, d_model=128, n_heads=4, n_layers=3,
                 mem_slots=4, max_len=160, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.sem_to_mem = SemanticToMemory(sem_dim, d_model, mem_slots)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4, batch_first=True,
                                            activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(n, device):
        return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)

    def forward(self, sem_vec, tgt_ids):
        memory = self.sem_to_mem(sem_vec)
        tgt = self.pos_enc(self.tok_emb(tgt_ids))
        mask = self.causal_mask(tgt.size(1), tgt.device)
        pad_mask = tgt_ids.eq(self.pad_id)
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
        return self.out_proj(h)

    @torch.no_grad()
    def generate(self, sem_vec, bos, eos, max_len=120, temperature=0.3):
        self.eval()
        device = sem_vec.device
        memory = self.sem_to_mem(sem_vec)
        ids = torch.tensor([[bos]], device=device)
        for _ in range(max_len):
            tgt = self.pos_enc(self.tok_emb(ids))
            mask = self.causal_mask(tgt.size(1), device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)
            logits = self.out_proj(h[:, -1]) / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == eos:
                break
        return ids.squeeze(0).tolist()


# ---------------------------------------------------------------------------
# 5. IMAGE ENCODER (small CNN, texture image -> global style embedding)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),   # 64->32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),  # 32->16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 16->8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        # img_batch: (B, 3, H, W)
        feat = self.net(img_batch).flatten(1)
        return self.head(feat)


# ---------------------------------------------------------------------------
# 6. TEXTURED POINTNET: fuses per-point RGB (local texture) with a global
#    image embedding (holistic style) alongside geometry.
# ---------------------------------------------------------------------------

class TexturedPointNetEncoder(nn.Module):
    def __init__(self, sem_dim: int, img_embed_dim: int = 32, hidden: int = 128):
        super().__init__()
        # per-point channels: x,y,z (centered) + r,g,b = 6
        self.point_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, hidden),
        )
        self.image_encoder = ImageEncoder(out_dim=img_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + img_embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, sem_dim),
        )

    def forward(self, colored_points: torch.Tensor, texture_images: torch.Tensor) -> torch.Tensor:
        # colored_points: (B, N, 6) = xyz + rgb ; texture_images: (B, 3, H, W)
        xyz, rgb = colored_points[..., :3], colored_points[..., 3:]
        xyz = xyz - xyz.mean(dim=1, keepdim=True)               # shape-only geometry, as before
        fused = torch.cat([xyz, rgb], dim=-1)                    # rgb stays in absolute [0,1], carries real appearance
        b, n, _ = fused.shape
        flat = fused.reshape(b * n, 6)
        point_feat = self.point_mlp(flat).reshape(b, n, -1)
        pooled, _ = point_feat.max(dim=1)                        # permutation-invariant local/texture summary

        style_embed = self.image_encoder(texture_images)         # global holistic appearance
        combined = torch.cat([pooled, style_embed], dim=-1)
        return self.head(combined)


# ---------------------------------------------------------------------------
# 7. CONCEPT TABLE  (same 3D motion signatures as gcsn_3d.py)
# ---------------------------------------------------------------------------

def build_concepts() -> Dict[str, Tuple[torch.Tensor, List[List[str]]]]:
    sem_dim = 32
    concepts = {}

    def emb():
        return torch.randn(sem_dim)

    signatures: Dict[str, List[List[str]]] = {
        "circle": [["G1", "X10", "Y0", "Z5", "F600"], ["G2", "X10", "Y0", "I-10", "J0", "F600"]],
        "stability": [["G1", "X18", "Y0", "Z0", "F300"], ["G1", "X18", "Y1", "Z0", "F300"]],
        "oscillation": [["G1", "X5", "Y5", "Z3", "F1200"], ["G1", "X10", "Y-5", "Z-3", "F1200"],
                         ["G1", "X15", "Y5", "Z3", "F1200"], ["G1", "X20", "Y-5", "Z-3", "F1200"]],
        "conflict": [["G1", "X3", "Y8", "Z6", "F2000"], ["G1", "X-4", "Y2", "Z-5", "F2000"],
                     ["G1", "X6", "Y-3", "Z4", "F2000"], ["G1", "X-2", "Y7", "Z-6", "F2000"]],
        "growth": [["G1", "X2", "Y0", "Z0", "F600"], ["G2", "X0", "Y2", "Z2", "I-2", "J0", "F600"],
                   ["G2", "X-4", "Y0", "Z4", "I0", "J-2", "F600"], ["G2", "X0", "Y-4", "Z6", "I4", "J0", "F600"],
                   ["G2", "X8", "Y0", "Z8", "I0", "J4", "F600"]],
        "convergence": [["G1", "X10", "Y0", "Z10", "F300"], ["G2", "X0", "Y5", "Z6", "I-10", "J0", "F300"],
                         ["G2", "X-2", "Y0", "Z3", "I2", "J-5", "F300"], ["G2", "X0", "Y0", "Z0", "I2", "J0", "F300"]],
        "divergence": [["G1", "X1", "Y0", "Z0", "F1200"], ["G2", "X0", "Y2", "Z3", "I-1", "J0", "F1200"],
                        ["G2", "X4", "Y0", "Z6", "I0", "J-2", "F1200"], ["G2", "X0", "Y-8", "Z9", "I-4", "J0", "F1200"]],
        "flow": [["G2", "X8", "Y0", "Z2", "I4", "J0", "F600"], ["G3", "X16", "Y0", "Z4", "I4", "J0", "F600"]],
        "boundary": [["G1", "X10", "Y0", "Z0", "F300"], ["G1", "X10", "Y10", "Z3", "F300"],
                     ["G1", "X0", "Y10", "Z0", "F300"], ["G1", "X0", "Y0", "Z3", "F300"]],
        "explosion": [["G0", "X8", "Y0", "Z0", "F2000"], ["G0", "X0", "Y0", "Z0", "F2000"],
                      ["G0", "X-8", "Y0", "Z4", "F2000"], ["G0", "X0", "Y0", "Z0", "F2000"],
                      ["G0", "X0", "Y8", "Z-4", "F2000"], ["G0", "X0", "Y0", "Z0", "F2000"],
                      ["G0", "X0", "Y-8", "Z8", "F2000"]],
    }
    for name, blocks in signatures.items():
        concepts[name] = (emb(), blocks)
    return concepts


def flatten_blocks(tok, blocks):
    ids = [tok.BOS]
    for b in blocks:
        ids.extend(tok.encode_block(b)); ids.append(tok.EOB)
    ids.append(tok.EOS)
    return ids

def pad_batch(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# 8. TRAINING
# ---------------------------------------------------------------------------

def train_decoder(decoder, tok, concepts, epochs=500, lr=3e-4, device="cpu"):
    decoder.to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    names = list(concepts.keys())
    sem_batch = torch.stack([concepts[n][0] for n in names]).to(device)
    target_ids = [flatten_blocks(tok, concepts[n][1]) for n in names]
    padded = pad_batch(target_ids, tok.PAD).to(device)
    inp, tgt = padded[:, :-1], padded[:, 1:]
    decoder.train()
    print("Stage 1 — training decoder (semantic vector -> G-code tokens)")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        logits = decoder(sem_batch, inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=tok.PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        if ep % 100 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}")
    return decoder


def train_textured_pointnet(model: TexturedPointNetEncoder, decoder, tok, concepts,
                             textures: Dict[str, torch.Tensor], epochs=300, lr=1e-3,
                             device="cpu", n_points=256):
    model.to(device)
    interp = GCodeInterpreter3D()
    names = list(concepts.keys())

    colored_clouds, texture_batch = [], []
    for name in names:
        sem_vec, _ = concepts[name]
        ids = decoder.generate(sem_vec.unsqueeze(0).to(device), tok.BOS, tok.EOS, temperature=0.3)
        tokens = tok.decode(ids)
        path = interp.run(tokens)
        xyz = densify_to_point_cloud(path, n_points)
        texture = textures[name]
        rgb = sample_texture_colors(xyz, texture)
        colored_clouds.append(torch.cat([xyz, rgb], dim=-1))     # (N, 6)
        texture_batch.append(texture)

    cloud_batch = torch.stack(colored_clouds).to(device)          # (C, N, 6)
    tex_batch = torch.stack(texture_batch).to(device)              # (C, 3, H, W)
    sem_batch = torch.stack([concepts[n][0] for n in names]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print("\nStage 2 — training textured PointNet (colored point cloud + image -> grounded vector)")
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        pred = model(cloud_batch, tex_batch)
        mse = F.mse_loss(pred, sem_batch)
        cos = 1 - F.cosine_similarity(pred, sem_batch, dim=-1).mean()
        loss = mse + 0.5 * cos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  mse {mse.item():.4f}  cos_dist {cos.item():.4f}")
    return model, cloud_batch, tex_batch, sem_batch, names


# ---------------------------------------------------------------------------
# 9. ABLATION: does the texture channel actually carry information the
#    geometry alone doesn't? Zero out RGB and the image embedding input and
#    compare round-trip fidelity. This is the honest check for this stage.
# ---------------------------------------------------------------------------

def evaluate(model, cloud_batch, tex_batch, sem_batch, names, concepts):
    model.eval()

    def cosine(a, b):
        d = a.norm() * b.norm()
        return float((a @ b) / d) if d > 1e-9 else 0.0

    with torch.no_grad():
        pred_full = model(cloud_batch, tex_batch)

        # ablation: geometry only (rgb zeroed, texture image zeroed)
        cloud_geo_only = cloud_batch.clone()
        cloud_geo_only[..., 3:] = 0.0
        tex_zero = torch.zeros_like(tex_batch)
        pred_geo_only = model(cloud_geo_only, tex_zero)

    print("\n--- Per-concept results (full texture+geometry vs. geometry-only ablation) ---")
    for idx, name in enumerate(names):
        sem_vec = concepts[name][0]
        full_sim = cosine(pred_full[idx], sem_vec)
        geo_sim = cosine(pred_geo_only[idx], sem_vec)
        print(f"  {name:12s}  full round-trip sim: {full_sim:.3f}   geometry-only sim: {geo_sim:.3f}")

    def pairwise_corr(vecs):
        sims = []
        sem_sims = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sims.append(cosine(vecs[i], vecs[j]))
                sem_sims.append(cosine(concepts[a][0], concepts[b][0]))
        st, vt = torch.tensor(sem_sims), torch.tensor(sims)
        return float(torch.corrcoef(torch.stack([st, vt]))[0, 1]) if st.std() > 1e-6 else float("nan")

    corr_full = pairwise_corr(pred_full)
    corr_geo = pairwise_corr(pred_geo_only)
    print(f"\nSemantic-vs-grounded correlation, full (geometry+texture): {corr_full:.3f}")
    print(f"Semantic-vs-grounded correlation, geometry-only ablation:   {corr_geo:.3f}")
    print("If these two numbers are close, the texture channel isn't doing real work yet —")
    print("expected at this dataset size (10 concepts), see self-critique.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GCodeTokenizer()
    concepts = build_concepts()
    textures = build_texture_library()
    sem_dim = 32

    print(f"Vocab size: {tok.vocab_size} | Concepts: {len(concepts)} | Textures: {len(textures)} | Device: {device}")

    decoder = GCodeDecoder(vocab_size=tok.vocab_size, sem_dim=sem_dim, pad_id=tok.PAD)
    train_decoder(decoder, tok, concepts, epochs=500, device=device)

    model = TexturedPointNetEncoder(sem_dim=sem_dim, img_embed_dim=32)
    model, cloud_batch, tex_batch, sem_batch, names = train_textured_pointnet(
        model, decoder, tok, concepts, textures, epochs=300, device=device
    )

    evaluate(model, cloud_batch, tex_batch, sem_batch, names, concepts)
