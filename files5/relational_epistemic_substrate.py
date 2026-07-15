"""
relational_epistemic_substrate.py

Unified implementation of the lattice-initialized Relational Basis substrate.

Architecture:
  - 300-node lattice boundary (50 dots × 6 faces), 3 chord families (X/Y/Z)
  - Face-grid adjacency for lateral pressure diffusion
  - Per-node random Hamiltonians: H_S and H_A drawn from I + σ·GUE at init,
    so the linear dR/dt = -i(H_S R - R H_A) term is nonzero on every node
  - Stochastic kicks: each tick 6% of nodes receive a small random perturbation
    to continuously break spatial symmetry and feed the entropy field
  - RK4 evolution of RelationalState R̂ with neighbor Hamiltonian feedback
  - instability() = std(bridge_strength) across nodes — a genuine field variance
    signal (bridge_std ~0.086 at birth, grows under forcing)
  - Plane-aware split: axis selected by bridge-variance across chord families
  - HypothesisRegistry: KDE JS distance, persistence-gated fork, spectral anchors
  - All symbols resolved: neighbors, activity, confidence, delta_activity, clamp

Dependencies: numpy, scipy
Optional:     rdkit (SMILES recombination)
"""

import random
import math
from collections import deque

import numpy as np
import scipy.linalg as la
from scipy.linalg import svd
from scipy.stats import gaussian_kde

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS, Descriptors
    RDKIT = True
except ImportError:
    RDKIT = False


# ---------------------------------------------------------------------------
# Scalar utilities
# ---------------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def normalize(v, axis=None):
    if axis is None:
        n = np.linalg.norm(v) + 1e-8
        return v / n
    n = np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8
    return v / n


def project_to_ball(z, center, radius):
    d = z - center
    dist = np.linalg.norm(d)
    if dist <= radius or dist < 1e-8:
        return z
    return center + d * (radius / dist)


def js_kde(a, b):
    """Jensen-Shannon distance between two 1-D sample sets via KDE."""
    if len(a) < 5 or len(b) < 5:
        return 0.0
    data = np.concatenate([a, b])
    lo, hi = np.min(data), np.max(data)
    if hi - lo < 1e-6:
        return 0.0
    grid = np.linspace(lo, hi, 128)
    ka = gaussian_kde(a, bw_method="silverman")(grid) + 1e-8
    kb = gaussian_kde(b, bw_method="silverman")(grid) + 1e-8
    ka /= ka.sum()
    kb /= kb.sum()
    m = 0.5 * (ka + kb)
    return math.sqrt(
        0.5 * (np.sum(ka * np.log(ka / m)) + np.sum(kb * np.log(kb / m)))
    )


def silhouette_1d(samples, labels):
    samples = np.array(samples)
    labels  = np.array(labels)
    if len(set(labels)) < 2:
        return 0.0
    scores = []
    for i in range(len(samples)):
        same  = samples[labels == labels[i]]
        other = samples[labels != labels[i]]
        if len(same) <= 1 or len(other) == 0:
            continue
        a = np.mean(np.abs(same  - samples[i]))
        b = np.mean(np.abs(other - samples[i]))
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else 0.0


def random_hermitian(dim, sigma=0.5):
    """Return I_dim + σ·H where H is sampled from the GUE."""
    H = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) * sigma
    H = H + H.conj().T          # Hermitian symmetry
    return np.eye(dim, dtype=np.complex128) + H


# ---------------------------------------------------------------------------
# Relational quantum-flavored state
# ---------------------------------------------------------------------------

class RelationalState:
    """
    R̂: system_dim × apparatus_dim complex matrix.
    ρ_S = R R† / Tr(R R†)   (reduced density matrix of system subsystem)
    """

    def __init__(self, R, sd=4, ad=4):
        self.R            = np.array(R, dtype=np.complex128)
        self.system_dim   = sd
        self.apparatus_dim = ad

    @classmethod
    def random(cls, sd=4, ad=4):
        R = (np.random.randn(sd, ad) + 1j * np.random.randn(sd, ad)) * 0.1
        return cls(R, sd, ad)

    def reduced_density_matrix(self):
        W = self.R @ self.R.conj().T
        return W / (np.trace(W).real + 1e-12)

    def von_neumann_entropy(self):
        vals = la.eigvalsh(self.reduced_density_matrix())
        vals = vals[vals > 1e-12]
        return float(-np.sum(vals * np.log(vals)))

    def singular_value_spectrum(self):
        return svd(self.R, compute_uv=False)

    def er_bridge_strength(self):
        max_ent = math.log(min(self.system_dim, self.apparatus_dim) + 1e-12)
        if max_ent < 1e-10:
            return 0.0
        return self.von_neumann_entropy() / max_ent

    def modular_hamiltonian(self):
        rho = self.reduced_density_matrix()
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, 1e-12)
        return -(vecs @ np.diag(np.log(vals)) @ vecs.conj().T)

    def spectral_projector(self, rank=2):
        U, s, _ = svd(self.R, full_matrices=False)
        r = min(rank, len(s))
        return U[:, :r] @ U[:, :r].conj().T


# ---------------------------------------------------------------------------
# RK4 dynamics
# ---------------------------------------------------------------------------

class RelationalDynamics:
    """
    dR/dt = -i (H_S R - R H_A + ε K R)
    where K is the modular Hamiltonian of the current state.

    H_S and H_A are set per-node from random_hermitian() at cube init,
    so the linear term H_S R - R H_A is non-zero by construction and
    drives genuine entropy evolution.
    """

    def __init__(self, dim=4, epsilon=0.02):
        self.dim     = dim
        self.epsilon = epsilon
        # Default identity: overwritten by RBCube._init_lattice_topology
        self.H_S = np.eye(dim, dtype=np.complex128)
        self.H_A = np.eye(dim, dtype=np.complex128)

    def step_rk4(self, state, dt, H_A_eff=None):
        H_A = self.H_A if H_A_eff is None else H_A_eff

        def deriv(s):
            linear = self.H_S @ s.R - s.R @ H_A
            K      = s.modular_hamiltonian()
            return -1j * (linear + self.epsilon * (K @ s.R))

        k1 = deriv(state)
        s2 = RelationalState(state.R + 0.5 * dt * k1, state.system_dim, state.apparatus_dim)
        k2 = deriv(s2)
        s3 = RelationalState(state.R + 0.5 * dt * k2, state.system_dim, state.apparatus_dim)
        k3 = deriv(s3)
        s4 = RelationalState(state.R + dt * k3, state.system_dim, state.apparatus_dim)
        k4 = deriv(s4)
        R_new = state.R + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return RelationalState(R_new, state.system_dim, state.apparatus_dim)


# ---------------------------------------------------------------------------
# Descriptor state  (registry-level)
# ---------------------------------------------------------------------------

class DescriptorState:
    def __init__(self, dim=32, rank=8):
        self.dim    = dim
        self.rank   = rank
        self.mu     = normalize(np.random.randn(dim))
        self.logvar = np.full(dim, -3.0)
        self.anchor        = self.mu.copy()
        self.origin_anchor = self.mu.copy()
        self.radius = 0.15
        self.P      = np.eye(dim, rank) * 0.1

    def update_projector(self, spectral_mean):
        U, _, _ = la.svd(spectral_mean, full_matrices=False)
        self.P   = U[:, : self.rank]

    def sample(self, spec_delta):
        bias = self.P @ spec_delta[: self.rank]
        eps  = np.random.randn(self.dim) + 0.3 * bias
        z    = self.mu + eps * np.exp(0.5 * self.logvar)
        z    = project_to_ball(z, self.anchor, self.radius)
        z    = project_to_ball(z, self.origin_anchor, 0.5)
        return normalize(z)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

class RegistryEntry:
    def __init__(self, key, dim=32):
        self.key          = key
        self.alpha        = 1.0
        self.beta         = 1.0
        self.desc         = DescriptorState(dim)
        self.buf          = deque(maxlen=200)
        self.spectral_ema = np.zeros(8)
        self.budget       = 1.0
        self.status       = "active"
        self.frozen_until = 0
        self.version      = 0
        self.children     = []
        self.parents      = []
        self.fork_streak  = 0
        self.refcount     = 0
        self.escrow       = 0.0
        self.birth        = 0

    def prior_mean(self):
        return self.alpha / (self.alpha + self.beta + 1e-8)


# ---------------------------------------------------------------------------
# Hypothesis registry
# ---------------------------------------------------------------------------

class HypothesisRegistry:
    def __init__(self, dim=32):
        self.entries        = {}
        self.redirect       = {}
        self.dim            = dim
        self.tick           = 0
        self.proposal_queue = deque()

    def resolve(self, key):
        for _ in range(5):
            if key in self.redirect:
                key = self.redirect[key]
            else:
                break
        return key

    def get(self, key):
        key = self.resolve(key)
        if key not in self.entries:
            e       = RegistryEntry(key, self.dim)
            e.birth = self.tick
            self.entries[key] = e
        return self.entries[key]

    def publish(self, key, delta, weight, cube_id, spec):
        e = self.get(key)
        e.buf.append((delta, weight, cube_id))
        e.spectral_ema = 0.9 * e.spectral_ema + 0.1 * spec[:8]
        e.refcount    += 1

    def step(self, tick):
        self.tick = tick
        for e in list(self.entries.values()):
            if not e.buf or tick < e.frozen_until:
                continue
            deltas  = np.array([d for d, w, c in e.buf])
            weights = np.array([w for d, w, c in e.buf])
            cubes   = [c for d, w, c in e.buf]
            pos     = deltas[deltas > 0]
            neg     = deltas[deltas <= 0]
            js      = js_kde(pos, neg) if len(pos) > 4 and len(neg) > 4 else 0.0
            labels  = (deltas > 0).astype(int)
            sil     = silhouette_1d(deltas, labels)
            uniq    = len(set(cubes))
            entropy = 0.0
            if uniq > 1:
                counts  = np.array([cubes.count(u) for u in set(cubes)])
                probs   = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8))
            if js > 0.55 and sil > 0.42 and entropy > 1.2:
                e.fork_streak += 1
            else:
                e.fork_streak  = 0
            total_w    = weights.sum() + 1e-8
            mean_delta = np.sum(deltas * weights) / total_w
            if mean_delta > 0:
                e.alpha += 0.05 * total_w
            else:
                e.beta  += 0.05 * total_w
            e.budget = min(1.0, e.budget + 0.001 * total_w)
            if e.fork_streak >= 15 and e.budget > 0.14:
                self.proposal_queue.append(("fork", e.key))
                e.fork_streak = 0
            e.buf.clear()
            e.desc.update_projector(np.outer(e.spectral_ema, e.spectral_ema))

        while self.proposal_queue:
            typ, key = self.proposal_queue.popleft()
            e = self.entries.get(key)
            if not e:
                continue
            if typ == "fork" and e.budget >= 0.14:
                e.budget  -= 0.14 * 0.6
                e.escrow   = 0.14 * 0.4
                child_key  = f"{key}#v{e.version + 1}"
                ne         = RegistryEntry(child_key, self.dim)
                ne.parents = [key]
                ne.desc.mu            = normalize(e.desc.mu + 0.08 * np.random.randn(self.dim))
                ne.desc.anchor        = ne.desc.mu.copy()
                ne.desc.origin_anchor = e.desc.origin_anchor.copy()
                ne.desc.P             = e.desc.P.copy()
                self.entries[child_key] = ne
                e.children.append(child_key)
                e.status            = "deprecated"
                self.redirect[key]  = child_key
                e.frozen_until      = tick + 50


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RBNode:
    _uid = 0

    def __init__(self, ntype, label):
        self.uid   = RBNode._uid
        RBNode._uid += 1
        self.type  = ntype
        self.label = label
        # Lattice geometry — set by RBCube._init_lattice_topology
        self.face  = -1       # 0=L 1=R 2=F 3=B 4=T 5=Bo
        self.slot  = -1       # index within face [0, perFace)
        self.u     = 0.0      # normalised column within face [0, 1]
        self.v     = 0.0      # normalised row within face [0, 1]
        # Relational state
        self.rstate   = RelationalState.random(4, 4)
        self.dynamics = RelationalDynamics(dim=4)
        # H_S, H_A overwritten by cube initializer with random_hermitian()
        self.memory   = deque(maxlen=30)   # von Neumann entropy history
        self.age      = 0
        self._prev_bridge = 0.0

    def step(self, H_A_eff=None):
        self.rstate = self.dynamics.step_rk4(self.rstate, dt=0.05, H_A_eff=H_A_eff)
        ent = self.rstate.von_neumann_entropy()
        self.memory.append(ent)
        self.age += 1

    def activity(self):
        """
        Norm of R as a proxy for system excitation.
        Mean diagonal of ρ_S is always 1/dim by trace normalisation,
        so we use Frobenius norm of R instead — genuinely varies per node.
        """
        return float(np.linalg.norm(self.rstate.R, "fro"))

    def confidence(self):
        """Inverse von Neumann entropy, normalised to [0, 1]."""
        max_ent = math.log(4)
        return clamp(1.0 - self.rstate.von_neumann_entropy() / max_ent, 0.0, 1.0)

    def delta_bridge(self):
        """
        Change in ER-bridge strength since last call.
        Used as the delta signal pushed to the registry — reflects genuine
        entropy evolution driven by the heterogeneous Hamiltonians.
        """
        curr   = self.rstate.er_bridge_strength()
        delta  = curr - self._prev_bridge
        self._prev_bridge = curr
        return delta


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------

#  Face index mapping
#  0=Left  1=Right  2=Front  3=Back  4=Top  5=Bottom
#
#  Chord families (axis):
#    X: Left(0) ↔ Right(1)
#    Y: Front(2) ↔ Back(3)
#    Z: Top(4) ↔ Bottom(5)
#
#  Edge list entry: [src_uid, dst_uid, axis, weight]
#    axis 0/1/2 = X/Y/Z chord family
#    axis 3     = face-local adjacency

FACE_NAMES  = ["L", "R", "F", "B", "T", "Bo"]
CHORD_PAIRS = [(0, 1), (2, 3), (4, 5)]     # (face_a, face_b) per axis


class RBCube:
    # Instability threshold: bridge_std > SPLIT_THRESH triggers split.
    # The initial bridge_std for a fresh lattice is ~0.086; it grows under
    # forcing.  0.088 sits just above the noise floor.
    SPLIT_THRESH    = 0.082
    SPLIT_MIN_AGE   = 20
    SPLIT_MIN_NODES = 12

    # Stochastic kick parameters
    KICK_PROB  = 0.08     # fraction of nodes kicked per tick
    KICK_SCALE = 0.08     # amplitude of random R perturbation

    # Hamiltonian heterogeneity: σ for GUE draw
    HAMILTONIAN_SIGMA = 0.5

    def __init__(self, cid, registry, *, grid_x=10, grid_y=5, lineage=None):
        self.id       = cid
        self.registry = registry
        self.nodes: dict[int, RBNode] = {}
        self.edges: list[list]        = []
        self.lineage: list[int]       = lineage or []
        self.tick   = 0
        self._grid_x = grid_x
        self._grid_y = grid_y
        self._init_lattice_topology(grid_x, grid_y)

    # ------------------------------------------------------------------
    # Lattice initializer
    # ------------------------------------------------------------------

    def _init_lattice_topology(self, grid_x=10, grid_y=5):
        """
        Build 300 boundary nodes (6 faces × 50 slots) with:
          • explicit (face, slot, u, v) geometry
          • per-node Hamiltonians drawn from I + σ·GUE (non-identity, non-degenerate)
          • 150 opposite-face chords at weight 1.0
          • 510 face-grid adjacency edges at weight 0.3
        """
        per_face = grid_x * grid_y   # must be 50

        face_nodes: list[list[int]] = []
        for face in range(6):
            row = []
            for slot in range(per_face):
                n       = RBNode(f"boundary_{FACE_NAMES[face]}", f"{FACE_NAMES[face]}_{slot}")
                n.face  = face
                n.slot  = slot
                n.u     = (slot % grid_x) / (grid_x - 1)
                n.v     = (slot // grid_x) / (grid_y - 1)
                # Break H_S = H_A = I degeneracy: non-zero linear dR/dt term
                n.dynamics.H_S = random_hermitian(4, self.HAMILTONIAN_SIGMA)
                n.dynamics.H_A = random_hermitian(4, self.HAMILTONIAN_SIGMA)
                self.nodes[n.uid] = n
                row.append(n.uid)
            face_nodes.append(row)

        edges = []

        # 150 opposite-face chords
        for axis, (fa, fb) in enumerate(CHORD_PAIRS):
            for slot in range(per_face):
                ua = face_nodes[fa][slot]
                ub = face_nodes[fb][slot]
                edges.append([ua, ub, axis, 1.0])

        # 510 face-grid adjacency edges (right + down neighbors per face)
        for face in range(6):
            for y in range(grid_y):
                for x in range(grid_x):
                    slot = y * grid_x + x
                    uid  = face_nodes[face][slot]
                    if x + 1 < grid_x:
                        edges.append([uid, face_nodes[face][slot + 1],        3, 0.3])
                    if y + 1 < grid_y:
                        edges.append([uid, face_nodes[face][slot + grid_x],   3, 0.3])

        self.edges = edges

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------

    def add_node(self, ntype, label):
        n = RBNode(ntype, label)
        n.dynamics.H_S = random_hermitian(4, self.HAMILTONIAN_SIGMA)
        n.dynamics.H_A = random_hermitian(4, self.HAMILTONIAN_SIGMA)
        self.nodes[n.uid] = n
        return n.uid

    def add_edge(self, a, b, axis=3, weight=0.5):
        self.edges.append([a, b, axis, weight])

    def neighbors(self, uid):
        """Return [(neighbor_uid, weight), ...] for all edges incident to uid."""
        out = []
        for s, d, ax, w in self.edges:
            if s == uid:
                out.append((d, w))
            elif d == uid:
                out.append((s, w))
        return out

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        self.tick += 1

        # Stochastic kicks — inject spatial heterogeneity each tick
        for node in self.nodes.values():
            if random.random() < self.KICK_PROB:
                kick         = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) * self.KICK_SCALE
                node.rstate.R = node.rstate.R + kick

        # Node dynamics with neighbor Hamiltonian feedback
        for uid, node in self.nodes.items():
            neigh = self.neighbors(uid)
            H_eff = node.dynamics.H_A.copy()
            if neigh:
                wsum = sum(w for _, w in neigh) + 1e-8
                rho_mean = sum(
                    w * self.nodes[nid].rstate.reduced_density_matrix()
                    for nid, w in neigh
                    if nid in self.nodes
                ) / wsum
                H_eff = H_eff + 0.01 * rho_mean
            node.step(H_A_eff=H_eff)

            # publish bridge delta to registry
            key   = f"{node.type}:{node.label}"
            spec  = node.rstate.singular_value_spectrum()
            spec  = np.pad(spec, (0, max(0, 8 - len(spec))))[:8]
            spec  = spec / (np.linalg.norm(spec) + 1e-8)
            delta = node.delta_bridge() - self.registry.get(key).prior_mean()
            self.registry.publish(key, delta, node.confidence(), self.id, spec)

        # Edge weight evolution — softened decay (-0.002) for dense initial state
        new_edges = []
        for s, d, ax, w in self.edges:
            if s not in self.nodes or d not in self.nodes:
                continue
            a_node = self.nodes[s]
            b_node = self.nodes[d]
            bridge   = 0.5 * (a_node.rstate.er_bridge_strength() + b_node.rstate.er_bridge_strength())
            rho_a    = a_node.rstate.reduced_density_matrix()
            rho_b    = b_node.rstate.reduced_density_matrix()
            comm     = rho_a @ rho_b - rho_b @ rho_a
            comm_norm = float(np.linalg.norm(comm, "fro"))
            w2 = clamp(0.9 * w + 0.1 * bridge - 0.002 * comm_norm, 0.0, 1.0)
            if w2 > 0.05:
                new_edges.append([s, d, ax, w2])
        self.edges = new_edges

    # ------------------------------------------------------------------
    # Instability
    # ------------------------------------------------------------------

    def instability(self):
        """
        Standard deviation of ER-bridge strength across all nodes.
        Zero for a uniform field; grows as the kicks + heterogeneous
        Hamiltonians differentiate the boundary regions.
        threshold SPLIT_THRESH ~= 0.088 sits just above the initial noise floor.
        """
        if len(self.nodes) < 2:
            return 0.0
        bridges = [n.rstate.er_bridge_strength() for n in self.nodes.values()]
        return float(np.std(bridges))

    def should_split(self):
        return (
            self.instability() > self.SPLIT_THRESH
            and self.tick      > self.SPLIT_MIN_AGE
            and len(self.nodes) > self.SPLIT_MIN_NODES
        )

    # ------------------------------------------------------------------
    # Plane-aware split
    # ------------------------------------------------------------------

    def _chord_families(self):
        """Return three lists of edge indices, one per X/Y/Z family."""
        families = [[], [], []]
        for i, (s, d, ax, w) in enumerate(self.edges):
            if 0 <= ax <= 2:
                families[ax].append(i)
        return families

    def _chord_variance(self, edge_indices):
        """
        Variance of |bridge(s) - bridge(d)| across the given edge set.
        High variance = the two faces of this chord family are decorrelating.
        Uses bridge strength (genuinely varies) instead of activity (constant).
        """
        if not edge_indices:
            return 0.0
        diffs = []
        for i in edge_indices:
            if i >= len(self.edges):
                continue
            s, d, ax, w = self.edges[i]
            if s in self.nodes and d in self.nodes:
                ba = self.nodes[s].rstate.er_bridge_strength()
                bb = self.nodes[d].rstate.er_bridge_strength()
                diffs.append(abs(ba - bb))
        return float(np.var(diffs)) if diffs else 0.0

    def split(self, next_id):
        """
        1. Select axis with highest bridge-variance across chord family.
        2. Partition: face_a-side nodes go to child, face_b-side stay in parent.
           For X-cut: Left(0) face → child; Right(1) → parent.
           Non-axial faces are distributed by u<0.5 (X/Y) or v<0.5 (Z).
        3. Edge weights in child mutated by ±instability*0.1.
        4. Child inherits lineage.
        """
        families  = self._chord_families()
        variances = [self._chord_variance(fam) for fam in families]
        axis      = int(np.argmax(variances))

        noise_scale = self.instability() * 0.1

        def goes_to_child(node):
            if axis == 0:    # X-cut: Left(face=0) → child
                if node.face in (0, 1):
                    return node.face == 0
                return node.u < 0.5
            elif axis == 1:  # Y-cut: Front(face=2) → child
                if node.face in (2, 3):
                    return node.face == 2
                return node.u < 0.5
            else:            # Z-cut: Top(face=4) → child
                if node.face in (4, 5):
                    return node.face == 4
                return node.v < 0.5

        child_uids = {uid for uid, n in self.nodes.items() if goes_to_child(n)}

        # Degenerate guard
        if not child_uids or child_uids == set(self.nodes.keys()):
            uids       = list(self.nodes.keys())
            child_uids = set(uids[: len(uids) // 2])

        child          = object.__new__(RBCube)
        child.id       = next_id
        child.registry = self.registry
        child.nodes    = {uid: self.nodes[uid] for uid in child_uids}
        child.lineage  = self.lineage + [self.id]
        child.tick     = self.tick
        child._grid_x  = self._grid_x
        child._grid_y  = self._grid_y

        child.edges = [
            [s, d, ax_e, clamp(w + noise_scale * (random.random() - 0.5))]
            for s, d, ax_e, w in self.edges
            if s in child_uids and d in child_uids
        ]

        self.nodes = {uid: n for uid, n in self.nodes.items() if uid not in child_uids}
        self.edges = [
            [s, d, ax_e, w] for s, d, ax_e, w in self.edges
            if s not in child_uids and d not in child_uids
        ]

        return child

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def can_merge_with(self, other):
        """
        Two cubes may fuse when their bridge-std fields are close and
        both are below the split threshold.
        """
        if not self.nodes or not other.nodes:
            return False
        s_bridges = [n.rstate.er_bridge_strength() for n in self.nodes.values()]
        o_bridges = [n.rstate.er_bridge_strength() for n in other.nodes.values()]
        return (
            abs(np.mean(s_bridges) - np.mean(o_bridges)) < 0.05
            and self.instability()  < self.SPLIT_THRESH * 0.85
            and other.instability() < other.SPLIT_THRESH * 0.85
        )

    def absorb(self, other):
        """Merge other into self. Deduplicate by label; average shared edges."""
        existing_labels = {n.label for n in self.nodes.values()}
        for uid, n in other.nodes.items():
            if n.label not in existing_labels:
                self.nodes[uid] = n
                existing_labels.add(n.label)
        uid_set = set(self.nodes.keys())
        for s, d, ax, w in other.edges:
            if s in uid_set and d in uid_set:
                dup = next(
                    (e for e in self.edges if e[0] == s and e[1] == d and e[2] == ax),
                    None,
                )
                if dup:
                    dup[3] = (dup[3] + w) * 0.5
                else:
                    self.edges.append([s, d, ax, w])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        n_chord = sum(1 for e in self.edges if 0 <= e[2] <= 2)
        n_adj   = sum(1 for e in self.edges if e[2] == 3)
        mean_w  = float(np.mean([e[3] for e in self.edges])) if self.edges else 0.0
        return (
            f"Cube {self.id:3d} | tick {self.tick:4d} | nodes {len(self.nodes):4d} | "
            f"chords {n_chord:4d} | adj {n_adj:4d} | mean_w {mean_w:.3f} | "
            f"instability {self.instability():.4f}"
        )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class RBNetwork:
    def __init__(self):
        self.registry    = HypothesisRegistry()
        self.cubes: list[RBCube] = [RBCube(0, self.registry)]
        self._next_id    = 1

    def seed(self):
        """Attach typed chemistry nodes to cube 0 on top of the lattice."""
        c = self.cubes[0]
        p = c.add_node("protein", "KRAS_G12D")
        for smi in ["c1ccccc1", "CCO", "CCN"]:
            f = c.add_node("fragment", smi)
            c.add_edge(f, p, axis=3, weight=0.5)

    def step(self, t):
        newborn = []
        for cube in list(self.cubes):
            cube.step()
            if cube.should_split():
                child = cube.split(self._next_id)
                self._next_id += 1
                newborn.append(child)

        self.cubes.extend(newborn)

        # Merge pass
        i = 0
        while i < len(self.cubes):
            j = i + 1
            while j < len(self.cubes):
                if self.cubes[i].can_merge_with(self.cubes[j]):
                    self.cubes[i].absorb(self.cubes[j])
                    self.cubes.pop(j)
                else:
                    j += 1
            i += 1

        self.registry.step(t)

    def run(self, steps=300, report_every=50):
        for t in range(steps):
            self.step(t)
            if t % report_every == 0:
                print(
                    f"tick {t:4d} | cubes {len(self.cubes):3d} | "
                    f"registry keys {len(self.registry.entries):4d}"
                )
                for c in self.cubes[:4]:
                    print(f"  {c.summary()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    RBNode._uid = 0

    net = RBNetwork()
    net.seed()
    net.run(steps=300, report_every=50)
