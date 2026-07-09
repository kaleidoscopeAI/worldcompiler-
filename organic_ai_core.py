"""organic_ai_core.py — Deterministic organic/evolutionary AI substrate. v4.

v4 thesis: the *learning rule itself* is evolvable, and SGD is a provable
special case of the evolved rule — so evolution can only match or beat it, and
the difference is measurable. This is the "learning to learn" direction (#4 of
the research critique) done honestly: no new capability is asserted, it is
either gated or measured.

Each weight update is a local plasticity rule with three evolved coefficients:

    Δw = lr * [ g_grad  * (pre ⊗ -grad_post)     # gradient-descent term
              + g_hebb  * (pre ⊗  act_post)      # Hebbian correlation term
              - g_decay *  w * act_post²   ]      # Oja-style stabilizing decay

where for each layer:
    pre       = presynaptic activation      (x for layer 1, h for layer 2)
    grad_post = backprop error signal       (d_h / d_y)   -> credit assignment
    act_post  = raw postsynaptic activation (h / y)       -> target-free Hebb

Special cases the evolved space *contains*:
    (g_grad, g_hebb, g_decay) = (1, 0, 0)  ->  EXACT SGD  (bit-exact, gated)
    (0, b, 0)                              ->  pure Hebbian (unsupervised)
    (0, b, e)                              ->  Oja's rule (normalized Hebbian)

Because SGD is interior to the search space, the scientific question is sharp
and falsifiable: does evolution discover a *rule* (not just weights) that
learns a held-out data source better than SGD? v4 answers by measurement, not
assertion — the champion rule from a plasticity-evolving population and the
champion from an SGD-frozen control are both replayed from scratch on unseen
sources (the #8 "multiple worlds" idea, realized as a transfer test).

Retained from v1->v3:
  * numeric-only genome (no pickle/exec)          * order-independent RNG
  * closed energy economy (conservation gate)     * deterministic iteration
  * comparative/tournament reward (substrate-      * universal ingestion layer
    invariant; see World.step)                       (array/csv/json/wav/text/
  * verification gates before any trusted output     bytes/generator/callable)

Runtime: Python 3.9+, numpy + stdlib only. No pandas, torch, or network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A divergent evolved plasticity rule can blow weights up. Rather than crash
# the run, the offending node is scored with this large *finite* error so it
# starves and dies through the ordinary energy economy — selection culls it.
_DEGENERATE_ERROR: float = 1.0e3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrganicError(Exception):
    """Base error for the organic substrate."""


class GenomeError(OrganicError):
    """Raised when a genome violates its schema/bounds."""


class IngestError(OrganicError):
    """Raised when input data cannot be turned into a valid DataSource."""


# ---------------------------------------------------------------------------
# Deterministic, order-independent RNG
# ---------------------------------------------------------------------------


def derive_rng(*keys: object) -> np.random.Generator:
    """Return a Generator whose stream is a pure function of ``keys``.

    Population dynamics change the *order* in which stochastic decisions occur.
    One global seeded Generator would entangle each node's randomness with that
    order. Hashing logical coordinates (node id, tick, purpose) into a 256-bit
    seed makes every decision's randomness order-independent.
    """
    payload = "|".join(repr(k) for k in keys).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=32).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _stable_hash_unit(token: str) -> float:
    """Map an arbitrary string to a deterministic value in [-1, 1].

    Used to embed categorical/text cells as bounded numeric features. Same
    token => same value on every machine and every run (blake2b, not hash()).
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big")
    return (raw / float(2**64 - 1)) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Universal ingestion layer  (verified in v3; unchanged in v4)
# ---------------------------------------------------------------------------


class DataSource:
    """Contract every source must satisfy. Subclasses implement ``_observe``.

    ``observe`` wraps ``_observe`` with I1/I2 validation so *no* source can
    leak a malformed frame into the world, including user FunctionSources.
    Invariants: I1 shape==(dim,), I2 finite & in [-1,1], I3 pure in tick,
    I4 total for all tick>=0.
    """

    dim: int

    def _observe(self, tick: int) -> np.ndarray:
        raise NotImplementedError

    def observe(self, tick: int) -> np.ndarray:
        if tick < 0:
            raise IngestError(f"tick must be >= 0, got {tick}")
        frame = np.asarray(self._observe(tick), dtype=np.float64)
        if frame.shape != (self.dim,):
            raise IngestError(
                f"{type(self).__name__}: frame shape {frame.shape} != ({self.dim},)")
        if not np.all(np.isfinite(frame)):
            raise IngestError(f"{type(self).__name__}: non-finite value at tick {tick}")
        if float(np.max(np.abs(frame))) > 1.0 + 1e-12:
            raise IngestError(
                f"{type(self).__name__}: frame outside [-1,1] at tick {tick}")
        return np.clip(frame, -1.0, 1.0)

    def fingerprint(self) -> str:
        """Identity of the data stream: hashed sample of frames + shape. Folded
        into World.state_hash so two runs on different data cannot collide."""
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{type(self).__name__}|dim={self.dim}".encode())
        for t in (0, 1, 2, 3, 5, 8, 13, 21, 34, 55):
            h.update(self.observe(t).tobytes())
        return h.hexdigest()


class _AffineNormalizer:
    """Per-dimension affine map to [-1, 1], fit once from the full dataset.

    y = clip((x - center) / half_range, -1, 1). Constant dims map to exactly 0.
    Fit-once keeps normalization a pure function of the data — a running
    normalizer would make observe(t) depend on request order (breaks I3).
    """

    __slots__ = ("center", "inv_half")

    def __init__(self, data: np.ndarray) -> None:
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        self.center = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        self.inv_half = np.where(half < 1e-12, 0.0,
                                 1.0 / np.where(half < 1e-12, 1.0, half))

    def __call__(self, row: np.ndarray) -> np.ndarray:
        return np.clip((row - self.center) * self.inv_half, -1.0, 1.0)

    def params_bytes(self) -> bytes:
        return self.center.tobytes() + self.inv_half.tobytes()


class ArraySource(DataSource):
    """Any finite (T, D) numeric matrix — the terminal form of every offline
    ingester below. Finite datasets are made total (I4) by deterministic wrap:
    'loop' (t mod T) or 'reflect' (triangle-wave index).

    Default is 'loop', by measurement, not taste: reflect avoids the one seam
    discontinuity but *reverses temporal direction* every period, so any
    direction-sensitive structure a node has learned becomes anti-knowledge for
    half of every cycle — observed as best-node error collapsing 0.28 -> 0.03
    then exploding to 0.36 at the first reflection. Loop pays one bad
    prediction per period and preserves the arrow of time.
    """

    __slots__ = ("dim", "_data", "_length", "_wrap", "_norm")

    def __init__(self, data: np.ndarray, wrap: str = "loop",
                 nan_policy: str = "raise") -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise IngestError(f"expected 1D/2D data, got ndim={arr.ndim}")
        if arr.shape[0] < 2:
            raise IngestError(
                f"need >= 2 rows (nodes predict t+1 from t), got {arr.shape[0]}")
        if wrap not in ("loop", "reflect"):
            raise IngestError(f"wrap must be 'loop' or 'reflect', got {wrap!r}")

        bad = ~np.isfinite(arr)
        if bad.any():
            if nan_policy == "raise":
                raise IngestError(
                    f"{int(bad.sum())} non-finite cells; pass nan_policy='impute' "
                    f"to replace them with per-column medians of finite values")
            if nan_policy != "impute":
                raise IngestError(
                    f"nan_policy must be 'raise' or 'impute', got {nan_policy!r}")
            for col in range(arr.shape[1]):
                mask = bad[:, col]
                if mask.all():
                    arr[:, col] = 0.0
                elif mask.any():
                    arr[mask, col] = float(np.median(arr[~mask, col]))

        self._data = arr
        self._length = arr.shape[0]
        self.dim = arr.shape[1]
        self._wrap = wrap
        self._norm = _AffineNormalizer(arr)

    def _index(self, tick: int) -> int:
        n = self._length
        if self._wrap == "loop":
            return tick % n
        period = 2 * (n - 1)
        k = tick % period
        return k if k < n else period - k

    def _observe(self, tick: int) -> np.ndarray:
        return self._norm(self._data[self._index(tick)])

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(f"ArraySource|{self._data.shape}|{self._wrap}".encode())
        h.update(self._norm.params_bytes())
        h.update(self._data[:64].tobytes())
        h.update(self._data[-64:].tobytes())
        return h.hexdigest()


class SyntheticSource(DataSource):
    """A drifting multi-sinusoid. One source among peers; also the default."""

    __slots__ = ("dim", "_freqs", "_phase", "_drift")

    def __init__(self, seed: int, dim: int = 8) -> None:
        if dim < 1:
            raise IngestError(f"dim must be >= 1, got {dim}")
        rng = derive_rng("synthetic-source", seed, dim)
        self.dim = dim
        self._freqs = rng.uniform(0.05, 0.5, size=dim)
        self._phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        self._drift = rng.uniform(1e-4, 1e-3, size=dim)

    def _observe(self, tick: int) -> np.ndarray:
        t = float(tick)
        return np.tanh(np.sin(self._freqs * t + self._phase + self._drift * t))


class FunctionSource(DataSource):
    """Wraps a user callable f(tick) -> array-like of length dim. Purity (I3)
    cannot be proven for arbitrary code, so it is spot-checked at construction
    and range/shape re-validated on every observe."""

    __slots__ = ("dim", "_fn")

    def __init__(self, fn: Callable[[int], Sequence[float]], dim: int) -> None:
        if dim < 1:
            raise IngestError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self._fn = fn
        for t in (0, 1, 7):
            if not np.array_equal(self.observe(t), self.observe(t)):
                raise IngestError(
                    "FunctionSource purity spot-check FAILED: f(t) is not a pure "
                    "function of t (stateful, clock-based, or unseeded-random)")

    def _observe(self, tick: int) -> np.ndarray:
        return np.asarray(self._fn(tick), dtype=np.float64)


def _rows_to_matrix(rows: List[List[object]], context: str) -> np.ndarray:
    """Cells: numbers pass through; strings/None embed via _stable_hash_unit /
    NaN. Ragged rows are an error — silent padding would fabricate data."""
    if len(rows) < 2:
        raise IngestError(f"{context}: need >= 2 data rows, got {len(rows)}")
    width = len(rows[0])
    if width == 0:
        raise IngestError(f"{context}: rows are empty")
    out = np.empty((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) != width:
            raise IngestError(f"{context}: ragged row {i} (len {len(row)} != {width})")
        for j, cell in enumerate(row):
            if isinstance(cell, bool):
                out[i, j] = 1.0 if cell else -1.0
            elif isinstance(cell, (int, float)):
                out[i, j] = float(cell)
            elif cell is None:
                out[i, j] = np.nan
            else:
                text = str(cell).strip()
                if text == "":
                    out[i, j] = np.nan
                    continue
                try:
                    out[i, j] = float(text)
                except ValueError:
                    out[i, j] = _stable_hash_unit(text)
    return out


def load_csv(path: os.PathLike, wrap: str = "loop",
             nan_policy: str = "impute") -> ArraySource:
    """CSV -> source. Header detected by 'first row has no parseable float'.
    Non-numeric cells become deterministic categorical embeddings; blanks
    become NaN handled by nan_policy (default impute, since real CSVs leak
    blanks and a hard-fail default would reject most field data)."""
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not raw:
        raise IngestError(f"CSV {path}: no rows")

    def _numeric_count(row: List[str]) -> int:
        n = 0
        for c in row:
            try:
                float(c)
                n += 1
            except ValueError:
                pass
        return n

    rows = raw[1:] if (_numeric_count(raw[0]) == 0 and len(raw) > 1) else raw
    return ArraySource(_rows_to_matrix([list(r) for r in rows], f"CSV {path}"),
                       wrap=wrap, nan_policy=nan_policy)


def _flatten_record(obj: object, prefix: str, into: Dict[str, object]) -> None:
    """Dict records flatten to dotted scalar keys; non-scalar leaves collapse to
    a hash of their canonical JSON — lossy but deterministic and honest about
    being an identity rather than a value."""
    if isinstance(obj, dict):
        for k in sorted(obj):
            _flatten_record(obj[k], f"{prefix}{k}." if prefix else f"{k}.", into)
        return
    key = prefix[:-1] if prefix.endswith(".") else prefix
    if isinstance(obj, (int, float, bool)) or obj is None or isinstance(obj, str):
        into[key] = obj
    else:
        into[key] = json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _records_to_matrix(records: List[object], context: str) -> np.ndarray:
    if records and all(isinstance(r, dict) for r in records):
        flats: List[Dict[str, object]] = []
        keys: set = set()
        for rec in records:
            flat: Dict[str, object] = {}
            _flatten_record(rec, "", flat)
            flats.append(flat)
            keys |= set(flat)
        ordered = sorted(keys)
        rows = [[f.get(k) for k in ordered] for f in flats]
        return _rows_to_matrix(rows, context)
    if records and all(isinstance(r, (list, tuple)) for r in records):
        return _rows_to_matrix([list(r) for r in records], context)
    if records and all(isinstance(r, (int, float)) and not isinstance(r, bool)
                       for r in records):
        return np.asarray(records, dtype=np.float64)[:, None]
    raise IngestError(f"{context}: records must be all-dicts, all-lists, or all-numbers")


def load_json(path: os.PathLike, wrap: str = "loop",
              nan_policy: str = "impute") -> ArraySource:
    """JSON (top-level array) or JSONL/NDJSON (one record per line) -> source."""
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    try:
        if suffix in (".jsonl", ".ndjson"):
            records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        else:
            doc = json.loads(text)
            if not isinstance(doc, list):
                raise IngestError(f"JSON {path}: top level must be an array")
            records = doc
    except json.JSONDecodeError as e:
        raise IngestError(f"JSON {path}: parse failure: {e}") from e
    return ArraySource(_records_to_matrix(records, f"JSON {path}"),
                       wrap=wrap, nan_policy=nan_policy)


def load_wav(path: os.PathLike, dim: int = 8, wrap: str = "loop") -> ArraySource:
    """PCM WAV -> frames of `dim` consecutive mono samples. Supports 8/16/24/
    32-bit widths via explicit decode (no scipy). Multi-channel is averaged."""
    with wave.open(str(path), "rb") as w:
        n_ch, width, n_frames = (w.getnchannels(), w.getsampwidth(),
                                 w.getnframes())
        raw = w.readframes(n_frames)
    if width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        samples = (samples - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        samples = v.astype(np.float64) / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise IngestError(f"WAV {path}: unsupported sample width {width}")
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    usable = (len(samples) // dim) * dim
    if usable < 2 * dim:
        raise IngestError(f"WAV {path}: too short for dim={dim}")
    return ArraySource(samples[:usable].reshape(-1, dim), wrap=wrap)


def load_text(text: str, dim: int = 16, window: int = 64, stride: int = 16,
              wrap: str = "loop") -> ArraySource:
    """Raw text -> hashed-trigram feature windows (deterministic embedding).
    Each char trigram is blake2b-hashed to (bucket, sign), accumulated per
    window, L2-normalized, tanh-squashed. No vocabulary, no training."""
    if len(text) < window + stride:
        raise IngestError(f"text too short: {len(text)} chars < window+stride")
    frames: List[np.ndarray] = []
    for start in range(0, len(text) - window + 1, stride):
        chunk = text[start:start + window]
        vec = np.zeros(dim, dtype=np.float64)
        for i in range(len(chunk) - 2):
            d = hashlib.blake2b(chunk[i:i + 3].encode("utf-8"),
                                digest_size=8).digest()
            vec[int.from_bytes(d[:4], "big") % dim] += 1.0 if d[4] & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        frames.append(np.tanh(vec / norm) if norm > 0 else vec)
    return ArraySource(np.stack(frames), wrap=wrap)


def load_bytes(blob: bytes, dim: int = 8, wrap: str = "loop") -> ArraySource:
    """Any binary blob -> frames of `dim` bytes mapped to [-1, 1]."""
    usable = (len(blob) // dim) * dim
    if usable < 2 * dim:
        raise IngestError(f"blob too short for dim={dim}: {len(blob)} bytes")
    arr = np.frombuffer(blob[:usable], dtype=np.uint8).astype(np.float64)
    return ArraySource(arr.reshape(-1, dim) / 127.5 - 1.0, wrap=wrap)


def from_iterable(it: Iterable[Sequence[float]], wrap: str = "loop",
                  nan_policy: str = "raise",
                  max_rows: int = 1_000_000) -> ArraySource:
    """Materialize a (possibly one-shot) iterable/generator of rows. Replay is
    required by the substrate's determinism, so buffering is correctness, not
    convenience; max_rows bounds memory."""
    rows: List[List[float]] = []
    for i, row in enumerate(it):
        if i >= max_rows:
            raise IngestError(f"iterable exceeded max_rows={max_rows}")
        rows.append(list(np.atleast_1d(np.asarray(row, dtype=np.float64))))
    return ArraySource(np.asarray(rows), wrap=wrap, nan_policy=nan_policy)


_EXT_LOADERS: Dict[str, Callable[..., ArraySource]] = {
    ".csv": load_csv, ".tsv": load_csv,
    ".json": load_json, ".jsonl": load_json, ".ndjson": load_json,
    ".wav": load_wav,
}


def ingest(source: object, *, dim: Optional[int] = None, **kwargs) -> DataSource:
    """Universal front door. Dispatch by type, then by file extension. A raw
    string is strictly a path — "path or prose?" is a guess, and guesses are
    how wrong data enters a pipeline; raw text must use load_text explicitly."""
    if isinstance(source, DataSource):
        return source
    if isinstance(source, np.ndarray) or (
            isinstance(source, (list, tuple)) and source):
        return ArraySource(np.asarray(source, dtype=np.float64), **kwargs)
    if callable(source):
        if dim is None:
            raise IngestError("callable source requires dim=")
        return FunctionSource(source, dim)
    if isinstance(source, (bytes, bytearray)):
        return load_bytes(bytes(source), dim=dim or 8, **kwargs)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise IngestError(
                f"not a file: {path}. Raw text must use load_text() explicitly.")
        loader = _EXT_LOADERS.get(path.suffix.lower())
        if loader is load_wav:
            return loader(path, dim=dim or 8, **kwargs)
        if loader is not None:
            return loader(path, **kwargs)
        if path.suffix.lower() in (".txt", ".md", ".log"):
            return load_text(path.read_text(encoding="utf-8"), dim=dim or 16, **kwargs)
        return load_bytes(path.read_bytes(), dim=dim or 8, **kwargs)
    if isinstance(source, Iterable):
        return from_iterable(source, **kwargs)
    raise IngestError(f"cannot ingest object of type {type(source).__name__}")


# ---------------------------------------------------------------------------
# Genome (pure data; now includes the evolvable plasticity rule)
# ---------------------------------------------------------------------------

# name -> (min, max). The plasticity block's SGD point (1,0,0) is inside the
# search space, which is what makes "did evolution beat SGD?" a sharp question.
_GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "learning_rate": (1e-4, 3e-1),
    "mutation_scale": (1e-3, 2.5e-1),
    "mutation_prob": (0.0, 1.0),
    "energy_efficiency": (0.5, 2.0),
    "reproduction_threshold": (1.5, 6.0),
    "hidden_size": (4.0, 16.0),
    "perception_gain": (0.25, 4.0),
    "exploration": (0.0, 0.5),
    # --- evolvable plasticity rule ---
    "plast_grad": (-1.0, 2.0),   # gradient-descent coefficient; SGD = 1.0
    "plast_hebb": (-1.0, 1.0),   # Hebbian correlation coefficient; SGD = 0.0
    "plast_decay": (0.0, 1.0),   # Oja-style weight decay coefficient; SGD = 0.0
}

# The three plasticity genes and their SGD values, named once so the control
# population (SGD-frozen) and the recovery gate share a single source of truth.
_PLASTICITY_GENES: Tuple[str, ...] = ("plast_grad", "plast_hebb", "plast_decay")
_SGD_RULE: Dict[str, float] = {"plast_grad": 1.0, "plast_hebb": 0.0,
                               "plast_decay": 0.0}
_SGD_RULE_TUPLE: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Genome:
    """Immutable, fully numeric genome. Reproduction produces a *new* Genome."""

    learning_rate: float
    mutation_scale: float
    mutation_prob: float
    energy_efficiency: float
    reproduction_threshold: float
    hidden_size: float
    perception_gain: float
    exploration: float
    plast_grad: float
    plast_hebb: float
    plast_decay: float

    def __post_init__(self) -> None:
        for name, (lo, hi) in _GENE_BOUNDS.items():
            value = getattr(self, name)
            if not np.isfinite(value):
                raise GenomeError(f"gene {name!r} is not finite: {value!r}")
            if not (lo <= value <= hi):
                raise GenomeError(f"gene {name!r}={value} outside [{lo}, {hi}]")

    @property
    def hidden(self) -> int:
        return int(round(self.hidden_size))

    @property
    def rule(self) -> Tuple[float, float, float]:
        """(g_grad, g_hebb, g_decay): the node's plasticity rule as a triple."""
        return (self.plast_grad, self.plast_hebb, self.plast_decay)

    def canonical_bytes(self) -> bytes:
        ordered = {name: round(getattr(self, name), 9) for name in _GENE_BOUNDS}
        return json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()

    def integrity_hash(self) -> str:
        """Lineage fingerprint — identifies a genome, does not authenticate it."""
        return hashlib.blake2b(self.canonical_bytes(), digest_size=8).hexdigest()

    def mutate(self, rng: np.random.Generator,
               evolve_plasticity: bool = True) -> "Genome":
        """Multiplicative log-normal jitter per gene, gated by mutation_prob,
        clamped to bounds. When ``evolve_plasticity`` is False the three
        plasticity genes are held fixed at their (SGD) values, so a control
        lineage keeps an exactly-SGD learning rule while its *other*
        hyperparameters still evolve — a fair control isolating the effect of
        letting the rule itself change."""
        genes: Dict[str, float] = {}
        for name, (lo, hi) in _GENE_BOUNDS.items():
            value = getattr(self, name)
            if not evolve_plasticity and name in _PLASTICITY_GENES:
                genes[name] = value
                continue
            if rng.random() < self.mutation_prob:
                value = value * float(np.exp(rng.normal(0.0, self.mutation_scale)))
            genes[name] = float(np.clip(value, lo, hi))
        return Genome(**genes)

    @staticmethod
    def seed(rng: np.random.Generator, evolve_plasticity: bool = True) -> "Genome":
        """Draw a random valid genome. With ``evolve_plasticity`` False the
        plasticity genes are pinned to the exact SGD rule."""
        genes = {name: float(rng.uniform(lo, hi))
                 for name, (lo, hi) in _GENE_BOUNDS.items()}
        if not evolve_plasticity:
            genes.update(_SGD_RULE)
        return Genome(**genes)


# ---------------------------------------------------------------------------
# Predictive network (deterministic numpy MLP; evolvable local plasticity)
# ---------------------------------------------------------------------------


class PredictiveNet:
    """2-layer tanh MLP: obs(t) -> prediction of obs(t+1). The weight update is
    a local plasticity rule parameterised by three coefficients (see module
    docstring). numpy, not torch: the workload is tiny and hand-rolled math is
    bit-exact across runs — required for the determinism gate."""

    __slots__ = ("w1", "b1", "w2", "b2", "lr", "g_grad", "g_hebb", "g_decay",
                 "dead")

    def __init__(self, in_dim: int, hidden: int, lr: float, spread: float,
                 rule: Tuple[float, float, float],
                 rng: np.random.Generator) -> None:
        s1 = spread * np.sqrt(1.0 / in_dim)
        s2 = spread * np.sqrt(1.0 / hidden)
        self.w1 = rng.normal(0.0, s1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, s2, size=(hidden, in_dim))
        self.b2 = np.zeros(in_dim)
        self.lr = lr
        self.g_grad, self.g_hebb, self.g_decay = rule
        self.dead = False   # set True if an evolved rule diverges to non-finite

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2), h

    def learn(self, x: np.ndarray, target: np.ndarray) -> float:
        """One plasticity step; returns the *pre-step* MSE so nodes are scored
        on what they knew before updating. A rule that drives weights to
        non-finite values retires the node (dead=True) with a large finite
        error instead of poisoning the run with NaN."""
        if self.dead:
            return _DEGENERATE_ERROR
        y, h = self.predict(x)
        if not np.all(np.isfinite(y)):
            self.dead = True
            return _DEGENERATE_ERROR

        err = y - target
        mse = float(np.mean(err * err))

        # Gradient (credit-assignment) signals: exact analytic backprop deltas.
        d_y = (2.0 / y.size) * err * (1.0 - y * y)   # output pre-activation grad
        d_h = (d_y @ self.w2.T) * (1.0 - h * h)      # hidden pre-activation grad

        g, hb, dc = self.g_grad, self.g_hebb, self.g_decay

        # Layer 2 (h -> y). At (g,hb,dc)=(1,0,0) the gradient term alone gives
        #   w2 += lr * outer(h, -d_y)  ==  w2 -= lr * outer(h, d_y)  == SGD.
        dw2 = (g * np.outer(h, -d_y)              # gradient descent
               + hb * np.outer(h, y)              # Hebbian (target-free corr.)
               - dc * self.w2 * (y * y)[None, :])  # Oja-style decay
        db2 = g * (-d_y)                           # biases: gradient-only (SGD-exact)

        # Layer 1 (x -> h).
        dw1 = (g * np.outer(x, -d_h)
               + hb * np.outer(x, h)
               - dc * self.w1 * (h * h)[None, :])
        db1 = g * (-d_h)

        nw2, nb2 = self.w2 + self.lr * dw2, self.b2 + self.lr * db2
        nw1, nb1 = self.w1 + self.lr * dw1, self.b1 + self.lr * db1
        if not (np.all(np.isfinite(nw2)) and np.all(np.isfinite(nw1))
                and np.all(np.isfinite(nb2)) and np.all(np.isfinite(nb1))):
            self.dead = True                       # revert: keep last finite weights
            return _DEGENERATE_ERROR
        self.w2, self.b2, self.w1, self.b1 = nw2, nb2, nw1, nb1
        return mse

    def clone_perturbed(self, scale: float,
                        rng: np.random.Generator) -> "PredictiveNet":
        """Lamarckian inheritance: acquired weights transfer; genome sets noise.
        Rule coefficients are overwritten by the caller from the child genome."""
        child = PredictiveNet.__new__(PredictiveNet)
        child.lr = self.lr
        child.g_grad, child.g_hebb, child.g_decay = self.g_grad, self.g_hebb, self.g_decay
        child.dead = False
        child.w1 = self.w1 + rng.normal(0.0, scale, size=self.w1.shape)
        child.b1 = self.b1.copy()
        child.w2 = self.w2 + rng.normal(0.0, scale, size=self.w2.shape)
        child.b2 = self.b2.copy()
        return child

    def checksum(self) -> float:
        return float(sum(np.abs(a).sum() for a in (self.w1, self.b1, self.w2, self.b2)))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class OrganicNode:
    """A living unit: perceives, predicts, earns/pays energy, may reproduce."""

    __slots__ = ("id", "genome", "net", "energy", "generation", "birth_tick",
                 "children_born", "last_error")

    def __init__(self, node_id: str, genome: Genome, energy: float,
                 generation: int, birth_tick: int, net: PredictiveNet) -> None:
        self.id = node_id
        self.genome = genome
        self.net = net
        self.energy = float(energy)
        self.generation = generation
        self.birth_tick = birth_tick
        self.children_born = 0
        self.last_error = float("nan")

    @classmethod
    def spawn_root(cls, index: int, genome: Genome, energy: float,
                   in_dim: int) -> "OrganicNode":
        node_id = f"root-{index}"
        rng = derive_rng("net-init", node_id, genome.integrity_hash())
        net = PredictiveNet(in_dim, genome.hidden, genome.learning_rate,
                            1.0 + genome.exploration, genome.rule, rng)
        return cls(node_id, genome, energy, 0, 0, net)

    def metabolic_cost(self, base_cost: float) -> float:
        capacity_factor = 1.0 + self.genome.hidden / 32.0
        return base_cost * capacity_factor / self.genome.energy_efficiency

    def can_reproduce(self) -> bool:
        return self.energy >= self.genome.reproduction_threshold

    def reproduce(self, tick: int, evolve_plasticity: bool = True) -> "OrganicNode":
        """Construct the child; the caller performs the conserved energy split."""
        sibling_idx = self.children_born
        self.children_born += 1
        child_id = f"{self.id}.{tick}.{sibling_idx}"

        child_genome = self.genome.mutate(derive_rng("mutate", child_id),
                                          evolve_plasticity)
        child_net = self.net.clone_perturbed(child_genome.mutation_scale,
                                             derive_rng("weights", child_id))
        child_net.lr = child_genome.learning_rate
        child_net.g_grad, child_net.g_hebb, child_net.g_decay = child_genome.rule
        return OrganicNode(child_id, child_genome, energy=0.0,
                           generation=self.generation + 1,
                           birth_tick=tick, net=child_net)


# ---------------------------------------------------------------------------
# World (closed energy economy over ANY DataSource)
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    seed: int = 0
    env_dim: int = 8                # used only when no source is supplied
    n_seed_nodes: int = 12
    total_energy: float = 60.0      # invariant: pool + Σ node energy == this
    seed_node_energy: float = 1.0
    base_metabolic_cost: float = 0.02
    max_reward: float = 0.06        # > base cost so good predictors net-gain
    error_scale: float = 1.0        # reward = max_reward*exp(-scale*mse/median)
    baseline_floor: float = 1e-3    # guards median division on near-constant data
    child_energy_fraction: float = 0.5
    death_energy: float = 1e-3
    max_population: int = 400
    evolve_plasticity: bool = True  # False => SGD-frozen control population


@dataclass
class StepMetrics:
    tick: int
    population: int
    pool: float
    mean_error: float
    max_generation: int
    total_energy: float


class World:
    """Deterministic evolutionary world over an arbitrary ingested source.
    Energy is conserved by construction: rewards withdraw from the pool, costs
    return to it, births split, deaths refund. Nothing created or destroyed."""

    def __init__(self, config: WorldConfig,
                 source: Optional[DataSource] = None) -> None:
        self.cfg = config
        self.tick = 0
        self.source = source if source is not None else SyntheticSource(
            config.seed, config.env_dim)
        self.nodes: Dict[str, OrganicNode] = {}

        seed_rng = derive_rng("seed-genomes", config.seed,
                              self.source.fingerprint(), config.evolve_plasticity)
        used = config.n_seed_nodes * config.seed_node_energy
        if used > config.total_energy:
            raise OrganicError("seed energy exceeds total_energy budget")
        self.pool = config.total_energy - used

        for i in range(config.n_seed_nodes):
            genome = Genome.seed(seed_rng, config.evolve_plasticity)
            node = OrganicNode.spawn_root(i, genome, config.seed_node_energy,
                                          self.source.dim)
            self.nodes[node.id] = node

    def total_energy(self) -> float:
        return self.pool + sum(n.energy for n in self.nodes.values())

    def step(self) -> StepMetrics:
        cfg = self.cfg
        obs = self.source.observe(self.tick)
        target = self.source.observe(self.tick + 1)

        # Pass 1 — perceive, predict, learn. Errors collected before any energy
        # moves, because the reward reference is the population itself.
        ordered_ids = sorted(self.nodes)       # order never depends on churn
        errors: List[float] = []
        for node_id in ordered_ids:
            node = self.nodes[node_id]
            scaled_obs = np.tanh(obs * node.genome.perception_gain)
            mse = node.net.learn(scaled_obs, target)
            node.last_error = mse
            errors.append(mse)

        # Pass 2 — comparative (tournament) economy. Reward is a function of
        # each node's error relative to the population *median*, not of absolute
        # accuracy. Any absolute reference is coupled to one dataset's noise
        # floor and caused extinction on others (v3 finding); the median is the
        # only reference invariant across substrates. Median predictor earns
        # subsistence; better-than-median accumulates; worse starves.
        median_mse = max(float(np.median(errors)), cfg.baseline_floor) \
            if errors else cfg.baseline_floor
        desired: Dict[str, float] = {}
        for node_id in ordered_ids:
            rel = self.nodes[node_id].last_error / median_mse
            desired[node_id] = cfg.max_reward * float(np.exp(-cfg.error_scale * rel))

        # Pro-rata settlement under scarcity: paying in iteration order until the
        # pool empties was measured to select by *alphabetical id*, not skill.
        total_desired = sum(desired.values())
        scale = 1.0 if total_desired <= self.pool else (
            self.pool / total_desired if total_desired > 0.0 else 0.0)
        for node_id in ordered_ids:
            node = self.nodes[node_id]
            reward = desired[node_id] * scale
            self.pool -= reward
            node.energy += reward
            cost = min(node.metabolic_cost(cfg.base_metabolic_cost), node.energy)
            node.energy -= cost
            self.pool += cost

        self._handle_reproduction()
        self._handle_death()

        self.tick += 1
        mean_error = float(np.mean(errors)) if errors else float("nan")
        max_gen = max((n.generation for n in self.nodes.values()), default=0)
        return StepMetrics(self.tick, len(self.nodes), self.pool, mean_error,
                           max_gen, self.total_energy())

    def _handle_reproduction(self) -> None:
        cfg = self.cfg
        if len(self.nodes) >= cfg.max_population:
            return
        parents = [self.nodes[nid] for nid in sorted(self.nodes)
                   if self.nodes[nid].can_reproduce()]
        for parent in parents:
            if len(self.nodes) >= cfg.max_population:
                break
            child = parent.reproduce(self.tick, cfg.evolve_plasticity)
            transfer = parent.energy * cfg.child_energy_fraction
            parent.energy -= transfer
            child.energy += transfer
            self.nodes[child.id] = child

    def _handle_death(self) -> None:
        dead = [nid for nid, n in self.nodes.items()
                if n.energy <= self.cfg.death_energy]
        for nid in dead:
            self.pool += self.nodes[nid].energy
            del self.nodes[nid]

    def champion(self) -> Optional[OrganicNode]:
        """Lowest-error survivor (ties broken by id for determinism)."""
        if not self.nodes:
            return None
        return min((self.nodes[nid] for nid in sorted(self.nodes)),
                   key=lambda n: (n.last_error, n.id))

    def state_hash(self) -> str:
        """128-bit digest of world state, bound to the data's fingerprint. Pool
        at full precision (hex) — %.6f once collided two distinct extinct
        worlds differing only past the 6th decimal."""
        parts = [f"tick={self.tick}", f"pool={float(self.pool).hex()}",
                 f"src={self.source.fingerprint()}"]
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            parts.append(
                f"{nid}|e={n.energy:.6f}|g={n.generation}"
                f"|dna={n.genome.integrity_hash()}|w={n.net.checksum():.6f}")
        return hashlib.blake2b(";".join(parts).encode(), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    history: List[StepMetrics]
    final_hash: str
    energy_drift: float
    champion_genome: Optional[Genome]   # economy champion (lowest live error)
    survivor_genomes: List[Genome]      # all survivors, for rule-quality selection

    @property
    def final(self) -> StepMetrics:
        return self.history[-1]


def run_simulation(config: WorldConfig, ticks: int,
                   source: Optional[DataSource] = None) -> SimulationResult:
    if ticks < 1:
        raise OrganicError(f"ticks must be >= 1, got {ticks}")
    world = World(config, source)
    initial_energy = world.total_energy()
    history = [world.step() for _ in range(ticks)]
    drift = abs(world.total_energy() - initial_energy)
    champ = world.champion()
    survivors = [world.nodes[nid].genome for nid in sorted(world.nodes)]
    return SimulationResult(history, world.state_hash(), drift,
                            champ.genome if champ else None, survivors)


# ---------------------------------------------------------------------------
# Rule evaluation (pure): how well does a *learning rule* learn a source from
# scratch, with no energy economy and no inherited weights?  This is the #8
# "multiple worlds" transfer harness: freeze a champion's rule, replay it cold
# on an unseen source, and measure the online forecast error it accumulates.
# ---------------------------------------------------------------------------


def evaluate_rule(genome: Genome, source: DataSource, ticks: int,
                  eval_key: str = "eval") -> float:
    """Mean online (predict-then-update) MSE of a fresh net driven by
    ``genome``'s rule over ``ticks`` of ``source``. Lower = the rule learns this
    source faster/better. Deterministic: init is keyed on genome + source, not
    on any evolutionary history."""
    rng = derive_rng(eval_key, genome.integrity_hash(), source.fingerprint())
    net = PredictiveNet(source.dim, genome.hidden, genome.learning_rate,
                        1.0 + genome.exploration, genome.rule, rng)
    total = 0.0
    for t in range(ticks):
        obs = source.observe(t)
        target = source.observe(t + 1)
        scaled = np.tanh(obs * genome.perception_gain)
        total += net.learn(scaled, target)
    return total / ticks


# ---------------------------------------------------------------------------
# Verification gates
# ---------------------------------------------------------------------------


def gate_source(source: DataSource,
                probe_ticks: Sequence[int] = (0, 1, 2, 17, 999, 12345)) -> None:
    """Enforce ingestion invariants I1–I4 on any source before it is trusted."""
    for t in probe_ticks:
        a = source.observe(t)          # I1/I2 validated inside observe
        if not np.array_equal(a, source.observe(t)):   # I3
            raise AssertionError(
                f"source gate FAILED: observe({t}) not repeatable "
                f"({type(source).__name__})")


def gate_sgd_recovery(steps: int = 200, in_dim: int = 6, hidden: int = 9,
                      tol: float = 0.0) -> None:
    """Hard gate: the plasticity rule at (1,0,0) must reproduce plain SGD
    *bit-exactly*. This is what makes the evolved rule a true superset of
    gradient descent rather than a loose analogy — if it fails, every
    'evolution beat SGD' comparison downstream is meaningless."""
    src = SyntheticSource(seed=7, dim=in_dim)

    # Net A: the general plasticity net pinned to the SGD rule.
    init_rng = derive_rng("recovery-init", in_dim, hidden)
    net = PredictiveNet(in_dim, hidden, lr=0.05, spread=1.0,
                        rule=_SGD_RULE_TUPLE, rng=init_rng)

    # Net B: an independent, literal SGD reference sharing the same init stream.
    ref_rng = derive_rng("recovery-init", in_dim, hidden)
    r_w1 = ref_rng.normal(0.0, 1.0 * np.sqrt(1.0 / in_dim), size=(in_dim, hidden))
    r_b1 = np.zeros(hidden)
    r_w2 = ref_rng.normal(0.0, 1.0 * np.sqrt(1.0 / hidden), size=(hidden, in_dim))
    r_b2 = np.zeros(in_dim)
    lr = 0.05

    for t in range(steps):
        obs = src.observe(t)
        target = src.observe(t + 1)
        net.learn(obs, target)

        # Literal reference backprop (the v2 update, verbatim).
        h = np.tanh(obs @ r_w1 + r_b1)
        y = np.tanh(h @ r_w2 + r_b2)
        err = y - target
        d_y = (2.0 / y.size) * err * (1.0 - y * y)
        d_h = (d_y @ r_w2.T) * (1.0 - h * h)
        r_w2 = r_w2 - lr * np.outer(h, d_y)
        r_b2 = r_b2 - lr * d_y
        r_w1 = r_w1 - lr * np.outer(obs, d_h)
        r_b1 = r_b1 - lr * d_h

    for name, a, b in (("w1", net.w1, r_w1), ("b1", net.b1, r_b1),
                       ("w2", net.w2, r_w2), ("b2", net.b2, r_b2)):
        max_dev = float(np.max(np.abs(a - b)))
        if max_dev > tol:
            raise AssertionError(
                f"SGD-recovery gate FAILED on {name}: max deviation {max_dev:.3e} "
                f"> tol {tol:.0e} — plasticity(1,0,0) is not exact SGD")


def _history_digest(result: SimulationResult) -> str:
    """Digest of the entire trajectory. Two seeds can legitimately converge to
    the same terminal state (e.g. both extinct) via different paths; seed
    sensitivity must be judged on the path, not the endpoint."""
    h = hashlib.blake2b(digest_size=16)
    for m in result.history:
        h.update(f"{m.tick}|{m.population}|{float(m.pool).hex()}"
                 f"|{float(m.mean_error).hex()}|{m.max_generation}".encode())
    h.update(result.final_hash.encode())
    return h.hexdigest()


def gate_determinism(config: WorldConfig, ticks: int,
                     source_factory: Callable[[], Optional[DataSource]]) -> None:
    a = run_simulation(config, ticks, source_factory())
    b = run_simulation(config, ticks, source_factory())
    if a.final_hash != b.final_hash or _history_digest(a) != _history_digest(b):
        raise AssertionError(
            f"determinism gate FAILED: {a.final_hash} != {b.final_hash}")
    c = run_simulation(replace(config, seed=config.seed + 1), ticks,
                       source_factory())
    if _history_digest(c) == _history_digest(a):
        raise AssertionError("determinism gate FAILED: seed has no effect")


def gate_energy_conservation(result: SimulationResult, tol: float = 1e-6) -> None:
    if result.energy_drift > tol:
        raise AssertionError(
            f"conservation gate FAILED: drift {result.energy_drift:.3e} > {tol:.0e}")


# ---------------------------------------------------------------------------
# Self-test + the evolvable-plasticity experiment
# ---------------------------------------------------------------------------


def _make_demo_files(root: Path) -> Dict[str, Path]:
    """Fabricate one exemplar per format with known content — the demo depends
    on no file already present on the machine."""
    rng = derive_rng("demo-files")
    paths: Dict[str, Path] = {}

    csv_path = root / "demo.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temp_c", "humidity", "site", "load"])
        for t in range(240):
            w.writerow([f"{20 + 8 * np.sin(0.11 * t):.3f}",
                        f"{55 + 20 * np.sin(0.05 * t + 1.0):.3f}",
                        ["north", "south", "east"][t % 3],
                        "" if t % 37 == 0 else f"{rng.uniform(0, 1):.4f}"])
    paths["csv"] = csv_path

    jsonl_path = root / "demo.jsonl"
    with open(jsonl_path, "w") as f:
        for t in range(200):
            f.write(json.dumps({"a": float(np.cos(0.2 * t)),
                                "b": {"x": t % 7, "y": float(np.sin(0.13 * t))},
                                "tag": "on" if t % 2 else "off"}) + "\n")
    paths["jsonl"] = jsonl_path

    wav_path = root / "demo.wav"
    t = np.arange(8000)
    tone = (0.6 * np.sin(2 * np.pi * 220 * t / 8000)
            + 0.3 * np.sin(2 * np.pi * 330 * t / 8000))
    pcm = (tone * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(pcm.tobytes())
    paths["wav"] = wav_path
    return paths


def _run_plasticity_experiment(config: WorldConfig, ticks: int,
                               train: DataSource,
                               transfers: Dict[str, DataSource],
                               eval_ticks: int) -> None:
    """Evolve two populations on ``train`` — one SGD-frozen control, one with
    evolvable plasticity — then transfer-test the champion *rule* from each on
    held-out ``transfers`` sources. Report the numbers honestly; the direction
    of the result is a measurement, not a gate.

    Champion selection is deliberately NOT the economy winner: the economy
    rewards accumulated Lamarckian weights, so an economy champion can be a
    mediocre rule riding lucky weights. Since we transfer-test the *rule from
    scratch*, we must also *select* the rule from scratch. The champion is the
    survivor whose rule attains the lowest cold-start online error on ``train``
    (a proper train/test split: select on train, judge on held-out). For the
    control population every survivor already carries the SGD rule, so this
    picks the best SGD *hyperparameters*.
    """
    control = run_simulation(replace(config, evolve_plasticity=False), ticks, train)
    plastic = run_simulation(replace(config, evolve_plasticity=True), ticks, train)

    def best_rule_on_train(res: SimulationResult) -> Optional[Genome]:
        if not res.survivor_genomes:
            return None
        # Deduplicate identical genomes to avoid redundant evaluations.
        uniq: Dict[str, Genome] = {g.integrity_hash(): g
                                   for g in res.survivor_genomes}
        return min(uniq.values(),
                   key=lambda g: (evaluate_rule(g, train, eval_ticks),
                                  g.integrity_hash()))

    print("-" * 70)
    print("EVOLVABLE-PLASTICITY EXPERIMENT (#4) + CROSS-WORLD TRANSFER (#8)")
    print(f"train source={train.fingerprint()[:12]}  ticks={ticks}  "
          f"eval_ticks={eval_ticks}  (champion = best cold-start rule on train)")

    control_champ = best_rule_on_train(control)
    plastic_champ = best_rule_on_train(plastic)
    for label, res, champ in (("control(SGD-frozen)", control, control_champ),
                              ("plastic(evolved)", plastic, plastic_champ)):
        rule = (f"({champ.plast_grad:+.3f},{champ.plast_hebb:+.3f},"
                f"{champ.plast_decay:+.3f})" if champ else "n/a (extinct)")
        print(f"  {label:<22} survivors={res.final.population:<3} "
              f"gen={res.final.max_generation}  champion (grad,hebb,decay)={rule}")

    if control_champ is None or plastic_champ is None:
        print("  transfer skipped: a population went extinct on the train source")
        return

    print(f"  {'source':<22} {'SGD champ':>12} {'evolved champ':>14} {'winner':>9}")
    for name, src in {"train": train, **transfers}.items():
        sgd_err = evaluate_rule(control_champ, src, eval_ticks)
        evo_err = evaluate_rule(plastic_champ, src, eval_ticks)
        margin = (sgd_err - evo_err) / sgd_err * 100.0
        winner = f"evolved+{margin:.1f}%" if evo_err < sgd_err else f"SGD{margin:.1f}%"
        print(f"  {name:<22} {sgd_err:>12.5f} {evo_err:>14.5f} {winner:>13}")


def main() -> None:
    import tempfile

    config = WorldConfig()
    ticks = 400

    with tempfile.TemporaryDirectory() as tmp:
        files = _make_demo_files(Path(tmp))

        sources: Dict[str, Callable[[], DataSource]] = {
            "synthetic": lambda: SyntheticSource(config.seed, config.env_dim),
            "ndarray":   lambda: ingest(np.column_stack(
                [np.sin(0.1 * np.arange(300)), np.cos(0.07 * np.arange(300))])),
            "csv":       lambda: ingest(files["csv"]),
            "jsonl":     lambda: ingest(files["jsonl"]),
            "wav":       lambda: ingest(files["wav"], dim=8),
            "text":      lambda: load_text(__doc__, dim=12),
            "bytes":     lambda: ingest(Path(__file__).read_bytes(), dim=8),
            "generator": lambda: from_iterable(
                ([np.sin(0.3 * k), np.cos(0.09 * k)] for k in range(150))),
            "function":  lambda: ingest(lambda k: np.sin(0.25 * k + np.arange(4)),
                                        dim=4),
        }

        print("=" * 70)
        print("ORGANIC AI CORE v4 — evolvable plasticity, universal ingestion")
        print("=" * 70)

        gate_sgd_recovery()
        print("GATE: SGD-recovery  PASS  (plasticity(1,0,0) == SGD, bit-exact)")

        print("-" * 70)
        print(f"{'source':<10} {'dim':>4}  {'gate':<6} {'fingerprint':<32}")
        print("-" * 70)
        for name, factory in sources.items():
            src = factory()
            gate_source(src)
            print(f"{name:<10} {src.dim:>4}  PASS   {src.fingerprint()}")

        for name in ("synthetic", "csv"):
            factory = sources[name]
            result = run_simulation(config, ticks, factory())
            gate_determinism(config, ticks, factory)
            gate_energy_conservation(result)
            f = result.final
            window = max(1, min(50, ticks // 4))
            head = float(np.mean([m.mean_error for m in result.history[:window]]))
            tail = float(np.mean([m.mean_error for m in result.history[-window:]]))
            print("-" * 70)
            print(f"[{name}] seed={config.seed} ticks={ticks} dim={factory().dim}")
            print(f"  final state hash : {result.final_hash}")
            print(f"  energy drift     : {result.energy_drift:.3e}")
            print(f"  population       : {f.population}   "
                  f"generations: {f.max_generation}")
            print(f"  fcast error (mean over {window}-tick window) : "
                  f"{head:.5f} -> {tail:.5f} ({100.0 * (1 - tail / head):.1f}% reduction)")
            print("  GATES: source PASS | determinism PASS | conservation PASS")

        train = SyntheticSource(config.seed, config.env_dim)          # dim 8
        transfers = {
            "synthetic(seed=999)": SyntheticSource(999, config.env_dim),  # new env
            "wav(dim=8)":          ingest(files["wav"], dim=8),           # new modality
        }
        _run_plasticity_experiment(config, ticks, train, transfers, eval_ticks=300)


if __name__ == "__main__":
    main()
