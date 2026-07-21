"""
GCSN-v4 — dictionary-scale training
======================================
Scales the v3 end-to-end system (real MiniLM embeddings, structured
Gumbel-softmax decoder, differentiable interpreter, PointNet++) from 20
hand-authored adjectives to a real English dictionary adjective list
(~21k raw entries from WordNet, filtered).

THE SUPERVISION PROBLEM, STATED HONESTLY
----------------------------------------
No dataset of (English word -> motion shape) exists. The only ground truth
is the 20 hand-authored anchor adjectives from v3. Every other word's
motion parameters are INFERRED by label propagation through the LM's own
semantic space:

    params(w) = sum_a softmax_a( cos(e_w, e_a) / T ) * params(a)

where e_* are MiniLM embeddings and a ranges over the 20 anchors. So
"turbulent" inherits mostly from "chaotic"/"explosive" because the LM
places it near them. The LM does the semantic generalization; I do not
hand-author 28k words. These propagated params are inferred labels, not
ground truth — the diagnostic printout during `prepare` shows the top
anchor attribution for sample words so the propagation quality is
inspectable, not asserted.

CHECKPOINT / RESUME
-------------------
Background processes do not survive turn boundaries in this environment
(observed empirically: the v3 run died with an empty log). So v4 trains in
resumable chunks: `train --seconds N` runs until the time budget expires,
checkpointing every epoch to ckpt_v4.pt. Repeated invocations continue
where the last one stopped.

Subcommands:
    prepare   — filter words, embed, propagate labels, build + freeze dataset
    train     — run/resume training for a bounded wall-clock budget
    eval      — evaluate train / held-out-combo / zero-shot-word splits

Paths note: this copy runs inside the World Compiler repo
(grv2_runtime's sibling "gcsn/" directory), not the original /home/claude
sandbox it was authored in -- WORDLIST_PATH/DATASET_PATH/CKPT_PATH below
are repo-relative instead of the original absolute paths, and
WORDLIST_PATH points at a real WordNet-derived adjective list (see
gcsn/data/adjectives.txt) rather than an external file that doesn't exist
here.
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcsn_v3_e2e import (  # reuse, don't duplicate
    ADJ, TEMPLATES, K_BLOCKS, SEM_DIM,
    StructuredGCodeDecoder, PointNetPlusPlusEncoder,
    differentiable_path, structured_ce_loss, synth_targets, quantize_targets,
    render_gcode_text,
)

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST_PATH = os.path.join(_HERE, "data", "adjectives.txt")
DATASET_PATH = os.path.join(_HERE, "data", "dataset_v4.pt")
CKPT_PATH = os.path.join(_HERE, "data", "ckpt_v4.pt")

KERNEL_TEMP = 0.08          # softmax temperature for label propagation
N_TRAIN_PAIRS = 3000
N_TEST_COMBO = 300
N_TEST_WORD_PAIRS = 300
HOLDOUT_WORD_FRAC = 0.05    # tier B: words never seen in any training sentence
PARAM_CLIP = 2.5            # propagated params live in a tighter range than anchor sums


# ---------------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------------

def load_words() -> List[str]:
    with open(WORDLIST_PATH) as f:
        raw = [w.strip() for w in f]
    words = sorted({
        w.lower() for w in raw
        if w and w.isalpha() and w.islower() and 3 <= len(w) <= 14
    })
    return words


def propagate_via_antonym_axes(anchor_emb: torch.Tensor, anchor_params: torch.Tensor,
                               target_emb: torch.Tensor, dim_names: List[str]
                               ) -> torch.Tensor:
    """Per-dimension antonym-axis projection, replacing the softmax-kernel
    (convex-combination-of-nearest-anchors) propagation this started with.

    WHY THE SWITCH, CONCRETELY: kernel propagation can only ever produce a
    convex combination of the anchors' params -- it has no way to express
    "opposite of the anchor I'm nearest to." That's precisely the failure
    mode found by inspecting the diagnostic spot-check: "turbulent" landed
    nearest "flowing" (turbulence param ~0.51 weight, wrong sign) and
    "erratic" nearest "stable" (0.32 weight, wrong sign) -- both are the
    well-documented phenomenon of antonyms sharing distributional context
    ("the weather was calm/turbulent") and therefore high cosine similarity
    despite opposite meaning. Ridge regression (params = W @ embedding,
    fit on the 20-24 anchors) was tried and rejected: with ~24 points in
    384 dimensions it has no real signal to learn and made previously-
    correct cases (serene, steady) wrong instead.

    The fix: for each param dimension, take the two anchors with the most
    extreme opposite values (e.g. explosive vs calm for turbulence), and
    use their embedding DIFFERENCE as the axis -- subtracting cancels the
    shared "topic" component (weather, motion) and isolates the polarity
    component, the same trick as king-man+woman=queen. Scale/offset are
    then calibrated with a least-squares fit over ALL anchors' known
    params on that axis (not just the 2 poles), so the calibration isn't
    purely a 2-point line.

    Verified before shipping (not asserted): this measurably fixes the two
    headline cases (turbulent -0.02->0.37, erratic 0.22->0.48, both now
    correctly positive) but is NOT a complete fix -- "steady" flips from
    correctly negative to incorrectly positive, and "serene"/"tranquil"
    stay weakly positive instead of going negative. Real, disclosed,
    partial improvement, not a solved problem. `prepare`'s own spot-check
    print (using the cosine-similarity kernel purely as a display
    diagnostic, not to drive propagation) is what keeps this inspectable
    each run, not an assertion here."""
    n_dims = anchor_params.shape[1]
    out = torch.zeros(target_emb.shape[0], n_dims)
    for d in range(n_dims):
        vals = anchor_params[:, d]
        pos_i, neg_i = vals.argmax().item(), vals.argmin().item()
        axis = anchor_emb[pos_i] - anchor_emb[neg_i]
        axis = axis / axis.norm().clamp(min=1e-8)
        mid = (anchor_emb[pos_i] + anchor_emb[neg_i]) / 2

        proj_anchors = (anchor_emb - mid) @ axis
        A = torch.stack([proj_anchors, torch.ones_like(proj_anchors)], dim=1)
        scale, offset = torch.linalg.lstsq(A, vals).solution.tolist()

        proj_targets = (target_emb - mid) @ axis
        out[:, d] = proj_targets * scale + offset
    return out


def prepare(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    lm = SentenceTransformer("all-MiniLM-L6-v2")

    words = load_words()
    print(f"Filtered dictionary adjectives: {len(words)} "
          f"(from raw list, lowercase alpha 3-14 chars)")

    anchors = list(ADJ.keys())
    anchor_params = torch.tensor([ADJ[a] for a in anchors], dtype=torch.float32)  # (24, 5)
    dim_names = ["vertical", "curvature", "radial", "turbulence", "size"]

    t0 = time.time()
    print("Embedding anchors + full word list with MiniLM (cached to disk after)...")
    anchor_emb = torch.tensor(lm.encode(anchors, show_progress_bar=False), dtype=torch.float32)
    word_emb = torch.tensor(
        lm.encode(words, batch_size=256, show_progress_bar=False), dtype=torch.float32
    )
    print(f"  embedded {len(words)} words in {time.time() - t0:.1f}s")

    # --- label propagation: antonym-axis projection (see docstring above
    # propagate_via_antonym_axes for why this replaced softmax-kernel
    # propagation over raw cosine similarity) ---
    print("  propagation axes (pos anchor - neg anchor per dimension):")
    for d, name in enumerate(dim_names):
        vals = anchor_params[:, d]
        pos_i, neg_i = vals.argmax().item(), vals.argmin().item()
        print(f"    {name:10s}: {anchors[pos_i]}(+{vals[pos_i]:.1f}) - {anchors[neg_i]}({vals[neg_i]:.1f})")
    word_params = propagate_via_antonym_axes(anchor_emb, anchor_params, word_emb, dim_names)
    word_params = word_params.clamp(-PARAM_CLIP, PARAM_CLIP)

    # Cosine-similarity nearest-anchor kernel kept ONLY as a display
    # diagnostic (top-3 attribution in the spot-check below) -- it no
    # longer drives the actual propagated params.
    a_n = F.normalize(anchor_emb, dim=-1)
    w_n = F.normalize(word_emb, dim=-1)
    sims = w_n @ a_n.T                                   # (W, len(anchors)) cosine sims
    kernel = F.softmax(sims / KERNEL_TEMP, dim=-1)        # (W, len(anchors))

    ent = -(kernel * (kernel + 1e-12).log()).sum(-1)
    print(f"  propagation kernel entropy: mean {ent.mean():.2f} nats "
          f"(uniform-over-20 would be {math.log(20):.2f}; near-zero would mean "
          f"collapse to a single anchor)")

    # Inspectable diagnostic: top anchors for a few intuitive probe words
    probes = ["turbulent", "serene", "spiral", "shattered", "steady", "swirling",
              "erratic", "tranquil", "widening", "collapsing"]
    print("\n  Propagation spot-check (top-3 anchors per probe word — judge for yourself):")
    for p in probes:
        if p in words:
            wi = words.index(p)
            top = kernel[wi].topk(3)
            attribution = ", ".join(
                f"{anchors[j]}:{top.values[i]:.2f}" for i, j in enumerate(top.indices.tolist())
            )
            print(f"    {p:12s} -> {attribution}   params={[round(v,2) for v in word_params[wi].tolist()]}")

    # --- splits: hold out whole words (tier B), then pair sampling ---
    rng = random.Random(SEED)
    idx_all = list(range(len(words)))
    rng.shuffle(idx_all)
    n_hold = int(len(words) * HOLDOUT_WORD_FRAC)
    holdout_word_idx = set(idx_all[:n_hold])
    train_word_idx = idx_all[n_hold:]
    print(f"\n  words fully held out (zero-shot tier B): {n_hold} | trainable words: {len(train_word_idx)}")

    def sample_pairs(pool: List[int], n: int, forbidden: set) -> List[Tuple[int, int]]:
        out, seen = [], set()
        while len(out) < n:
            a, b = rng.sample(pool, 2)
            key = (min(a, b), max(a, b))
            if key in seen or key in forbidden:
                continue
            seen.add(key)
            out.append(key)
        return out

    train_pairs = sample_pairs(train_word_idx, N_TRAIN_PAIRS, forbidden=set())
    train_pair_set = set(train_pairs)
    combo_pairs = sample_pairs(train_word_idx, N_TEST_COMBO, forbidden=train_pair_set)
    hold_list = list(holdout_word_idx)
    word_pairs = []
    seen = set()
    while len(word_pairs) < N_TEST_WORD_PAIRS:
        a = rng.choice(hold_list)
        b = rng.choice(idx_all)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        word_pairs.append(key)

    def build_split(pairs: List[Tuple[int, int]]):
        texts, sem_list, tgt_list, wordpairs = [], [], [], []
        for (i, j) in pairs:
            a1, a2 = words[i], words[j]
            text = rng.choice(TEMPLATES).format(a1=a1, a2=a2)
            params = (word_params[i] + word_params[j]).clamp(-PARAM_CLIP, PARAM_CLIP)
            blocks = synth_targets(tuple(params.tolist()), seed=0)
            texts.append(text)
            tgt_list.append(quantize_targets(blocks))
            wordpairs.append((a1, a2))
        sem = torch.tensor(lm.encode(texts, batch_size=256, show_progress_bar=False),
                            dtype=torch.float32)
        return dict(texts=texts, words=wordpairs, sem=sem, tgt=torch.stack(tgt_list))

    print(f"\n  building splits: train {len(train_pairs)} | held-out combos {len(combo_pairs)} "
          f"| zero-shot word pairs {len(word_pairs)} ...")
    t0 = time.time()
    data = dict(
        train=build_split(train_pairs),
        test_combo=build_split(combo_pairs),
        test_word=build_split(word_pairs),
        meta=dict(n_words=len(words), n_holdout_words=n_hold,
                   kernel_temp=KERNEL_TEMP, anchors=anchors),
    )
    torch.save(data, DATASET_PATH)
    print(f"  dataset built + frozen to {DATASET_PATH} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# TRAIN (bounded wall-clock, checkpoint/resume)
# ---------------------------------------------------------------------------

def train(seconds: int, batch_size: int = 64, lr: float = 2e-4,
          lambda_cycle: float = 1.0, device: str = "cpu"):
    data = torch.load(DATASET_PATH, weights_only=False)
    sem = data["train"]["sem"].to(device)
    tgt = data["train"]["tgt"].to(device)
    n = sem.size(0)

    decoder = StructuredGCodeDecoder().to(device)
    encoder = PointNetPlusPlusEncoder().to(device)
    params = list(decoder.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    start_epoch = 0

    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, weights_only=False)
        decoder.load_state_dict(ck["decoder"])
        encoder.load_state_dict(ck["encoder"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    else:
        print("Fresh training run (no checkpoint found)")

    deadline = time.time() + seconds
    ep = start_epoch
    decoder.train(); encoder.train()
    while time.time() < deadline:
        ep += 1
        perm = torch.randperm(n, device=device)
        total_ce = total_cyc = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            bidx = perm[start:start + batch_size]
            sem_b, tgt_b = sem[bidx], tgt[bidx]
            opt.zero_grad()
            out = decoder(sem_b)
            ce_loss = structured_ce_loss(out, tgt_b)
            tau = max(0.3, 0.9 * (0.99 ** ep))
            cloud = differentiable_path(out, device, tau=tau)
            grounded = encoder(cloud)
            cyc = F.mse_loss(grounded, sem_b) + 0.5 * (1 - F.cosine_similarity(grounded, sem_b, dim=-1).mean())
            loss = ce_loss + lambda_cycle * cyc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_ce += ce_loss.item(); total_cyc += cyc.item(); nb += 1
            if time.time() > deadline + 30:   # hard escape if a single epoch overruns badly
                break

        print(f"  epoch {ep:4d}  ce {total_ce/nb:.4f}  cycle {total_cyc/nb:.4f}  "
              f"({time.time() - (deadline - seconds):.0f}s elapsed)")
        tmp = CKPT_PATH + ".tmp"
        torch.save(dict(decoder=decoder.state_dict(), encoder=encoder.state_dict(),
                         opt=opt.state_dict(), epoch=ep), tmp)
        os.replace(tmp, CKPT_PATH)   # atomic: a kill mid-save can never corrupt the live ckpt
    print(f"Checkpoint saved at epoch {ep} -> {CKPT_PATH}")


# ---------------------------------------------------------------------------
# EVAL
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(device: str = "cpu", show: int = 4):
    data = torch.load(DATASET_PATH, weights_only=False)
    ck = torch.load(CKPT_PATH, weights_only=False)
    decoder = StructuredGCodeDecoder().to(device)
    encoder = PointNetPlusPlusEncoder().to(device)
    decoder.load_state_dict(ck["decoder"]); encoder.load_state_dict(ck["encoder"])
    decoder.eval(); encoder.eval()
    print(f"Evaluating checkpoint at epoch {ck['epoch']}\n")

    results = {}
    for split_name, label in [("train", "TRAIN"),
                               ("test_combo", "HELD-OUT COMBO (tier A)"),
                               ("test_word", "ZERO-SHOT WORD (tier B)")]:
        split = data[split_name]
        sem = split["sem"].to(device)
        tgt = split["tgt"].to(device)

        sims_all, tok_acc_all = [], []
        outs_for_display = None
        bs = 128
        for start in range(0, sem.size(0), bs):
            sem_b, tgt_b = sem[start:start + bs], tgt[start:start + bs]
            out = decoder(sem_b)
            if outs_for_display is None:
                outs_for_display = out
            cloud = differentiable_path(out, device, tau=0.3)
            grounded = encoder(cloud)
            sims_all.append(F.cosine_similarity(grounded, sem_b, dim=-1))
            # structured accuracy: fraction of argmax fields matching target
            preds = torch.stack([
                out["is_arc"].argmax(-1), out["clockwise"].argmax(-1),
                out["x"].argmax(-1), out["y"].argmax(-1), out["z"].argmax(-1),
                out["i"].argmax(-1), out["j"].argmax(-1),
            ], dim=-1)
            tok_acc_all.append((preds == tgt_b).float().mean(dim=(1, 2)))
        sims = torch.cat(sims_all); tok_acc = torch.cat(tok_acc_all)
        print(f"--- {label}: n={sem.size(0)}  round-trip cosine {sims.mean():.3f} "
              f"(min {sims.min():.3f})  |  structured-field accuracy {tok_acc.mean():.3f} ---")
        for i in range(min(show, len(split['texts']))):
            print(f"  \"{split['texts'][i]}\"  sim={sims[i]:.3f}  field-acc={tok_acc[i]:.3f}")
            print("\n".join("    " + l for l in render_gcode_text(outs_for_display, i).splitlines()))
        print()
        results[split_name] = (sims.mean().item(), tok_acc.mean().item())

    tr, ca, wa = results["train"], results["test_combo"], results["test_word"]
    print("=== Summary (round-trip sim / structured-field acc) ===")
    print(f"train:           {tr[0]:.3f} / {tr[1]:.3f}")
    print(f"held-out combo:  {ca[0]:.3f} / {ca[1]:.3f}   (sim gap {tr[0]-ca[0]:+.3f})")
    print(f"zero-shot word:  {wa[0]:.3f} / {wa[1]:.3f}   (sim gap {tr[0]-wa[0]:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "eval"])
    ap.add_argument("--seconds", type=int, default=300)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "prepare":
        prepare(device)
    elif args.cmd == "train":
        train(seconds=args.seconds, device=device)
    else:
        evaluate(device)
