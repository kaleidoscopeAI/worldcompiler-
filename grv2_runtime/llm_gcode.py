"""grv2_runtime/llm_gcode.py -- ask an LLM to hand-author G-code for any word.

Fourth and final wiring tier, added by explicit user choice: unlike the
other three tiers (gcode/atlas/neural), which are all free/offline/
deterministic by design, this one makes a real network call to an LLM and
costs money per genuinely new word. It exists because film_siren's
hash-seeded deformation is distinctive but not anatomically informed -- an
LLM actually knows what a castle, a dragon, or a lighthouse should look
like, in a way a deterministic primitive deformation never will.

WiringBank hard-caches every result (one call ever, per process, per
resolved subject word) and this module fails silently (returns None) on
any missing API key, missing dependency, network error, or malformed
response -- the caller always has a free fallback (film_siren) to drop to,
exactly like texture.py's image search already does for the texture layer.

Reuses grv2_runtime.wiring's own G-code text format (G0/G1/G28, X/Y/Z, E
for extrusion) so the LLM's output flows through the exact same
parse_gcode/normalize_pts/voxelize/anneal_crystal pipeline as the
hand-authored GCODE_LIBRARY tier -- same quality bar, same density.
"""
from __future__ import annotations

import os
from typing import Optional

_SYSTEM_PROMPT = """You are a semantic G-code synthesizer for a 3D printer.

Printer bed: X/Y in [0, 40], Z in [0, 24]. Layer height 0.2mm.

Commands available:
  G28          home all axes (always first line)
  G1 F<speed>  set feedrate
  G0 X.. Y.. Z..            travel move, no extrusion
  G1 X.. Y.. Z.. E<amount>  extrusion move (E must be > 0 and increasing)
  G2/G3 X.. Y.. I.. J.. E.. arc moves (optional)

Output 80-160 G-code lines that trace out a recognizable, anatomically
plausible 3D structure for the given object, built up layer by layer
(increasing Z). Use multiple closed contours per layer where the object
has multiple parts (e.g. a body plus a head, or a trunk plus a canopy).

Output ONLY raw G-code lines. No comments, no markdown, no explanation."""

_MODEL = "claude-sonnet-5"
_MAX_TOKENS = 1800
_TIMEOUT_SECONDS = 20.0


def generate_gcode_via_llm(word: str, context: str = "") -> Optional[str]:
    """Best-effort: ask an LLM for G-code describing `word`. Returns None
    (never raises) if ANTHROPIC_API_KEY is unset, the SDK isn't installed,
    the network is unreachable, the call errors or times out, or the
    response doesn't look like G-code."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    prompt = f"Generate geometry for: {word}"
    if context and context.strip().lower() != word.strip().lower():
        prompt += f" (appearing in context: {context[:120]!r})"

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=_TIMEOUT_SECONDS)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
    except Exception:
        return None

    if "G1" not in text and "G0" not in text:
        return None
    return text
