"""Tests for world_compiler.py — wraps both end-to-end gate functions."""
import pytest

import world_compiler as wc

_TEXT_A = (
    "The ocean tide carried the whale past the coral reef. "
    "A current swept the reef fish beneath the drifting kelp. "
    "The tide pulled the coral and kelp along the current. "
    "Whales and reef fish share the same deep ocean current. " * 6
)

_TEXT_B = (
    "The ridge trail climbed past the frozen glacier. "
    "A steep trail wound along the rocky mountain ridge. "
    "The glacier carved a path beneath the summit ridge. "
    "Climbers followed the ridge trail toward the summit. " * 6
)


def test_determinism():
    """Same text + same seed must produce a byte-identical scene."""
    wc.gate_determinism(_TEXT_A)


def test_diverges():
    """Different texts must compile to different worlds."""
    wc.gate_diverges(_TEXT_A, _TEXT_B)
