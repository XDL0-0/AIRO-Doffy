"""Compatibility command for :mod:`scripts.diagnostics.camera_stream`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._compat import run_moved_module

if __name__ == "__main__":
    run_moved_module("scripts.diagnostics.camera_stream", __file__)
