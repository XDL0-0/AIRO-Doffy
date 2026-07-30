"""Compatibility command for :mod:`scripts.diagnostics.force_visualizer`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._compat import run_moved_module

if __name__ == "__main__":
    run_moved_module("scripts.diagnostics.force_visualizer", __file__)
