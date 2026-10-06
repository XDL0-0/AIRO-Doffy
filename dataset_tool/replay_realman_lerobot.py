#!/usr/bin/env python3
"""Compatibility launcher for ``doffy_teleop.recording.replay``."""

import logging
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import doffy_teleop.recording.replay as _implementation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
