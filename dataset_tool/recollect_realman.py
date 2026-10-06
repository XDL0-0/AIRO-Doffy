#!/usr/bin/env python
"""Launch the RealMan LeRobot recollection workflow from dataset_tool."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doffy_teleop.runtime.recollect import main


if __name__ == "__main__":
    raise SystemExit(main())
