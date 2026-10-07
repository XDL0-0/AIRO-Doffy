#!/usr/bin/env python
"""Launch the LeRobot v3.0 tightness annotator from dataset_tool."""
# python -m dataset_tool.annotate_tightness --dataset-root datasets/WRM_grasp_cylinder_different_sizes_lero_recollect_gray
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_tool.tightness_annotator.__main__ import main

if __name__ == "__main__":
    main()
