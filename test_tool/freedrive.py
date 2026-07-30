"""Safety stop for the retired hard-coded UR freedrive command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._compat import block_unsafe_legacy_command

if __name__ == "__main__":
    block_unsafe_legacy_command("scripts.diagnostics.ur_freedrive", __file__)
