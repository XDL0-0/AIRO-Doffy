#!/usr/bin/env python3
"""Compatibility launcher; use ``python -m doffy_teleop.runtime.recollect``."""

import sys
import doffy_teleop.runtime.recollect as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
