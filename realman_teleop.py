"""Launch the packaged AIRO-Doffy runtime."""

import sys
import doffy_teleop.runtime.realman_cli as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
