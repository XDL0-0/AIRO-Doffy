"""Compatibility alias for the packaged RealMan recollection UI."""

import sys
import doffy_teleop.recording.recollect_ui.app as _implementation

sys.modules[__name__] = _implementation
