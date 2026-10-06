"""Compose the BODY receiver, skeleton dashboard, and GUI lifecycle.

No robot runtime is imported here. CLI and optional teleop process ownership
live in :mod:`doffy_teleop.runtime.body_visualizer`.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Callable

from doffy_teleop.media.body import BodyReceiver
from doffy_teleop.visualization.body import BodyDashboard
from doffy_teleop.visualization.body_demo import demo_frame


def run_visualizer(bind_ip: str = "0.0.0.0", port: int = 8015, hz: float = 30.0,
                   stale_after: float = 0.5, demo: bool = False, save_preview: str | Path | None = None,
                   on_ready: Callable[[], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> None:
    """Start the GUI; on_ready runs only after imports, bind and initial draw succeed.

    ``save_preview`` renders a labelled demo PNG without network or robot access.
    ``should_stop`` lets the launcher close the GUI when its teleop child exits.
    """
    if not math.isfinite(hz) or not 1 <= hz <= 120:
        raise ValueError("Visualization rate must be between 1 and 120 Hz")
    if not math.isfinite(stale_after) or stale_after <= 0:
        raise ValueError("Stale timeout must be positive")
    if not 0 <= port <= 65535:
        raise ValueError("Body port must be between 0 and 65535")
    import matplotlib
    if save_preview is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    receiver = None
    dashboard = None
    timer = None
    try:
        if not demo and save_preview is None:
            receiver = BodyReceiver(bind_ip, port)
        endpoint = f"UDP {bind_ip}:{receiver.address[1] if receiver else port}"
        dashboard = BodyDashboard(plt, stale_after=stale_after, demo=demo or save_preview is not None, endpoint=endpoint)
        started = time.monotonic()

        def update():
            if should_stop is not None and should_stop():
                plt.close(dashboard.fig)
                return False
            if demo or save_preview is not None:
                frame = demo_frame(time.monotonic() - started + 0.6)
                dashboard.update(frame, receive_hz=hz, source="synthetic UI preview")
            else:
                frame = receiver.poll()
                source = f"{receiver.source[0]}:{receiver.source[1]}" if receiver.source else endpoint
                dashboard.update(frame, receive_hz=receiver.receive_hz(time.monotonic_ns()), source=source, rejected=receiver.rejected)
            dashboard.fig.canvas.draw_idle()
            return True

        update()
        dashboard.fig.canvas.draw()
        if save_preview is not None:
            destination = Path(save_preview).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            dashboard.fig.savefig(destination, dpi=150, facecolor=dashboard.fig.get_facecolor())
            return
        if not matplotlib.is_interactive() and str(matplotlib.get_backend()).lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            raise RuntimeError("An interactive Matplotlib backend is required. Use MPLBACKEND=TkAgg with a desktop, or --save-preview for a PNG.")
        timer = dashboard.fig.canvas.new_timer(interval=max(1, round(1000 / hz)))
        timer.add_callback(update)
        if on_ready is not None:
            on_ready()
        timer.start()
        plt.show(block=True)
    finally:
        if timer is not None:
            timer.stop()
        if dashboard is not None:
            plt.close(dashboard.fig)
        if receiver is not None:
            receiver.close()
