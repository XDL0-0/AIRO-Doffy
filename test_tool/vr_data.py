"""Standalone VR data receiver — connects to VR headset via UDP and prints
incoming controller / hand-tracking data without requiring cameras or robot
hardware.

Also provides HandVisualizer for real-time 3-D visualization of hand bone
landmarks received from the VR headset.
"""

from __future__ import annotations

import time
import threading
import sys
from typing import Dict

import numpy as np

import utils
import udp_comms as U
from config import Config
from parse_vr import parse_data, parse_hand_data, detect_packet_type

VR_RECEIVE_HZ = 100
CONNECTION_TIMEOUT = 30.0


# ── VR Data Receiver ──────────────────────────────────────────────────────

class VRDataReceiver:
    def __init__(self):
        cfg = Config()
        self.running = True
        self.pc_ip = cfg.PC_IP
        self.vr_ip = cfg.VR_IP
        self.ip_port = cfg.IP_PORT
        self.tracking_mode = cfg.TRACKING_MODE

        self._lock = threading.Lock()
        self.data: list[dict] | None = None
        self.hand_data: Dict[str, dict] = {}  # {"L": {...}, "R": {...}}
        self.fine_mode: str | None = None
        self.current_mode: str | None = None  # "controller" or "hand"
        self.data_collecting_state = False
        self.data_export_state = False

        self.socket_list = self._create_sockets()

    def _create_sockets(self) -> Dict[str, U.UdpComms]:
        """Allocate the same 3 UDP socket pairs used by CameraUDPManager."""
        sockets: Dict[str, U.UdpComms] = {}
        port = self.ip_port

        for i in range(3):
            name = f"socket_{i}"
            sock = U.UdpComms(
                udp_ip=self.pc_ip,
                send_ip=self.vr_ip,
                port_tx=port,
                port_rx=port + 1,
                enable_rx=True,
                suppress_warnings=True,
            )
            sockets[name] = sock
            utils.logger.info(f"{name}: TX={port}, RX={port + 1}")
            port += 2

        return sockets

    def wait_for_connection(self) -> list[dict]:
        """Block until VR sends initial data."""
        utils.logger.info(
            f"Waiting for VR connection (PC={self.pc_ip}, VR={self.vr_ip}) ..."
        )
        t0 = time.time()
        while True:
            raw = self.socket_list["socket_0"].read()
            if raw is not None:
                ptype = detect_packet_type(raw)

                if ptype == "controller":
                    parsed = parse_data(raw)
                    if parsed:
                        with self._lock:
                            self.data = parsed
                            self.current_mode = "controller"
                        utils.logger.info("VR connected! (controller mode)")
                        return parsed

                elif ptype in ("hand_text", "hand_binary"):
                    hand = parse_hand_data(raw)
                    if hand:
                        with self._lock:
                            self.hand_data[hand["side"]] = hand
                            self.current_mode = "hand"
                        utils.logger.info(f"VR connected! (hand mode, {hand['side']})")
                        return [
                            _dummy_controller("LTouch"),
                            _dummy_controller("RTouch"),
                        ]

            if time.time() - t0 > CONNECTION_TIMEOUT:
                raise TimeoutError(
                    f"VR did not respond within {CONNECTION_TIMEOUT}s"
                )
            time.sleep(0.05)

    def _receive_loop(self) -> None:
        target_dt = 1.0 / VR_RECEIVE_HZ

        while self.running:
            t0 = time.time()
            try:
                for raw_data in self.socket_list["socket_0"].read_all():
                    ptype = detect_packet_type(raw_data)

                    if ptype == "controller":
                        parsed = parse_data(raw_data)
                        if parsed:
                            with self._lock:
                                self.data = parsed
                                self.current_mode = "controller"

                    elif ptype in ("hand_text", "hand_binary"):
                        hand = parse_hand_data(raw_data)
                        if hand:
                            with self._lock:
                                self.hand_data[hand["side"]] = hand
                                self.current_mode = "hand"

                record_ctl = self.socket_list["socket_1"].read()
                resolution_ctl = self.socket_list["socket_2"].read()

                if record_ctl:
                    utils.logger.info(f"Record control: {record_ctl}")
                    with self._lock:
                        if record_ctl == "Start":
                            self.data_collecting_state = True
                            self.data_export_state = False
                        elif record_ctl == "Stop":
                            self.data_collecting_state = False
                            self.data_export_state = True

                if resolution_ctl:
                    utils.logger.info(f"Resolution control: {resolution_ctl}")

            except Exception as e:
                utils.logger.error(f"Receive error: {e}")
                time.sleep(0.1)

            elapsed = time.time() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def start(self) -> None:
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()
        utils.logger.info(f"VR receive thread started ({VR_RECEIVE_HZ} Hz)")

    def close(self) -> None:
        self.running = False
        for name, sock in self.socket_list.items():
            try:
                sock.close()
            except Exception as e:
                utils.logger.warning(f"Error closing {name}: {e}")
        utils.logger.info("VR sockets closed.")


# ── Hand Visualizer ───────────────────────────────────────────────────────
# Follows the same visual style as the HTS visualizer:
#   https://github.com/wengmister/hand-tracking-streamer/blob/main/scripts/visualizer.py

# OpenXR XR_EXT_hand_tracking (26 joints, 0-indexed; Palm=0, Wrist=1)
JOINT_NAMES = [
    "Palm",                # 0
    "Wrist",               # 1
    "ThumbMetacarpal",     # 2
    "ThumbProximal",       # 3
    "ThumbDistal",         # 4
    "ThumbTip",            # 5
    "IndexMetacarpal",     # 6
    "IndexProximal",       # 7
    "IndexIntermediate",   # 8
    "IndexDistal",         # 9
    "IndexTip",            # 10
    "MiddleMetacarpal",    # 11
    "MiddleProximal",      # 12
    "MiddleIntermediate",  # 13
    "MiddleDistal",        # 14
    "MiddleTip",           # 15
    "RingMetacarpal",      # 16
    "RingProximal",        # 17
    "RingIntermediate",    # 18
    "RingDistal",          # 19
    "RingTip",             # 20
    "LittleMetacarpal",    # 21
    "LittleProximal",      # 22
    "LittleIntermediate",  # 23
    "LittleDistal",        # 24
    "LittleTip",           # 25
]

WRIST_BONE_INDEX = 1  # Palm=0, Wrist=1

_FINGER_CHAINS = (
    (1, 2, 3, 4, 5),              # Thumb
    (1, 6, 7, 8, 9, 10),          # Index
    (1, 11, 12, 13, 14, 15),      # Middle
    (1, 16, 17, 18, 19, 20),      # Ring
    (1, 21, 22, 23, 24, 25),      # Little
)

_NUM_SEGMENTS = sum(len(c) - 1 for c in _FINGER_CHAINS)

# Unity LH (x right, y up, z forward) → RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=float,
)


def _finger_segments(landmarks: np.ndarray):
    """Return line segments for OpenXR 26-joint finger chains."""
    segments = []
    n = landmarks.shape[0]
    for chain in _FINGER_CHAINS:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if a < n and b < n:
                segments.append((landmarks[a], landmarks[b]))
    return tuple(segments)


def _init_finger_lines(ax, color: str) -> list:
    """Pre-allocate Line3D artists for one hand (HTS-style)."""
    lines = []
    for _ in range(_NUM_SEGMENTS):
        (line,) = ax.plot([], [], [], color=color, linewidth=2)
        lines.append(line)
    return lines


def _update_finger_lines(lines: list, segments) -> None:
    """Push segment data into Line3D artists."""
    for line, (start, end) in zip(lines, segments):
        line.set_data([start[0], end[0]], [start[1], end[1]])
        line.set_3d_properties([start[2], end[2]])


def _set_axes_from_bounds(ax, center: np.ndarray, limit: float) -> None:
    """Set symmetric axis limits and equal aspect (HTS-style)."""
    ax.set_xlim(center[0] - limit, center[0] + limit)  # type: ignore[attr-defined]
    ax.set_ylim(center[1] - limit, center[1] + limit)  # type: ignore[attr-defined]
    ax.set_zlim(center[2] - limit, center[2] + limit)  # type: ignore[attr-defined]
    try:
        ax.set_box_aspect([1.0, 1.0, 1.0])
    except Exception:
        pass


class HandVisualizer:
    """Real-time 3-D hand skeleton visualizer (matplotlib).

    Visual style matches the HTS project ``scripts/visualizer.py``:
    scatter landmarks + wrist cross-marker + finger bone lines + EMA
    dynamic axis scaling.

    Usage::

        receiver = VRDataReceiver()
        receiver.wait_for_connection()
        receiver.start()

        vis = HandVisualizer()
        vis.run(receiver)   # blocking — close window or Ctrl+C to stop
    """

    def __init__(
        self,
        convert_coords: bool = True,
        axis_limit: float = 0.4,
        ema_alpha: float = 0.1,
    ):
        self.convert_coords = convert_coords
        self.axis_limit = axis_limit
        self.ema_alpha = ema_alpha

    @staticmethod
    def _to_rh(points: np.ndarray) -> np.ndarray:
        return (_UNITY_TO_RH @ points.T).T

    def _points_from(self, hand_dict: dict | None) -> np.ndarray | None:
        if hand_dict is None:
            return None
        pts = np.asarray(hand_dict["bones"], dtype=float)
        if self.convert_coords:
            pts = self._to_rh(pts)
        return pts

    def run(self, receiver: VRDataReceiver) -> None:
        """Blocking visualizer loop — close the figure or Ctrl+C to exit."""
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure("Hand Tracking Visualizer", figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore[attr-defined]
        try:
            ax.view_init(elev=10, azim=-170, roll=0)  # type: ignore[attr-defined]
        except TypeError:
            ax.view_init(elev=10, azim=-170)  # type: ignore[attr-defined]

        # --- artists (same colour palette as HTS visualizer) ---
        right_scatter = ax.scatter([], [], [], c="#E45756", label="Right")
        left_scatter  = ax.scatter([], [], [], c="#4C78A8", label="Left")
        right_wrist   = ax.scatter([], [], [], c="#B2333C", marker="x")
        left_wrist    = ax.scatter([], [], [], c="#2D5E8D", marker="x")
        ax.scatter([0.0], [0.0], [0.0], c="#222222", marker="o")
        right_scatter.set_sizes([20])
        left_scatter.set_sizes([20])
        right_wrist.set_sizes([60])
        left_wrist.set_sizes([60])
        ax.legend(loc="upper right")

        right_lines = _init_finger_lines(ax, color="#FFE692")
        left_lines  = _init_finger_lines(ax, color="#94FFDF")

        plt.show(block=False)

        cached_right: np.ndarray | None = None
        cached_left:  np.ndarray | None = None
        ema_center: np.ndarray | None = None
        ema_limit = self.axis_limit

        try:
            while plt.fignum_exists(fig.number):  # type: ignore[attr-defined]
                with receiver._lock:
                    if receiver.current_mode == "controller":
                        utils.logger.info(
                            "Controller data detected — closing visualizer."
                        )
                        break
                    left_dict  = receiver.hand_data.get("L")
                    right_dict = receiver.hand_data.get("R")

                # ---- right hand ----
                rp = self._points_from(right_dict)
                if rp is not None:
                    right_scatter._offsets3d = (rp[:, 0], rp[:, 1], rp[:, 2])
                    cached_right = rp
                    wrist_pt = rp[WRIST_BONE_INDEX]
                    right_wrist._offsets3d = ([wrist_pt[0]], [wrist_pt[1]], [wrist_pt[2]])
                    segs = _finger_segments(rp)
                    _update_finger_lines(right_lines, segs)

                # ---- left hand ----
                lp = self._points_from(left_dict)
                if lp is not None:
                    left_scatter._offsets3d = (lp[:, 0], lp[:, 1], lp[:, 2])
                    cached_left = lp
                    wrist_pt = lp[WRIST_BONE_INDEX]
                    left_wrist._offsets3d = ([wrist_pt[0]], [wrist_pt[1]], [wrist_pt[2]])
                    segs = _finger_segments(lp)
                    _update_finger_lines(left_lines, segs)

                # ---- dynamic axis scaling (HTS-style EMA) ----
                bounds_pts = []
                if cached_right is not None:
                    bounds_pts.append(cached_right)
                if cached_left is not None:
                    bounds_pts.append(cached_left)

                if bounds_pts:
                    all_pts = np.vstack(bounds_pts)
                    mins = all_pts.min(axis=0)
                    maxs = all_pts.max(axis=0)
                    center = (mins + maxs) * 0.5
                    extent = (maxs - mins).max() * 0.5
                    padding = max(extent * 0.3, 0.02)
                    target_limit = max(extent + padding, 0.05)
                    if ema_center is None:
                        ema_center = center
                        ema_limit = target_limit
                    else:
                        a = self.ema_alpha
                        ema_center = (1.0 - a) * ema_center + a * center
                        ema_limit  = (1.0 - a) * ema_limit  + a * target_limit
                    assert ema_center is not None
                    _set_axes_from_bounds(ax, ema_center, float(ema_limit))

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

        except KeyboardInterrupt:
            utils.logger.info("Visualizer interrupted.")
        finally:
            plt.close(fig)


# ── Pretty-printing helpers ───────────────────────────────────────────────

def _format_controller(ctrl: dict) -> str:
    pos = ctrl["Position"]
    rot = ctrl["Rotation"]
    joy = ctrl["Joystick"]
    ts = ctrl.get("Timestamp", "?")
    fid = ctrl.get("FrameId", 0)
    return (
        f"  {ctrl['ControllerType']:>6s}  f={fid}  ts={ts}  "
        f"pos=({pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f})  "
        f"rot=({rot[0]:+6.3f}, {rot[1]:+6.3f}, {rot[2]:+6.3f}, {rot[3]:+6.3f})  "
        f"joy=({joy[0]:+5.2f}, {joy[1]:+5.2f})  "
        f"idx={ctrl['IndexTrigger']:.2f}  grip={ctrl['GripTrigger']:.2f}  "
        f"AX={ctrl['Button_AX']}  BY={ctrl['Button_BY']}  "
        f"JoyPress={ctrl['Joystick_Press']}"
    )


def _format_hand(side: str, hand: dict) -> str:
    ts = hand.get("timestamp", "?")
    fid = hand.get("frame_id", 0)
    n = len(hand.get("bones", []))
    wp = hand.get("wrist_pose")
    wrist_str = ""
    if wp is not None:
        p = wp["position"]
        wrist_str = f"  wrist=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
    return f"  Hand {side}  f={fid}  ts={ts}  bones={n}{wrist_str}"


def _dummy_controller(name: str) -> dict:
    return {
        "ControllerType": name,
        "Timestamp": 0,
        "Position": (0.0, 0.0, 0.0),
        "Rotation": (0.0, 0.0, 0.0, 1.0),
        "Joystick": (0.0, 0.0),
        "IndexTrigger": 0.0,
        "GripTrigger": 0.0,
        "Button_AX": 0,
        "Button_BY": 0,
        "Joystick_Press": 0,
    }


# ── Auto mode switching helpers ──────────────────────────────────────────

def _controller_terminal_loop(receiver: VRDataReceiver) -> None:
    """Print controller data to terminal; returns when mode switches to hand."""
    print_dt = 1.0 / 10
    while True:
        with receiver._lock:
            if receiver.current_mode != "controller":
                return
            ctrl_snapshot = receiver.data

        if ctrl_snapshot:
            lines = [_format_controller(c) for c in ctrl_snapshot]
            sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
        time.sleep(print_dt)


def _auto_mode_loop(receiver: VRDataReceiver) -> None:
    """Switch between controller terminal and hand visualizer.

    The hand visualizer opens only on a **controller → hand** transition (e.g. user
    switches tracking mode on the headset). It does not re-open every frame while
    hand data is streaming, and not on initial connect if the stream starts in hand mode.
    """
    vis = HandVisualizer()
    prev_mode: str | None = None

    while receiver.running:
        with receiver._lock:
            mode = receiver.current_mode

        if mode == "hand" and prev_mode == "controller":
            utils.logger.info("Controller → hand — opening visualizer.")
            vis.run(receiver)
            prev_mode = "hand"
        elif mode == "controller":
            utils.logger.info("Controller mode — showing data in terminal.")
            _controller_terminal_loop(receiver)
            prev_mode = "controller"
        else:
            time.sleep(0.1)
            if mode is not None:
                prev_mode = mode


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    receiver = VRDataReceiver()
    receiver.wait_for_connection()
    receiver.start()

    try:
        _auto_mode_loop(receiver)
    except KeyboardInterrupt:
        utils.logger.info("Stopping...")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
