#!/usr/bin/env python3
"""Manual VR-app -> BrainCo Revo2 hand-only integration test.

This script deliberately does not create a RealMan arm teleop backend and does
not send any arm joint, TCP, CAN-FD, camera, tactile, or dataset commands.  It
only receives OpenXR hand packets from the VR app and forwards the right-hand
pose to the BrainCo hand through RM_ARM+.

Run from the repository root:

    conda run -n airo-doffy python tests/test_vr_brainco_hand_teleop.py

Use ``--dry-run`` to verify VR reception and the six-DoF mapping without
connecting to the RealMan controller.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# Running a file inside tests/ puts tests/ rather than the repository root on
# sys.path. Add the root so this remains directly executable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from brainco_hand import BrainCoHandDriver, openxr_to_brainco_joints
from config import Config
from parse_vr import detect_packet_type, parse_hand_data


def hand_test_config() -> Config:
    """Use hand mode for this isolated test without changing normal defaults."""
    return Config(TRACKING_MODE="hand", TCP_TOOL="Hand")


def parse_args() -> argparse.Namespace:
    cfg = hand_test_config()
    parser = argparse.ArgumentParser(
        description="Drive only the BrainCo hand from VR OpenXR hand tracking."
    )
    parser.add_argument("--robot-ip", default=cfg.ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=cfg.REALMAN_PORT)
    parser.add_argument("--pc-ip", default=cfg.PC_IP)
    parser.add_argument("--pose-port", type=int, default=cfg.POSE_PORT)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 runs until Ctrl-C.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mapped hand joints without connecting to BrainCo.",
    )
    return parser.parse_args()


def validate_config(cfg: Config, *, dry_run: bool) -> None:
    if cfg.TRACKING_MODE != "hand":
        raise ValueError(
            "Set Config.TRACKING_MODE='hand' before running this test."
        )
    if not dry_run and cfg.TCP_TOOL != "Hand":
        raise ValueError(
            "Set Config.TCP_TOOL='Hand' before running the hardware test."
        )
    if not dry_run and cfg.ROBOT_TYPE != "realman":
        raise ValueError(
            "The BrainCo hardware test requires Config.ROBOT_TYPE='realman'."
        )


def connect_hand(cfg: Config, robot_ip: str, robot_port: int):
    """Create the shared RM_ARM+ connection without enabling arm motion."""
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(robot_ip, robot_port)
    if handle.id < 0:
        arm.rm_delete_robot_arm()
        raise RuntimeError(
            f"Could not connect to RealMan at {robot_ip}:{robot_port}; "
            f"handle={handle.id}."
        )

    try:
        hand = BrainCoHandDriver(
            arm,
            baudrate=cfg.BRAINCO_HAND_BAUDRATE,
            read_retries=cfg.BRAINCO_HAND_READ_RETRIES,
            retry_delay=cfg.BRAINCO_HAND_RETRY_DELAY,
            mode_settle_delay=cfg.BRAINCO_HAND_MODE_SETTLE_DELAY,
            max_send_hz=cfg.BRAINCO_HAND_MAX_SEND_HZ,
            filter_cutoff_hz=cfg.BRAINCO_HAND_FILTER_CUTOFF_HZ,
            dead_zone=cfg.BRAINCO_HAND_DEAD_ZONE,
            thumb_rotate_progress_range=(
                cfg.BRAINCO_THUMB_ROTATE_PROGRESS_RANGE
            ),
        )
    except Exception:
        arm.rm_delete_robot_arm()
        raise
    return arm, hand


def receive_loop(
    udp_socket: socket.socket,
    hand: BrainCoHandDriver | None,
    *,
    duration: float,
) -> None:
    started = time.monotonic()
    last_status = 0.0
    valid_frames = 0
    ignored_packets = 0
    last_frame_id: Any = None

    print("Waiting for right-hand OpenXR packets from the VR app...")
    while duration <= 0.0 or time.monotonic() - started < duration:
        try:
            payload, source = udp_socket.recvfrom(65535)
        except socket.timeout:
            continue

        try:
            raw = payload.decode("utf-8")
        except UnicodeDecodeError:
            ignored_packets += 1
            continue
        if detect_packet_type(raw) not in {"hand_text", "hand_binary"}:
            ignored_packets += 1
            continue

        packet = parse_hand_data(raw)
        if packet is None or packet.get("side") != "R":
            ignored_packets += 1
            continue
        if packet.get("frame_id") == last_frame_id:
            continue
        last_frame_id = packet.get("frame_id")

        bones = packet.get("bones")
        try:
            raw_joints = (
                hand.map_openxr_hand(bones)
                if hand is not None
                else openxr_to_brainco_joints(bones)
            )
            filtered_joints = raw_joints
            if hand is not None:
                hand.send_normalized(raw_joints)
                filtered_joints = hand.filtered_joints.copy()
        except ValueError as exc:
            ignored_packets += 1
            print(f"Ignoring incomplete hand frame: {exc}")
            continue

        valid_frames += 1
        now = time.monotonic()
        if now - last_status >= 0.5:
            if hand is None:
                hardware = "dry-run"
            else:
                hardware = hand.target.tolist()
            print(
                f"source={source[0]} frame={last_frame_id} "
                f"raw={np.round(raw_joints, 3).tolist()} "
                f"filtered={np.round(filtered_joints, 3).tolist()} "
                f"hardware={hardware} valid={valid_frames} "
                f"ignored={ignored_packets}"
            )
            last_status = now

    print(
        f"Test complete: {valid_frames} valid right-hand frames, "
        f"{ignored_packets} ignored packets."
    )


def main() -> int:
    args = parse_args()
    cfg = hand_test_config()
    validate_config(cfg, dry_run=args.dry_run)
    if args.robot_ip is None and not args.dry_run:
        raise ValueError("A RealMan --robot-ip is required.")

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.settimeout(0.1)
    try:
        udp_socket.bind((args.pc_ip, args.pose_port))
    except OSError as exc:
        udp_socket.close()
        raise RuntimeError(
            f"Could not listen on {args.pc_ip}:{args.pose_port}. Stop any "
            "running teleop process and verify Config.PC_IP."
        ) from exc

    arm = None
    hand = None
    try:
        if args.dry_run:
            print("Dry run: BrainCo hardware commands are disabled.")
        else:
            print(
                "HAND-ONLY TEST: the robot arm will not receive motion commands.\n"
                "Keep the BrainCo hand clear of people and objects."
            )
            arm, hand = connect_hand(cfg, args.robot_ip, args.robot_port)
            print(
                "BrainCo connected: "
                f"limits={hand.lower.tolist()}..{hand.upper.tolist()}"
            )

        print(f"Listening on {args.pc_ip}:{args.pose_port}")
        receive_loop(udp_socket, hand, duration=args.duration)
        return 0
    finally:
        udp_socket.close()
        if hand is not None:
            hand.close()
        if arm is not None:
            arm.rm_delete_robot_arm()
            print("Disconnected from RealMan RM_ARM+.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
