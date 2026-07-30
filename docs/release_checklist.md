# v2.0 Release Candidate Checklist

Status: not ready to tag.

The codebase remains `2.0.0.dev0`. This checklist prevents a documentation-only
refactor from being mistaken for a hardware-validated production release.

## Required automated checks

- [x] Minimal editable install succeeds.
- [x] Base package imports without optional hardware SDKs.
- [x] Unit suite passes in the current minimal environment.
- [x] Complete mock integration session passes.
- [x] Package dependency-boundary tests pass.
- [x] Hardware tests skip by default.
- [ ] `python -m pytest` passes with dev dependencies.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m ruff format --check .` passes.
- [ ] `python -m pyright` passes.
- [ ] `python -m pre_commit run --all-files` passes.
- [ ] HDF5 tests pass with `h5py`.
- [ ] Build wheel and sdist in a clean environment.
- [ ] Install built wheel into a clean environment and rerun smoke tests.

## Compatibility checks

- [x] Legacy JPEG/UDP header has golden tests.
- [x] VR legacy text and binary migration protocols have tests.
- [x] State and command protocols have serialization, stale, retry, and dedupe
      tests.
- [x] HDF5 and LeRobot schema declarations have tests.
- [x] Episode numbering and rollback have tests.
- [x] Root compatibility facades remain present.
- [ ] Compare real pre-v2 HDF5 episodes with v2 writer output.
- [ ] Open migrated LeRobot data in the supported LeRobot version.
- [ ] Validate the selected Unity client against frozen and new channels.

## Hardware checks

- [ ] H-UR: UR3e/UR5e connect, state, low-speed motion, HOLD, STOP, wrench,
      gripper, disconnect, and close.
- [ ] H-RM: RM75 state push/poll fallback, CAN-FD rate gate, low-speed joint/TCP
      motion, HOLD, STOP, force frame, disconnect, and close.
- [ ] H-CAM: all selected RealSense serials, color/depth, retry, disconnect,
      and multi-camera rate.
- [ ] H-VR: controller and hand input, sequence wrap, packet loss, stale hold,
      reference reset, and Unity coordinate conventions.
- [ ] H-TAC: BLE4 calibration, `(4, 3)` units, disconnect, recalibration, and
      recording.
- [ ] H-NET: WebRTC signaling, H.264 negotiation, state/command DataChannels,
      ACK retry, UDP/RTP MTU, firewall, and reconnect.
- [ ] H-STOP: emergency/controlled stop and shutdown under every supported
      operating mode.

## Performance evidence

- [ ] Video queue depth remains bounded at configured FPS.
- [ ] Hardware/software encoder selection and fallback are recorded.
- [ ] Multi-camera end-to-end latency is measured.
- [ ] State loss/reordering behavior is measured.
- [ ] Reliable command ACK latency and retry are measured.
- [ ] Teleoperation cycle jitter and deadline misses are measured.
- [ ] RealMan CAN-FD stays above its configured safety gate.
- [ ] Recording export does not block the control path.

## Documentation and deployment

- [x] README is v2-first.
- [x] Architecture, communication, extension, and migration documents exist.
- [x] Breaking changes and compatibility wrappers are documented.
- [x] No wrapper was removed without approval.
- [x] Changelog identifies the development status and blockers.
- [ ] Deployment composition factory is reviewed for SDK thread ownership.
- [ ] Workcell IPs, serials, tool geometry, force frames, and safety limits are
      stored outside the repository.
- [ ] Operator startup, stop, recovery, and rollback procedures are reviewed.

## Tagging rule

Do not create `v2.0.0-rcN` or `v2.0.0` while any deployment-required automated,
compatibility, hardware, safety, or performance item above remains unchecked.
A tag must identify the exact commit and built artifacts that passed the release
evidence.
