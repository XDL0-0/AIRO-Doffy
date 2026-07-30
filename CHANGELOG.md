# Changelog

All notable changes to AIRO-Doffy are documented here.

The repository currently identifies as `2.0.0.dev0`. No final v2.0 release or
release-candidate tag has been created.

## Unreleased

### Added

- Installable `airo_doffy` package with side-effect-free base import.
- Immutable typed domain models, clock domains, validation errors, and
  sequence-aware latest buffers.
- Typed configuration sections, layered YAML loading, environment/CLI
  overrides, and lazy focused factories.
- UR, RealMan, mock robot, generic action executor, RealMan CAN-FD executor,
  Robotiq 2F-85, and null gripper components.
- BLE4 MagTouch acquisition/filtering, mock tactile sources, and modular wrench
  processing.
- RealSense and mock camera sources plus dependency-free packed-frame
  processing.
- Legacy JPEG/UDP, RTP/H.264 UDP, WebRTC H.264, bounded encoding, and video
  benchmarking components.
- Legacy-compatible VR parsers, binary VR v2, transport-neutral receiver, and
  deterministic mock VR.
- Binary latest-state protocol and a separate reliable ordered command protocol
  with acknowledgement and deduplication.
- Pure teleoperation transforms, controller/hand mappings, gripper mappings,
  safety filters, and stale-input watchdog.
- Immutable recording schema/samples, episode state machine, HDF5 and LeRobot
  writers, rollback, and bounded export worker.
- Typed visualization snapshots, latest-only consumer, mock renderer, and
  runtime-command outbox.
- Deterministic lifecycle manager, managed workers, composable teleoperation and
  data-collection sessions, and installed CLI entry points.
- Hardware-free unit and complete mock integration suites.
- Default-skipped UR, RealMan, RealSense, and BLE4 hardware smoke tests.
- Ruff, Pyright, pytest, and pre-commit configuration.
- v2 architecture, communication, extension, migration, and phase reports.

### Changed

- Hardware and optional dependencies load only inside selected adapters.
- Camera acquisition, frame processing, encoding, and transport are independent.
- VR parsing, mapping, safety, robot execution, recording, and visualization no
  longer share one manager.
- Real-time queues are bounded/latest-only and expose stale/drop metrics.
- Missing or stale VR input produces a safe HOLD path.
- Runtime startup and shutdown are explicit, ordered, reversible, and
  health-checked.
- Recording export is asynchronous and failures remain observable/retryable.
- Device IPs, camera serials, secrets, and machine-specific paths are absent
  from v2 defaults.
- The legacy RealMan characterization suite skips cleanly when optional math or
  vision dependencies are absent.

### Compatibility

- Root entry points and import facades remain available during migration.
- Legacy controller/hand messages and JPEG/UDP packet bytes remain supported.
- WebRTC signaling envelope shape is preserved.
- HDF5/LeRobot schemas, episode numbering, and rollback semantics are protected
  by compatibility tests.
- Supported v2 modules do not import from `deprecated/`.

### Breaking changes when adopting v2 APIs

- Components require explicit `start()` and lifecycle ownership.
- Configuration is immutable, sectioned, and has no default device addresses.
- Cross-component values are immutable typed samples rather than mutable shared
  attributes.
- Applications require a deployment composition factory.
- Realtime state and reliable commands use different channels.
- Stale sequences and invalid protocol/config values fail explicitly.
- Serial/legacy tactile paths are not supported as new v2 backends.

### Release blockers

- Install and run Ruff, Pyright, pytest, and pre-commit in a dev environment.
- Install `h5py` and run real HDF5 integration tests.
- Validate production composition and SDK thread ownership.
- Run supervised UR, RealMan, RealSense, Quest, BLE4, gripper, network, stop,
  disconnect, force-frame, and timing checks applicable to the deployment.
- Validate existing datasets and Unity builds end to end.
- Record deployment latency/throughput evidence.
- Tag a tested release candidate only after the above checks pass.
