"""One-time feature migration of the rebuilt project, preserving Unity GUIDs."""
from pathlib import Path
import json
import re
import uuid


PROJECT = Path('/home/yuyuan/UNITY_Project/Codex')
GROUPS = {
    'Core': ['New/AppManager', 'RecordingController', 'TeleopConfig', 'TeleopSessionState'],
    'Input': ['DualControllerSender', 'HandTrackingSender', 'TrackingModeManager', 'TeleopTrackingGuard'],
    'Networking': ['UdpSocket'],
    'Calibration': ['TeleopWorld', 'TeleopReferenceFrame', 'TeleopControlModeManager',
                    'TeleopViewModeMapper', 'RobotBasePlacementTool',
                    'Calibration/CalibrationTool', 'Calibration/CalibrationGizmo'],
    'Feedback': ['TCP/TCPPoseReceiver', 'Force/ForceArrow', 'Force/ForceSensorReceiver', 'Viz/RobotTcpAxes'],
    'Media/UDP': ['MultipleVideoStream/UDPManager', 'MultipleVideoStream/VideoCreateManager',
                  'MultipleVideoStream/UdpSocketMultiHD'],
    'Media/WebRTC': ['New/VideoSignalingClient', 'New/VideoStreamManager', 'New/WebRTCVideoReceiver'],
    'UpperLimb': ['UpperLimb/UpperLimbAkmManager', 'UpperLimb/UpperLimbAkmBootstrap'],
    'Visualization/Tactile': ['Tactile Visualization/TactileArrow', 'Tactile Visualization/TactileUIManager',
                              'Tactile Visualization/TactileSensorGenerator'],
    'Visualization/VirtualRobot': ['VirtualRobot/VirtualRobotJointStateReceiver',
                                   'VirtualRobot/VirtualUrdfJointDriver', 'VirtualRobot/Robotiq2F85MimicJointDriver'],
    'UI': ['Tactile Visualization/GrabbleWindows', 'RuntimeUITheme'],
    'Diagnostics': ['New/LogManager'],
    'Legacy/Presentation': ['FoldingCanvas', 'User_Manual_control', 'PassthroughManager', 'AppController'],
    'Legacy/Debug': ['Controller', 'PythonTest'],
    'Legacy/Video': ['MultipleVideoStream/UDPSocket_MULTI'],
    'Legacy/UI': ['UpperLimb/UpperLimbVrKeyboard', 'UpperLimb/UpperLimbAkmUi'],
    'Legacy': ['Position_transfer', 'Unused/CanvasMovement', 'Unused/A_button_laser_control',
               'Unused/X_button_laser_control', 'Unused/WindowJoystickHoldMove', 'Unused/Position_transfer_Multi'],
}
RENAMES = {'UDPManager': 'UdpWindowManager', 'VideoCreateManager': 'VideoWindowController',
           'GrabbleWindows': 'DraggableWindowWorld', 'FoldingCanvas': 'CollapsibleCanvas',
           'User_Manual_control': 'WindowToggle', 'PassthroughManager': 'PassthroughToggle',
           'Controller': 'MetaControllerData', 'UDPSocket_MULTI': 'UdpSocketMulti',
           'Position_transfer': 'PositionTransfer', 'CanvasMovement': 'CanvasMoveProvider',
           'A_button_laser_control': 'AButtonLaserControl', 'X_button_laser_control': 'XButtonLaserControl'}


def main():
    moved = []
    for group, files in GROUPS.items():
        for name in files:
            source = PROJECT / 'Assets/Scripts' / (name + '.cs')
            if not source.exists():
                continue
            target = PROJECT / 'Assets/Teleop' / group / (RENAMES.get(source.stem, source.stem) + '.cs')
            if target.exists():
                raise RuntimeError(f'Target exists: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            meta = Path(str(source) + '.meta')
            guid = re.search(r'^guid: (\w+)', meta.read_text(), re.M)[1]
            source.rename(target)
            meta.rename(Path(str(target) + '.meta'))
            moved.append({'from': str(source.relative_to(PROJECT)),
                          'to': str(target.relative_to(PROJECT)), 'guid': guid})
    old = PROJECT / 'Assets/Scripts'
    if old.exists():
        for directory in sorted([old, *[p for p in old.rglob('*') if p.is_dir()]],
                                key=lambda p: len(p.parts), reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()
                Path(str(directory) + '.meta').unlink(missing_ok=True)
    # New resources get stable path-derived GUIDs; migrated references keep their originals.
    for path in [PROJECT / 'Assets/Teleop', *(PROJECT / 'Assets/Teleop').rglob('*')]:
        if path.suffix == '.meta':
            continue
        meta = Path(str(path) + '.meta')
        if meta.exists():
            continue
        guid = uuid.uuid5(uuid.NAMESPACE_URL, 'doffy-teleop:' + str(path.relative_to(PROJECT))).hex
        if path.is_dir():
            body = 'folderAsset: yes\nDefaultImporter:\n  externalObjects: {}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n'
        elif path.suffix == '.cs':
            body = ('MonoImporter:\n  externalObjects: {}\n  serializedVersion: 2\n  defaultReferences: []\n'
                    '  executionOrder: 0\n  icon: {instanceID: 0}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n')
        else:
            body = 'DefaultImporter:\n  externalObjects: {}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n'
        meta.write_text('fileFormatVersion: 2\nguid: ' + guid + '\n' + body)
    report = Path(__file__).resolve().parents[2] / 'docs/teleop_refactor/unity-source-migration.json'
    if moved:
        report.write_text(json.dumps(moved, indent=2) + '\n')
    print(f'Moved {len(moved)} scripts with their original GUIDs.')


if __name__ == '__main__':
    main()
