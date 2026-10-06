"""Render recorded LeRobot v3 data using the real teach-collect dashboard (no hardware)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-airo-doffy')
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from doffy_teleop.visualization.dashboard import TeleopDashboard, TeleopSample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=124)
    parser.add_argument('--frame', type=int, default=220)
    parser.add_argument('--output-dir', type=Path, default=Path('output/teleoperation_ui'))
    parser.add_argument('--half-page', action='store_true', help='Hide force panels and export a 182 × 104 mm compact layout')
    parser.add_argument('--display-status', default='DATASET REPLAY', help='Presentation label only; does not start live collection')
    args = parser.parse_args()
    root, out = args.dataset_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = [r for f in sorted(root.glob('meta/episodes/**/*.parquet')) for r in pq.read_table(f).to_pylist()]
    episode = next(r for r in episodes if r['episode_index'] == args.episode)
    path = root / info['data_path'].format(chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
    rows = [r for r in pq.read_table(path).to_pylist() if r['episode_index'] == args.episode]
    row = next(r for r in rows if r['frame_index'] == args.frame)
    image_key = 'observation.images.camera_0'
    video_path = root / info['video_path'].format(video_key=image_key, chunk_index=episode[f'videos/{image_key}/chunk_index'], file_index=episode[f'videos/{image_key}/file_index'])
    cap = cv2.VideoCapture(str(video_path))
    video_time = episode[f'videos/{image_key}/from_timestamp'] + row['timestamp']
    video_frame = round(video_time * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot decode video frame {video_frame}')
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    distance = np.asarray(row['observation.beaver.distance_mm'], dtype=np.float32)
    status = np.asarray(row['observation.beaver.target_status'])
    present = np.asarray(row['observation.beaver.present'])
    layout = ((0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3))
    dashboard = TeleopDashboard(queue.Queue(), queue.Queue(), hz=info['fps'],
        title='RealMan Teach / Replay Dataset Collector', camera_num=1,
        show_teach_controls=True, show_tactile_panel=False, beaver_enabled=True,
        beaver_layout=layout, force_panel_range=5)
    for r in rows:
        if r['frame_index'] > args.frame:
            break
        dashboard.times.append(r['timestamp'])
        dashboard.values.append(np.array(r['observation.force'] + r['observation.torque']))
    sample = TeleopSample(timestamp=row['timestamp'],
        wrench=np.array(row['observation.force'] + row['observation.torque']),
        joints=np.array(row['observation.state']), tcp_translation=np.array(row['extra.tcp_pose'][4:7]),
        images={'camera_0': image}, camera_count=1,
        dataset={'dataset_type':'LeRobot', 'recorded_episodes':info['total_episodes'], 'current_episode_frames':args.frame+1, 'last_episode_length':len(rows), 'collecting':False, 'collect_rate_hz':info['fps']},
        teach={'state':'idle','trajectory_frames':0, 'message':f'Recorded dataset preview | episode {args.episode} | frame {args.frame} | t = {row["timestamp"]:.3f} s'},
        beaver={'distance_mm':distance,'target_status':status,'present':present,'connected':True, 'dataset_replay':True, 'frame_count':args.frame})
    dashboard._update_plots(sample, sample.wrench)
    dashboard._update_status(sample, sample.wrench)
    dashboard._update_vector(sample.wrench)
    dashboard._update_camera(sample)
    dashboard._update_pose(sample)
    dashboard._update_beaver(sample)
    dashboard.status_text.set_text('DATASET REPLAY  |  recorded observations')
    dashboard.status_text.set_fontsize(10)
    # Recorded files contain observations, not live teach-control state.
    dashboard.dataset_text.set_text(f'LeRobot  {info["fps"]} Hz\nepisode {args.episode:03d}\nframe {args.frame:04d} / {len(rows):04d}')
    dashboard.rollback_button_ax.set_alpha(.35)
    dashboard.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().update(left=0.05)
    dashboard.workflow_message_text.set_position((0.035, 0.012))
    dpi = 240
    if args.half_page:
        dpi = 600
        dashboard.fig.set_size_inches(3.0, 4.1)
        for ax in dashboard.axes + [dashboard.vector_ax]:
            ax.set_visible(False)
        dashboard.force_mag_text.set_visible(False)
        dashboard.torque_mag_text.set_visible(False)
        dashboard.status_ax.set_position([0.06, 0.69, 0.90, 0.20])
        dashboard.camera_axes[0].set_position([0.06, 0.25, 0.90, 0.40])
        dashboard.pose_ax.set_position([0.06, 0.06, 0.90, 0.14])
        dashboard.fig._suptitle.set_text('Teach / Replay Dataset Collector')
        dashboard.fig._suptitle.set_fontsize(9)
        dashboard.fig._suptitle.set_position((0.06, 0.975))
        dashboard.fig._suptitle.set_verticalalignment('top')
        dashboard.status_text.set_text(args.display_status)
        dashboard.status_text.set_fontsize(7)
        dashboard.dataset_text.set_fontsize(6)
        dashboard.pose_text.set_fontsize(7)
        for ax in [dashboard.status_ax, dashboard.pose_ax, *dashboard.camera_axes]:
            ax.set_title(ax.get_title(loc='left'), loc='left', fontsize=7, pad=3)
        for button in [dashboard.teach_button, dashboard.reteach_button, dashboard.replay_button, dashboard.initial_pose_button, dashboard.rollback_button]:
            button.label.set_fontsize(5)
        dashboard.workflow_message_text.set_text(f'Episode {args.episode} · frame {args.frame} · {row["timestamp"]:.2f} s')
        dashboard.workflow_message_text.set_fontsize(6)
        dashboard.workflow_message_text.set_position((0.06, 0.015))
        dashboard.beaver_fig.set_size_inches(4.16, 4.1)
        dashboard.beaver_fig._suptitle.set_text(f'Beaver distance · {args.display_status.lower()}')
        dashboard.beaver_fig._suptitle.set_fontsize(9)
        dashboard.beaver_fig.get_layout_engine().set(rect=(0.01, 0.055, 0.98, 0.93), w_pad=0.015, h_pad=0.035)
        for ax in dashboard.beaver_axes:
            ax.set_title(ax.get_title().replace(' · recorded', '').replace(' · avg', '\navg'), fontsize=6.5, pad=3)
        cbar_ax = dashboard.beaver_fig.axes[-1]
        cbar_ax.tick_params(labelsize=5)
        cbar_ax.yaxis.label.set_fontsize(6)
        for label in dashboard.beaver_fig.texts:
            if label is not dashboard.beaver_fig._suptitle:
                label.set_fontsize(5.5)
                label.set_text('0 mm: red · 1–400 mm: blue · >400 mm: grey')
    dashboard.fig.savefig(out/'teachcollect_dashboard.png',dpi=dpi)
    dashboard.beaver_fig.savefig(out/'beaver_distance.png',dpi=dpi)
    dashboard.fig.savefig(out/'teachcollect_dashboard.pdf')
    dashboard.beaver_fig.savefig(out/'beaver_distance.pdf')
    left = Image.open(out/'teachcollect_dashboard.png').convert('RGB')
    right = Image.open(out/'beaver_distance.png').convert('RGB')
    right = right.resize((round(right.width*left.height/right.height),left.height),Image.Resampling.LANCZOS)
    canvas = Image.new('RGB',(left.width+right.width,left.height),'#080b10')
    canvas.paste(left,(0,0)); canvas.paste(right,(left.width,0))
    canvas.save(out/'teleoperation_ui_latest.png', dpi=(dpi, dpi))
    if args.half_page:
        canvas.save(out/'teleoperation_ui_half_page.pdf', resolution=dpi)
    np.save(out/'distance_mm.npy',distance)
    report = {'dataset':str(root),'episode':args.episode,'frame':args.frame,'timestamp':row['timestamp'], 'video':str(video_path),'video_frame':video_frame,'distance_storage_dtype':info['features']['observation.beaver.distance_mm']['dtype'], 'distance_shape':list(distance.shape), 'distance_range_mm':[float(distance.min()),float(distance.max())], 'nonmultiples_of_10':int(np.count_nonzero(distance % 10)), 'note':'Actual dashboard rendered offline. Original distance millimetres retained; no uint8 conversion. Both windows are the actual TeleopDashboard used by realman_teachcollect.main. Beaver figure is exported directly with no custom drawing or annotations. Live controls disabled.'}
    report['display_status'] = args.display_status
    report['half_page'] = args.half_page
    if args.half_page:
        report['layout_note'] = 'Force/torque plots, magnitudes and force vector hidden; existing teachcollect panels rearranged. 7.16 × 4.1 inches at 600 dpi.'
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
