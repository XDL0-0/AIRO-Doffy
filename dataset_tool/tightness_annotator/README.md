# LeRobot v3 Tightness Annotator

This local web application labels the first episode-local frame at which a grasp
is securely tightened. A transition at frame `k` writes zero to frames before
`k` and one to frame `k` and every later frame. **No tight grasp** writes all
zeros while remaining explicitly distinguishable from an unfinished episode.

## Safety and v3 alignment

- The source dataset is opened read-only and is never modified.
- A missing target is built in a staging directory with LeRobot's official
  `dataset_tools.add_features` API, then renamed into place only after it loads
  successfully as a v3.0 dataset.
- Videos are byte-copied by LeRobot rather than re-encoded.
- Frames are decoded lazily with LeRobot's video decoder using the row timestamp,
  the episode's official `from_timestamp`, and the video shard selected by
  episode metadata. An MP4 filename or its physical start is never treated as an
  episode boundary.
- Each save first writes a recovery journal, atomically replaces the one shared
  Parquet shard containing that episode, atomically replaces progress metadata,
  and then clears the journal. Restarting finishes an interrupted operation.
- `meta/tightness_annotations.json` is authoritative for annotated/unfinished
  state. It stores `annotated`, `has_tight_grasp`, `tight_frame_index`, and a UTC
  timestamp for every episode.

## Launch

Install the package in the LeRobot environment and run:

```bash
python -m dataset_tool.annotate_tightness \
  --dataset-root /path/to/my_dataset
```

The default output is the sibling directory
`/path/to/my_dataset_tightness`. Override it when needed:

```bash
python -m dataset_tool.annotate_tightness \
  --dataset-root /path/to/my_dataset \
  --output-dataset /other/path/my_dataset_tightness \
  --no-browser
```

`pyav` is the default decoder because it is reliable in the supported LeRobot
environment. `--video-backend` can select another backend explicitly.

LeRobot 0.4.4 and 0.5.x are supported. Pip selects 0.4.4 on Python 3.10/3.11
(newer LeRobot releases require Python 3.12) and a compatible 0.5.x release on
Python 3.12+.

If the target already exists, it is validated against the source fingerprint and
opened without recreation. The UI resumes at the first unfinished episode.

## Controls

| Key | Action |
| --- | --- |
| Left / Right | Previous / next frame |
| Shift + Left / Right | Move 10 frames |
| Space | Play / pause |
| T | Mark the visible frame as the tightness transition |
| S | Save this episode |
| Enter | Save and move to the next episode |
| N / P | Next / previous episode |

Shortcuts are ignored while editing the episode input or camera selector. Leaving
an episode with an unsaved draft requires confirmation; normal Save & Next does
not show a modal.

## Verification

```bash
python -m unittest tests.test_tightness_annotator -v
python -m unittest discover -s tests -v
```

The focused suite covers transition boundaries, no-tight annotations, shared
Parquet shards, edits, recovery, resume, v3 validity, and source immutability.
