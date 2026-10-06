"""Command line launcher for the local annotator."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .app import AnnotatorController, create_app
from .dataset_adapter import TightnessDatasetAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate tight-grasp transitions in a LeRobot v3.0 dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path)
    parser.add_argument(
        "--repo-id",
        help="Source LeRobot repo ID (defaults to the dataset directory name)",
    )
    parser.add_argument("--output-repo-id")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    adapter = TightnessDatasetAdapter(
        source_root=args.dataset_root,
        output_root=args.output_dataset,
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        video_backend=args.video_backend,
    )
    controller = AnnotatorController(adapter)
    application = create_app(controller)
    url = f"http://{args.host}:{args.port}"
    print(f"Source: {adapter.source_root}")
    print(f"Output: {adapter.output_root}")
    print(
        f"Progress: {adapter.store.annotated_count}/{adapter.total_episodes}; "
        f"opening episode {adapter.store.first_unfinished}"
    )
    print(f"Open {url}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
