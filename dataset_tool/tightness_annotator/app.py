"""FastAPI backend for the local tightness annotation UI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dataset_adapter import TightnessDatasetAdapter
from .video_reader import EpisodeVideoReader

STATIC_ROOT = Path(__file__).with_name("static")


class AnnotationRequest(BaseModel):
    has_tight_grasp: bool
    tight_frame_index: int | None = None


class AnnotatorController:
    def __init__(
        self,
        adapter: TightnessDatasetAdapter,
        reader: EpisodeVideoReader | None = None,
    ) -> None:
        self.adapter = adapter
        self.reader = reader or EpisodeVideoReader(adapter)

    def state(self) -> dict:
        records = self.adapter.store.records()
        annotated = self.adapter.store.annotated_count
        total = self.adapter.total_episodes
        return {
            "source_dataset": str(self.adapter.source_root),
            "output_dataset": str(self.adapter.output_root),
            "dataset_name": self.adapter.output_root.name,
            "fps": self.adapter.fps,
            "total_episodes": total,
            "total_frames": self.adapter.total_frames,
            "annotated_count": annotated,
            "remaining_count": total - annotated,
            "progress_percent": (annotated / total * 100.0) if total else 0.0,
            "resume_episode": self.adapter.store.first_unfinished,
            "cameras": self.adapter.camera_keys,
            "episodes": [asdict(record) for record in records],
        }

    def episode(self, episode_index: int) -> dict:
        length = self.adapter.episode_length(episode_index)
        return {
            "episode_index": episode_index,
            "frame_count": length,
            "last_frame_index": length - 1,
            "fps": self.adapter.fps,
            "timestamps": self.reader.episode_timestamps(episode_index),
            "cameras": self.adapter.camera_keys,
            "annotation": asdict(self.adapter.store.get(episode_index)),
        }

    def save(self, episode_index: int, request: AnnotationRequest) -> dict:
        try:
            record = self.adapter.save_annotation(
                episode_index,
                has_tight_grasp=request.has_tight_grasp,
                tight_frame_index=request.tight_frame_index,
            )
        except (IndexError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "annotation": asdict(record),
            "annotated_count": self.adapter.store.annotated_count,
            "remaining_count": self.adapter.total_episodes
            - self.adapter.store.annotated_count,
            "progress_percent": self.adapter.store.annotated_count
            / self.adapter.total_episodes
            * 100.0,
            "resume_episode": self.adapter.store.first_unfinished,
        }


def create_app(controller: AnnotatorController) -> FastAPI:
    app = FastAPI(title="LeRobot Tightness Annotator", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/api/state")
    def state() -> dict:
        return controller.state()

    @app.get("/api/episodes/{episode_index}")
    def episode(
        episode_index: Annotated[int, ApiPath(ge=0)],
    ) -> dict:
        try:
            return controller.episode(episode_index)
        except IndexError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/episodes/{episode_index}/frames/{frame_index}")
    def frame(
        episode_index: Annotated[int, ApiPath(ge=0)],
        frame_index: Annotated[int, ApiPath(ge=0)],
        camera: str,
    ) -> Response:
        try:
            content = controller.reader.jpeg(episode_index, frame_index, camera)
            timestamp = controller.reader.episode_timestamps(episode_index)[frame_index]
        except (IndexError, KeyError, ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content,
            media_type="image/jpeg",
            headers={"X-LeRobot-Timestamp": f"{timestamp:.9f}"},
        )

    @app.put("/api/episodes/{episode_index}/annotation")
    def save_annotation(
        request: AnnotationRequest,
        episode_index: Annotated[int, ApiPath(ge=0)],
    ) -> dict:
        return controller.save(episode_index, request)

    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    return app
