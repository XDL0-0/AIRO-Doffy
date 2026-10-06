"""FastAPI controller for the RealMan recollection workflow."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image


STATIC_ROOT = Path(__file__).with_name("static")


def _jpeg_rgb(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Expected RGB camera image, got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size:
            if float(np.nanmax(array)) <= 1.5:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="JPEG", quality=88)
    return output.getvalue()


def _jpeg_bgr(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Expected BGR camera image, got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size:
            if float(np.nanmax(array)) <= 1.5:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    success, encoded = cv2.imencode(
        ".jpg",
        array,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not success:
        raise ValueError("Could not encode source camera frame.")
    return encoded.tobytes()


class RecollectController:
    def __init__(self, recollector) -> None:
        self.recollector = recollector

    def state(self) -> dict:
        return self.recollector.status()

    def beaver(self) -> dict:
        return self.recollector.beaver_visualization()

    def command(self, name: str) -> dict:
        actions = {
            "continue": self.recollector.request_replay,
            "teach": self.recollector.request_teach,
            "end_teach": self.recollector.request_end_teach,
            "reteach": self.recollector.request_reteach,
            "cancel_teach": self.recollector.request_cancel_teach,
            "teach_collect": self.recollector.request_teach_collect,
            "retry": self.recollector.request_retry,
            "rollback": self.recollector.request_rollback,
            "stop": self.recollector.request_stop,
        }
        if name not in actions:
            raise HTTPException(status_code=404, detail=f"Unknown command {name!r}.")
        if not actions[name]():
            state = self.recollector.status()
            reason = "; ".join(state.get("readiness_issues", []))
            suffix = f": {reason}" if reason else " in the current workflow state"
            raise HTTPException(
                status_code=409,
                detail=f"Command {name!r} is not available{suffix}.",
            )
        return {"accepted": True, "command": name}

    def camera_jpeg(self, kind: str, camera_name: str) -> bytes:
        if kind == "live":
            return _jpeg_rgb(self.recollector.live_image(camera_name))
        if kind == "reference":
            return _jpeg_bgr(self.recollector.reference_image(camera_name))
        raise KeyError(f"Unknown camera stream kind {kind!r}.")


def create_app(controller: RecollectController) -> FastAPI:
    app = FastAPI(title="RealMan Recollect", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/api/state")
    def state() -> dict:
        return controller.state()

    @app.get("/api/beaver")
    def beaver() -> dict:
        return controller.beaver()

    @app.get("/api/cameras/{kind}/{camera_name}.jpg")
    def camera(kind: str, camera_name: str) -> Response:
        try:
            content = controller.camera_jpeg(kind, camera_name)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/commands/{name}")
    def command(name: str) -> dict:
        return controller.command(name)

    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    return app
