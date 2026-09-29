"""FastAPI app: serves the dashboard and streams pipeline output over a websocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .pipeline import Pipeline

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

Bundle = tuple[str, list[bytes]]


class _Client:
    """Per-connection outbox. Frames are dropped (newest kept) so slow clients never lag;
    control messages (hello/status) are always delivered."""

    def __init__(self) -> None:
        self.control: deque[Bundle] = deque()
        self.frames: deque[Bundle] = deque(maxlen=2)
        self.ready = asyncio.Event()

    def offer(self, bundle: Bundle, droppable: bool) -> None:
        (self.frames if droppable else self.control).append(bundle)
        self.ready.set()

    def pop(self) -> Bundle | None:
        if self.control:
            return self.control.popleft()
        return self.frames.popleft() if self.frames else None


class Hub:
    def __init__(self) -> None:
        self.clients: set[_Client] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, msg: dict, binaries: list[bytes]) -> None:
        """Thread-safe: called from the pipeline thread."""
        if self.loop is None:
            return
        bundle = (json.dumps(msg, separators=(",", ":")), binaries)
        try:
            self.loop.call_soon_threadsafe(self._fanout, bundle, msg.get("type") == "frame")
        except RuntimeError:
            pass  # event loop already closed: the server is shutting down

    def _fanout(self, bundle: Bundle, droppable: bool) -> None:
        for c in self.clients:
            c.offer(bundle, droppable)


def create_app(cfg: Config) -> FastAPI:
    hub = Hub()
    pipeline = Pipeline(cfg, hub.publish)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        pipeline.start()
        yield
        pipeline.stop()

    app = FastAPI(title="Vigil", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/api/mesh/faces")
    def mesh_faces() -> Response:
        if pipeline.faces is None:
            return Response(status_code=404)
        return Response(pipeline.faces, media_type="application/octet-stream")

    @app.get("/api/probe/model.glb")
    def probe_model() -> Response:
        path = cfg.resolve(cfg.probe.model_path)
        if not path.is_file():
            return Response(status_code=404)
        return FileResponse(path, media_type="model/gltf-binary")

    @app.get("/api/probe/tag/{marker_id}.png")
    def probe_tag(marker_id: int) -> Response:
        from .probe.aruco import generate_tag_image

        if not 0 <= marker_id < 5:
            return Response(status_code=404)
        ok, png = cv2.imencode(".png", generate_tag_image(cfg.probe.dictionary, marker_id))
        return Response(png.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        client = _Client()
        if pipeline.hello:
            client.offer((json.dumps(pipeline.hello), []), droppable=False)
        client.offer((json.dumps(pipeline.status), []), droppable=False)
        hub.clients.add(client)

        async def receive() -> None:
            # Dashboard → pipeline commands (probe filter, rig calibration).
            while True:
                try:
                    msg = json.loads(await socket.receive_text())
                except (ValueError, TypeError):
                    continue
                except (WebSocketDisconnect, RuntimeError):
                    return  # client went away; the send loop notices via this task ending
                if isinstance(msg, dict) and "cmd" in msg:
                    pipeline.command(msg)

        receiver = asyncio.create_task(receive())
        try:
            while True:
                # Wake on new data *or* on the client going away, so a closed connection
                # never keeps the handler (and server shutdown) waiting for the next frame.
                ready = asyncio.create_task(client.ready.wait())
                await asyncio.wait({ready, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if receiver.done():
                    ready.cancel()
                    break
                client.ready.clear()
                while (bundle := client.pop()) is not None:
                    text, binaries = bundle
                    # Binaries first so the JSON frame can reference them by seq.
                    for b in binaries:
                        await socket.send_bytes(b)
                    await socket.send_text(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()
            hub.clients.discard(client)

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Always revalidate the dashboard files, so an update is never masked by a stale cache.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
