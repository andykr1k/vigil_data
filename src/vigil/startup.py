"""Startup checklist for the dashboard: each step's state plus overall progress and time left.

Steps run one after another; the estimate for each is how long it took on the last successful
start (defaults the first time), so the bar gets accurate after one run.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

DEFAULT_S = {"clarius": 5.0, "cameras": 4.0, "pose": 8.0, "tracker": 0.5, "detector": 10.0,
             "compile": 30.0}


class Startup:
    def __init__(self, steps: list[tuple[str, str]], emit: Callable[[dict], None], history: Path,
                 extra: dict, watch: Callable[[], None] | None = None):
        self.checks = [{"id": i, "label": label, "state": "pending", "detail": ""}
                       for i, label in steps]
        self._by_id = {c["id"]: c for c in self.checks}
        self._emit, self._history, self._extra, self._watch = emit, history, extra, watch
        try:
            last = json.loads(history.read_text())
        except (OSError, ValueError):
            last = {}
        self._expected = {i: float(last.get(i, DEFAULT_S.get(i, 5.0))) for i, _ in steps}
        self._took: dict[str, float] = {}
        self._active: tuple[str, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._tick, name="startup", daemon=True).start()

    def set(self, step: str, state: str | None = None, detail: str | None = None) -> None:
        """Update a check without timing it (e.g. live detail from a background connection)."""
        with self._lock:
            c = self._by_id[step]
            if c["state"] in ("done", "warn", "error"):
                return
            if state is not None:
                c["state"] = state
            if detail is not None:
                c["detail"] = detail

    def begin(self, step: str, detail: str = "") -> None:
        with self._lock:
            self._by_id[step].update(state="active", detail=detail)
            self._active = (step, time.monotonic())
        self.publish()

    def done(self, step: str, detail: str | None = None, state: str = "done") -> None:
        with self._lock:
            c = self._by_id[step]
            c["state"] = state
            if detail is not None:
                c["detail"] = detail
            if self._active and self._active[0] == step:
                self._took[step] = time.monotonic() - self._active[1]
                self._active = None
        self.publish()

    def fail(self, message: str) -> None:
        with self._lock:
            if self._active:
                self._by_id[self._active[0]].update(state="error", detail=message)
                self._active = None
        self.close()

    def finish(self) -> None:
        """All steps done: remember how long each took for the next estimate."""
        self.close()
        try:
            self._history.parent.mkdir(parents=True, exist_ok=True)
            self._history.write_text(json.dumps({k: round(v, 2) for k, v in self._took.items()}))
        except OSError:
            pass

    def close(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            total = sum(self._expected.values()) or 1.0
            spent = 0.0
            for c in self.checks:
                if c["state"] in ("done", "warn", "error"):
                    spent += self._expected[c["id"]]
            if self._active:
                step, t0 = self._active
                spent += min(time.monotonic() - t0, 0.95 * self._expected[step])
                message = self._by_id[step]["label"] + "…"
            else:
                message = "Starting…"
            return {"type": "status", "state": "loading", "message": message, **self._extra,
                    "checks": [dict(c) for c in self.checks],
                    "progress": round(min(spent / total, 1.0), 3),
                    "eta_s": round(max(total - spent, 0.0), 1)}

    def publish(self) -> None:
        if not self._stop.is_set():
            self._emit(self.status())

    def _tick(self) -> None:
        while not self._stop.wait(0.25):  # keeps the bar and ETA moving during long waits
            if self._watch:
                self._watch()
            self.publish()
