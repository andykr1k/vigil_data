"""Startup checklist, progress/ETA, and the procedure choice that gates loading."""

import json

from vigil.config import Config
from vigil.pipeline import Pipeline
from vigil.startup import Startup


def test_progress_eta_and_history(tmp_path):
    hist = tmp_path / "startup.json"
    hist.write_text(json.dumps({"a": 2.0, "b": 6.0}))
    sent = []
    s = Startup([("a", "A"), ("b", "B")], sent.append, hist, {"procedure": "Cardiac"})
    st = s.status()
    assert st["progress"] == 0 and st["eta_s"] == 8.0 and st["procedure"] == "Cardiac"
    s.begin("a")
    s.done("a", "ok")
    s.set("a", "active", "ignored once finished")
    st = s.status()
    assert st["progress"] == 0.25 and st["checks"][0] == {"id": "a", "label": "A", "state": "done",
                                                          "detail": "ok"}
    s.begin("b")
    s.done("b", state="warn")
    s.finish()
    assert s.status()["progress"] == 1.0 and set(json.loads(hist.read_text())) == {"a", "b"}
    assert sent  # every step is published


def test_nothing_loads_until_a_procedure_is_chosen():
    p = Pipeline(Config(), lambda msg, bins: None)
    p.command({"cmd": "start", "procedure": "knee"})  # unknown: ignored
    assert not p._chosen.is_set()
    p.command({"cmd": "start", "procedure": "lower_limb"})
    assert p._chosen.is_set() and p.procedure == "lower_limb"
    assert p.cfg.clarius.procedures[p.procedure] == "dvt"


def test_cloud_worker_runs_one_job_at_a_time_off_the_loop():
    import threading
    import time

    from vigil.pipeline import _Worker

    stop = threading.Event()
    w = _Worker(lambda job: (time.sleep(0.05), job * 2)[1], stop)
    assert w.idle() and w.take() is None
    w.offer(21)
    assert not w.idle()  # busy: the loop won't hand it newer frames yet
    deadline = time.monotonic() + 2
    while not w.idle() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert w.take() == 42 and w.take() is None
    stop.set()
