"""Clarius probe over the Solum SDK (libsolum, bound with ctypes).

Connects over the probe's Wi-Fi (this machine joins the probe's access point), applies
the certificate, loads a preset, streams B-mode images as JPEG, and reports battery,
temperature and frame rate. Depth and gain can be changed live.
"""

from __future__ import annotations

import ctypes as C
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Enum values from solum_def.h
CONNECTED, DISCONNECTED, CONNECTION_FAILED, SW_UPDATE, OS_UPDATE = 0, 1, 2, 3, 4
IMAGING_READY, CERT_EXPIRED = 1, 2
IMAGING_STATES = ["not ready", "ready", "certificate expired", "poor Wi-Fi", "no contact",
                  "charging changed", "low bandwidth", "motion sensor", "no tee", "tee expired"]
PARAM_DEPTH, PARAM_GAIN = 0, 1  # CusParam: ImageDepth (cm), Gain (%)
FORMAT_JPEG = 2  # CusImageFormat.Jpeg

_Vp = C.c_void_p
_ConnectFn = C.CFUNCTYPE(None, C.c_int, C.c_int, C.c_char_p)
_CertFn = C.CFUNCTYPE(None, C.c_int)
_PowerDownFn = C.CFUNCTYPE(None, C.c_int, C.c_int)
_ImagingFn = C.CFUNCTYPE(None, C.c_int, C.c_int)
_ButtonFn = C.CFUNCTYPE(None, C.c_int, C.c_int)
_ErrorFn = C.CFUNCTYPE(None, C.c_int, C.c_char_p)


class _TgcInfo(C.Structure):
    _fields_ = [("depth", C.c_double), ("gain", C.c_double)]


class _ProcessedImageInfo(C.Structure):
    _fields_ = [("width", C.c_int), ("height", C.c_int), ("bitsPerPixel", C.c_int),
                ("imageSize", C.c_int), ("micronsPerPixel", C.c_double), ("originX", C.c_double),
                ("originY", C.c_double), ("tm", C.c_longlong), ("angle", C.c_double),
                ("fps", C.c_double), ("overlay", C.c_int), ("format", C.c_int),
                ("tgc", _TgcInfo * 10)]


_ImageFn = C.CFUNCTYPE(None, _Vp, C.POINTER(_ProcessedImageInfo), C.c_int, _Vp)


class _Args(C.Structure):
    _fields_ = [("argc", C.c_int), ("argv", C.POINTER(C.c_char_p))]


class _InitParams(C.Structure):
    _fields_ = [("args", _Args), ("storeDir", C.c_char_p), ("connectFn", _ConnectFn),
                ("certFn", _CertFn), ("powerDownFn", _PowerDownFn), ("imagingFn", _ImagingFn),
                ("buttonFn", _ButtonFn), ("errorFn", _ErrorFn), ("elemTestFn", _Vp),
                ("newProcessedImageFn", _ImageFn), ("newRawImageFn", _Vp),
                ("newSpectralImageFn", _Vp), ("newImuPortFn", _Vp), ("newImuDataFn", _Vp),
                ("width", C.c_int), ("height", C.c_int)]


class _ConnectionParams(C.Structure):
    _fields_ = [("ipAddress", C.c_char_p), ("port", C.c_uint), ("networkId", C.c_longlong)]


class _StatusInfo(C.Structure):
    _fields_ = [("battery", C.c_int), ("temperature", C.c_int), ("frameRate", C.c_double),
                ("teeTimeRemaining", C.c_double), ("fan", C.c_int), ("guide", C.c_int),
                ("charger", C.c_int)]


class _Range(C.Structure):
    _fields_ = [("min", C.c_double), ("max", C.c_double)]


@dataclass
class ClariusState:
    connected: bool = False
    imaging: bool = False
    state: str = "disconnected"  # human-readable connection/imaging state
    battery: int | None = None  # %
    temperature: int | None = None  # % of the probe's thermal limit
    charging: bool = False
    fps: float | None = None
    cert_days: int | None = None
    depth_cm: float | None = None
    gain: float | None = None
    depth_range: tuple[float, float] | None = None
    gain_range: tuple[float, float] | None = None
    error: str | None = None
    image_seq: int = 0
    image: bytes | None = field(default=None, repr=False)  # latest JPEG
    image_size: tuple[int, int] = (0, 0)
    microns_per_pixel: float | None = None

    def json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("image",)}


class ClariusProbe:
    """Owns the Solum SDK session. SDK callbacks arrive on SDK threads; all state goes
    through a lock and the pipeline reads a snapshot each frame."""

    def __init__(self, sdk_path: Path, store_dir: Path, ip: str, port: int, model: str,
                 application: str, cert: str | None, width: int = 640, height: int = 480):
        self.ip, self.port, self.model, self.application = ip, port, model, application
        self.cert = cert
        self._lib = C.CDLL(str(sdk_path))
        self._lock = threading.Lock()
        self._s = ClariusState()
        self._stop = threading.Event()
        self._keep = []  # C callbacks must outlive the SDK session
        self._trying = port

        lib = self._lib
        lib.solumDefaultInitParams.restype = _InitParams
        lib.solumGetParam.restype = C.c_double
        p = lib.solumDefaultInitParams()
        self._argv = (C.c_char_p * 1)(b"vigil")
        p.args.argc, p.args.argv = 1, self._argv
        store_dir.mkdir(parents=True, exist_ok=True)
        self._store = str(store_dir).encode()
        p.storeDir = self._store
        p.connectFn = self._cb(_ConnectFn, self._on_connect)
        p.certFn = self._cb(_CertFn, self._on_cert)
        p.powerDownFn = self._cb(_PowerDownFn, self._on_power_down)
        p.imagingFn = self._cb(_ImagingFn, self._on_imaging)
        p.buttonFn = self._cb(_ButtonFn, lambda btn, clicks: None)
        p.errorFn = self._cb(_ErrorFn, self._on_error)
        p.newProcessedImageFn = self._cb(_ImageFn, self._on_image)
        p.width, p.height = width, height
        if lib.solumInit(C.byref(p)) != 0:
            raise RuntimeError("solumInit failed")
        lib.solumSetFormat(FORMAT_JPEG)  # frames arrive ready to forward to the dashboard
        self._thread = threading.Thread(target=self._supervise, name="clarius", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- public
    def snapshot(self) -> ClariusState:
        with self._lock:
            return ClariusState(**self._s.__dict__)

    def set_param(self, name: str, value: float) -> None:
        param = {"depth": PARAM_DEPTH, "gain": PARAM_GAIN}[name]
        if name == "gain":
            self._lib.solumSetParam(2, C.c_double(0))  # AutoGain off, or it overrides the user
        self._lib.solumSetParam(param, C.c_double(float(value)))

    def set_running(self, run: bool) -> None:
        self._lib.solumRun(1 if run else 0)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        try:
            if self._lib.solumIsConnected() == 1:  # disconnect blocks ~3 s even with no link
                self._lib.solumRun(0)
                self._lib.solumDisconnect()
        finally:
            self._lib.solumDestroy()

    # ---------------------------------------------------------------- internals
    def _cb(self, ftype, fn):
        f = ftype(fn)
        self._keep.append(f)
        return f

    def _update(self, **kv) -> None:
        with self._lock:
            for k, v in kv.items():
                setattr(self._s, k, v)

    def _status(self, text: str) -> None:
        """Supervisor progress messages; never override what the SDK callbacks reported."""
        with self._lock:
            if not self._s.connected:
                self._s.state = text

    def _supervise(self) -> None:
        """Reconnect while disconnected; poll status once a second while connected."""
        last_try = 0.0
        candidates: list[int] = []
        while not self._stop.is_set():
            connected = self._lib.solumIsConnected() == 1
            if not connected and time.monotonic() - last_try > 3.0:
                if not on_probe_network(self.ip):
                    self._status(f"not on the probe's Wi-Fi ({self.ip})")
                    self._stop.wait(3.0)
                    continue
                if not self.port and not candidates:
                    # No Bluetooth to be told the control port: find the probe's open ports.
                    self._status(f"searching {self.ip} for the probe's control port…")
                    candidates = scan_ports(self.ip)
                    if not candidates:
                        self._status(f"probe not reachable at {self.ip} — join its Wi-Fi")
                        self._stop.wait(3.0)
                        continue
                port = self.port or candidates.pop(0)
                last_try = time.monotonic()
                self._status(f"connecting to {self.ip}:{port}…")
                self._trying = port
                self._lib.solumConnect(C.byref(_ConnectionParams(self.ip.encode(), port, 0)))
            if connected:
                if not self.port:
                    self.port = self._trying  # found it; remember for reconnects
                    log.info("clarius control port is %d (set clarius.port to skip the scan)", self.port)
                self._poll()
            self._stop.wait(1.0)

    def _poll(self) -> None:
        st = _StatusInfo()
        if self._lib.solumStatusInfo(C.byref(st)) == 0:
            self._update(battery=st.battery, temperature=st.temperature,
                         fps=round(st.frameRate, 1), charging=st.charger == 1)
        depth = self._lib.solumGetParam(PARAM_DEPTH)
        gain = self._lib.solumGetParam(PARAM_GAIN)
        self._update(depth_cm=None if depth < 0 else round(depth, 1),
                     gain=None if gain < 0 else round(gain, 1),
                     imaging=self._lib.solumIsImaging() == 1)

    def _ranges(self) -> None:
        for name, param in (("depth_range", PARAM_DEPTH), ("gain_range", PARAM_GAIN)):
            r = _Range()
            if self._lib.solumGetRange(param, C.byref(r)) == 0:
                self._update(**{name: (r.min, r.max)})

    # SDK callbacks (SDK threads) ------------------------------------------------
    def _on_connect(self, res: int, port: int, status: bytes) -> None:
        msg = (status or b"").decode(errors="replace")
        if res == CONNECTED:
            self._update(connected=True, state="connected", error=None)
            if self.cert:
                self._lib.solumSetCert(self.cert.encode())
            else:
                self._update(error="No probe certificate (set CLARIUS_CERT in .env)")
            # Loading the preset triggers imagingFn(ImagingReady) when done.
            self._lib.solumLoadApplication(self.model.encode(), self.application.encode())
            self._update(state=f"loading {self.model} / {self.application}…")
        elif res == SW_UPDATE:
            self._update(connected=False, state="probe firmware update required",
                         error="Probe firmware doesn't match this SDK; update it with the Clarius app.")
        elif res == OS_UPDATE:
            self._update(connected=False, state="probe OS update required")
        else:
            self._update(connected=False, imaging=False,
                         state="disconnected" if res == DISCONNECTED else "connection failed",
                         error=msg or None)
        log.info("clarius connect %s %s", res, msg)

    def _on_cert(self, days: int) -> None:
        self._update(cert_days=days)
        if days < 0:
            self._update(error="Probe certificate invalid")

    def _on_power_down(self, reason: int, seconds: int) -> None:
        reasons = ["idle", "too hot", "low battery", "button", "docked", "software"]
        self._update(state=f"powering down ({reasons[reason] if 0 <= reason < 6 else reason})")

    def _on_imaging(self, state: int, imaging: int) -> None:
        name = IMAGING_STATES[state] if 0 <= state < len(IMAGING_STATES) else str(state)
        self._update(imaging=bool(imaging), state=f"imaging: {name}" if imaging else name)
        if state == IMAGING_READY:
            self._ranges()
            if not imaging:
                self._lib.solumRun(1)  # start streaming as soon as the preset is loaded
        elif state == CERT_EXPIRED:
            self._update(error="Probe certificate expired")

    def _on_error(self, code: int, msg: bytes) -> None:
        text = (msg or b"").decode(errors="replace")
        log.warning("clarius error %s: %s", code, text)
        self._update(error=text)

    def _on_image(self, img, info_p, npos, pos) -> None:
        info = info_p.contents
        data = C.string_at(img, info.imageSize)  # copy out of the SDK's buffer
        with self._lock:
            s = self._s
            s.image, s.image_seq = data, s.image_seq + 1
            s.image_size = (info.width, info.height)
            s.microns_per_pixel = info.micronsPerPixel


def on_probe_network(ip: str) -> bool:
    """Is one of our interfaces on the probe's access-point subnet (/24)?"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((ip, 9))  # no packet is sent; this just asks the routing table
            local = s.getsockname()[0]
        except OSError:
            return False
    return local.rsplit(".", 1)[0] == ip.rsplit(".", 1)[0]


def scan_ports(ip: str, timeout: float = 0.4, concurrency: int = 1024,
               ports: range = range(1, 65536)) -> list[int]:
    """Open TCP ports on the probe (its control port among them), lowest first."""
    import asyncio

    async def probe(port: int, sem: asyncio.Semaphore) -> int | None:
        async with sem:
            try:
                _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout)
                w.close()
                return port
            except (OSError, asyncio.TimeoutError):
                return None

    async def run() -> list[int]:
        sem = asyncio.Semaphore(concurrency)
        found = await asyncio.gather(*(probe(p, sem) for p in ports))
        return [p for p in found if p]

    return asyncio.run(run())


def from_config(cfg) -> ClariusProbe | None:
    c = cfg.clarius
    if not c.enabled:
        return None
    sdk = cfg.resolve(c.sdk_path)
    if not sdk.is_file():
        raise RuntimeError(f"Solum SDK not found at {sdk}. Run `uv run vigil setup`.")
    cert = os.environ.get("CLARIUS_CERT") or None
    cert_path = os.environ.get("CLARIUS_CERT_PATH")
    if not cert and cert_path:
        path = cfg.resolve(Path(cert_path))
        if not path.is_file():
            raise RuntimeError(f"CLARIUS_CERT_PATH not found: {path}")
        cert = path.read_bytes().decode()  # exactly as issued (CRLF line endings kept)
    return ClariusProbe(sdk, cfg.resolve(c.store_dir), c.ip, c.port, c.model, c.application, cert,
                        c.width, c.height)
