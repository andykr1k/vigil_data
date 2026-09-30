"""Clarius probe over the Solum SDK (libsolum, bound with ctypes).

Connects over the probe's Wi-Fi (this machine joins the probe's access point), applies
the certificate, loads a preset, streams B-mode images as JPEG, and reports battery,
temperature and frame rate. Depth and gain can be changed live. The startup sequence
follows vigil-system's solumultrasoundnode (sensors package).
"""

from __future__ import annotations

import ctypes as C
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Enum values from solum_def.h
CONNECTED, DISCONNECTED, CONNECTION_FAILED, SW_UPDATE = 0, 1, 2, 3
IMAGING_READY, CERT_EXPIRED = 1, 2
IMAGING_STATES = ["not ready", "ready", "certificate expired", "poor Wi-Fi", "no contact",
                  "charging changed", "low bandwidth", "motion sensor", "no tee", "tee expired"]
PARAM_DEPTH, PARAM_GAIN, PARAM_AUTO_GAIN, PARAM_ECO = 0, 1, 2, 24  # CusParam
MODE_B = 0  # CusMode.BMode
ERROR_VERSION_MISMATCH = 4  # reported while the probe still verifies its firmware
FORMAT_JPEG = 2  # CusImageFormat.Jpeg
TIMEOUT = 20.0  # seconds to wait for each SDK callback during startup
LOAD_ATTEMPTS, CONNECT_ATTEMPTS = 3, 4

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


class _ProbeSettings(C.Structure):
    _fields_ = [(n, C.c_int) for n in (
        "contactDetection autoFreeze keepAwake deepSleep stationary powerFan autoBoot "
        "wifiOptimization htWifi keepAwakeCharging powerOn sounds wakeOnShake "
        "bandwidthOptimization forceLogSend imageOnUndock alarmOnUndock up down handle "
        "upHold downHold").split()]


# Never freeze or sleep on its own (timeouts 0); holding a button powers it off.
_SETTINGS = _ProbeSettings(powerFan=1, powerOn=1, sounds=1, wakeOnShake=1, upHold=1, downHold=1)


class _Range(C.Structure):
    _fields_ = [("min", C.c_double), ("max", C.c_double)]


class _Failed(RuntimeError):
    pass


@dataclass
class ClariusState:
    connected: bool = False
    imaging: bool = False
    state: str = "disconnected"  # human-readable connection/imaging state
    battery: int | None = None  # %
    temperature: int | None = None  # % of the probe's thermal limit (Solum has no °C)
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
    """Owns the Solum SDK session. Callbacks arrive on SDK threads and only record state
    and signal events; the session thread makes every SDK call. The pipeline reads a
    snapshot each frame."""

    def __init__(self, sdk_path: Path, store_dir: Path, ip: str, port: int, model: str,
                 application: str, cert: str | None, width: int = 640, height: int = 480):
        self.ip, self.port, self.model, self.application = ip, port, model, application
        self.cert, self.size = cert, (width, height)
        store_dir.mkdir(parents=True, exist_ok=True)
        self._store = str(store_dir).encode()
        self._lock = threading.Lock()
        self._s = ClariusState()
        self._stop = threading.Event()
        self._ev = {k: threading.Event() for k in ("connect", "cert", "app", "imaging")}
        self._result = self._error_code = None
        self._update_required = False

        lib = self._lib = C.CDLL(str(sdk_path))
        lib.solumDefaultInitParams.restype = _InitParams
        lib.solumGetParam.restype = C.c_double
        lib.solumSetParam.argtypes = [C.c_int, C.c_double]
        self._keep = [_ConnectFn(self._on_connect), _CertFn(self._on_cert),
                      _PowerDownFn(self._on_power_down), _ImagingFn(self._on_imaging),
                      _ButtonFn(lambda btn, clicks: None), _ErrorFn(self._on_error),
                      _ImageFn(self._on_image)]  # C callbacks must outlive the SDK
        self._init()
        self._thread = threading.Thread(target=self._run, name="clarius", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- public
    def snapshot(self) -> ClariusState:
        with self._lock:
            return ClariusState(**self._s.__dict__)

    def set_param(self, name: str, value: float) -> None:
        param = {"depth": PARAM_DEPTH, "gain": PARAM_GAIN}[name]
        if name == "gain":
            self._lib.solumSetParam(PARAM_AUTO_GAIN, 0)  # or auto gain overrides the user
        self._lib.solumSetParam(param, float(value))

    def set_running(self, run: bool) -> None:
        self._lib.solumRun(1 if run else 0)

    def close(self) -> None:
        self._stop.set()
        for e in self._ev.values():
            e.set()  # release a startup wait
        self._thread.join(timeout=10)  # the session thread releases the SDK

    # ---------------------------------------------------------------- session
    def _init(self) -> None:
        p = self._lib.solumDefaultInitParams()
        p.args.argc, p.args.argv = 0, None
        p.storeDir = self._store
        (p.connectFn, p.certFn, p.powerDownFn, p.imagingFn, p.buttonFn, p.errorFn,
         p.newProcessedImageFn) = self._keep
        p.width, p.height = self.size
        if self._lib.solumInit(C.byref(p)) != 0:
            raise RuntimeError("solumInit failed")
        self._lib.solumSetProbeSettings(C.byref(_SETTINGS))

    def _release(self) -> None:
        if self._lib.solumIsConnected() == 1:  # disconnect blocks ~3 s even with no link
            self._lib.solumRun(0)
            self._lib.solumDisconnect()
        self._lib.solumDestroy()

    def _run(self) -> None:
        """Start a session, stream until it drops, then start over with a fresh SDK."""
        try:
            self._sessions()
        finally:
            self._release()

    def _sessions(self) -> None:
        while not self._stop.is_set():
            if not on_probe_network(self.ip):
                self._update(state=f"not on the probe's Wi-Fi ({self.ip})")
                self._stop.wait(3.0)
                continue
            try:
                self._start()
                while not self._stop.is_set() and self._lib.solumIsConnected() == 1:
                    self._poll()
                    self._stop.wait(1.0)
            except _Failed as e:
                if self._stop.is_set():
                    return
                log.warning("clarius: %s", e)
                self._update(error=str(e))
            if self._stop.is_set():
                return
            self._update(connected=False, imaging=False)
            self._release()  # like vigil-system: retry on a freshly initialised SDK
            self._stop.wait(30.0 if self._update_required else 2.0)
            self._init()

    def _start(self) -> None:
        """connect → JPEG output → certificate → preset (retried while the probe
        verifies its firmware) → imaging."""
        self._update_required = False
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            self._update(state=f"connecting to {self.ip}:{self.port}…")
            self._result = None
            msg = self._call("connect", lambda: self._lib.solumConnect(
                C.byref(_ConnectionParams(self.ip.encode(), self.port, 0))))
            if self._result == CONNECTED:
                break
            if "refused" not in (msg or "").lower() or attempt == CONNECT_ATTEMPTS:
                raise _Failed(f"connection failed: {msg or self._s.error}")
            self._stop.wait(2.0)
        self._lib.solumSeparateOverlays(0)
        self._lib.solumSetFormat(FORMAT_JPEG)  # frames arrive ready for the dashboard

        if not self.cert:
            raise _Failed("No probe certificate (set CLARIUS_CERT_PATH in .env)")
        self._update(cert_days=None)
        self._call("cert", lambda: self._lib.solumSetCert(self.cert.encode()))
        if not (self._s.cert_days or 0) > 0:
            raise _Failed(f"probe rejected the certificate (days valid: {self._s.cert_days})")

        for attempt in range(1, LOAD_ATTEMPTS + 1):
            self._update(state=f"loading {self.model} / {self.application}…")
            self._error_code = None
            self._call("app", lambda: self._lib.solumLoadApplication(
                self.model.encode(), self.application.encode()))
            if self._error_code != ERROR_VERSION_MISMATCH:
                break
            if attempt == LOAD_ATTEMPTS:
                raise _Failed("probe still reports a software version mismatch")
            log.info("clarius: firmware verification pending; retrying the preset")
            self._stop.wait(1.0)
        self._update(error=None)
        self._ranges()
        self._call("imaging", lambda: self._lib.solumRun(1))
        self._lib.solumSetParam(PARAM_ECO, 0)
        self._lib.solumSetMode(MODE_B)  # as vigil-system: never left in Color Doppler

    def _call(self, event: str, fn) -> str | None:
        """Make an SDK call and wait for the callback it triggers."""
        ev = self._ev[event]
        ev.clear()
        if fn() != 0:
            raise _Failed(f"SDK refused '{event}'")
        if not ev.wait(TIMEOUT):
            raise _Failed(f"timed out waiting for '{event}' ({self._s.error or 'no reply'})")
        if self._update_required:
            raise _Failed("probe requires a firmware update for this SDK")
        if self._stop.is_set():
            raise _Failed("stopped")
        return self._s.error

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

    def _update(self, **kv) -> None:
        with self._lock:
            for k, v in kv.items():
                setattr(self._s, k, v)

    # SDK callbacks (SDK threads): record and signal, never call back into the SDK -------
    def _on_connect(self, res: int, port: int, status: bytes) -> None:
        msg = (status or b"").decode(errors="replace")
        self._result = res
        if res == CONNECTED:
            self._update(connected=True, state="connected", error=None)
        else:
            self._update_required |= res == SW_UPDATE
            self._update(connected=False, imaging=False, error=msg or None,
                         state={DISCONNECTED: "disconnected", SW_UPDATE: "firmware update "
                                "required"}.get(res, "connection failed"))
            for e in self._ev.values():
                e.set()  # nothing else is coming for a pending wait
        self._ev["connect"].set()
        log.info("clarius connect %s %s", res, msg)

    def _on_cert(self, days: int) -> None:
        self._update(cert_days=days)
        self._ev["cert"].set()

    def _on_power_down(self, reason: int, seconds: int) -> None:
        reasons = ["idle", "too hot", "low battery", "button", "docked", "software"]
        self._update(state=f"powering down ({reasons[reason] if 0 <= reason < 6 else reason})")

    def _on_imaging(self, state: int, imaging: int) -> None:
        name = IMAGING_STATES[state] if 0 <= state < len(IMAGING_STATES) else str(state)
        self._update(imaging=bool(imaging), state=f"imaging: {name}" if imaging else name)
        if state == IMAGING_READY:
            self._ev["app"].set()
        if imaging:
            self._ev["imaging"].set()
        if state == CERT_EXPIRED:
            self._update(error="Probe certificate expired")

    def _on_error(self, code: int, msg: bytes) -> None:
        text = (msg or b"").decode(errors="replace")
        log.warning("clarius error %s: %s", code, text)
        self._error_code = code
        self._update(error=text)
        if code == ERROR_VERSION_MISMATCH:
            self._ev["app"].set()  # the preset load won't report ready

    def _on_image(self, img, info_p, npos, pos) -> None:
        info = info_p.contents
        data = C.string_at(img, info.imageSize)  # copy out of the SDK's buffer
        with self._lock:
            s = self._s
            s.image, s.image_seq = data, s.image_seq + 1
            s.image_size = (info.width, info.height)
            s.microns_per_pixel = info.micronsPerPixel


def on_probe_network(ip: str) -> bool:
    """Does the routing table reach the probe's subnet (i.e. are we on its Wi-Fi)?"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((ip, 9))  # no packet is sent; this just asks the routing table
            local = s.getsockname()[0]
        except OSError:
            return False
    return local.rsplit(".", 1)[0] == ip.rsplit(".", 1)[0]


def from_config(cfg, application: str) -> ClariusProbe | None:
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
    return ClariusProbe(sdk, cfg.resolve(c.store_dir), c.ip, c.port, c.model, application, cert,
                        c.width, c.height)
