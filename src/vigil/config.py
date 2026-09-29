"""YAML configuration + .env secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DepthFilters(_Strict):
    spatial: bool = False
    temporal: bool = True
    hole_filling: bool = False


class CamerasConfig(_Strict):
    serials: list[str] = []  # [] = every connected RealSense, sorted; the first is the world frame
    modes: list[tuple[int, int, int]] = [(848, 480, 30), (640, 480, 30), (640, 480, 15)]
    usb2_modes: list[tuple[int, int, int]] = [(640, 480, 30)]
    color_exposure_ms: float | None = 5.0  # None = auto-exposure
    color_gain: float | None = 128.0
    depth_min_m: float = 0.2
    depth_max_m: float = 6.0
    depth_filters: DepthFilters = DepthFilters()
    extrinsics_path: Path = Path("configs/extrinsics.yaml")


class DetectorConfig(_Strict):
    model_id: str = "PekingU/rtdetr_v2_r50vd"
    score_threshold: float = 0.5
    select: Literal["closest", "largest", "confident"] = "closest"
    max_distance_m: float = 5.0


class Sam3dBodyConfig(_Strict):
    hf_repo_id: str = "facebook/sam-3d-body-dinov3"
    repo_dir: Path = Path("third_party/sam-3d-body")
    inference_type: Literal["body", "full"] = "body"
    use_camera_intrinsics: bool = True
    anchor_to_depth: bool = True
    limb_radius_m: float = 0.06
    send_mesh: bool = True


class VitposeDepthConfig(_Strict):
    model_id: str = "usyd-community/vitpose-plus-base"
    min_score: float = 0.3
    depth_patch_px: int = 7
    limb_radius_m: float = 0.06
    max_depth_jump_m: float = 1.0
    compile: bool = True   # torch.compile + CUDA graphs (~4x faster, ~25 s at startup)
    half: bool = True


class EstimatorConfig(_Strict):
    backend: Literal["sam3d_body", "vitpose_depth"] = "sam3d_body"
    device: str = "cuda"
    cameras: Literal["all", "primary"] = "all"
    triangulate: bool = True  # 2+ calibrated views: triangulate joints instead of depth lookup
    triangulation_min_conf: float = 0.4
    triangulation_max_px: float = 15.0
    sam3d_body: Sam3dBodyConfig = Sam3dBodyConfig()
    vitpose_depth: VitposeDepthConfig = VitposeDepthConfig()


class ProbeConfig(_Strict):
    enabled: bool = True
    dictionary: str = "DICT_6X6_50"
    marker_length_m: float = 0.04
    cube_size_m: tuple[float, float, float] = (0.05, 0.05, 0.05)
    tip_in_object_m: tuple[float, float, float] = (0.0, 0.1925096, -0.003594)
    model_path: Path = Path("assets/probe/ClariusFinal.glb")
    filter_preset: Path = Path("configs/probe-filter.json")
    filter_method: Literal["raw", "kalman", "ekf", "one_euro"] | None = None
    calibration_samples: int = 60
    joint_solve: bool = True  # one solve over every corner in every calibrated camera
    use_depth: bool = True  # add measured depth at the tag corners to that solve
    max_tag_error_px: float = 3.0  # tags that don't fit the others by this much are dropped


class SmoothingConfig(_Strict):
    enabled: bool = True
    min_cutoff: float = 1.5
    beta: float = 0.3


class SceneConfig(_Strict):
    point_cloud: bool = True
    point_cloud_stride: int = 4
    point_cloud_every_n: int = 2
    fused_voxel_m: float = 0.01
    floor_detection: bool = True
    floor_max_tilt_deg: float = 75.0
    floor_every_n: int = 15


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    preview_width: int = 424
    preview_jpeg_quality: int = 70


class Config(_Strict):
    cameras: CamerasConfig = CamerasConfig()
    detector: DetectorConfig = DetectorConfig()
    estimator: EstimatorConfig = EstimatorConfig()
    probe: ProbeConfig = ProbeConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    scene: SceneConfig = SceneConfig()
    server: ServerConfig = ServerConfig()

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_env() -> None:
    """Load secrets from the project's .env (existing env vars win)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    # An empty placeholder (HF_TOKEN=) must not shadow a token cached by `hf auth login`.
    if os.environ.get("HF_TOKEN") == "":
        del os.environ["HF_TOKEN"]


def load_config(path: str | Path | None = None) -> Config:
    load_env()
    path = Path(path or os.environ.get("VIGIL_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
