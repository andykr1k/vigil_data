"""`vigil` command line: run the dashboard, fetch models, check the camera."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

from .config import load_config

# Pinned upstream commit of facebookresearch/sam-3d-body (not pip-installable).
SAM3D_REPO = "https://github.com/facebookresearch/sam-3d-body"
SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"


def cmd_run(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    cfg = load_config(args.config)
    if args.backend:
        cfg.estimator.backend = args.backend
    host, port = args.host or cfg.server.host, args.port or cfg.server.port
    print(f"\n  Vigil dashboard → http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}\n")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


def cmd_setup(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    cfg = load_config(args.config)
    s3 = cfg.estimator.sam3d_body
    repo_dir = cfg.resolve(s3.repo_dir)

    if not (repo_dir / ".git").is_dir():
        print(f"→ cloning SAM 3D Body into {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", SAM3D_REPO, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", SAM3D_COMMIT], check=True)
    print(f"✓ SAM 3D Body code @ {SAM3D_COMMIT[:8]}")

    print("→ caching DINOv3 backbone code (torch.hub)")
    torch.hub.list("facebookresearch/dinov3", trust_repo=True, verbose=False)

    print("→ downloading transformers models")
    snapshot_download(cfg.detector.model_id)
    snapshot_download(cfg.estimator.vitpose_depth.model_id)
    print("✓ detector + ViTPose")

    print(f"→ downloading {s3.hf_repo_id} (~2 GB, gated)")
    try:
        snapshot_download(s3.hf_repo_id)
        print("✓ SAM 3D Body weights")
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print(
            f"✗ could not download {s3.hf_repo_id}: {type(e).__name__}\n"
            f"  1. Request access at https://huggingface.co/{s3.hf_repo_id}\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Put it in .env as HF_TOKEN=hf_... and re-run `uv run vigil setup`\n"
            "  Meanwhile: `uv run vigil run --backend vitpose_depth`",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_camera(args: argparse.Namespace) -> None:
    import numpy as np

    from .camera import RealSenseCamera, connected_serials

    cfg = load_config(args.config)
    serials = connected_serials()
    print(f"{len(serials)} RealSense camera(s) connected: {', '.join(serials) or '—'}")
    for serial in serials:
        cam = RealSenseCamera(cfg.cameras, serial)
        w, h, fps = cam.mode
        print(f"\n{cam.name} {serial}  USB {cam.usb}  {w}x{h}@{fps}")
        print(f"  {cam.intrinsics}")
        frame, n, t0 = None, 0, time.time()
        while time.time() - t0 < 3:
            f = cam.latest(frame.index if frame else -1)
            if f is not None:
                frame, n = f, n + 1
        cam.close()
        if frame is None:
            print("  no frames!")
            continue
        d = frame.depth
        print(f"  {n / 3:.1f} fps  valid depth {100 * (d > 0).mean():.0f}%  "
              f"median {np.median(d[d > 0]):.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vigil", description="RealSense leg pose dashboard")
    parser.add_argument("-c", "--config", help="YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="start the pose pipeline and web dashboard")
    run.add_argument("--backend", choices=["sam3d_body", "vitpose_depth"])
    run.add_argument("--host")
    run.add_argument("--port", type=int)
    run.set_defaults(func=cmd_run)

    sub.add_parser("setup", help="fetch SAM 3D Body code + all model weights").set_defaults(func=cmd_setup)
    sub.add_parser("camera", help="list and check the RealSense cameras").set_defaults(func=cmd_camera)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)
