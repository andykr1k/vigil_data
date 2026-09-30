# Vigil

Live leg (lower-limb) or chest (cardiac) pose and ArUco-cube probe tracking from one or more Intel RealSense cameras, rendered into a dark 3D scene in the browser at ≥30 fps.

```
camera process ×N ──► shared memory ──┬──► detector process (RT-DETR, transformers) ──┐
 (capture, align)                     │                                               ▼
                                      ├──► body thread: ViTPose (compiled, batched) / SAM 3D Body
                                      │      → world frame → multi-camera fusion → One Euro
                                      └──► fast loop @ camera rate: ArUco probe (per camera → fused
                                             → ESKF), feeds, point clouds, floor ──► websocket ──► three.js
```

## Quick start

```bash
cp .env.example .env        # then add HF_TOKEN (see below)
uv sync
uv run vigil setup          # clones SAM 3D Body code, downloads all weights
uv run vigil run            # → http://localhost:8000
```

No SAM 3D Body access yet? Run the transformers-only backend. It needs no gated weights:

```bash
uv run vigil run --backend vitpose_depth
```

Other commands: `uv run vigil camera` lists and checks every RealSense, and `uv run vigil -c my.yaml run` uses a different config. Tests: `uv run pytest`.

The first start compiles the pose model with CUDA graphs (about 30 s). Later starts reuse torch's compile cache.

## Pose backends

| backend | model | how 3D is obtained | notes |
|---|---|---|---|
| `sam3d_body` (default) | Meta **SAM 3D Body** (DINOv3-H+, MHR body model) | full mesh + 70 keypoints regressed from RGB, using the RealSense intrinsics; distance/scale corrected with measured depth | Most accurate, and robust to occlusion. Returns a body mesh plus feet (heels, toes), so ankle angles work. Gated weights. |
| `vitpose_depth` | **ViTPose+** via `transformers` | COCO-17 2D keypoints deprojected through the aligned depth map | Metric positions, but it breaks when a joint is occluded because the depth hits whatever is in front. No feet or mesh. |

**Why SAM 3D Body over SMPL:** SMPL is a body *model*, not an estimator. You'd still need an SMPL regressor (HMR2.0, CameraHMR, NLF, …). SAM 3D Body outperforms those on the standard benchmarks (3DPW MPJPE 54.8 mm), and SMPL itself needs a separate license. SAM 3D Body isn't in `transformers` yet, so it's loaded from Meta's repo (pinned commit, cloned into `third_party/`). Person detection and the ViTPose backend use `transformers`.

### Getting SAM 3D Body weights

1. Request access at <https://huggingface.co/facebook/sam-3d-body-dinov3> (and/or `-vith` for a smaller, faster model).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in `.env` as `HF_TOKEN=hf_...` and run `uv run vigil setup`.

## Multiple cameras

Every connected RealSense is used (or the `cameras.serials` list). The first serial, in sorted order, defines the **world frame**. The others need extrinsics, which the ArUco probe provides:

1. Hold the probe where **every** camera sees at least one tag.
2. Press **CALIBRATE RIG** (top-right RIG panel).
3. Keep it still at a spot for a second or two, then move to a new spot. Repeat until the bar fills (60 still frames per camera by default).

Each still frame gives `T_world←cam = T_world←cube · T_cam←cube⁻¹`. Frames where the cube moved, flipped poses, and outliers are rejected. The rest are averaged (chordal mean for rotation). The result goes to `configs/extrinsics.yaml` with its standard error. Once calibrated, a camera's point cloud and frustum join the fused scene, its body keypoints are triangulated with the others, and its tag corners join the probe solve.

**Calibration health:** whenever the world camera and another camera each see the probe on their own, the RIG panel shows how well their poses agree (rolling median, e.g. "agrees with CAM 1 to 4 mm / 1.1° (GOOD)"). If it turns FAIR or POOR, a camera was probably bumped, so recalibrate.

**Time sync:** frames carry the cameras' own capture timestamps (librealsense global time), and each other camera's frame is paired with the world camera's by capture time. If the other camera's next frame is closer and only milliseconds away, the pairing waits for it. The RIG header shows the remaining skew.

A camera on a USB 2 link uses `cameras.usb2_modes` (640×480@30). If a camera stops delivering frames (unplugged, link reset), its process reconnects automatically once it re-enumerates.

## Probe (ArUco cube)

Ported from the DataCollection project into `src/vigil/probe/`: DICT_6X6_50 tags 0–4 on a 50 mm cube (front, back, right, left, bottom; the Clarius attachment is on +Y). Per frame (`probe/tracker.py`, `probe/solver.py`):

1. **Detect** tags in every camera, in parallel, searching around the last known cube position (with a full-frame pass every 15 frames or when it's lost).
2. **Clean each view.** With 3+ tags, a tag that the *others* fail to predict (leave-one-out) is dropped: a bad detection or a loose tag. Two tags that disagree keep the larger one. A single tag has two mirror-image poses (the wrong one can even fit better), so pick the one consistent with the previous pose, or with another camera that sees 2+ tags. Rejected tags show red in the probe panel.
3. **Joint solve:** one Gauss–Newton fit of the cube's world pose over every kept corner in every calibrated camera, plus depth measured *inside* each tag (edges often hit the background). It uses an analytic Jacobian and robust weighting and takes about 2 ms. A single tag is weak along its viewing ray, and the other camera and the depth sensor constrain exactly that. The panel shows the fit (px, corners, depth residual).
4. **Filter:** SE(3) error-state EKF (default, tuned in `configs/probe-filter.json`), One Euro, position Kalman, or raw.

**Exposure matters most.** The colour cameras run a fixed 5 ms exposure with high gain (`cameras.color_exposure_ms`, `color_gain`). Auto-exposure picks ~17 ms indoors, which smears a moving tag's ~3 px code cells. Measured with the probe moving: tracked in 99.6% of frames at 5 ms vs 41% on auto. If the image is too dark in your room, raise `color_gain` before lengthening the exposure.



The dashboard shows the Clarius model, the five textured tags on their faces, the cube axes, and the **red tip dot** (`probe.tip_in_object_m`, 192.5 mm along +Y) with a trail. It also reports the tip's XYZ above the floor and its distance to the nearest leg bone (thigh / shin / foot axis), drawn as a gold dashed line. The feeds show detected tag outlines and the projected tip.

## Body pose across cameras

With 2+ calibrated cameras, each joint whose 2D keypoint is confident in at least two views is **triangulated** from the rays rather than read from the depth map. Depth lookup fails when the joint is occluded (it hits whatever is in front) or sits at a silhouette edge. Views are chosen by consensus: every pair proposes a point, and the largest set that reprojects within 15 px wins, so one camera confusing left and right can't drag the result. A two-view point must also land within 30 cm of some camera's depth estimate. Joints seen by one camera keep the depth-based position. Body and probe run in separate loops, so the "nearest bone" distance uses the joints extrapolated to the probe frame's capture time.

**Floor:** the lowest large horizontal surface in the world camera's cloud (seats and tables are horizontal too), tracked over time. A different plane replaces it only after three consistent fits, and it works with cameras looking steeply down (up to 75°).

## Recording and replay

The **RECORDING** box records everything to `recordings/<date>_<procedure>/`, in formats other tools open directly:

- `cloud/*.ply`: the fused point cloud (world frame, metres, RGB, binary PLY), for CloudCompare, MeshLab, Blender, Open3D or PCL.
- `cam<i>/color/*.jpg` and `cam<i>/depth/*.png`: each RealSense's colour image, plus depth as uint16 millimetres aligned to it (the TUM / Open3D RGB-D convention).
- `meta.json`: intrinsics and `T_world_camera` per camera. `frames.jsonl`: per-frame timestamps, probe pose and tip, and body joints.
- `ultrasound/*.jpg`: the Clarius frames.
- Each recording has a `README.txt` with an Open3D example.

Files share one frame number, so a frame's images, cloud, probe pose and ultrasound line up. Writes run on a thread pool. If the disk can't keep up, frames are dropped (and counted) rather than slowing the live view. That's about 60 MB/s for two cameras.

**REPLAY** plays a recording back through the dashboard in place of live data: 3D scene, point cloud, skeleton, probe, camera feeds and ultrasound. It has play/pause, speed, and a slider for scrubbing. **BACK TO LIVE** returns to the cameras.

## Configuration

- `configs/default.yaml` holds all tunables: cameras and stream modes, depth filters, detector and person selection, procedures (probe preset per procedure), backend options, probe geometry and filter, smoothing, point cloud and floor detection, server.
- `configs/probe-filter.json` holds the probe filter method and noise tuning. `configs/extrinsics.yaml` is written by rig calibration.
- `.env` holds secrets only (`HF_TOKEN`) and is git-ignored. Optionally set `VIGIL_CONFIG` there to choose a different YAML.

Useful knobs:

- `detector.select`: `closest` | `largest` | `confident`, with `max_distance_m`. Once a subject is picked, it stays tracked while visible.
- `clarius.procedures`: the startup choices and the probe preset each loads (`cardiac: cardiac`, `lower_limb: dvt`). Lower Limb tracks the legs (with joint angles); Cardiac tracks the chest (shoulders, hips and a derived chest centre).
- `estimator.sam3d_body.inference_type: full` also refines hands (slower). Swap `hf_repo_id` to `facebook/sam-3d-body-vith` for speed.
- `smoothing.min_cutoff` / `beta`: lower min_cutoff means steadier; higher beta means less lag on fast moves.
- `scene.floor_detection`: RANSAC ground plane from depth, which levels the scene and puts the grid on the real floor.

## Dashboard

- **Top-left:** link status, stream and body FPS (amber below 30), latency, per-stage timings, target lock (leg or chest). Below that, hip / knee / ankle angles (L/R) with a 10 s knee-flexion trace, then the PROBE panel: visible tag chips, tip XYZ, nearest bone and distance, filter selector, and reset.
- **Top-right:** display toggles, camera presets, the RIG panel (cameras, calibration state, CALIBRATE RIG), and one feed per camera with keypoints, tag outlines and the tip dot.
- Drag to orbit, scroll to zoom, right-drag to pan. Toggle and panel states are remembered per browser.
- `window.vigil` in the browser console exposes the scene, camera and latest frame for debugging.

Angle conventions: knee flexion 0° = straight leg; hip flexion 0° = thigh in line with the trunk; ankle = shin–foot angle (≈90° standing; SAM 3D Body only).

## Layout

```
configs/default.yaml      tunables
src/vigil/
  cli.py                  vigil run | setup | camera
  config.py               YAML + .env loading (pydantic)
  camera.py               RealSense capture, filters, depth→color alignment, reconnect
  capture.py              one capture process per camera, triple-buffered shared memory
  rig.py                  multi-camera world frame, extrinsics file, rig calibration
  detector.py             transformers RT-DETR person detector
  detect_worker.py        detector in its own process (reads camera shared memory)
  recorder.py             session recording (PLY clouds, RGB-D, meta, per-frame JSONL)
  replay.py               replays a recording through the dashboard protocol
  startup.py              startup checklist, progress and time left for the dashboard
  body.py                 body thread: subject tracking, batched pose, triangulation, smoothing
  estimators/             sam3d_body.py, vitpose_depth.py (compiled, batched)
  probe/                  ArUco detection, cube pose, per-view cleaning + joint solve (solver.py),
                          filters (EKF/One Euro/Kalman), tracker
  skeleton.py             canonical joints, bones, leg/chest regions, leg angles
  geometry.py             depth sampling, deprojection, point cloud, floor detection
  filters.py              vectorised One Euro filter
  pipeline.py             camera-rate loop: probe, feeds, clouds, floor → websocket
  server.py               FastAPI app + websocket hub
  web/                    index.html, style.css, app.js (three.js from CDN)
```

Websocket protocol: JSON `hello` / `status` / `calibration` / `frame` messages from the server, and `{"cmd": ...}` from the dashboard (`start` with the chosen procedure, `probe_filter`, `probe_reset`, `calibrate_rig`, `cancel_calibration`, `depth_range`, `record`, `recordings`, `replay`, `replay_ctl`, `replay_stop`). Binary messages have an 8-byte header (`u8 kind, u8 camera, 2 pad, u32 seq`): kind 1 is mesh vertices (int16 mm), 2 is a JPEG preview, 3 is a point cloud (int16 mm xyz + rgb, in that camera's frame). Other coordinates are in the world frame: the first camera's color optical frame (x right, y down, z forward, metres).

## Performance notes (RTX 3090, 848×480)

Measured with two D435s (both 848×480@30 on USB 3) and the ViTPose backend, probe in view of both: stream ≥ 29.9 fps (median 31.8), body 30.4 fps, probe tracking ~19 ms per frame for both cameras, paired-frame skew ~0.5 ms. Probe at rest: tip jitter 0.7 / 0.6 / 2.0 mm (x / y / z), joint-solve fit ~1.9 px, and the two cameras' independent poses agree to 3.6 mm / 1.1°.

- librealsense's `align`/filters hold the Python GIL (about 8 ms per frame per camera), so capture runs in **one process per camera**. The spatial filter (15 ms) is off by default.
- RT-DETR (about 45 ms, launch-bound) runs in **its own process**. The body loop never waits for it: it uses the newest matched box, or a box around the last keypoints.
- ViTPose runs **batched across cameras** and compiled with CUDA graphs: about 4 ms for two crops instead of about 16 ms per crop eager. Crops use cv2 instead of scipy, and heatmap decoding uses one multi-channel cv2 blur. Keypoints match the transformers processor to within 0.15 px.
- SAM 3D Body (840M parameters) will run slower than 30 fps on the body thread. The stream, probe, and feeds still run at camera rate.
- The dashboard loads three.js from jsDelivr, so the browser needs internet access.
