# TSDF reconstruction and mesh unfolding: file guide

This guide separates project integration and processing code from third-party algorithms.
“Project code” describes its role in this repository, not verified personal authorship:
some modules explicitly document ports from earlier projects. Paths below are relative to
the repository root. Generated meshes, model weights, native build products and individual
transitive vendor headers are not application implementation files.

## Execution paths

Live (`testing.mode: 0`):

```text
web/app.js: SCAN + UNFOLD, selected mesh and unfolding settings
  → pipeline.py: short background capture through Recorder; live frames keep publishing
  → meshing/auto.py: run_recording()
      → autoleg.py: candidate analysis and automatic thigh/knee/hip prompts
      → tsdf.py: build_tsdf()
          → legmask.py: SAM 2 masks
          → alignment, support/section cuts, TSDF fusion, marching cubes
          → holes.py: hole filling; tsdf.py: smoothing and normals
      → library.py: save original 3D mesh
  → web/app.js: display the mesh in the live world frame, with no replay command
  → flatten.py: flatten_mesh() on that saved mesh
      → preparation → native/planar_solver.cpp → libigl solver → validation
  → library.py: save flat mesh, metadata and 2D↔3D correspondence
  → web/app.js: display the unfolded result beside the live overlay
```

Replay/testing (`testing.mode: 1`) retains its existing path: `replay.py` supplies recorded
frames; `routes.py` handles BUILD MESH and FLATTEN requests. The CLI can also run automatic
meshing over recordings through `auto.py`. Those paths share the same processing functions.

The live workflow changes orchestration and presentation, not the reconstruction or
unfolding algorithms. The live overlay is fixed at the reconstructed world position;
following a moving leg would require separate registration/tracking. Saved-mesh browsing
still supports opening recordings; the live scan result uses a separate display path.

## 1. Reconstruction: review these first

| File | Responsibility and useful entry points |
| --- | --- |
| [src/vigil/meshing/tsdf.py](../src/vigil/meshing/tsdf.py) | Main reconstruction implementation. `TsdfParams` defines settings; `Mesh` holds geometry and serialization; `build_tsdf()` orchestrates reading views, segmentation, alignment, cuts, fusion and cleanup. `_icp()`/`_align()` align frames, `_Volume.integrate()` performs TSDF fusion, and `_Volume.extract()` calls scikit-image marching cubes. Also contains support-plane fitting, anatomical crops, smoothing and normals. |
| [src/vigil/meshing/holes.py](../src/vigil/meshing/holes.py) | Shared surface repair: `boundary_loops()` identifies open edges; `fill_holes()` patches small holes while retaining the outer boundary. Used by reconstruction and unfolding preparation. |
| [src/vigil/meshing/legmask.py](../src/vigil/meshing/legmask.py) | `Prompts` parses clicks/boxes; `segment()` runs SAM 2 tracking for the relevant camera sequences; `_transfer()` carries prompts between calibrated views; `click_point()` converts anatomical clicks to world coordinates. SAM 2 itself is external. |
| [src/vigil/meshing/autoleg.py](../src/vigil/meshing/autoleg.py) | Automatic prompts. Loads Sapiens2 segmentation and pose models, interprets their outputs with measured depth, combines camera observations, and proposes thigh, knee and hip prompts through `propose()`. Also caches model analysis. Sapiens2 itself is external. |
| [src/vigil/meshing/auto.py](../src/vigil/meshing/auto.py) | Automatic recording-to-mesh orchestration. `score_frames()` ranks candidates; `precheck()` and `postcheck()` assess inputs/results; `run_recording()` tries candidates, calls the normal TSDF builder and saves accepted meshes; `run_batch()` handles multiple recordings. This is also the processing path used by live scans. |

Within `tsdf.py`, a useful reading order is `TsdfParams` → `build_tsdf()` → the helpers it
calls → `_Volume` → cleanup. This separates the reconstruction mathematics from the
application-specific choice of which leg pixels to reconstruct.

## 2. Unfolding

| File | Responsibility and useful entry points |
| --- | --- |
| [src/vigil/meshing/flatten.py](../src/vigil/meshing/flatten.py) | Python unfolding pipeline. `_prepare()` cleans topology, orients triangles, splits pinched vertices and patches small holes; the input must become a disk. `_solve()` invokes the native solver. `flatten_mesh()` checks finite coordinates, triangle orientation, boundary overlap and area consistency, computes distortion, and records source vertex/face correspondence in `FlatResult`. `flat_binary()` packages data for the browser. |
| [src/vigil/meshing/native/planar_solver.cpp](../src/vigil/meshing/native/planar_solver.cpp) | Project C++ wrapper. Reads vertices/faces, constructs the initial planar map, applies method-specific normalization/constraints, calls the selected libigl optimizer, reports progress and writes UV coordinates in physical units. The three optimization algorithms are not implemented by this wrapper. |
| [src/vigil/meshing/native/CMakeLists.txt](../src/vigil/meshing/native/CMakeLists.txt) | Builds the executable and Triangle dependency, locates Eigen and configures compiler options. |
| [src/vigil/meshing/native/SOURCES.md](../src/vigil/meshing/native/SOURCES.md) | Records algorithm references, vendor provenance/versions, solver configuration and build requirements. Start here when comparing the wrapper against the reference implementations. |

Underlying third-party implementations:

| File or dependency | Role |
| --- | --- |
| [vendor/libigl/igl/triangle/scaf.cpp](../src/vigil/meshing/native/vendor/libigl/igl/triangle/scaf.cpp), [scaf.h](../src/vigil/meshing/native/vendor/libigl/igl/triangle/scaf.h) | SCAF implementation and interface. |
| [vendor/libigl/igl/slim.cpp](../src/vigil/meshing/native/vendor/libigl/igl/slim.cpp), [slim.h](../src/vigil/meshing/native/vendor/libigl/igl/slim.h) | SLIM implementation and interface. |
| [vendor/libigl/igl/arap.cpp](../src/vigil/meshing/native/vendor/libigl/igl/arap.cpp), [arap.h](../src/vigil/meshing/native/vendor/libigl/igl/arap.h) | ARAP implementation and interface. |
| [vendor/libigl/igl/](../src/vigil/meshing/native/vendor/libigl/igl/) | Other libigl routines used by these methods: harmonic initialization, boundary handling, energies, sparse solves and geometry utilities. |
| [vendor/triangle/triangle.cpp](../src/vigil/meshing/native/vendor/triangle/triangle.cpp) | Third-party triangulation dependency used by SCAF. |
| Eigen, installed as a build dependency | C++ matrix/vector operations and linear algebra used by libigl and the wrapper. |
| scikit-image, installed as a Python dependency | Provides the marching-cubes implementation called by TSDF extraction. |

## 3. Storage, orchestration and display

| File | Responsibility |
| --- | --- |
| [src/vigil/meshing/library.py](../src/vigil/meshing/library.py) | `MeshLibrary`: saves/loads original and flattened meshes, JSON metadata, thumbnails, library indexes and the NPZ correspondence arrays. `save_flat()` links each unfolding to its source 3D mesh. |
| [src/vigil/meshing/routes.py](../src/vigil/meshing/routes.py) | HTTP endpoints for BUILD MESH, automatic prompts, mask previews, FLATTEN, library listing and mesh files. Coordinates the shared build lock and progress events. Manual reconstruction uses the active replay; unfolding only needs a saved mesh. |
| [src/vigil/meshing/__init__.py](../src/vigil/meshing/__init__.py) | Exposes the meshing router to the server. |
| [src/vigil/pipeline.py](../src/vigil/pipeline.py) | Main live loop, command handling and scan job. `_scan()` snapshots settings; `_finish_scan()` saves the capture, calls existing reconstruction and unfolding, and publishes mesh-ready/complete/partial-failure events. Live frame publication remains active. Startup checks and testing-mode selection are also here. |
| [src/vigil/server.py](../src/vigil/server.py) | Registers HTTP routes and transports live frames, commands, progress and scan results over WebSocket. Supplies the latest scan status to reconnecting dashboards. |
| [src/vigil/cli.py](../src/vigil/cli.py) | CLI entry points, including automatic batch reconstruction through `meshing/auto.py`. |
| [src/vigil/web/app.js](../src/vigil/web/app.js) | Mesh settings and requests, live scan controller, 3D rendering, library browsing and the 2D unfolding viewer. `showLiveScan()` overlays a scan without replay; `showSurface()` renders the 3D mesh; `loadFlat()` displays the 2D mapping. Also implements selection correspondence between the flat sheet and the 3D surface. |
| [src/vigil/web/index.html](../src/vigil/web/index.html) | MESH METHODS, LIVE SCAN, FLATTEN MESH and saved-mesh controls and layout. |
| [src/vigil/web/style.css](../src/vigil/web/style.css) | Presentation of those controls, progress, results and viewers. No reconstruction math. |

## 4. Shared input and coordinate-system code

These are upstream dependencies, not separate TSDF/unfolding implementations.

| File | Why it matters |
| --- | --- |
| [src/vigil/camera.py](../src/vigil/camera.py) | RealSense capture, color intrinsics, depth alignment/filtering and `Frame`/`Intrinsics` types. Defines the input sensor measurements. |
| [src/vigil/capture.py](../src/vigil/capture.py) | Camera processes, shared frame buffers and timestamp pairing through `CameraClient`. |
| [src/vigil/rig.py](../src/vigil/rig.py) | Camera-to-world transforms, extrinsic loading/calibration and `transform()`. Reconstruction and the live overlay must use the same world frame. |
| [configs/extrinsics.yaml](../configs/extrinsics.yaml) | Saved camera calibration data, not algorithm code. |
| [src/vigil/geometry.py](../src/vigil/geometry.py) | Depth sampling, deprojection, live point clouds and floor estimation. |
| [src/vigil/skeleton.py](../src/vigil/skeleton.py) | Joint names and bone definitions used by the pose-based leg crop. |
| [src/vigil/body.py](../src/vigil/body.py) | Produces live body joints, including multi-camera fusion/triangulation. Recorded joints can drive the pose-based meshing route; automatic SAM thigh reconstruction uses `autoleg.py` prompts instead. |
| [src/vigil/recorder.py](../src/vigil/recorder.py) | Saves color/depth frames, capture metadata, camera poses and frame records consumed by reconstruction. Also supplies the live scan's saved input. |
| [src/vigil/replay.py](../src/vigil/replay.py) | Recording discovery, loading and playback for the testing/manual workflow. Live scan display does not start `Player`. |
| [src/vigil/config.py](../src/vigil/config.py), [configs/default.yaml](../configs/default.yaml) | Application configuration and testing mode. TSDF parameter definitions themselves are in `tsdf.py`; dashboard selections are managed in `app.js`. |
| [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock) | Python dependencies and their resolved versions. |

For the optional recorded-body-joint input, the estimator adapters are
[estimators/__init__.py](../src/vigil/estimators/__init__.py),
[base.py](../src/vigil/estimators/base.py),
[vitpose_depth.py](../src/vigil/estimators/vitpose_depth.py) and
[sam3d_body.py](../src/vigil/estimators/sam3d_body.py).
The SAM 3D Body visualization mesh is distinct from the depth-derived TSDF thigh mesh.

## 5. Tests to review alongside the implementation

| File | Coverage |
| --- | --- |
| [tests/test_meshing.py](../tests/test_meshing.py) | Synthetic cylinder reconstruction accuracy, support removal, anatomical cropping, hole filling and mesh library round trips. |
| [tests/test_legmask.py](../tests/test_legmask.py) | Prompt handling, camera transfer and SAM-region/crop geometry helpers. |
| [tests/test_autoleg.py](../tests/test_autoleg.py) | Automatic prompt interpretation, agreement checks and automatic meshing orchestration. |
| [tests/test_flatten.py](../tests/test_flatten.py) | All three unfolding methods on known geometry, distortion/dimensions, correspondence, hole-patch bookkeeping and saved outputs. |
| [tests/test_scan.py](../tests/test_scan.py) | Live scan orchestration, selected settings, worker lifetime, no replay call, successful unfolding, partial failure and testing-mode guard. Models/capture are stand-ins in these workflow tests. |
| [tests/scan_ui.test.cjs](../tests/scan_ui.test.cjs) | Runs the actual live-scan JavaScript controller with mocked DOM/network/rendering: direct overlay, unchanged viewpoint, no replay commands, partial success and stale-result rejection. Run with `node tests/scan_ui.test.cjs`. |
| [tests/test_recording.py](../tests/test_recording.py) | Saved frame format and replay behavior. |
| [tests/test_rig.py](../tests/test_rig.py), [tests/test_calibration.py](../tests/test_calibration.py) | Camera extrinsics and calibration-health behavior supporting the shared world frame. |
| [tests/test_floor.py](../tests/test_floor.py), [tests/test_body.py](../tests/test_body.py) | Floor and body geometry supplying upstream orientation/joint information. |
| [tests/test_startup.py](../tests/test_startup.py) | Startup checks and background worker behavior. |

Suggested review sequence: `tsdf.py` → `holes.py` → `legmask.py` → `autoleg.py` → `auto.py`
→ `flatten.py` → `native/SOURCES.md` → `native/planar_solver.cpp` → the selected libigl
solver → `library.py` → `pipeline.py` / `routes.py` → the browser controller/viewers.
