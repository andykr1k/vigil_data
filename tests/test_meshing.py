"""MESH METHODS: TSDF leg-surface meshes from a recording, hole filling, the Meshes/ library.

The recording is synthetic: one camera looking at a horizontal cylinder (the "leg"), written
with vigil's own Recorder and read back with its Player, so the mesh code is tested against
the real recording format.
"""

import json

import numpy as np

from vigil.camera import Frame
from vigil.config import RecordingConfig
from vigil.meshing.holes import boundary_loops, fill_holes
from vigil.meshing.library import MeshLibrary
from vigil.meshing.tsdf import TsdfParams, build_tsdf
from vigil.recorder import Recorder
from vigil.replay import Player
from vigil.skeleton import BONES

W, H, F = 160, 120, 120.0
Z0, R = 0.6, 0.05            # cylinder axis (along x, at y = 0) depth and radius
LIMB = 0.05                  # joints placed this far behind the surface, like vitpose_depth
CAMS = [{"index": 0, "serial": "A", "width": W, "height": H, "fx": F, "fy": F, "cx": W / 2,
         "cy": H / 2, "T_world_camera": np.eye(4).ravel().tolist(), "dist": [0.0] * 5}]
HELLO = {"type": "hello", "backend": "vitpose_depth", "cameras": CAMS, "bones": BONES}


def cylinder_depth(seat: bool = False) -> np.ndarray:
    """Depth (m) of the cylinder's camera-facing side; 0 elsewhere. `seat`: the leg rests on a
    flat seat (a plane facing the camera, touching the cylinder's underside)."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dx, dy = (u - W / 2) / F, (v - H / 2) / F
    # ray s·(dx, dy, 1) meets y² + (z - Z0)² = R²
    a = dy ** 2 + 1
    disc = Z0 ** 2 - a * (Z0 ** 2 - R ** 2)
    s = (Z0 - np.sqrt(np.maximum(disc, 0))) / a
    depth = np.where((disc > 0) & (np.abs(s * dx) < 0.2), s, 0.0)
    if seat:
        depth = np.where(depth > 0, depth, Z0 + R)
    return depth.astype(np.float32)


def joints() -> dict:
    # On the axis = surface point + LIMB along the ray (top-of-leg view), conf 0.9.
    return {"left_hip": [-0.18, 0.0, Z0, 0.9], "left_knee": [-0.05, 0.0, Z0, 0.9],
            "left_ankle": [0.12, 0.0, Z0, 0.9]}


def make_recording(tmp_path, n=3, seat=False):
    rec = Recorder(RecordingConfig(), tmp_path, CAMS,
                   {"procedure": "lower_limb", "world_frame": "cam 1", "depth_window": [0.2, 1.5],
                    "hello": HELLO})
    depth, color = cylinder_depth(seat), np.full((H, W, 3), 128, np.uint8)
    for seq in range(1, n + 1):
        f = Frame(color, depth, 100.0 + seq / 30, seq)
        frame = {"type": "frame", "seq": seq, "person": {"joints": joints()},
                 "floor": {"normal": [0.0, 0.0, -1.0], "height": 1.0}}
        rec.add_frame(seq, [f], [True], {"t": f.timestamp, "frame": frame})
    rec.close()
    player = Player(rec.path, lambda *a: None)  # loads frames.jsonl; not started
    return rec.path, json.loads((rec.path / "meta.json").read_text()), player.lines


def surface_error(verts: np.ndarray) -> np.ndarray:
    return np.abs(np.hypot(verts[:, 1], verts[:, 2] - Z0) - R)


def test_tsdf_meshes_the_camera_facing_side(tmp_path):
    path, meta, lines = make_recording(tmp_path)
    mesh = build_tsdf(path, meta, lines, 1, TsdfParams(section="whole", voxel_mm=2, trunc_mm=8),
                      limb_radius=LIMB)
    v = mesh.vertices
    assert len(mesh.faces) > 200
    assert np.median(surface_error(v)) < 0.002 and np.percentile(surface_error(v), 95) < 0.005
    assert (v[:, 2] < Z0 + 0.005).mean() > 0.99            # only the half the camera sees
    assert boundary_loops(mesh.faces.astype(np.int64))      # an open sheet, not a closed tube
    assert mesh.stats["views"] == {0: 3}


def test_support_seat_is_cut_away(tmp_path):
    path, meta, lines = make_recording(tmp_path, seat=True)
    # BELOW reaches down to the seat and every piece is kept, so only the support cut removes it.
    params = dict(section="whole", voxel_mm=2, trunc_mm=8, below_cm=8, largest=False)
    with_seat = build_tsdf(path, meta, lines, 1, TsdfParams(support=False, **params), limb_radius=LIMB)
    assert (with_seat.vertices[:, 2] > Z0 + R - 0.01).mean() > 0.3   # seat is in there

    mesh = build_tsdf(path, meta, lines, 1, TsdfParams(**params), limb_radius=LIMB)
    assert mesh.stats["support"] is not None
    assert mesh.vertices[:, 2].max() < Z0 + R - 0.01                # seat gone
    assert np.median(surface_error(mesh.vertices)) < 0.002            # leg kept


def test_above_knee_keeps_only_the_thigh(tmp_path):
    path, meta, lines = make_recording(tmp_path)
    # Seen straight from above the synthetic leg is a thin band; no seat here to remove.
    mesh = build_tsdf(path, meta, lines, 1,
                      TsdfParams(section="above", voxel_mm=2, trunc_mm=8, support=False),
                      limb_radius=LIMB)
    sec = mesh.stats["section"]
    assert sec["knees_used"] >= 1
    assert abs(sec["knee"][0] - (-0.05)) < 0.01             # cut at the knee
    assert mesh.vertices[:, 0].max() < -0.03                # the hip side (-x): ankle is at +x


def test_fill_holes_closes_small_holes_only():
    # A 30 x 30 grid of 1 cm squares in z = 0, with a 2 x 2 square hole in the middle.
    n = 31
    xs, ys = np.meshgrid(np.arange(n) * 0.01, np.arange(n) * 0.01)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], 1).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            if 14 <= i < 16 and 14 <= j < 16:
                continue
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, d], [a, d, c]]
    faces = np.array(faces)
    colors = np.full((len(verts), 3), 100, np.uint8)
    assert len(boundary_loops(faces)) == 2

    v2, f2, c2, stats = fill_holes(verts, faces, colors, max_diameter=0.05)
    assert stats["holes_filled"] == 1 and len(boundary_loops(f2)) == 1   # outer edge stays open
    assert np.abs(v2[:, 2]).max() < 1e-6                                 # patch lies in the plane

    _, f3, _, stats = fill_holes(verts, faces, colors, max_diameter=0.01)  # too big to fill
    assert stats["holes_filled"] == 0 and len(f3) == len(faces)


def test_library_saves_lists_loads_and_deletes(tmp_path):
    path, meta, lines = make_recording(tmp_path / "rec")
    mesh = build_tsdf(path, meta, lines, 1, TsdfParams(section="whole"), limb_radius=LIMB)
    lib = MeshLibrary(tmp_path / "Meshes")
    entry = lib.save(mesh, path, lines, 1, window_s=0.5)
    assert entry["frame"] == 2 and entry["stem"].startswith("frame00002_")

    [item] = lib.list(path.name)
    assert item["stem"] == entry["stem"] and item["faces"] == len(mesh.faces)
    loaded, side = lib.load(path.name, entry["stem"])
    assert np.allclose(loaded.vertices, mesh.vertices) and side["seq"] == lines[1]["seq"]
    assert (tmp_path / "Meshes" / "index.csv").read_text().count("\n") == 2

    lib.delete(path.name, entry["stem"])
    assert lib.list() == [] and not (tmp_path / "Meshes" / path.name).exists()
