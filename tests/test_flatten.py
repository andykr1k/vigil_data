"""FLATTEN MESH: SCAF / ARAP unrolling of a saved surface, its checks, and the Meshes/ files.

The surface is a third of a cylinder (like the top of a thigh). A cylinder unrolls without
any stretch, so both methods must give a flat sheet the size of the arc × the length.
The native solver is built with CMake on first use; the tests skip if that isn't possible.
"""

import json
import shutil

import numpy as np
import pytest

from vigil.meshing.flatten import flat_binary, flatten_mesh
from vigil.meshing.library import MeshLibrary
from vigil.meshing.tsdf import Mesh, MeshError, _normals

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs CMake to build the solver")

R, LENGTH, ARC = 0.07, 0.2, np.radians(120)


def cylinder_patch(n_around: int = 24, n_along: int = 30, hole: bool = False) -> Mesh:
    a = np.linspace(-ARC / 2, ARC / 2, n_around)
    x = np.linspace(0, LENGTH, n_along)
    A, X = np.meshgrid(a, x, indexing="ij")
    verts = np.stack([X, R * np.sin(A), R * np.cos(A)], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(n_around * n_along).reshape(n_around, n_along)
    faces = []
    for i in range(n_around - 1):
        for j in range(n_along - 1):
            if hole and abs(i - n_around // 2) <= 1 and abs(j - n_along // 2) <= 1:
                continue  # a small square hole in the middle
            a0, a1, b0, b1 = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            faces += [[a0, b0, b1], [a0, b1, a1]]
    faces = np.array(faces, np.uint32)
    colors = np.full((len(verts), 3), 180, np.uint8)
    normals, _ = _normals(verts, faces)
    return Mesh(verts, faces, normals, colors, {"method": "tsdf", "frame": 4, "seq": 104, "time_s": 1.5})


@pytest.mark.parametrize("method", ["scaf", "arap"])
def test_cylinder_patch_unrolls_without_stretch(method):
    result = flatten_mesh(cylinder_patch(), method, 50)
    s, a = result.flat.stats, result.arrays
    assert s["method"] == "flat" and s["flatten_method"] == method
    d = s["distortion"]
    assert d["flipped_triangles"] == 0 and d["min_jacobian"] > 0
    assert d["stretch_p95"] < 1.05
    assert abs(d["area_ratio"] - 1) < 0.01
    # The sheet is the unrolled arc by the length (in some orientation).
    assert sorted(s["size_cm"]) == pytest.approx(sorted([R * ARC * 100, LENGTH * 100]), rel=0.03)
    # Every flat vertex still maps to its 3D vertex: same triangles, original vertex IDs.
    assert len(a["uv"]) == len(a["vertices"]) and (a["source_vertex"] >= 0).all()
    assert np.allclose(a["vertices"], cylinder_patch().vertices[a["source_vertex"]])
    assert (result.flat.vertices[:, 2] == 0).all()


def test_flat_point_maps_back_to_the_surface():
    """A point inside a flat triangle, taken with the same barycentric weights in 3D, lies on
    the cylinder (what the dashboard does when a point is clicked)."""
    a = flatten_mesh(cylinder_patch(), "arap", 50).arrays
    f = a["faces"][len(a["faces"]) // 3]
    w = np.array([0.2, 0.5, 0.3])
    p = w @ a["vertices"][f].astype(float)
    assert abs(np.hypot(p[1], p[2]) - R) < 0.002  # on the surface (chord sag only)


def test_small_holes_are_patched_and_marked():
    result = flatten_mesh(cylinder_patch(hole=True), "arap", 50)
    a, s = result.arrays, result.flat.stats
    assert s["repair"]["hole_patch_faces"] > 0
    assert (a["face_kind"] == 1).sum() == s["repair"]["hole_patch_faces"]
    assert (a["source_face"][a["face_kind"] == 1] == -1).all()
    assert len(result.flat.faces) == (a["face_kind"] == 0).sum()  # the flat PLY leaves patches out


def test_rejects_bad_requests():
    with pytest.raises(MeshError):
        flatten_mesh(cylinder_patch(), "slim")
    with pytest.raises(MeshError):
        flatten_mesh(cylinder_patch(), "scaf", 0)
    flat = flatten_mesh(cylinder_patch(), "arap", 10).flat
    with pytest.raises(MeshError, match="already flat"):
        flatten_mesh(flat, "arap")


def test_library_saves_flat_next_to_its_source(tmp_path):
    lib = MeshLibrary(tmp_path)
    rec = tmp_path / "2026-10-08_rec"
    rec.mkdir()
    source = {"recording": rec.name, "stem": "frame00005_t0001.50s_tsdf_20261008-120000", "frame": 5,
              "index": 4, "seq": 104, "time_s": 1.5, "method": "tsdf", "created": "2026-10-08T12:00:00",
              "stats": {"method": "tsdf"}}
    (rec / f"{source['stem']}.ply").write_bytes(cylinder_patch().to_ply())
    (rec / f"{source['stem']}.json").write_text(json.dumps(source))

    mesh, entry = lib.load(rec.name, source["stem"])
    result = flatten_mesh(mesh, "scaf", 30)
    saved = lib.save_flat(result, entry)
    assert saved["stem"].startswith("frame00005_t0001.50s_flat-scaf_")
    assert saved["source"] == {"recording": rec.name, "stem": source["stem"]}
    for ext in ("ply", "json", "npz"):
        assert (rec / f"{saved['stem']}.{ext}").is_file()

    items = {e["stem"]: e for e in lib.list(rec.name)}
    assert items[saved["stem"]]["method"] == "flat" and items[saved["stem"]]["flatten_method"] == "scaf"
    assert "flat-scaf" in (tmp_path / "index.csv").read_text()

    arrays, side = lib.load_flat(rec.name, saved["stem"])
    assert side["stats"]["distortion"]["stretch_p95"] < 1.05
    blob = flat_binary(arrays)
    nv, nf = np.frombuffer(blob[:8], "<u4")
    pad = lambda n: n + (-n % 4)
    assert len(blob) == 8 + nv * 8 + nv * 12 + pad(nv * 3) + nf * 12 + nf * 4 + pad(nf)
    with pytest.raises(ValueError):
        lib.load_flat(rec.name, source["stem"])  # a surface, not a flat mesh

    lib.delete(rec.name, saved["stem"])
    assert not list(rec.glob(f"{saved['stem']}.*"))
