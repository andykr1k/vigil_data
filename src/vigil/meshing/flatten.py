"""FLATTEN MESH: unroll a saved leg-surface mesh onto the plane (SCAF or ARAP).

The surface is an open sheet (a disk), so it's mapped straight to the plane: no cuts and no
completion into a tube. The optimisation itself is done by the authors' libigl code, in
native/planar_solver.cpp, built with CMake the first time it's needed (native/SOURCES.md).
This module prepares the sheet (drops debris, splits pinched vertices, orients it, closes
small holes), runs the solver, checks what comes back, and keeps the correspondence: every
flat point maps to a point on the 3D surface through the same triangle.

Ported from the test_legs viewer (planar.py + unwrap.py), SCAF and ARAP only.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .holes import boundary_loops, fill_holes
from .tsdf import Mesh, MeshError, _normals

METHODS = {
    "scaf": {"title": "SCAF · Jiang 2017", "metric": "symmetric_dirichlet",
             "paper": "https://cims.nyu.edu/gcl/papers/Scaffold-2017.pdf",
             "implementation": "libigl 2.5.0 / igl::triangle::scaf_solve"},
    "arap": {"title": "ARAP · Liu 2008", "metric": "as_rigid_as_possible",
             "paper": "https://cs.harvard.edu/~sjg/papers/arap.pdf",
             "implementation": "libigl 2.5.0 / igl::arap_solve (2D, element energy)"},
}
NATIVE = Path(__file__).with_name("native")


@dataclass
class FlatResult:
    """`flat`: the flat mesh (x, y in metres, z = 0; hole patches left out).
    `arrays`: the full correspondence, saved as the .npz next to it:
      vertices (V, 3) world, faces (F, 3), colors (V, 3), uv (V, 2) metres,
      face_kind (F,) 0 = scanned / 1 = hole patch, face_stretch (F,),
      source_vertex (V,) / source_face (F,) index in the saved mesh (-1 = added)."""
    flat: Mesh
    arrays: dict[str, np.ndarray]


# ---------------------------------------------------------------------- topology
def _edges(faces):
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    return edges, inverse.ravel(), counts


def _topology(faces, n):
    edges, _, counts = _edges(faces)
    boundary = edges[counts == 1]
    _, degree = np.unique(boundary, return_counts=True)
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    components = connected_components(graph, directed=False, return_labels=False)
    return {"vertices": int(n), "faces": len(faces), "components": int(components),
            "euler": int(n - len(edges) + len(faces)),
            "boundary_loops": len(boundary_loops(faces)),
            "nonmanifold_edges": int(np.sum(counts > 2)),
            "irregular_boundary_vertices": int(np.sum(degree != 2))}


def _orient(faces):
    """Orient adjacent triangles consistently; reject non-orientable topology."""
    f = faces.copy()
    edges, inv, counts = _edges(f)
    if np.any(counts > 2):
        raise MeshError("The mesh has edges shared by more than two triangles. Rebuild a cleaner leg patch.")
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    ids = np.tile(np.arange(len(f)), 3)
    order = np.argsort(inv, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    adjacency = [[] for _ in f]
    for e in np.flatnonzero(counts == 2):
        a, b = order[offsets[e]:offsets[e + 1]]
        fa, fb = int(ids[a]), int(ids[b])
        flip = bool(directed[a, 0] == directed[b, 0])
        adjacency[fa].append((fb, flip))
        adjacency[fb].append((fa, flip))
    parity = np.full(len(f), -1, np.int8)
    for seed in range(len(f)):
        if parity[seed] >= 0:
            continue
        parity[seed] = 0
        stack = [seed]
        while stack:
            a = stack.pop()
            for b, flip in adjacency[a]:
                expected = parity[a] ^ flip
                if parity[b] < 0:
                    parity[b] = expected
                    stack.append(b)
                elif parity[b] != expected:
                    raise MeshError("The mesh is not an orientable surface.")
    f[parity == 1] = f[parity == 1, ::-1]
    return f


def _prepare(mesh: Mesh):
    """Remove zero-area debris, split pinched fans, keep the largest face component, close
    small holes. The result must be a disk. Original vertex / face IDs are kept (-1 = added).

    Genuine handles or large holes are rejected instead of silently cut through the scan.
    """
    v = mesh.vertices.astype(float).copy()
    f = mesh.faces.astype(np.int64).copy()
    c = mesh.colors.copy()
    vertex_ids = np.arange(len(v), dtype=np.int64)
    face_ids = np.arange(len(f), dtype=np.int64)
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    positive = area2[area2 > 0]
    if not len(positive):
        raise MeshError("This mesh has no non-degenerate triangles.")
    keep = area2 > max(1e-16, np.median(positive) * 1e-7)
    f, face_ids = f[keep], face_ids[keep]
    _, unique = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    unique.sort()
    f, face_ids = f[unique], face_ids[unique]
    edges, inv, counts = _edges(f)
    if np.any(counts > 2):
        raise MeshError("The mesh has non-manifold edges. Rebuild it with Largest only and Fill holes on.")

    # Components by shared edges, not shared vertices (pinched fragments).
    order = np.argsort(inv, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    interior = np.flatnonzero(counts == 2)
    a = order[offsets[interior]] % len(f)
    b = order[offsets[interior] + 1] % len(f)
    graph = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(len(f), len(f)))
    _, labels = connected_components(graph, directed=False)
    areas = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    main = int(np.argmax(np.bincount(labels, weights=areas)))
    keep = labels == main
    f, face_ids = f[keep], face_ids[keep]

    # A vertex may still have disconnected incident fans within that component: split it.
    incident = [[] for _ in v]
    for fi, tri in enumerate(f):
        for x in tri:
            incident[x].append(fi)
    copies = []
    for x, fan in enumerate(incident):
        if len(fan) < 2:
            continue
        by_neighbor = {}
        for fi in fan:
            for y in f[fi]:
                if y != x:
                    by_neighbor.setdefault(int(y), []).append(fi)
        links = {fi: set() for fi in fan}
        for neighbors in by_neighbor.values():
            for fi in neighbors:
                links[fi].update(neighbors)
        remaining = set(fan)
        first = True
        while remaining:
            stack, group = [min(remaining)], []
            while stack:
                fi = stack.pop()
                if fi not in remaining:
                    continue
                remaining.remove(fi)
                group.append(fi)
                stack.extend(links[fi] & remaining)
            if first:
                first = False
            else:
                new = len(v) + len(copies)
                copies.append(x)
                for fi in group:
                    f[fi, f[fi] == x] = new
    if copies:
        v = np.concatenate([v, v[copies]])
        c = np.concatenate([c, c[copies]])
        vertex_ids = np.concatenate([vertex_ids, vertex_ids[copies]])
    used, inverse = np.unique(f, return_inverse=True)
    v, c, vertex_ids, f = v[used], c[used], vertex_ids[used], inverse.reshape(-1, 3)
    f = _orient(f)
    topology = _topology(f, len(v))
    if topology["irregular_boundary_vertices"] or topology["nonmanifold_edges"]:
        raise MeshError("The boundary could not be repaired into a simple surface. Try a cleaner saved mesh.")
    original_count = len(f)
    if topology["boundary_loops"] > 1:
        v, f, c, _ = fill_holes(v, f, c, 0.05)
        vertex_ids = np.pad(vertex_ids, (0, len(v) - len(vertex_ids)), constant_values=-1)
        face_ids = np.pad(face_ids, (0, len(f) - len(face_ids)), constant_values=-1)
        f = _orient(f)
        topology = _topology(f, len(v))
    if topology["euler"] != 1 or topology["boundary_loops"] != 1 or topology["components"] != 1:
        handles = (2 - topology["boundary_loops"] - topology["euler"]) // 2
        if topology["boundary_loops"] > 1:  # holes over 5 cm (maybe handles too)
            fix = (f"It still has {topology['boundary_loops'] - 1} hole(s) larger than 5 cm: rebuild it with a "
                   "bigger MAX HOLE, or pick another mesh.")
        else:  # one edge, so the Euler characteristic below 1 is all handles (tunnels through the sheet)
            fix = (f"It has {handles} handle(s): small tunnels through the surface, made when it was extracted, so "
                   "SMOOTH and MAX HOLE don't remove them. Rebuild it with a larger TRUNC (e.g. 20 mm) or "
                   "VOXEL (e.g. 4 mm), or pick another mesh.")
        raise MeshError("The surface must be a single sheet (a disk) to flatten; found Euler characteristic "
                        f"{topology['euler']} and {topology['boundary_loops']} boundaries. " + fix)
    repair = {"input_faces": len(mesh.faces), "retained_source_faces": original_count,
              "removed_source_faces": len(mesh.faces) - original_count,
              "split_vertices": len(copies), "hole_patch_faces": len(f) - original_count}
    return v.astype(float), f, c, vertex_ids, face_ids, repair


# ---------------------------------------------------------------------- map quality
def _intrinsic(v, f):
    """Gradients of each triangle's barycentric basis functions in its own tangent plane
    (true 3D edge lengths), and the triangle areas."""
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    length = np.linalg.norm(e1, axis=1)
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    if np.any(area2 <= 1e-18) or np.any(length <= 1e-12):
        raise MeshError("The surface contains degenerate triangles; choose a cleaner mesh.")
    x = np.einsum("ij,ij->i", e1, e2) / length
    y = area2 / length
    grad = np.empty((len(f), 3, 2))
    grad[:, 1, 0], grad[:, 1, 1] = 1 / length, -x / (length * y)
    grad[:, 2, 0], grad[:, 2, 1] = 0, 1 / y
    grad[:, 0] = -grad[:, 1] - grad[:, 2]
    return grad, area2 / 2


def _jacobian(uv, f, grad):
    return np.einsum("fki,fkj->fij", uv[f], grad)


def _det(J):
    return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def _simple_boundary(points):
    """Reject crossings and non-adjacent touching/overlapping boundary edges."""
    n = len(points)
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    p = (points - points.mean(axis=0)) / scale
    q = np.roll(p, -1, axis=0)
    if np.any(np.linalg.norm(q - p, axis=1) <= 1e-12):
        return False

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for i in range(n - 2):
        js = np.arange(i + 2, n - (1 if i == 0 else 0))
        a, b, c, d = p[i], q[i], p[js], q[js]
        boxes = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                       np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-13, axis=1)
        c, d = c[boxes], d[boxes]
        if not len(c):
            continue
        u, w = cross(b - a, c - a), cross(b - a, d - a)
        x, y = cross(d - c, a - c), cross(d - c, b - c)
        if np.any((np.minimum(u, w) <= 1e-14) & (np.maximum(u, w) >= -1e-14) &
                  (np.minimum(x, y) <= 1e-14) & (np.maximum(x, y) >= -1e-14)):
            return False
    return True


# ---------------------------------------------------------------------- native solver
def _native_binary(progress) -> Path:
    """native/build/planar_solver, (re)built with CMake when missing or older than its source."""
    executable = NATIVE / "build" / "planar_solver"
    sources = [NATIVE / "planar_solver.cpp", NATIVE / "CMakeLists.txt"]
    if not executable.is_file() or any(p.stat().st_mtime > executable.stat().st_mtime for p in sources):
        progress("building the solver (first run)", 0.02)
        for command in (["cmake", "-S", str(NATIVE), "-B", str(NATIVE / "build"), "-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake", "--build", str(NATIVE / "build"), "-j2"]):
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise MeshError("Could not build the flattening solver: CMake, a C++17 compiler and "
                                "Eigen3 are required (see src/vigil/meshing/native/SOURCES.md).") from e
            if p.returncode:
                raise MeshError("Flattening solver build failed: " + (p.stderr or p.stdout)[-1800:])
    return executable


def _solve(v, f, method, iterations, progress):
    executable = _native_binary(progress)
    title = METHODS[method]["title"]
    with tempfile.TemporaryDirectory(prefix="vigil-flatten-") as directory:
        source, target = Path(directory) / "mesh.txt", Path(directory) / "uv.txt"
        with source.open("w") as out:
            out.write(f"{len(v)} {len(f)}\n")
            np.savetxt(out, v, fmt="%.17g")
            np.savetxt(out, f, fmt="%d")
        progress("optimizing " + title, 0.12)
        history, tail = [], []
        with subprocess.Popen([str(executable), method, str(source), str(target), str(iterations)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                tail = (tail + [line.strip()])[-12:]
                if line.startswith("PROGRESS "):
                    _, step, energy = line.split()
                    history.append(float(energy))
                    progress("optimizing " + title, 0.12 + 0.78 * int(step) / iterations)
            code = process.wait()
        if code or not target.is_file():
            raise MeshError(f"{title} failed: " + " ".join(tail)[-1800:])
        with target.open() as result:
            count = int(result.readline())
            uv = np.loadtxt(result, ndmin=2)
        if count != len(v) or uv.shape != (len(v), 2):
            raise MeshError("The solver returned a different number of vertices.")
    return uv, {"iterations": len(history) if method == "scaf" else iterations, "iteration_limit": iterations,
                "solver": method, "implementation": METHODS[method]["implementation"],
                "paper": METHODS[method]["paper"], "metric": METHODS[method]["metric"],
                "boundary_fixed": False, "translation_anchor_vertex": 0 if method == "arap" else None,
                "energy_history": history if method == "scaf" else []}


# ---------------------------------------------------------------------- entry point
def flatten_mesh(mesh: Mesh, method: str = "scaf", iterations: int = 50, progress=None) -> FlatResult:
    """Unroll a saved leg-surface mesh. Raises MeshError (nothing to save) if the result has
    flipped triangles, an overlapping boundary or fails the area check."""
    progress = progress or (lambda stage, fraction: None)
    started = time.monotonic()
    if method not in METHODS:
        raise MeshError(f"Unknown flattening method {method!r} (SCAF or ARAP).")
    if mesh.stats.get("method") == "flat":
        raise MeshError("This mesh is already flat. Pick the surface mesh it came from.")
    try:
        iterations = int(iterations)
    except (TypeError, ValueError, OverflowError) as e:
        raise MeshError("Iterations must be a whole number.") from e
    if not 1 <= iterations <= 1000:
        raise MeshError("Iterations must be between 1 and 1000.")
    title = METHODS[method]["title"]

    progress("preparing surface", 0.0)
    v, f, c, vertex_ids, face_ids, repair = _prepare(mesh)
    uv, optimization = _solve(v, f, method, iterations, progress)

    progress("checking the map", 0.93)
    if not np.isfinite(uv).all():
        raise MeshError(f"{title} produced non-finite coordinates; nothing was saved.")
    grad, areas = _intrinsic(v, f)
    J = _jacobian(uv, f, grad)
    determinant = _det(J)
    if np.all(determinant < 0):  # a uniform reflection only mirrors the chart
        uv[:, 1] *= -1
        J, determinant = _jacobian(uv, f, grad), -determinant
    bad = determinant <= 1e-12
    if bad.any():
        hint = ("ARAP keeps triangles' shapes but nothing stops one turning over, typically a small one in a "
                "fold. More iterations won't change that; SCAF can't flip triangles, so use SCAF for this mesh."
                if method == "arap" else "Try more iterations, or ARAP.")
        raise MeshError(f"{title} flipped or collapsed {int(bad.sum())} of {len(f)} triangles "
                        f"({float(areas[bad].sum()) * 1e4:.2f} cm² of surface); nothing was saved. " + hint)
    loop = boundary_loops(f)[0]
    if not _simple_boundary(uv[loop]):
        raise MeshError(f"{title} produced an overlapping boundary; nothing was saved. Try the other method.")
    boundary = uv[loop]
    polygon_area = abs(float(np.sum(boundary[:, 0] * np.roll(boundary[:, 1], -1) -
                                    boundary[:, 1] * np.roll(boundary[:, 0], -1)))) / 2
    mapped_area = float(np.sum(areas * determinant))
    coverage_error = abs(mapped_area - polygon_area) / max(mapped_area, 1e-30)
    if coverage_error > 1e-6:
        raise MeshError("The flat map failed its boundary-area check; nothing was saved.")
    # Turn the sheet so its long axis runs along x, then move it to start at (0, 0). A rotation
    # and a shift only: no rescaling (uv stays in metres) and no mirroring.
    centred = uv - uv.mean(axis=0)
    axes = np.linalg.svd(centred[:: max(1, len(uv) // 20000)], full_matrices=False)[2]
    if np.linalg.det(axes) < 0:
        axes[1] *= -1
    uv = centred @ axes.T
    uv -= uv.min(axis=0)

    # Stretch per triangle: the larger of its longest stretch and its strongest squash (1 = none).
    singular = np.linalg.svd(J, compute_uv=False)
    stretch = np.maximum(singular[:, 0], 1 / singular[:, 1])
    scanned = face_ids >= 0
    ratios, weights = stretch[scanned], areas[scanned]
    order = np.argsort(ratios)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    percentile = lambda q: float(np.interp(q, cumulative, ratios[order]))  # area-weighted
    flat_area = float(np.sum(areas[scanned] * determinant[scanned]))
    used, inverse = np.unique(f[scanned], return_inverse=True)
    flat_v = np.column_stack([uv[used], np.zeros(len(used))])
    flat_f = inverse.reshape(-1, 3).astype(np.uint32)
    normals, _ = _normals(flat_v.astype(np.float32), flat_f)

    warnings = []
    if repair["removed_source_faces"]:
        warnings.append(f"Left out {repair['removed_source_faces']} degenerate or disconnected triangles "
                        "(the saved surface is unchanged).")
    if percentile(0.95) > 1.5:
        warnings.append("Some regions stretch substantially: check the STRETCH view before using the flat map.")
    size = np.ptp(uv, axis=0)
    stats = {"method": "flat", "flatten_method": method, "title": title,
             "coordinate_system": "flat x, y in metres (z = 0)",
             "vertices": len(flat_v), "faces": len(flat_f), "size_cm": [round(float(s) * 100, 2) for s in size],
             "source_area_cm2": round(float(weights.sum()) * 1e4, 1), "area_cm2": round(flat_area * 1e4, 1),
             "distortion": {"area_ratio": flat_area / float(weights.sum()), "stretch_p50": percentile(0.5),
                            "stretch_p95": percentile(0.95), "stretch_max": float(ratios.max()),
                            "flipped_triangles": 0, "min_jacobian": float(determinant.min()),
                            "simple_boundary": True, "boundary_coverage_error": coverage_error},
             "optimization": optimization, "repair": repair, "topology": _topology(f, len(v)),
             "build_s": round(time.monotonic() - started, 2), "warnings": warnings}
    for key in ("frame", "seq", "time_s"):
        if key in mesh.stats:
            stats[key] = mesh.stats[key]
    arrays = {"vertices": v.astype(np.float32), "faces": f.astype(np.uint32), "colors": c.astype(np.uint8),
              "uv": uv.astype(np.float32), "face_kind": np.where(scanned, 0, 1).astype(np.uint8),
              "face_stretch": stretch.astype(np.float32),
              "source_vertex": vertex_ids, "source_face": face_ids}
    progress("done", 1.0)
    return FlatResult(Mesh(flat_v.astype(np.float32), flat_f, normals, c[used], stats), arrays)


def flat_binary(arrays: dict[str, np.ndarray]) -> bytes:
    """What the dashboard's flat view loads: u32 V, u32 F, f32 uv[V*2], f32 xyz[V*3] (world),
    u8 rgb[V*3] (padded to 4), u32 faces[F*3], f32 stretch[F], u8 kind[F] (padded to 4)."""
    def pad(b: bytes) -> bytes:
        return b + b"\0" * (-len(b) % 4)
    a = arrays
    return (np.array([len(a["vertices"]), len(a["faces"])], "<u4").tobytes()
            + a["uv"].astype("<f4").tobytes() + a["vertices"].astype("<f4").tobytes()
            + pad(a["colors"].astype("u1").tobytes()) + a["faces"].astype("<u4").tobytes()
            + a["face_stretch"].astype("<f4").tobytes() + pad(a["face_kind"].astype("u1").tobytes()))
