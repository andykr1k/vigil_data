"""Small-hole filling for an open triangle mesh (verts (V, 3), faces (F, 3), consistently wound)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------- hole filling
def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """Closed loops of boundary edges (edges used by one face), each as an ordered vertex
    array following the faces' winding (v[i] → v[i+1] is an edge of an existing face)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.minimum(e[:, 0], e[:, 1]).astype(np.int64) << 32 | np.maximum(e[:, 0], e[:, 1])
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    b = e[counts[inv] == 1]
    nxt: dict[int, list[int]] = {}
    for a, c in b:
        nxt.setdefault(int(a), []).append(int(c))
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, v = [start], start
        while True:
            outs = nxt.get(v)
            if not outs:
                break
            w = outs.pop()
            if not outs:
                del nxt[v]
            if w == start:
                loops.append(np.array(loop))
                break
            loop.append(w)
            v = w
            if len(loop) > len(b):  # malformed (non-manifold): give up on this one
                break
    return loops


def fill_holes(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, max_diameter: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Close every hole smaller than `max_diameter` (≈ perimeter / π), never the outer edge.

    Each hole is patched with concentric rings shrinking to its centre (ring spacing ≈ the
    mesh's edge length, winding matching the surroundings), then the patch's interior
    vertices are faired by a thin-plate (bi-Laplacian) solve with everything else fixed, so the
    patch continues the curvature around it instead of lying flat.
    """
    loops = boundary_loops(faces)
    stats = {"holes_found": max(len(loops) - 1, 0), "holes_filled": 0, "largest_filled_cm": 0.0}
    if len(loops) < 2:
        return verts, faces, colors, stats
    perim = [float(np.linalg.norm(verts[l] - verts[np.roll(l, -1)], axis=1).sum()) for l in loops]
    outer = int(np.argmax(perim))  # the sheet's own edge: always left open
    el = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=1).mean()

    new_v, new_c, new_f, interior = [verts], [colors.astype(np.float32)], [faces], []
    n_v = len(verts)
    for k, loop in enumerate(loops):
        if k == outer or len(loop) < 3 or perim[k] / np.pi > max_diameter:
            continue
        ring0 = verts[loop]
        centre = ring0.mean(0)
        radius = np.linalg.norm(ring0 - centre, axis=1).mean()
        n_rings = int(np.clip(round(radius / el) - 1, 0, 30))
        rings = [loop]
        for r in range(1, n_rings + 1):
            f = r / (n_rings + 1)
            pts = ring0 + f * (centre - ring0)
            idx = np.arange(n_v, n_v + len(loop))
            n_v += len(loop)
            new_v.append(pts)
            new_c.append(colors[loop].astype(np.float32))
            interior.append(idx)
            rings.append(idx)
        c_idx = n_v
        n_v += 1
        new_v.append(centre[None])
        new_c.append(colors[loop].astype(np.float32).mean(0, keepdims=True))
        interior.append(np.array([c_idx]))
        tris = []
        for o, i_ in zip(rings[:-1], rings[1:]):  # outer ring o → inner ring i_ (reversed winding)
            o1, i1 = np.roll(o, -1), np.roll(i_, -1)
            tris.append(np.stack([o1, o, i_], 1))
            tris.append(np.stack([o1, i_, i1], 1))
        last = rings[-1]
        tris.append(np.stack([np.roll(last, -1), last, np.full(len(last), c_idx)], 1))
        new_f.append(np.concatenate(tris))
        stats["holes_filled"] += 1
        stats["largest_filled_cm"] = max(stats["largest_filled_cm"], round(perim[k] / np.pi * 100, 1))
    if not interior:
        return verts, faces, colors, stats

    V = np.concatenate(new_v).astype(np.float64)
    F = np.concatenate(new_f).astype(np.int64)
    C = np.concatenate(new_c)
    I = np.concatenate(interior)
    # Thin-plate fairing: L² x = 0 on the new interior vertices, all others fixed.
    L = _uniform_laplacian(F, len(V))
    L2 = (L @ L).tocsr()
    is_int = np.zeros(len(V), bool)
    is_int[I] = True
    K = np.flatnonzero(~is_int)
    A = L2[I][:, I].tocsc()
    rhs = -(L2[I][:, K] @ V[K])
    try:
        V[I] = np.column_stack([spsolve(A, rhs[:, d]) for d in range(3)])
    except Exception:  # singular (degenerate patch): keep the linear interpolation
        pass
    return V.astype(np.float32), F, C.clip(0, 255).astype(np.uint8), stats


def _uniform_laplacian(faces: np.ndarray, n: int) -> sparse.csr_matrix:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return (sparse.diags(1.0 / np.maximum(deg, 1)) @ A - sparse.eye(n)).tocsr()
