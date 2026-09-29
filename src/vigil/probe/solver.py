"""Probe pose estimation across the rig: tag cleaning, flip resolution, and one joint
solve over every tag corner seen by every calibrated camera, with measured depth.

Per camera (`clean_view`):
  * 2+ tags: joint PnP, then drop tags whose corners don't reproject under it (a bad
    detection or a tag knocked off the cube) and re-solve.
  * 1 tag: square-tag PnP is two-fold ambiguous (a mirrored tilt fits almost equally
    well). Keep the candidate consistent with the previous pose, else the lower error.

Across cameras (`refine_multiview`): least squares over the cube's world pose, with
  * reprojection residuals of every kept corner in every calibrated camera, and
  * depth residuals: measured depth at each corner vs its predicted distance.
Tag PnP is precise across the image but weak along the viewing ray; other cameras and
the depth sensor constrain exactly that direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .aruco import DetectedMarkerPose
from .geometry import ProbeGeometry, marker_object_points
from .object_pose import ObjectPose, _single_marker_object_pose, estimate_object_pose


_ZERO3 = np.zeros(3)


def _to_h(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def reprojection_errors(pose: ObjectPose, markers: list[DetectedMarkerPose],
                        geometry: ProbeGeometry, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """RMS corner reprojection error (px) of each tag under a cube pose (camera frame)."""
    rvec, _ = cv2.Rodrigues(pose.rotation_matrix)
    errs = []
    for m in markers:
        proj, _ = cv2.projectPoints(geometry.marker_corners_in_object(m.marker_id), rvec,
                                    pose.position, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - m.image_corners) ** 2, 1)))))
    return np.array(errs)


@dataclass
class CleanView:
    markers: list[DetectedMarkerPose]  # tags kept
    pose: ObjectPose | None  # cube in this camera's frame
    rms_px: float = float("nan")
    rejected: list[int] = field(default_factory=list)  # tag ids dropped as outliers
    flip_resolved: bool = False  # single tag, chosen by consistency with the previous pose


def clean_view(markers: list[DetectedMarkerPose], geometry: ProbeGeometry, K: np.ndarray,
               dist: np.ndarray, prev_pose: ObjectPose | None = None,
               max_err_px: float = 3.0) -> CleanView:
    markers = [m for m in markers if m.marker_id in geometry.mounts]
    if not markers:
        return CleanView([], None)
    if len(markers) == 1:
        return _single_tag(markers[0], geometry, K, dist, prev_pose)

    rejected: list[int] = []
    # 3+ tags: leave-one-out. A bad tag drags a joint fit towards itself and spreads its
    # error over the good ones, so judge each tag by how well the *others* predict it.
    while len(markers) >= 3:
        pose = estimate_object_pose(markers, geometry, K, dist)
        if pose is None or reprojection_errors(pose, markers, geometry, K, dist).max() <= max_err_px:
            break
        loo = []
        for i in range(len(markers)):
            others = markers[:i] + markers[i + 1:]
            p_i = estimate_object_pose(others, geometry, K, dist)
            loo.append(np.inf if p_i is None else
                       reprojection_errors(p_i, [markers[i]], geometry, K, dist)[0])
        worst = int(np.argmax(loo))
        if loo[worst] <= 2 * max_err_px:
            break  # no clear outlier: the residual is just noise spread over the tags
        rejected.append(markers[worst].marker_id)
        markers = markers[:worst] + markers[worst + 1:]

    if len(markers) == 2:
        pose = estimate_object_pose(markers, geometry, K, dist)
        errs = None if pose is None else reprojection_errors(pose, markers, geometry, K, dist)
        if errs is None or errs.max() > max_err_px:
            # Two tags disagree (a joint fit spreads one bad tag's error over both, so this
            # shows on both) and there's no majority: trust the larger (sharper) one.
            keep = max(markers, key=lambda m: m.pixel_area)
            rejected += [m.marker_id for m in markers if m is not keep]
            markers = [keep]

    if len(markers) == 1:
        view = _single_tag(markers[0], geometry, K, dist, prev_pose)
        view.rejected = rejected
        return view
    pose = estimate_object_pose(markers, geometry, K, dist)
    if pose is None:
        return CleanView(markers, None, rejected=rejected)
    errs = reprojection_errors(pose, markers, geometry, K, dist)
    return CleanView(markers, pose, float(np.sqrt(np.mean(errs ** 2))), rejected)


def _single_tag(m: DetectedMarkerPose, geometry: ProbeGeometry, K: np.ndarray,
                dist: np.ndarray, prev_pose: ObjectPose | None) -> CleanView:
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        marker_object_points(geometry.marker_length), m.image_corners.reshape(4, 2),
        K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n == 0:
        return CleanView([m], None)
    candidates = [
        _single_marker_object_pose(DetectedMarkerPose(m.marker_id, r, t.reshape(3), m.pixel_area,
                                                      m.image_corners), geometry)
        for r, t in zip(rvecs, tvecs)
    ]
    errors = [float(e) for e in np.asarray(errs).reshape(-1)] if errs is not None else [0.0] * n
    best = int(np.argmin(errors))
    resolved = False
    if prev_pose is not None and len(candidates) == 2:
        angles = [_angle_deg(c.rotation_matrix, prev_pose.rotation_matrix) for c in candidates]
        if abs(angles[0] - angles[1]) > 10:  # the two solutions are genuinely different
            best, resolved = int(np.argmin(angles)), True
    return CleanView([m], candidates[best], errors[best], flip_resolved=resolved)


@dataclass
class CameraView:
    """Everything the joint solve needs from one calibrated camera."""

    T_world_cam: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    markers: list[DetectedMarkerPose]
    depth: np.ndarray | None = None  # metres, aligned to the colour image


@dataclass
class Refined:
    pose: ObjectPose  # world frame
    rms_px: float
    corners: int
    depth_samples: int
    depth_rms_mm: float


def _corner_depth(depth: np.ndarray, uv: np.ndarray, r: int = 2) -> float:
    h, w = depth.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (r <= u < w - r and r <= v < h - r):
        return 0.0
    win = depth[v - r: v + r + 1, u - r: u + r + 1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size >= 5 else 0.0


def refine_multiview(views: list[CameraView], geometry: ProbeGeometry, init: ObjectPose,
                     use_depth: bool = True, sigma_px: float = 1.0,
                     depth_rel_sigma: float = 0.01, iterations: int = 8) -> Refined | None:
    """Joint least squares of the cube's world pose over all cameras' corners (+ depth).

    Gauss–Newton with an analytic Jacobian and robust (Huber) reweighting: the pose is
    updated by a small world-frame twist (ω, v), under which a corner X moves by ω×X + v.
    (~1 ms for two cameras, vs ~25 ms with a finite-difference solver.)
    """
    P_obj, uv_obs, cam_idx = [], [], []
    D_obj, z_obs, d_cam = [], [], []  # depth samples: object point, measured depth, camera
    cams = []  # (R_cw, t_cw, K, dist)
    half = geometry.marker_length / 2
    square = np.array([[-half, half], [half, half], [half, -half], [-half, -half]], np.float32)
    # Depth is sampled *inside* each tag (centre + halfway to each corner): corners sit on the
    # cube's edges, where on a small, obliquely seen cube the depth often hits the background.
    inner_local = np.vstack([[0.0, 0.0], 0.5 * square]).astype(np.float32)
    for v in views:
        tags = [m for m in v.markers if m.marker_id in geometry.mounts]
        if not tags:
            continue
        T_cw = np.linalg.inv(v.T_world_cam)
        cams.append((T_cw[:3, :3], T_cw[:3, 3], v.K, v.dist))
        c = len(cams) - 1
        for m in tags:
            P_obj.append(geometry.marker_corners_in_object(m.marker_id))
            uv_obs.append(m.image_corners)
            cam_idx.append(np.full(4, c))
            if use_depth and v.depth is not None:
                H = cv2.getPerspectiveTransform(square, m.image_corners.astype(np.float32))
                uv_in = cv2.perspectiveTransform(inner_local[None], H)[0]
                mount = geometry.mounts[m.marker_id]
                obj_in = (mount.rotation_object_from_marker
                          @ np.c_[inner_local, np.zeros(5)].T).T + np.asarray(mount.center)
                for P, uv in zip(obj_in, uv_in):
                    z = _corner_depth(v.depth, uv)
                    if z > 0:
                        D_obj.append(P)
                        z_obs.append(z)
                        d_cam.append(c)
    if not P_obj:
        return None
    P_obj, uv_obs, cam_idx = np.concatenate(P_obj), np.concatenate(uv_obs), np.concatenate(cam_idx)
    D_obj = np.asarray(D_obj, dtype=np.float64).reshape(-1, 3)
    z_obs, d_cam = np.asarray(z_obs, dtype=np.float64), np.asarray(d_cam, dtype=int)
    n = len(P_obj)
    R, t = init.rotation_matrix.copy(), np.asarray(init.position, dtype=np.float64).copy()
    if len(D_obj):
        # Drop background hits: samples far (>10 sigma) from the initial pose's prediction.
        z0 = np.array([(cams[c][0] @ (init.rotation_matrix @ P + t) + cams[c][1])[2]
                       for P, c in zip(D_obj, d_cam)])
        keep = np.abs(z_obs - z0) < 10 * depth_rel_sigma * np.maximum(z0, 0.05)
        D_obj, z_obs, d_cam = D_obj[keep], z_obs[keep], d_cam[keep]

    def twist_jacobian(Xw):
        """d(X)/d(ω, v) for a small world-frame twist: [-[X]× | I]."""
        m = len(Xw)
        skew = np.zeros((m, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -Xw[:, 2], Xw[:, 1], -Xw[:, 0]
        skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = Xw[:, 2], -Xw[:, 1], Xw[:, 0]
        return np.concatenate([-skew, np.broadcast_to(np.eye(3), (m, 3, 3))], axis=2)

    def evaluate(R, t):
        """Residuals (px and depth, in sigma units) and their Jacobian wrt the twist."""
        Xw = P_obj @ R.T + t
        r_px = np.zeros((n, 2))
        J_px = np.zeros((n, 2, 6))
        dX = twist_jacobian(Xw)
        for c, (R_cw, t_cw, K, dist) in enumerate(cams):
            sel = cam_idx == c
            Xc = Xw[sel] @ R_cw.T + t_cw
            proj, _ = cv2.projectPoints(Xc, _ZERO3, _ZERO3, K, dist)
            r_px[sel] = (proj.reshape(-1, 2) - uv_obs[sel]) / sigma_px
            x, y, z = Xc[:, 0], Xc[:, 1], np.maximum(Xc[:, 2], 1e-3)
            fx, fy = K[0, 0], K[1, 1]
            dproj = np.zeros((len(Xc), 2, 3))  # pinhole d(u,v)/d(Xc)
            dproj[:, 0, 0], dproj[:, 0, 2] = fx / z, -fx * x / z ** 2
            dproj[:, 1, 1], dproj[:, 1, 2] = fy / z, -fy * y / z ** 2
            J_px[sel] = dproj @ (R_cw @ dX[sel]) / sigma_px
        r, J = [r_px.ravel()], [J_px.reshape(-1, 6)]
        if len(D_obj):
            Dw = D_obj @ R.T + t
            dD = twist_jacobian(Dw)
            r_z, J_z = np.zeros(len(Dw)), np.zeros((len(Dw), 6))
            for c, (R_cw, t_cw, _, _) in enumerate(cams):
                sel = d_cam == c
                if not sel.any():
                    continue
                zp = np.maximum((Dw[sel] @ R_cw.T + t_cw)[:, 2], 0.05)
                sig = depth_rel_sigma * zp
                r_z[sel] = (zp - z_obs[sel]) / sig
                J_z[sel] = (R_cw @ dD[sel])[:, 2, :] / sig[:, None]
            r.append(r_z)
            J.append(J_z)
        return np.concatenate(r), np.concatenate(J)

    for _ in range(iterations):
        r, J = evaluate(R, t)
        w = np.minimum(1.0, 2.0 / np.maximum(np.abs(r), 1e-9))  # Huber weights (k = 2 sigma)
        JW = J * w[:, None]
        H = JW.T @ J + 1e-6 * np.eye(6)
        delta = -np.linalg.solve(H, JW.T @ r)
        dR, _ = cv2.Rodrigues(delta[:3])
        R, t = dR @ R, dR @ t + delta[3:]
        if np.linalg.norm(delta) < 1e-6:  # ~1 µm / 1 µrad
            break

    # Plain (unweighted) errors for the HUD.
    Xw = P_obj @ R.T + t
    px, dz = [], []
    for c, (R_cw, t_cw, K, dist) in enumerate(cams):
        sel = cam_idx == c
        Xc = Xw[sel] @ R_cw.T + t_cw
        proj, _ = cv2.projectPoints(Xc, _ZERO3, _ZERO3, K, dist)
        px.append(proj.reshape(-1, 2) - uv_obs[sel])
        dsel = d_cam == c
        if dsel.any():
            zp = ((D_obj[dsel] @ R.T + t) @ R_cw.T + t_cw)[:, 2]
            dz.extend((z_obs[dsel] - zp) * 1000)
    px = np.concatenate(px)
    ids = tuple(sorted({m.marker_id for v in views for m in v.markers}))
    return Refined(ObjectPose(ids, t, R), float(np.sqrt(np.mean(np.sum(px ** 2, 1)))), n,
                   len(dz), float(np.sqrt(np.mean(np.square(dz)))) if dz else float("nan"))
