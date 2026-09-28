"""Physical description of the ArUco cube probe (ported from DataCollection)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MarkerMount:
    """Where one tag sits on the cube. Marker +Z points out of the face, +Y to the printed top."""

    marker_id: int
    face: str
    center: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]

    @property
    def rotation_object_from_marker(self) -> np.ndarray:
        x_axis = np.asarray(self.x_axis, dtype=np.float64)
        y_axis = np.asarray(self.y_axis, dtype=np.float64)
        z_axis = np.cross(x_axis, y_axis)
        return np.column_stack((x_axis, y_axis, z_axis))


def marker_object_points(marker_length: float) -> np.ndarray:
    """Tag corners in the tag's own frame, in OpenCV's detection order (TL, TR, BR, BL)."""
    h = marker_length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass(frozen=True)
class ProbeGeometry:
    marker_length: float = 0.04
    cube_size: tuple[float, float, float] = (0.05, 0.05, 0.05)
    # Probe tip in the cube frame; the probe body extends along +Y (unused face).
    tip: tuple[float, float, float] = (0.0, 0.1925096, -0.003594)
    mounts: dict[int, MarkerMount] = field(default_factory=dict)

    @classmethod
    def clarius(cls, marker_length: float, cube_size: tuple[float, float, float],
                tip: tuple[float, float, float]) -> "ProbeGeometry":
        """Five tags on the exposed faces; the attachment occupies +Y."""
        hx, hy, hz = (v / 2 for v in cube_size)
        mounts = {
            0: MarkerMount(0, "front (+Z)", (0, 0, hz), (1, 0, 0), (0, 1, 0)),
            1: MarkerMount(1, "back (-Z)", (0, 0, -hz), (-1, 0, 0), (0, 1, 0)),
            2: MarkerMount(2, "right (+X)", (hx, 0, 0), (0, 0, -1), (0, 1, 0)),
            3: MarkerMount(3, "left (-X)", (-hx, 0, 0), (0, 0, 1), (0, 1, 0)),
            4: MarkerMount(4, "bottom (-Y)", (0, -hy, 0), (1, 0, 0), (0, 0, 1)),
        }
        return cls(marker_length, tuple(cube_size), tuple(tip), mounts)

    def marker_corners_in_object(self, marker_id: int) -> np.ndarray:
        mount = self.mounts[marker_id]
        corners = marker_object_points(self.marker_length)
        return (mount.rotation_object_from_marker @ corners.T).T + np.asarray(mount.center)

    @property
    def tip_array(self) -> np.ndarray:
        return np.asarray(self.tip, dtype=np.float64)
