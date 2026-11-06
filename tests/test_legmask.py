"""Leg region "sam" without the model: click parsing, prompt transfer between cameras, the
knee-click cut, the thigh tube, the BOX crop and the mask in _segment."""
import unittest

import numpy as np

from vigil.camera import Intrinsics
from vigil.meshing import legmask
from vigil.meshing.tsdf import (MeshError, TsdfParams, _box_crop, _click_section, _deproject, _in_tube, _segment,
                               _thigh_tube, _View)


def camera(T=np.eye(4)):
    return {"fx": 400.0, "fy": 400.0, "cx": 160.0, "cy": 120.0, "w": 320, "h": 240,
            "K": Intrinsics(320, 240, 400.0, 400.0, 160.0, 120.0), "T": T, "T_inv": np.linalg.inv(T)}


def plane_depth(c, z_world=1.0):
    """Depth image of the plane z = z_world (world frame) seen by camera c."""
    v, u = np.mgrid[0:240, 0:320].astype(float)
    ray = np.stack([(u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"], np.ones_like(u)], -1) @ c["T"][:3, :3].T
    return ((z_world - c["T"][2, 3]) / ray[..., 2]).astype(np.float32)


class PromptsTest(unittest.TestCase):
    def test_needs_a_leg_click(self):
        with self.assertRaises(MeshError):
            legmask.Prompts.parse({"index": 0, "clicks": [{"cam": 0, "x": 1, "y": 2, "kind": "not"}]}, 10)
        with self.assertRaises(MeshError):
            legmask.Prompts.parse(None, 10)

    def test_rejects_bad_input(self):
        for d in ({"index": 10, "clicks": [{"cam": 0, "x": 1, "y": 2, "kind": "leg"}]},
                  {"index": 0, "clicks": [{"cam": 0, "x": 1, "y": 2, "kind": "elbow"}]},
                  {"index": 0, "clicks": [{"cam": 0, "x": "a", "y": 2, "kind": "leg"}]}):
            with self.assertRaises(MeshError):
                legmask.Prompts.parse(d, 10)

    def test_points_per_camera_skip_the_knee(self):
        p = legmask.Prompts.parse({"index": 3, "clicks": [
            {"cam": 0, "x": 10, "y": 20, "kind": "leg"}, {"cam": 0, "x": 30, "y": 40, "kind": "not"},
            {"cam": 0, "x": 50, "y": 60, "kind": "knee"}, {"cam": 1, "x": 70, "y": 80, "kind": "leg"}]}, 10)
        self.assertEqual(p.points(0), ([[10.0, 20.0], [30.0, 40.0]], [1, 0]))
        self.assertEqual(p.points(1), ([[70.0, 80.0]], [1]))
        self.assertEqual(p.first("knee")["x"], 50.0)


    def test_a_box_alone_is_enough(self):
        p = legmask.Prompts.parse({"index": 0, "clicks": [
            {"cam": 1, "x": 200, "y": 150, "x2": 40, "y2": 30, "kind": "box"},
            {"cam": 0, "x": 5, "y": 5, "x2": 9, "y2": 9, "kind": "box"}]}, 10)  # too small: ignored
        self.assertEqual(p.boxes(), {1: [40.0, 30.0, 200.0, 150.0]})
        self.assertEqual(p.points(1), ([], []))  # not a point prompt
        with self.assertRaises(MeshError):  # a box needs its second corner
            legmask.Prompts.parse({"index": 0, "clicks": [{"cam": 0, "x": 1, "y": 2, "kind": "box"}]}, 10)


class TransferTest(unittest.TestCase):
    def setUp(self):
        self.a = camera()
        T = np.eye(4)
        T[0, 3] = 0.2  # second camera 20 cm to the side, same direction
        self.b = camera(T)
        self.da, self.db = plane_depth(self.a), plane_depth(self.b)
        self.mask = np.zeros((240, 320), bool)
        self.mask[80:160, 100:220] = True

    def test_points_land_inside_the_projected_mask(self):
        pts = legmask._transfer(self.mask, self.da, self.a, self.db, self.b)
        self.assertEqual(len(pts), legmask.TRANSFER_POINTS)
        # Moving the camera +0.2 m in x shifts the plane at 1 m by -fx * 0.2 = -80 px.
        for x, y in pts:
            self.assertTrue(100 - 80 <= x < 220 - 80 and 80 <= y < 160, (x, y))
        self.assertGreater(np.ptp(np.array(pts)[:, 0]), 40)  # spread out, not clumped

    def test_hidden_surface_gives_no_prompts(self):
        occluded = self.db * 0.7  # something nearer than the plane everywhere in camera b
        self.assertEqual(legmask._transfer(self.mask, self.da, self.a, occluded, self.b), [])


class SegmentAndSectionTest(unittest.TestCase):
    def test_mask_limits_the_kept_pixels(self):
        c = camera()
        depth = plane_depth(c)
        mask = np.zeros(depth.shape, bool)
        mask[50:100, 50:150] = True
        view = _View(0, None, None, {}, np.array([0, -1.0, 0]), 0.0)
        seg = _segment(depth, view, c, TsdfParams(region="sam", edge_filter=False), [], (0.2, 3.0), mask)
        kept = seg > 0
        self.assertFalse(kept[~mask].any())
        self.assertTrue(kept[60:90, 60:140].all())  # all but the eroded rim
        self.assertFalse(kept[50, 50:150].any())

    def test_knee_cut_points_into_the_clicked_side(self):
        # A bent leg: thigh along +x from the knee, shin going down-and-away along +y.
        thigh = np.stack([np.linspace(0.02, 0.4, 400), np.zeros(400), np.zeros(400)], 1)
        shin = np.stack([np.zeros(400), np.linspace(0.02, 0.4, 400), np.zeros(400)], 1)
        pts = np.concatenate([thigh, shin])
        e = {"view": _View(0, None, None, {}, None, 0.0), "cam": camera(), "orig_T": np.eye(4)}
        knee = {"point": np.zeros(3), "cam": 0, "t": 0.0}
        keep = {"point": np.array([0.3, 0.0, 0.0]), "cam": 0, "t": 0.0}
        s = _click_section([e], pts, TsdfParams(knee_shift_cm=2.0), knee, keep)
        np.testing.assert_allclose(s["axis"], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(s["knee"], [0.02, 0, 0], atol=1e-6)
        self.assertEqual(s["warnings"], [])

    def test_knee_and_leg_click_on_the_same_spot(self):
        e = {"view": _View(0, None, None, {}, None, 0.0), "cam": camera(), "orig_T": np.eye(4)}
        click = {"point": np.zeros(3), "cam": 0, "t": 0.0}
        with self.assertRaises(MeshError):
            _click_section([e], np.random.rand(500, 3), TsdfParams(), click, click)



class ThighTubeTest(unittest.TestCase):
    def test_keeps_the_thigh_and_drops_the_other_one(self):
        # The visible half of a thigh along +x from the knee (radius 6 cm), the other thigh
        # alongside it 22 cm away, and a blob of buttock past it; all one SAM mask.
        rng = np.random.default_rng(0)
        def half_tube(n, y0, r):
            s, a = rng.uniform(0, 0.4, n), rng.uniform(0, np.pi, n)
            return np.stack([s, y0 + r * np.cos(a), r * np.sin(a)], 1)
        thigh, other = half_tube(4000, 0.0, 0.06), half_tube(4000, 0.22, 0.07)
        blob = rng.normal([0.3, -0.2, 0.05], 0.04, (1500, 3))
        pts = np.concatenate([thigh, other, blob])
        section = {"centre": np.zeros(3), "axis": np.array([1.0, 0.05, 0]) / np.hypot(1, 0.05), "t": 0.0, "t_hip": 0.4}
        tube = _thigh_tube(pts, section)
        np.testing.assert_allclose(tube["axis"], [1, 0, 0], atol=0.03)  # refitted to the thigh
        self.assertGreater(_in_tube(thigh, tube).mean(), 0.97)
        self.assertLess(_in_tube(other, tube).mean(), 0.05)
        self.assertLess(_in_tube(blob, tube).mean(), 0.1)

    def test_too_little_thigh_above_the_knee(self):
        section = {"centre": np.zeros(3), "axis": np.array([1.0, 0, 0]), "t": 0.0, "t_hip": None}
        self.assertIsNone(_thigh_tube(np.random.rand(50, 3) * 0.05, section))


class BoxCropTest(unittest.TestCase):
    def test_keeps_what_projects_inside_the_box(self):
        c = camera()
        depth = plane_depth(c)
        world = _deproject(depth, c)
        e = {"view": _View(0, None, None, {}, None, 0.0), "cam": c,
             "seg": np.round(depth * 1000).astype(np.uint16), "pts": world.reshape(-1, 3)[::7].astype(float),
             "nrm": np.tile([0, 0, -1.0], (len(world.reshape(-1, 3)[::7]), 1))}
        warnings = []
        stats = _box_crop([e], {0: [100, 80, 220, 160]}, 0.0, warnings)
        kept = e["seg"] > 0
        self.assertTrue(kept[82:158, 102:218].all())
        self.assertFalse(kept[:78].any() or kept[:, :98].any() or kept[:, 222:].any())
        self.assertEqual(len(e["pts"]), len(e["nrm"]))
        self.assertAlmostEqual(stats["removed"], 1 - 120 * 80 / (320 * 240), delta=0.02)
        self.assertEqual(warnings, [])
        _box_crop([e], {3: [0, 0, 10, 10]}, 0.0, warnings)  # a camera with no frames in the mesh
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
