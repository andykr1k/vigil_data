"""Automatic leg prompts and the autonomous pipeline without the models: a synthetic two-camera
recording, with Sapiens2's output faked per camera."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vigil.camera import Intrinsics
from vigil.meshing import auto, autoleg
from vigil.meshing.library import MeshLibrary
from vigil.meshing.tsdf import Mesh, MeshError, TsdfParams, _click_section, _View

W, H = 320, 240
CAM = {"fx": 400.0, "fy": 400.0, "cx": 160.0, "cy": 120.0}
P = autoleg.PART_NAMES.index


def camera(T=np.eye(4)):
    return {**CAM, "w": W, "h": H, "K": Intrinsics(W, H, CAM["fx"], CAM["fy"], CAM["cx"], CAM["cy"]),
            "T": T, "T_inv": np.linalg.inv(T)}


def bare_leg_parts():
    """A left leg lying along the image's x axis: shin, thigh (knee at u=160), torso (hip at u=300),
    and a smaller right leg below it."""
    parts = np.zeros((H, W), np.uint8)
    parts[100:140, 40:160] = P("Left_Lower_Leg")
    parts[100:140, 160:300] = P("Left_Upper_Leg")
    parts[90:150, 300:320] = P("Torso")
    parts[190:210, 40:160] = P("Right_Lower_Leg")
    parts[190:210, 160:300] = P("Right_Upper_Leg")
    return parts


def leg_depth():
    d = np.full((H, W), 1.5, np.float32)  # floor
    d[100:140, 40:320] = 1.0
    d[185:215, 40:320] = 1.0
    return d


def analysis(parts, kps=None):
    kp, score = np.zeros((308, 2)), np.zeros(308)
    for i, (xy, s) in (kps or {}).items():
        kp[i], score[i] = xy, s
    return {"parts": parts, "kp": kp, "score": score}


def view(a, cam=0):
    return autoleg.View(cam, 0, np.zeros((H, W, 3), np.uint8), leg_depth(), camera(), a)


def make_recording(root: Path, n=16, dt=0.1) -> Path:
    """Two calibrated cameras (both at the world origin), n frames of colour + depth."""
    path = root / "2026-01-01_00-00-00_lower_limb"
    lines = []
    for cam in (0, 1):
        (path / f"cam{cam}" / "color").mkdir(parents=True)
        (path / f"cam{cam}" / "depth").mkdir(parents=True)
    for i in range(n):
        files = []
        for cam in (0, 1):
            bgr = np.full((H, W, 3), 60, np.uint8)
            bgr[0, 0, 2] = 100 * cam  # tells the fake analysis which camera it's looking at
            cv2.imwrite(str(path / f"cam{cam}/color/{i}.png"), bgr)
            cv2.imwrite(str(path / f"cam{cam}/depth/{i}.png"), (leg_depth() * 1000).astype(np.uint16))
            files.append({"cam": cam, "color": f"cam{cam}/color/{i}.png", "depth": f"cam{cam}/depth/{i}.png"})
        lines.append({"t": 100 + i * dt, "seq": i, "frame": {}, "files": files})
    meta = {"hello": {"cameras": []}, "cameras": [
        {"index": cam, **CAM, "width": W, "height": H, "T_world_camera": np.eye(4).ravel().tolist()} for cam in (0, 1)]}
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "frames.jsonl").write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    return path


def fake_analyse(per_cam):
    return lambda images, *_: [per_cam[int(im[0, 0, 0]) // 100] for im in images]


class ReadLegTest(unittest.TestCase):
    def test_bare_leg_parts_give_knee_hip_and_prompts(self):
        r = autoleg.read_leg(view(analysis(bare_leg_parts())), "left")
        self.assertEqual(r["source"], "parts")
        self.assertAlmostEqual(r["knee"][0], 0.0, delta=0.01)  # u=160 at z=1 m is x=0
        self.assertAlmostEqual(r["hip"][0], 0.35, delta=0.01)  # u=300
        parts = bare_leg_parts()
        self.assertTrue(all(parts[int(y), int(x)] == P("Left_Upper_Leg") for x, y in r["leg"]))
        self.assertTrue(r["not"] and all(parts[int(y), int(x)] in (P("Right_Upper_Leg"), P("Right_Lower_Leg"))
                                         for x, y in r["not"]))

    def test_hem_is_not_the_hip(self):
        # Shorts / trousers pushed up: a bare strip above the knee, clothing over the rest of the
        # thigh. The strip's edge against the clothing is a hem; the hip is the keypoint.
        parts = bare_leg_parts()
        parts[100:140, 210:300] = P("Lower_Clothing")
        kps = {9: ((300, 120), 0.8), 11: ((160, 120), 0.9)}  # left hip, knee
        r = autoleg.read_leg(view(analysis(parts, kps)), "left")
        self.assertEqual(r["source"], "parts")
        self.assertAlmostEqual(r["hip"][0], 0.35, delta=0.01)  # the keypoint, not the hem at u=210 (x=0.125)
        on_clothing = [p for p in r["leg"] if parts[int(p[1]), int(p[0])] == P("Lower_Clothing")]
        self.assertGreaterEqual(len(on_clothing), 2)  # + prompts on the covered thigh too

    def test_torso_speck_by_the_knee_is_not_the_hip(self):
        # Shorts: no thigh–torso edge in view, but a few pixels next to the knee labelled torso.
        parts = bare_leg_parts()
        parts[90:150, 300:320] = P("Lower_Clothing")
        parts[95:105, 165:180] = P("Torso")
        kps = {9: ((300, 120), 0.8), 11: ((160, 120), 0.9)}
        r = autoleg.read_leg(view(analysis(parts, kps)), "left")
        self.assertAlmostEqual(r["hip"][0], 0.35, delta=0.01)  # the keypoint, not the speck at x≈0.03

    def test_trousers_fall_back_to_keypoints(self):
        parts = np.full((H, W), P("Lower_Clothing"), np.uint8)
        kps = {11: ((160, 120), 0.9), 9: ((300, 120), 0.8), 13: ((40, 120), 0.9),  # left knee, hip, ankle
               12: ((160, 200), 0.9),   # right knee, clearly apart: a − prompt
               14: ((100, 122), 0.9)}   # right ankle on the left shin (a mix-up): not a − prompt
        r = autoleg.read_leg(view(analysis(parts, kps)), "left")
        self.assertEqual(r["source"], "pose")
        self.assertEqual(len(r["leg"]), 4)  # three along the thigh, one on the shin
        self.assertEqual(r["not"], [[160.0, 200.0]])
        self.assertAlmostEqual(r["hip"][0], 0.35, delta=0.01)
        self.assertIsNone(autoleg.read_leg(view(analysis(parts, {11: ((160, 120), 0.3)})), "left"))

    def test_no_leg(self):
        self.assertIsNone(autoleg.read_leg(view(analysis(np.zeros((H, W), np.uint8))), "left"))


class VoteTest(unittest.TestCase):
    axis = np.array([1.0, 0, 0])

    def votes(self, s0, s1, jitter=0.0, across=0.0):
        return [(0, np.array([s0 + jitter * (-1) ** k, 0, 1])) for k in range(4)] + \
               [(1, np.array([s1, across, 1])) for _ in range(4)]

    def test_consensus_is_the_mean_of_camera_medians(self):
        v = autoleg.vote(self.votes(0.0, 0.04), self.axis)
        self.assertAlmostEqual(v["s"], 0.02)
        self.assertAlmostEqual(v["disagree"], 0.04)
        self.assertEqual(autoleg.check_vote(v, "knee"), [])

    def test_rejections(self):
        for votes, why in ((self.votes(0.0, 0.1), "disagree"), (self.votes(0.0, 0.0, jitter=0.05), "jumps"),
                           (self.votes(0.0, 0.0, across=0.3), "different legs")):
            with self.assertRaisesRegex(autoleg.AutoError, why):
                autoleg.check_vote(autoleg.vote(votes, self.axis), "knee")

    def test_outlier_frame_is_dropped(self):
        votes = [(0, np.array([s, 0, 1])) for s in (0.0, 0.002, -0.002, 0.3)]
        v = autoleg.vote(votes, self.axis)
        self.assertEqual(v["cameras"][0]["dropped"], 1)
        self.assertAlmostEqual(v["s"], 0.0, places=3)


class WidthKneeTest(unittest.TestCase):
    def test_narrowest_point(self):
        rng = np.random.default_rng(0)
        s = np.repeat(np.arange(-0.2, 0.2, 0.002), 80)
        half = 0.045 + 0.3 * np.abs(s + 0.02)  # narrowest 2 cm below the knee
        pts = np.stack([s, rng.uniform(-1, 1, len(s)) * half, np.ones_like(s)], 1)
        self.assertAlmostEqual(autoleg.width_knee(pts, np.zeros(3), np.array([1.0, 0, 0])), -0.02, delta=0.015)


class ProposeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = make_recording(Path(self.tmp.name) / "recordings")
        self.meta, self.lines = auto.load_recording(self.path)
        autoleg._cache.clear()

    def tearDown(self):
        self.tmp.cleanup()
        autoleg._cache.clear()

    def test_prompts_from_bare_legs(self):
        a = analysis(bare_leg_parts())
        info = autoleg.propose(self.path, self.meta, self.lines, 8, analyse_fn=fake_analyse({0: a, 1: a}))
        self.assertEqual((info["side"], info["source"]), ("left", "parts"))
        kinds = [c["kind"] for c in info["prompts"]["clicks"]]
        self.assertEqual((kinds.count("leg"), kinds.count("not"), kinds.count("knee"), kinds.count("hip")), (6, 4, 1, 1))
        knee = next(c for c in info["prompts"]["clicks"] if c["kind"] == "knee")
        hip = next(c for c in info["prompts"]["clicks"] if c["kind"] == "hip")
        self.assertAlmostEqual(knee["x"], 160, delta=6)
        self.assertAlmostEqual(hip["x"], 300, delta=6)
        self.assertAlmostEqual(info["hip"]["length_cm"], 35, delta=2)
        np.testing.assert_allclose(info["axis"], [1, 0, 0], atol=0.02)

    def test_wrong_side_or_nothing(self):
        empty = analysis(np.zeros((H, W), np.uint8))
        with self.assertRaisesRegex(autoleg.AutoError, "No camera sees"):
            autoleg.propose(self.path, self.meta, self.lines, 8, analyse_fn=fake_analyse({0: empty, 1: empty}))
        with self.assertRaises(MeshError):
            autoleg.propose(self.path, self.meta, self.lines, 8, side="middle")


class RunRecordingTest(unittest.TestCase):
    def test_retries_and_non_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_recording(Path(tmp) / "recordings")
            library = MeshLibrary(Path(tmp) / "Meshes")
            a = analysis(bare_leg_parts())
            calls = []

            def build(method, path_, meta, lines, idx, params):
                calls.append(idx)
                self.assertEqual((params["region"], params["section"]), ("sam", "knee"))
                if len(calls) == 1:
                    raise MeshError("first window fails")
                v = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]], np.float32)
                return Mesh(v, np.array([[0, 1, 2]], np.uint32), np.tile([0, 0, -1], (3, 1)).astype(np.float32),
                            np.full((3, 3), 128, np.uint8),
                            {"method": "tsdf", "area_cm2": 800.0, "warnings": [],
                             "segmentation": {"cameras": {"0": {"mask_px_cv": 0.05}}, "support_removed": 0.1}})

            autoleg._cache.clear()
            with patch.object(autoleg, "analyse", fake_analyse({0: a, 1: a})), \
                    patch.object(auto, "precheck", return_value={"warnings": []}):
                r = auto.run_recording(path, library, count=2, attempts=3, window_s=0.2, stride_s=0.3,
                                       build_fn=build, preview_dir=Path(tmp) / "previews")
            autoleg._cache.clear()
            self.assertEqual([x["status"] for x in r["attempts"]], ["rejected", "accepted", "accepted"])
            self.assertEqual(r["attempts"][0]["reason"], "first window fails")
            times = [x["time_s"] for x in r["attempts"] if x["status"] == "accepted"]
            self.assertGreaterEqual(abs(times[1] - times[0]), 0.4)  # windows don't overlap
            self.assertEqual(len(library.list()), 2)
            saved = json.loads(library.path(path.name, r["accepted"][0], "json").read_text())
            self.assertEqual(saved["stats"]["auto"]["side"], "left")
            self.assertTrue((Path(tmp) / "previews" / r["attempts"][1]["preview"]).is_file())

    def test_postcheck(self):
        ok = {"area_cm2": 800, "warnings": [], "segmentation": {"cameras": {"0": {"mask_px_cv": 0.1}}}}
        self.assertEqual(auto.postcheck(ok), [])
        for bad, why in (({"segmentation": {"cameras": {"1": {"mask_px_cv": 0.6}}}}, "varies"),
                         ({"segmentation": {"cameras": {}, "support_removed": 0.7}}, "support cut"),
                         ({"area_cm2": 20}, "Implausible"),
                         ({"warnings": ["The KNEE click isn't on the leg surface (more than 3 cm off)"]}, "knee cut")):
            with self.assertRaisesRegex(autoleg.AutoError, why):
                auto.postcheck(ok | bad)


class HipCutTest(unittest.TestCase):
    def test_hip_click_cuts_the_thigh(self):
        pts = np.stack([np.linspace(-0.3, 0.6, 900), np.zeros(900), np.zeros(900)], 1)
        e = {"view": _View(0, None, None, {}, None, 0.0), "cam": camera(), "orig_T": np.eye(4)}
        click = lambda x: {"point": np.array([x, 0.0, 0.0]), "cam": 0, "t": 0.0}
        s = _click_section([e], pts, TsdfParams(), click(0.0), click(0.2), click(0.4))
        self.assertAlmostEqual(s["t_hip"], 0.4, places=6)
        s = _click_section([e], pts, TsdfParams(), click(0.0), click(0.05), click(0.1))
        self.assertIsNone(s["t_hip"])
        self.assertIn("HIP click", s["warnings"][0])


if __name__ == "__main__":
    unittest.main()
