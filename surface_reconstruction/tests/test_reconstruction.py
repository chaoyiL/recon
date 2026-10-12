"""Geometry truth, derivatives, tracking recovery and video I/O regressions."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import unittest
from pathlib import Path
import tempfile
import csv
import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import cv2
import numpy as np
from surface import SurfaceModel, align, local_coordinates
from tracking import TrackingSeed, MarkerTracker, select_complete_grid

ROOT = Path(__file__).resolve().parents[1]
XY = np.array([(3.0 * c, 6.0 * r) for r in range(8) for c in range(6)])
K = np.array([[425.0, 0.0, 305.0], [0.0, 425.0, 244.0], [0.0, 0.0, 1.0]])


def project(q):
    return cv2.projectPoints(q, np.zeros(3), np.zeros(3), K, np.zeros(5))[0].reshape(-1, 2)


def cylinder(
    sign=1.0, angle=0.0, radius=30.0, rotation=(0.3, -0.25, 0.7), translation=(-5.0, 3.0, 100.0)
):
    xy = XY - XY.mean(0)
    axis = np.array([np.sin(angle), np.cos(angle)])
    cross = np.array([-axis[1], axis[0]])
    s = xy @ axis
    t = xy @ cross
    curved = radius * np.sin(s / radius) if sign else s
    Q = np.column_stack(
        (curved[:, None] * axis + t[:, None] * cross, sign * radius * (1 - np.cos(s / radius)))
    )
    return Q @ cv2.Rodrigues(np.array(rotation, float))[0].T + translation


class GeometryTests(unittest.TestCase):
    def test_plane_arbitrary_pose(self):
        for rot in [(0.0, 0.0, 0.0), (0.5, -0.4, 1.9), (-0.4, 0.3, -1.5)]:
            truth = cylinder(0, rotation=rot)
            q, _, info = SurfaceModel(XY, K, np.zeros(5)).fit(project(truth))
            np.testing.assert_allclose(q, truth, atol=1e-5)
            self.assertFalse(info["status"] == "ambiguous")

    def test_bent_first_frame_both_signs_and_axes(self):
        for sign in (-1.0, 1.0):
            for angle in (0.0, 0.7, np.pi / 2):
                with self.subTest(sign=sign, axis=angle):
                    truth = cylinder(sign, angle)
                    q, _, info = SurfaceModel(XY, K, np.zeros(5)).fit(project(truth))
                    self.assertLess(np.sqrt(np.mean(np.sum((q - truth) ** 2, axis=1))), 0.4)
                    # A mirrored bend cannot pass this camera-coordinate test.
                    self.assertLess(info["metric_max_error"], 0.02)

    def test_free_motion_and_deformation_sequence(self):
        m = SurfaceModel(XY, K, np.zeros(5))
        for step in range(12):
            truth = cylinder(
                1.0,
                angle=0.2,
                radius=40 - step * 0.8,
                rotation=(0.2 + step * 0.025, -0.2, step * 0.05),
                translation=(-5 + step, step * 0.3, 100 + step),
            )
            q, _, _ = m.fit(project(truth))
            self.assertLess(np.sqrt(np.mean(np.sum((q - truth) ** 2, axis=1))), 0.6)

    def test_noise_and_missing_observations(self):
        truth = cylinder(-1.0)
        p = project(truth) + np.random.default_rng(8).normal(0, 0.12, (48, 2))
        m = SurfaceModel(XY, K, np.zeros(5))
        m.fit(p)
        p[[2, 3, 15, 16, 30, 31, 39, 45]] = np.nan
        q, _, info = m.fit(p)
        self.assertEqual(info["visible_count"], 40)
        self.assertLess(np.sqrt(np.mean(np.sum((q - truth) ** 2, axis=1))), 1.0)

    def test_analytic_jacobian(self):
        m = SurfaceModel(XY, K, np.zeros(5))
        truth = cylinder()
        p = project(truth)
        visible = np.ones(48, bool)
        coefficients = np.linalg.lstsq(m.B, truth, rcond=None)[0].ravel()
        for prior in (None, truth):
            f = m.residual_function(p, visible, np.ones(48), prior)
            h = 1e-5
            numeric = np.column_stack(
                [
                    (f(coefficients + np.eye(48)[i] * h) - f(coefficients - np.eye(48)[i] * h))
                    / (2 * h)
                    for i in range(48)
                ]
            )
            np.testing.assert_allclose(f(coefficients, True), numeric, atol=2e-5, rtol=2e-5)

    def test_shape_alignment_is_rigid_and_right_handed(self):
        q = cylinder()
        local, R, origin = local_coordinates(q, XY)
        np.testing.assert_allclose(local @ R + origin, q, atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        moved = q @ cv2.Rodrigues(np.array([0.8, -0.2, 0.4]))[0].T + [8, 3, 7]
        np.testing.assert_allclose(local_coordinates(moved, XY)[0], local, atol=1e-8)
        np.testing.assert_allclose(align(moved, q), q, atol=1e-8)

    def test_insufficient_and_nonfinite_rejected(self):
        p = project(cylinder())
        p[:20] = np.nan
        with self.assertRaises(ValueError):
            SurfaceModel(XY, K, np.zeros(5)).fit(p)
        with self.assertRaises(ValueError):
            SurfaceModel(XY, K, np.zeros(5)).fit(p[:20])

    def test_outlier_shape_rejected(self):
        p = project(cylinder())
        p[20] += [40.0, 20.0]
        q, _, info = SurfaceModel(XY, K, np.zeros(5)).fit(p)
        self.assertEqual(info["rejected_observation_ids"], [20])
        self.assertLess(np.sqrt(np.mean(np.sum((q - cylinder()) ** 2, axis=1))), 0.5)


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = cv2.imread(str(ROOT / "data/reference.jpg"))
        cls.seed, _, _ = TrackingSeed.build(cls.frame, np.arange(48), XY)

    def test_current_stills(self):
        for name in ("reference.jpg", "test1.jpg", "test2.jpg"):
            with self.subTest(image=name):
                frame = cv2.imread(str(ROOT / "data" / name))
                seed, _, _ = TrackingSeed.build(frame, np.arange(48), XY)
                self.assertEqual(seed.pixels.shape, (48, 2))

    def test_bent_grid_with_outlier(self):
        p = project(cylinder(rotation=(0.15, 0.1, 0.15)))
        ordered, _ = select_complete_grid(np.vstack((p, [320, 250])), 8, 6)
        np.testing.assert_allclose(ordered, p, atol=1e-8)

    def test_missing_grid_not_invented(self):
        with self.assertRaises(ValueError):
            select_complete_grid(self.seed.pixels[:-1], 8, 6)
        with self.assertRaises(ValueError):
            select_complete_grid(np.vstack((self.seed.pixels, self.seed.pixels[20] + 0.05)), 8, 6)

    def test_larger_translation_persistent_ids(self):
        tracker = MarkerTracker(self.seed)
        for step in range(1, 8):
            move = np.array([6.0 * step, 2.0 * step])
            image = cv2.warpAffine(
                self.frame, np.array([[1.0, 0, move[0]], [0, 1.0, move[1]]]), (640, 480)
            )
            result = tracker.update(image)
            self.assertTrue(result["valid"].all())
            np.testing.assert_allclose(result["pixels"], self.seed.pixels + move, atol=0.8)

    def test_rotation_persistent_ids(self):
        tracker = MarkerTracker(self.seed)
        for step in range(1, 9):
            matrix = cv2.getRotationMatrix2D((320, 240), step * 2.0, 1.0)
            image = cv2.warpAffine(self.frame, matrix, (640, 480))
            result = tracker.update(image)
            truth = np.column_stack((self.seed.pixels, np.ones(48))) @ matrix.T
            self.assertGreaterEqual(result["valid"].sum(), 46)
            np.testing.assert_allclose(
                result["pixels"][result["valid"]], truth[result["valid"]], atol=0.9
            )

    def test_occluded_mark_is_not_observation(self):
        image = self.frame.copy()
        u, v = np.rint(self.seed.pixels[20]).astype(int)
        cv2.rectangle(image, (u - 5, v - 5), (u + 5, v + 5), tuple(map(int, image[v + 8, u])), -1)
        tracker = MarkerTracker(self.seed)
        result = tracker.update(image)
        self.assertFalse(result["valid"][20])
        self.assertTrue(np.isnan(result["pixels"][20]).all())
        self.assertTrue(tracker.update(self.frame)["valid"][20])

    def test_no_frame_border_fallback(self):
        tracker = MarkerTracker(self.seed)
        with patch("tracking.detector.find_inner_roi", side_effect=RuntimeError("clipped")):
            result = tracker.update(self.frame)
        self.assertTrue(result["valid"].all())

    def test_blank_and_resolution_change(self):
        tracker = MarkerTracker(self.seed)
        self.assertFalse(tracker.update(np.full((480, 640), 255, np.uint8))["valid"].any())
        with self.assertRaises(ValueError):
            tracker.update(np.zeros((240, 320), np.uint8))


class PipelineTests(unittest.TestCase):
    def test_video_from_bent_frame_records_and_recovers(self):
        import run_reconstruction as app

        with tempfile.TemporaryDirectory(prefix="surface_test_") as directory:
            root = Path(directory)
            video = root / "input.avi"
            frame = cv2.imread(str(ROOT / "data/test2.jpg"))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
            self.assertTrue(writer.isOpened())
            for step in range(14):
                image = frame if step < 3 or step >= 10 else np.full_like(frame, 255)
                writer.write(image)
            writer.release()
            args = argparse.Namespace(
                config=ROOT / "config.json",
                video=str(video),
                image=None,
                out=root / "result",
                max_frames=None,
                preview=False,
                no_preview=True,
                no_record=False,
                row_direction="auto",
            )
            with patch.object(app, "ROOT", root), redirect_stdout(io.StringIO()):
                summary = app.run(args)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["frames"], 14)
            self.assertGreaterEqual(summary["segments"], 2)
            self.assertGreaterEqual(summary["reconstructed_frames"], 6)
            with (root / "result/frame_points.csv").open() as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 14 * 48)
            self.assertTrue(all(not r["X_mm"] for r in records if 3 <= int(r["frame"]) < 10))
            capture = cv2.VideoCapture(str(root / "result/reconstruction.avi"))
            ok, image = capture.read()
            self.assertTrue(ok)
            self.assertEqual(image.shape[:2], (480, 1280))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 14)
            capture.release()
            self.assertTrue((root / "marker_detection/result/detected_markers.avi").exists())

    def test_resolution_mismatch_not_silently_resized(self):
        import run_reconstruction as app

        with tempfile.TemporaryDirectory(prefix="surface_test_") as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "wrong.png"), np.zeros((240, 320, 3), np.uint8))
            args = argparse.Namespace(
                config=ROOT / "config.json",
                image=str(root / "wrong.png"),
                video=None,
                out=root / "result",
                max_frames=None,
                preview=False,
                no_preview=True,
                no_record=True,
                row_direction="auto",
            )
            with patch.object(app, "ROOT", root), self.assertRaises(ValueError):
                app.run(args)


if __name__ == "__main__":
    unittest.main()
