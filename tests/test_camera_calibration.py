import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import (
    calibrate_with_outlier_rejection,
    find_board_points,
    find_circles,
    make_object_points,
    parse_calibration_config,
    select_inlier_indices,
)
from utils.config import ConfigError


def _config(section: dict) -> object:
    return parse_calibration_config(section, Path("/tmp/config.yaml"))


def _render_symmetric_white_dots(
    cols: int,
    rows: int,
    spacing: int = 48,
    radius: int = 14,
    margin: int = 72,
) -> np.ndarray:
    width = margin * 2 + (cols - 1) * spacing
    height = margin * 2 + (rows - 1) * spacing
    image = np.zeros((height, width), np.uint8)
    for row in range(rows):
        for col in range(cols):
            center = (margin + col * spacing, margin + row * spacing)
            cv2.circle(image, center, radius, 255, -1, cv2.LINE_AA)
    return image


def _render_asymmetric_white_dots(
    cols: int,
    rows: int,
    spacing: int = 36,
    radius: int = 12,
    margin: int = 72,
) -> np.ndarray:
    width = margin * 2 + (2 * (cols - 1) + 1) * spacing
    height = margin * 2 + (rows - 1) * spacing
    image = np.zeros((height, width), np.uint8)
    for row in range(rows):
        for col in range(cols):
            center = (
                margin + (2 * col + row % 2) * spacing,
                margin + row * spacing,
            )
            cv2.circle(image, center, radius, 255, -1, cv2.LINE_AA)
    return image


class CameraCalibrationConfigTest(unittest.TestCase):
    def test_default_board_type_is_chessboard(self):
        config = _config({
            "board_cols": 6,
            "board_rows": 6,
            "square_size_mm": 25.0,
            "min_samples": 12,
            "output": "camera.yaml",
        })
        self.assertEqual(config.board_type, "chessboard")
        self.assertEqual(config.circle_pattern, "symmetric")
        self.assertFalse(config.uses_circles)

    def test_parses_white_dot_circle_board(self):
        config = _config({
            "board_type": "circles",
            "circle_pattern": "asymmetric",
            "board_cols": 4,
            "board_rows": 11,
            "square_size_mm": 15.0,
            "min_samples": 8,
            "output": "camera.yaml",
        })
        self.assertTrue(config.uses_circles)
        self.assertEqual(config.circle_pattern, "asymmetric")
        self.assertEqual(config.board_size, (4, 11))

    def test_rejects_unknown_board_type(self):
        with self.assertRaisesRegex(ConfigError, "board_type"):
            _config({"board_type": "charuco", "output": "camera.yaml"})

    def test_rejects_unknown_circle_pattern(self):
        with self.assertRaisesRegex(ConfigError, "circle_pattern"):
            _config({
                "board_type": "circles",
                "circle_pattern": "hex",
                "output": "camera.yaml",
            })

    def test_parses_outlier_rejection_settings(self):
        config = _config({
            "max_view_error_px": 1.5,
            "outlier_sigma": 3,
            "outlier_max_rounds": 4,
            "output": "camera.yaml",
        })
        self.assertEqual(config.max_view_error_px, 1.5)
        self.assertEqual(config.outlier_sigma, 3.0)
        self.assertEqual(config.outlier_max_rounds, 4)
        self.assertTrue(config.rejects_outliers)

    def test_zero_sigma_disables_relative_rejection(self):
        config = _config({
            "outlier_sigma": 0,
            "max_view_error_px": None,
            "outlier_max_rounds": 5,
            "output": "camera.yaml",
        })
        self.assertIsNone(config.outlier_sigma)
        self.assertFalse(config.rejects_outliers)

    def test_rejects_invalid_outlier_settings(self):
        with self.assertRaisesRegex(ConfigError, "max_view_error_px"):
            _config({"max_view_error_px": 0, "output": "camera.yaml"})
        with self.assertRaisesRegex(ConfigError, "outlier_sigma"):
            _config({"outlier_sigma": -1, "output": "camera.yaml"})
        with self.assertRaisesRegex(ConfigError, "outlier_max_rounds"):
            _config({"outlier_max_rounds": -1, "output": "camera.yaml"})


class CameraCalibrationGeometryTest(unittest.TestCase):
    def test_symmetric_object_points_match_chessboard_layout(self):
        chess = _config({
            "board_cols": 4,
            "board_rows": 3,
            "square_size_mm": 10.0,
            "output": "camera.yaml",
        })
        circles = _config({
            "board_type": "circles",
            "circle_pattern": "symmetric",
            "board_cols": 4,
            "board_rows": 3,
            "square_size_mm": 10.0,
            "output": "camera.yaml",
        })
        np.testing.assert_array_equal(
            make_object_points(chess),
            make_object_points(circles),
        )
        points = make_object_points(circles)
        self.assertEqual(points.shape, (12, 3))
        np.testing.assert_array_equal(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(points[4], [0.0, 10.0, 0.0])

    def test_asymmetric_object_points_follow_opencv_convention(self):
        config = _config({
            "board_type": "circles",
            "circle_pattern": "asymmetric",
            "board_cols": 4,
            "board_rows": 3,
            "square_size_mm": 10.0,
            "output": "camera.yaml",
        })
        points = make_object_points(config)
        np.testing.assert_array_equal(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(points[1], [20.0, 0.0, 0.0])
        np.testing.assert_array_equal(points[4], [10.0, 10.0, 0.0])
        np.testing.assert_array_equal(points[5], [30.0, 10.0, 0.0])


class WhiteDotDetectionTest(unittest.TestCase):
    def test_finds_symmetric_white_dots_on_black(self):
        cols, rows = 5, 4
        image = _render_symmetric_white_dots(cols, rows)
        found, centers = find_circles(image, (cols, rows), asymmetric=False)
        self.assertTrue(found)
        self.assertIsNotNone(centers)
        assert centers is not None
        self.assertEqual(len(centers), cols * rows)

    def test_finds_asymmetric_white_dots_on_black(self):
        cols, rows = 4, 5
        image = _render_asymmetric_white_dots(cols, rows)
        found, centers = find_circles(image, (cols, rows), asymmetric=True)
        self.assertTrue(found)
        self.assertIsNotNone(centers)
        assert centers is not None
        self.assertEqual(len(centers), cols * rows)

    def test_find_board_points_uses_circle_config(self):
        cols, rows = 5, 4
        image = _render_symmetric_white_dots(cols, rows)
        with tempfile.TemporaryDirectory() as directory:
            config = parse_calibration_config(
                {
                    "board_type": "circles",
                    "board_cols": cols,
                    "board_rows": rows,
                    "square_size_mm": 20.0,
                    "output": "camera.yaml",
                },
                Path(directory) / "config.yaml",
            )
        found, centers = find_board_points(image, config)
        self.assertTrue(found)
        self.assertIsNotNone(centers)


class OutlierRejectionTest(unittest.TestCase):
    def test_absolute_threshold_drops_high_error_views(self):
        kept, rejected, threshold = select_inlier_indices(
            [0.20, 0.30, 0.25, 2.00, 0.22],
            max_error_px=1.0,
            outlier_sigma=None,
        )
        self.assertEqual(threshold, 1.0)
        self.assertEqual(kept, [0, 1, 2, 4])
        self.assertEqual(rejected, [3])

    def test_statistical_threshold_drops_extreme_view(self):
        kept, rejected, threshold = select_inlier_indices(
            [0.20, 0.21, 0.19, 0.22, 0.20, 1.50],
            max_error_px=None,
            outlier_sigma=2.5,
        )
        self.assertEqual(rejected, [5])
        self.assertEqual(kept, [0, 1, 2, 3, 4])
        self.assertLess(threshold, 1.50)
        self.assertGreater(threshold, 0.22)

    def test_keeps_minimum_views_when_all_exceed_threshold(self):
        kept, rejected, _ = select_inlier_indices(
            [5.0, 4.0, 6.0, 3.5],
            max_error_px=1.0,
            outlier_sigma=None,
            min_keep=3,
        )
        self.assertEqual(kept, [0, 1, 3])
        self.assertEqual(rejected, [2])

    def test_recalibrates_after_dropping_perturbed_view(self):
        config = _config({
            "board_cols": 4,
            "board_rows": 3,
            "square_size_mm": 25.0,
            "min_samples": 3,
            "max_view_error_px": 1.0,
            "outlier_sigma": 2.5,
            "outlier_max_rounds": 5,
            "output": "camera.yaml",
        })
        object_template = make_object_points(config)
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5)
        poses = (
            (np.array([0.00, 0.00, 0.00]), np.array([0.0, 0.0, 500.0])),
            (np.array([0.25, 0.00, 0.05]), np.array([-20.0, 10.0, 480.0])),
            (np.array([0.00, 0.20, -0.04]), np.array([15.0, -12.0, 520.0])),
            (np.array([-0.18, 0.12, 0.03]), np.array([-10.0, 8.0, 460.0])),
            (np.array([0.12, -0.22, 0.02]), np.array([8.0, 16.0, 540.0])),
            (np.array([-0.10, -0.15, 0.06]), np.array([12.0, -6.0, 490.0])),
        )
        object_points = []
        image_points = []
        outlier_index = 3
        for index, (rotation, translation) in enumerate(poses):
            projected, _ = cv2.projectPoints(
                object_template,
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            if index == outlier_index:
                rng = np.random.default_rng(0)
                projected = projected + rng.normal(0.0, 10.0, projected.shape)
            object_points.append(object_template.copy())
            image_points.append(np.asarray(projected, dtype=np.float32))

        fit = calibrate_with_outlier_rejection(
            object_points,
            image_points,
            (640, 480),
            config,
        )
        self.assertIn(outlier_index, fit.rejected_indices)
        self.assertNotIn(outlier_index, fit.kept_indices)
        self.assertGreater(fit.outlier_rounds, 0)
        self.assertLess(fit.calibration_rms, fit.initial_calibration_rms)
        self.assertLess(fit.reprojection_rms, 0.2)

    def test_disabled_rejection_keeps_all_views(self):
        config = _config({
            "board_cols": 4,
            "board_rows": 3,
            "square_size_mm": 25.0,
            "min_samples": 3,
            "max_view_error_px": None,
            "outlier_sigma": 0,
            "outlier_max_rounds": 5,
            "output": "camera.yaml",
        })
        object_template = make_object_points(config)
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5)
        object_points = []
        image_points = []
        for offset in (0.0, 0.15, -0.12):
            projected, _ = cv2.projectPoints(
                object_template,
                np.array([offset, 0.1, 0.0]),
                np.array([0.0, 0.0, 500.0]),
                camera_matrix,
                distortion,
            )
            object_points.append(object_template.copy())
            image_points.append(np.asarray(projected, dtype=np.float32))

        fit = calibrate_with_outlier_rejection(
            object_points,
            image_points,
            (640, 480),
            config,
        )
        self.assertEqual(fit.rejected_indices, ())
        self.assertEqual(fit.kept_indices, (0, 1, 2))
        self.assertEqual(fit.outlier_rounds, 0)


if __name__ == "__main__":
    unittest.main()
