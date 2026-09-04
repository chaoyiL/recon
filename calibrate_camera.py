"""使用棋盘格或黑底白圆点标定板完成单目相机标定。"""

from __future__ import annotations

import argparse
import functools
import math
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from utils.camera import open_camera
from utils.config import ConfigError, load_config_sections, parse_camera_config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BOARD_TYPES = ("chessboard", "circles")
CIRCLE_PATTERNS = ("symmetric", "asymmetric")
MIN_CALIBRATION_VIEWS = 3
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class CalibrationConfig:
    """标定板及标定输出配置。"""

    board_type: str
    circle_pattern: str
    board_cols: int
    board_rows: int
    square_size_mm: float
    min_samples: int
    max_view_error_px: float | None
    outlier_sigma: float | None
    outlier_max_rounds: int
    output: Path

    @property
    def board_size(self) -> tuple[int, int]:
        return self.board_cols, self.board_rows

    @property
    def uses_circles(self) -> bool:
        return self.board_type == "circles"

    @property
    def rejects_outliers(self) -> bool:
        has_absolute = self.max_view_error_px is not None
        has_relative = self.outlier_sigma is not None and self.outlier_sigma > 0
        return self.outlier_max_rounds > 0 and (has_absolute or has_relative)


@dataclass(frozen=True)
class CalibrationFit:
    """一次完整标定（含可选的异常帧剔除）结果。"""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    calibration_rms: float
    per_view_errors: list[float]
    reprojection_rms: float
    kept_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    rejected_errors_px: tuple[float, ...]
    outlier_rounds: int
    initial_sample_count: int
    initial_calibration_rms: float
    initial_reprojection_rms: float
    outlier_threshold_px: float | None


def parse_calibration_config(
    section: Mapping[str, Any],
    config_path: Path,
) -> CalibrationConfig:
    known = {
        "board_type",
        "circle_pattern",
        "board_cols",
        "board_rows",
        "square_size_mm",
        "min_samples",
        "max_view_error_px",
        "outlier_sigma",
        "outlier_max_rounds",
        "output",
    }
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"calibration 包含未知字段: {sorted(unknown)}")

    board_type = section.get("board_type", "chessboard")
    circle_pattern = section.get("circle_pattern", "symmetric")
    board_cols = section.get("board_cols", 9)
    board_rows = section.get("board_rows", 6)
    square_size_mm = section.get("square_size_mm", 25.0)
    min_samples = section.get("min_samples", 12)
    max_view_error_px = section.get("max_view_error_px")
    outlier_sigma = section.get("outlier_sigma", 2.5)
    outlier_max_rounds = section.get("outlier_max_rounds", 5)
    output = section.get("output", "camera_calibration.yaml")

    if board_type not in BOARD_TYPES:
        raise ConfigError(
            "calibration.board_type 必须是 chessboard 或 circles"
        )
    if circle_pattern not in CIRCLE_PATTERNS:
        raise ConfigError(
            "calibration.circle_pattern 必须是 symmetric 或 asymmetric"
        )
    for name, value in (
        ("board_cols", board_cols),
        ("board_rows", board_rows),
        ("min_samples", min_samples),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"calibration.{name} 必须是正整数")
    if board_cols < 2 or board_rows < 2:
        raise ConfigError("标定板横向和纵向特征点数都必须至少为 2")
    if min_samples < 3:
        raise ConfigError("calibration.min_samples 必须至少为 3")
    if (
        not isinstance(square_size_mm, Real)
        or isinstance(square_size_mm, bool)
        or float(square_size_mm) <= 0
    ):
        raise ConfigError("calibration.square_size_mm 必须是正数")
    if not isinstance(output, str) or not output.strip():
        raise ConfigError("calibration.output 必须是非空字符串")
    if max_view_error_px is not None:
        if (
            not isinstance(max_view_error_px, Real)
            or isinstance(max_view_error_px, bool)
            or float(max_view_error_px) <= 0
        ):
            raise ConfigError("calibration.max_view_error_px 必须是正数或 null")
        max_view_error_px = float(max_view_error_px)
    if outlier_sigma is not None:
        if (
            not isinstance(outlier_sigma, Real)
            or isinstance(outlier_sigma, bool)
            or float(outlier_sigma) < 0
        ):
            raise ConfigError("calibration.outlier_sigma 必须是非负数或 null")
        outlier_sigma = float(outlier_sigma)
        if outlier_sigma == 0:
            outlier_sigma = None
    if (
        not isinstance(outlier_max_rounds, int)
        or isinstance(outlier_max_rounds, bool)
        or outlier_max_rounds < 0
    ):
        raise ConfigError("calibration.outlier_max_rounds 必须是非负整数")

    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = config_path.parent / output_path

    return CalibrationConfig(
        board_type=board_type,
        circle_pattern=circle_pattern,
        board_cols=board_cols,
        board_rows=board_rows,
        square_size_mm=float(square_size_mm),
        min_samples=min_samples,
        max_view_error_px=max_view_error_px,
        outlier_sigma=outlier_sigma,
        outlier_max_rounds=outlier_max_rounds,
        output=output_path,
    )


def make_object_points(config: CalibrationConfig) -> np.ndarray:
    """生成标定板特征点在板坐标系中的三维坐标，单位为毫米。"""
    point_count = config.board_cols * config.board_rows
    points = np.zeros((point_count, 3), np.float32)
    if config.uses_circles and config.circle_pattern == "asymmetric":
        row_index, col_index = np.indices((config.board_rows, config.board_cols))
        points[:, 0] = (
            (2 * col_index + row_index % 2) * config.square_size_mm
        ).reshape(-1)
        points[:, 1] = (row_index * config.square_size_mm).reshape(-1)
        return points

    points[:, :2] = np.mgrid[
        0 : config.board_cols,
        0 : config.board_rows,
    ].T.reshape(-1, 2)
    points[:, :2] *= config.square_size_mm
    return points


@functools.lru_cache(maxsize=1)
def _white_dot_blob_detector() -> cv2.SimpleBlobDetector:
    """检测反色后的黑圆点（对应原图中的白圆点）。"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10
    params.minRepeatability = 2
    params.minDistBetweenBlobs = 5
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 12
    params.maxArea = 1e5
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)


def find_corners(
    gray: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """查找棋盘格角点，并进行亚像素精化。"""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def find_circles(
    gray: np.ndarray,
    board_size: tuple[int, int],
    asymmetric: bool,
) -> tuple[bool, np.ndarray | None]:
    """查找黑底白圆点阵列的圆心。"""
    inverted = cv2.bitwise_not(gray)
    base_flags = (
        cv2.CALIB_CB_ASYMMETRIC_GRID
        if asymmetric
        else cv2.CALIB_CB_SYMMETRIC_GRID
    )
    detector = _white_dot_blob_detector()
    attempts = (base_flags, base_flags | cv2.CALIB_CB_CLUSTERING)
    for flags in attempts:
        found, centers = cv2.findCirclesGrid(
            inverted,
            board_size,
            flags=flags,
            blobDetector=detector,
        )
        if found and centers is not None:
            return True, np.asarray(centers, dtype=np.float32)
    return False, None


def find_board_points(
    gray: np.ndarray,
    config: CalibrationConfig,
) -> tuple[bool, np.ndarray | None]:
    """按配置的标定板类型查找图像特征点。"""
    if config.uses_circles:
        return find_circles(
            gray,
            config.board_size,
            config.circle_pattern == "asymmetric",
        )
    return find_corners(gray, config.board_size)


def calculate_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[list[float], float]:
    """计算每张图和全部角点的 RMS 重投影误差（像素）。"""
    per_view: list[float] = []
    total_squared_error = 0.0
    total_points = 0

    for object_set, image_set, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            object_set,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )
        squared_error = float(
            cv2.norm(
                image_set.reshape(-1, 2),
                projected.reshape(-1, 2),
                cv2.NORM_L2SQR,
            )
        )
        point_count = len(object_set)
        per_view.append(float(np.sqrt(squared_error / point_count)))
        total_squared_error += squared_error
        total_points += point_count

    overall = float(np.sqrt(total_squared_error / total_points))
    return per_view, overall


def compute_outlier_threshold(
    errors: Sequence[float],
    max_error_px: float | None,
    outlier_sigma: float | None,
) -> float:
    """计算本轮剔除阈值：绝对上限与 median + σ×1.4826×MAD 中的较严者。"""
    threshold = math.inf
    if max_error_px is not None:
        threshold = float(max_error_px)
    if outlier_sigma is not None and outlier_sigma > 0 and len(errors) >= MIN_CALIBRATION_VIEWS:
        values = np.asarray(errors, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 0:
            statistical = median + float(outlier_sigma) * MAD_TO_SIGMA * mad
            threshold = min(threshold, statistical)
    return threshold


def select_inlier_indices(
    errors: Sequence[float],
    *,
    max_error_px: float | None,
    outlier_sigma: float | None,
    min_keep: int = MIN_CALIBRATION_VIEWS,
) -> tuple[list[int], list[int], float]:
    """按阈值划分保留/剔除下标；剔除后至少保留 min_keep 帧。"""
    count = len(errors)
    if count == 0:
        return [], [], math.inf
    min_keep = min(count, max(MIN_CALIBRATION_VIEWS, min_keep))
    threshold = compute_outlier_threshold(errors, max_error_px, outlier_sigma)
    if not math.isfinite(threshold):
        return list(range(count)), [], threshold

    outliers = [index for index, error in enumerate(errors) if error > threshold]
    if not outliers:
        return list(range(count)), [], threshold

    inliers = [index for index in range(count) if index not in set(outliers)]
    if len(inliers) >= min_keep:
        return inliers, outliers, threshold

    ranked = sorted(range(count), key=lambda index: (errors[index], index))
    kept = ranked[:min_keep]
    kept_set = set(kept)
    rejected = [
        index for index in ranked[min_keep:]
        if errors[index] > threshold and index not in kept_set
    ]
    return sorted(kept), rejected, threshold


def save_calibration(
    output_path: Path,
    config: CalibrationConfig,
    image_size: tuple[int, int],
    fit: CalibrationFit,
) -> None:
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "board_type": config.board_type,
        "circle_pattern": config.circle_pattern if config.uses_circles else None,
        "board_cols": config.board_cols,
        "board_rows": config.board_rows,
        "square_size_mm": config.square_size_mm,
        "sample_count": len(fit.per_view_errors),
        "initial_sample_count": fit.initial_sample_count,
        "rejected_sample_count": len(fit.rejected_indices),
        "kept_view_indices": list(fit.kept_indices),
        "rejected_view_indices": list(fit.rejected_indices),
        "rejected_view_errors_px": list(fit.rejected_errors_px),
        "outlier_rounds": fit.outlier_rounds,
        "outlier_threshold_px": fit.outlier_threshold_px,
        "initial_calibration_rms": fit.initial_calibration_rms,
        "initial_reprojection_rms_px": fit.initial_reprojection_rms,
        "calibration_rms": fit.calibration_rms,
        "reprojection_rms_px": fit.reprojection_rms,
        "per_view_errors_px": fit.per_view_errors,
        "camera_matrix": fit.camera_matrix.tolist(),
        "distortion_coefficients": fit.distortion.reshape(-1).tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(data, output_file, allow_unicode=True, sort_keys=False)
    temporary_path.replace(output_path)


def calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, list[float], float]:
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_view_errors, reprojection_rms = calculate_reprojection_errors(
        object_points,
        image_points,
        rotations,
        translations,
        camera_matrix,
        distortion,
    )
    return (
        camera_matrix,
        distortion,
        float(rms),
        per_view_errors,
        reprojection_rms,
    )


def calibrate_with_outlier_rejection(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    config: CalibrationConfig,
) -> CalibrationFit:
    """标定后按重投影误差剔除异常帧并重标定，直到收敛或达到轮数上限。"""
    if len(object_points) != len(image_points):
        raise ValueError("object_points 与 image_points 数量必须一致")
    initial_count = len(image_points)
    active = list(range(initial_count))
    rejected_indices: list[int] = []
    rejected_errors: list[float] = []
    last_threshold: float | None = None
    rounds = 0
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    calibration_rms = 0.0
    per_view_errors: list[float] = []
    reprojection_rms = 0.0
    initial_calibration_rms = 0.0
    initial_reprojection_rms = 0.0

    while True:
        current_objects = [object_points[index] for index in active]
        current_images = [image_points[index] for index in active]
        (
            camera_matrix,
            distortion,
            calibration_rms,
            per_view_errors,
            reprojection_rms,
        ) = calibrate(current_objects, current_images, image_size)
        if rounds == 0:
            initial_calibration_rms = calibration_rms
            initial_reprojection_rms = reprojection_rms
        if not config.rejects_outliers or rounds >= config.outlier_max_rounds:
            break

        kept_local, rejected_local, threshold = select_inlier_indices(
            per_view_errors,
            max_error_px=config.max_view_error_px,
            outlier_sigma=config.outlier_sigma,
        )
        last_threshold = threshold if math.isfinite(threshold) else None
        if not rejected_local:
            break

        rounds += 1
        for local_index in rejected_local:
            rejected_indices.append(active[local_index])
            rejected_errors.append(per_view_errors[local_index])
        active = [active[local_index] for local_index in kept_local]

    assert camera_matrix is not None and distortion is not None
    return CalibrationFit(
        camera_matrix=camera_matrix,
        distortion=distortion,
        calibration_rms=calibration_rms,
        per_view_errors=per_view_errors,
        reprojection_rms=reprojection_rms,
        kept_indices=tuple(active),
        rejected_indices=tuple(rejected_indices),
        rejected_errors_px=tuple(rejected_errors),
        outlier_rounds=rounds,
        initial_sample_count=initial_count,
        initial_calibration_rms=initial_calibration_rms,
        initial_reprojection_rms=initial_reprojection_rms,
        outlier_threshold_px=last_threshold,
    )


def format_rejection_summary(fit: CalibrationFit) -> str:
    initial = (
        f"初始 {fit.initial_sample_count} 帧，"
        f"OpenCV RMS {fit.initial_calibration_rms:.4f} px，"
        f"重投影 RMS {fit.initial_reprojection_rms:.4f} px"
    )
    if not fit.rejected_indices:
        return initial + "，全部保留"
    details = ", ".join(
        f"第 {index + 1} 张 ({error:.4f} px)"
        for index, error in zip(fit.rejected_indices, fit.rejected_errors_px)
    )
    threshold = (
        f"{fit.outlier_threshold_px:.4f} px"
        if fit.outlier_threshold_px is not None
        else "无"
    )
    return (
        f"{initial}；剔除 {len(fit.rejected_indices)} 帧（{details}），"
        f"剩余 {len(fit.kept_indices)} 帧，共 {fit.outlier_rounds} 轮，"
        f"末轮阈值 {threshold}"
    )


def _outlier_policy_text(config: CalibrationConfig) -> str:
    if not config.rejects_outliers:
        return "异常帧剔除: 关闭"
    parts: list[str] = []
    if config.outlier_sigma is not None:
        parts.append(f"σ={config.outlier_sigma:g}")
    if config.max_view_error_px is not None:
        parts.append(f"绝对上限 {config.max_view_error_px:g} px")
    parts.append(f"最多 {config.outlier_max_rounds} 轮")
    return "异常帧剔除: 开启（" + "，".join(parts) + "）"


def _board_status_label(config: CalibrationConfig) -> str:
    if config.uses_circles:
        pattern = (
            "ASYM CIRCLES" if config.circle_pattern == "asymmetric" else "CIRCLES"
        )
        return pattern
    return "CHESSBOARD"


def draw_status(
    frame: np.ndarray,
    found: bool,
    sample_count: int,
    min_samples: int,
    undistorting: bool,
    config: CalibrationConfig,
) -> None:
    color = (0, 220, 0) if found else (0, 0, 255)
    lines = [
        f"{_board_status_label(config)}: {'FOUND' if found else 'NOT FOUND'}",
        f"Samples: {sample_count}/{min_samples}",
        "SPACE: capture   ENTER: calibrate   Q: quit",
    ]
    if undistorting:
        lines.append("Undistortion preview: ON (press U to disable)")
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (15, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color if index == 0 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="棋盘格或黑底白圆点标定板单目相机标定"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        camera_section, calibration_section = load_config_sections(
            config_path,
            "camera",
            "calibration",
        )
        camera = parse_camera_config(camera_section)
        calibration_config = parse_calibration_config(
            calibration_section,
            config_path,
        )
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(2)

    try:
        cap = open_camera(
            camera.device,
            camera.exposure,
            camera.white_balance_temperature,
            camera.width,
            camera.height,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    object_template = make_object_points(calibration_config)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    undistorting = False
    image_size: tuple[int, int] | None = None
    window = "camera calibration"

    if calibration_config.uses_circles:
        pattern_name = (
            "交错" if calibration_config.circle_pattern == "asymmetric" else "矩形"
        )
        print(
            f"黑底白圆点: {calibration_config.board_cols} x "
            f"{calibration_config.board_rows}，圆心间距: "
            f"{calibration_config.square_size_mm:g} mm，阵列: {pattern_name}"
        )
        print("将圆点标定板放在画面中不同位置和角度，圆心被识别后按空格采集。")
    else:
        print(
            f"棋盘格内角点: {calibration_config.board_cols} x "
            f"{calibration_config.board_rows}，方格边长: "
            f"{calibration_config.square_size_mm:g} mm"
        )
        print("将棋盘放在画面中不同位置和角度，角点被识别后按空格采集。")
    print("按 Enter 开始标定，按 u 切换去畸变预览，按 q 或 Esc 退出。")
    print(_outlier_policy_text(calibration_config))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取相机帧失败", file=sys.stderr)
                break

            current_size = (frame.shape[1], frame.shape[0])
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                print("相机分辨率在采集过程中发生变化，无法继续标定", file=sys.stderr)
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_board_points(gray, calibration_config)
            display = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(
                    display,
                    calibration_config.board_size,
                    corners,
                    found,
                )

            if undistorting and camera_matrix is not None and distortion is not None:
                display = cv2.undistort(display, camera_matrix, distortion)
            draw_status(
                display,
                found,
                len(image_points),
                calibration_config.min_samples,
                undistorting,
                calibration_config,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not found or corners is None:
                    print("未检测到完整标定板，本帧未采集")
                    continue
                object_points.append(object_template.copy())
                image_points.append(corners.copy())
                print(f"已采集第 {len(image_points)} 张")
                continue
            if key in (10, 13):
                if len(image_points) < MIN_CALIBRATION_VIEWS:
                    print(
                        f"样本不足：当前 {len(image_points)} 张，"
                        f"至少需要 {MIN_CALIBRATION_VIEWS} 张"
                    )
                    continue
                if (
                    camera_matrix is None
                    and len(image_points) < calibration_config.min_samples
                ):
                    print(
                        f"样本不足：当前 {len(image_points)} 张，至少需要 "
                        f"{calibration_config.min_samples} 张"
                    )
                    continue
                assert image_size is not None
                try:
                    fit = calibrate_with_outlier_rejection(
                        object_points,
                        image_points,
                        image_size,
                        calibration_config,
                    )
                    save_calibration(
                        calibration_config.output,
                        calibration_config,
                        image_size,
                        fit,
                    )
                except cv2.error as error:
                    print(f"OpenCV 标定失败: {error}", file=sys.stderr)
                    continue

                if fit.rejected_indices:
                    kept_objects = [object_points[index] for index in fit.kept_indices]
                    kept_images = [image_points[index] for index in fit.kept_indices]
                    object_points.clear()
                    image_points.clear()
                    object_points.extend(kept_objects)
                    image_points.extend(kept_images)

                camera_matrix = fit.camera_matrix
                distortion = fit.distortion
                print(format_rejection_summary(fit))
                print(f"标定完成，OpenCV RMS: {fit.calibration_rms:.4f} px")
                print(f"重投影 RMS: {fit.reprojection_rms:.4f} px")
                remaining = ", ".join(
                    f"{error:.4f}" for error in fit.per_view_errors
                )
                print(f"保留帧重投影误差 (px): {remaining}")
                print(
                    f"fx={camera_matrix[0, 0]:.6f}, fy={camera_matrix[1, 1]:.6f}"
                )
                print(
                    f"cx={camera_matrix[0, 2]:.6f}, cy={camera_matrix[1, 2]:.6f}"
                )
                print(f"畸变系数: {distortion.reshape(-1).tolist()}")
                print(f"结果已保存到: {calibration_config.output}")
                undistorting = True
                continue
            if key == ord("u"):
                if camera_matrix is None or distortion is None:
                    print("请先完成标定")
                else:
                    undistorting = not undistorting
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
