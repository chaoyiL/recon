"""使用普通黑白棋盘格完成单目相机标定。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from utils.camera import open_camera
from utils.config import ConfigError, load_config_sections, parse_camera_config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class CalibrationConfig:
    """棋盘格及标定输出配置。"""

    board_cols: int
    board_rows: int
    square_size_mm: float
    min_samples: int
    output: Path

    @property
    def board_size(self) -> tuple[int, int]:
        return self.board_cols, self.board_rows


def parse_calibration_config(
    section: Mapping[str, Any],
    config_path: Path,
) -> CalibrationConfig:
    known = {
        "board_cols",
        "board_rows",
        "square_size_mm",
        "min_samples",
        "output",
    }
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"calibration 包含未知字段: {sorted(unknown)}")

    board_cols = section.get("board_cols", 9)
    board_rows = section.get("board_rows", 6)
    square_size_mm = section.get("square_size_mm", 25.0)
    min_samples = section.get("min_samples", 12)
    output = section.get("output", "camera_calibration.yaml")

    for name, value in (
        ("board_cols", board_cols),
        ("board_rows", board_rows),
        ("min_samples", min_samples),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"calibration.{name} 必须是正整数")
    if board_cols < 2 or board_rows < 2:
        raise ConfigError("棋盘格横向和纵向内角点数都必须至少为 2")
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

    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = config_path.parent / output_path

    return CalibrationConfig(
        board_cols=board_cols,
        board_rows=board_rows,
        square_size_mm=float(square_size_mm),
        min_samples=min_samples,
        output=output_path,
    )


def make_object_points(config: CalibrationConfig) -> np.ndarray:
    """生成棋盘角点在棋盘坐标系中的三维坐标，单位为毫米。"""
    points = np.zeros((config.board_cols * config.board_rows, 3), np.float32)
    points[:, :2] = np.mgrid[
        0 : config.board_cols,
        0 : config.board_rows,
    ].T.reshape(-1, 2)
    points[:, :2] *= config.square_size_mm
    return points


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


def save_calibration(
    output_path: Path,
    config: CalibrationConfig,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    calibration_rms: float,
    per_view_errors: list[float],
    reprojection_rms: float,
) -> None:
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "board_cols": config.board_cols,
        "board_rows": config.board_rows,
        "square_size_mm": config.square_size_mm,
        "sample_count": len(per_view_errors),
        "calibration_rms": calibration_rms,
        "reprojection_rms_px": reprojection_rms,
        "per_view_errors_px": per_view_errors,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
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


def draw_status(
    frame: np.ndarray,
    found: bool,
    sample_count: int,
    min_samples: int,
    undistorting: bool,
) -> None:
    color = (0, 220, 0) if found else (0, 0, 255)
    lines = [
        f"Chessboard: {'FOUND' if found else 'NOT FOUND'}",
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
    parser = argparse.ArgumentParser(description="普通黑白棋盘格单目相机标定")
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

    print(
        f"棋盘格内角点: {calibration_config.board_cols} x "
        f"{calibration_config.board_rows}，方格边长: "
        f"{calibration_config.square_size_mm:g} mm"
    )
    print("将棋盘放在画面中不同位置和角度，角点被识别后按空格采集。")
    print("按 Enter 开始标定，按 u 切换去畸变预览，按 q 或 Esc 退出。")

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
            found, corners = find_corners(gray, calibration_config.board_size)
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
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not found or corners is None:
                    print("未检测到完整棋盘格，本帧未采集")
                    continue
                object_points.append(object_template.copy())
                image_points.append(corners.copy())
                print(f"已采集第 {len(image_points)} 张")
                continue
            if key in (10, 13):
                if len(image_points) < calibration_config.min_samples:
                    print(
                        f"样本不足：当前 {len(image_points)} 张，至少需要 "
                        f"{calibration_config.min_samples} 张"
                    )
                    continue
                assert image_size is not None
                try:
                    (
                        camera_matrix,
                        distortion,
                        calibration_rms,
                        per_view_errors,
                        reprojection_rms,
                    ) = calibrate(object_points, image_points, image_size)
                    save_calibration(
                        calibration_config.output,
                        calibration_config,
                        image_size,
                        camera_matrix,
                        distortion,
                        calibration_rms,
                        per_view_errors,
                        reprojection_rms,
                    )
                except cv2.error as error:
                    print(f"OpenCV 标定失败: {error}", file=sys.stderr)
                    continue

                print(f"标定完成，OpenCV RMS: {calibration_rms:.4f} px")
                print(f"重投影 RMS: {reprojection_rms:.4f} px")
                print(f"fx={camera_matrix[0, 0]:.6f}, fy={camera_matrix[1, 1]:.6f}")
                print(f"cx={camera_matrix[0, 2]:.6f}, cy={camera_matrix[1, 2]:.6f}")
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
