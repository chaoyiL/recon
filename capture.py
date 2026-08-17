"""从相机实时读取图像帧并显示，固定曝光度。"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2

from utils.camera import open_camera
from utils.config import (
    ConfigError,
    load_config_sections,
    parse_camera_config,
    require_keys,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def capture_output_path(configured_path: Path) -> Path:
    """文件配置直接使用；目录配置生成带微秒时间戳且不覆盖的 PNG。"""
    if configured_path.suffix.lower() in IMAGE_SUFFIXES:
        return configured_path
    configured_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = configured_path / f"capture_{timestamp}.png"
    counter = 1
    while candidate.exists():
        candidate = configured_path / f"capture_{timestamp}_{counter:02d}.png"
        counter += 1
    return candidate


def video_output_path(configured_path: Path) -> Path:
    """文件配置直接使用；目录配置生成带微秒时间戳且不覆盖的无损 AVI。"""
    if configured_path.suffix.lower() in VIDEO_SUFFIXES:
        return configured_path
    configured_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = configured_path / f"capture_{timestamp}.avi"
    counter = 1
    while candidate.exists():
        candidate = configured_path / f"capture_{timestamp}_{counter:02d}.avi"
        counter += 1
    return candidate


def recording_fps(camera_fps: float, fallback: float = 30.0) -> float:
    """过滤部分相机后端返回的 0、NaN 等无效帧率。"""
    if math.isfinite(camera_fps) and 1.0 <= camera_fps <= 240.0:
        return float(camera_fps)
    return float(fallback)


def open_video_writer(
    output_path: Path,
    frame_size: tuple[int, int],
    fps: float,
) -> cv2.VideoWriter:
    """按文件扩展名选择编码器并创建录像写入器。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # FFV1 保留精确像素颜色，适合后续光场标定；显式选择 MP4 时使用 mp4v。
    codec = "FFV1" if output_path.suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"无法创建录像文件 {output_path}（编码器 {codec}，{fps:.2f} FPS）"
        )
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="从相机实时显示帧（固定曝光）")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        camera_section, capture_section = load_config_sections(
            config_path,
            "camera",
            "capture",
        )
        camera = parse_camera_config(camera_section)
        require_keys(capture_section, "capture", "save", "video_save")
        raw_save_path = capture_section["save"]
        if raw_save_path is not None and not isinstance(raw_save_path, str):
            raise ConfigError("capture.save 必须是字符串或 null")
        save_path = Path(raw_save_path).expanduser() if raw_save_path else None
        if save_path is not None and not save_path.is_absolute():
            save_path = config_path.parent/save_path
        raw_video_save_path = capture_section["video_save"]
        if not isinstance(raw_video_save_path, str) or not raw_video_save_path.strip():
            raise ConfigError("capture.video_save 必须是非空字符串")
        video_save_path = Path(raw_video_save_path).expanduser()
        if not video_save_path.is_absolute():
            video_save_path = config_path.parent/video_save_path
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(2)

    cap = open_camera(
        camera.device,
        camera.exposure,
        camera.white_balance_temperature,
        camera.width,
        camera.height,
    )

    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    actual_wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"相机已打开: device={camera.device}, 分辨率={w}x{h}")
    print(f"自动曝光={actual_auto}, 曝光值={actual_exposure}")
    print(f"自动白平衡={actual_auto_wb}, 白平衡色温={actual_wb} K")
    print("按 v 开始录像；按 s 停止录像；按 p 保存当前帧；按 q 或 Esc 退出")

    window = "camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    video_writer: cv2.VideoWriter | None = None
    active_video_path: Path | None = None
    recorded_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("读取帧失败", file=sys.stderr)
                break

            if video_writer is not None:
                video_writer.write(frame)
                recorded_frames += 1

            display_frame = frame
            if video_writer is not None:
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"REC  {recorded_frames}",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q 或 Esc
                break
            if key == ord("v"):
                if video_writer is not None:
                    print(f"正在录像: {active_video_path}")
                    continue
                height, width = frame.shape[:2]
                active_video_path = video_output_path(video_save_path)
                fps = recording_fps(float(cap.get(cv2.CAP_PROP_FPS)))
                try:
                    video_writer = open_video_writer(
                        active_video_path,
                        (width, height),
                        fps,
                    )
                except RuntimeError as error:
                    print(f"开始录像失败: {error}", file=sys.stderr)
                    active_video_path = None
                    continue
                recorded_frames = 0
                print(
                    f"开始录像: {active_video_path} "
                    f"({width}x{height}, {fps:.2f} FPS)"
                )
            elif key == ord("s"):
                if video_writer is None:
                    print("当前没有正在进行的录像。按 v 开始录像。")
                    continue
                video_writer.release()
                video_writer = None
                print(f"录像已保存: {active_video_path}（{recorded_frames} 帧）")
                active_video_path = None
                recorded_frames = 0
            elif key == ord("p") and save_path:
                output_path = capture_output_path(save_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(output_path), frame):
                    print(f"已保存到 {output_path}")
                else:
                    print(f"保存失败: {output_path}", file=sys.stderr)
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"录像已保存: {active_video_path}（{recorded_frames} 帧）")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
