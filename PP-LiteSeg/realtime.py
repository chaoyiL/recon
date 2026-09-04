#!/usr/bin/env python3
"""从主 config.yaml 指定相机实时运行 PP-LiteSeg-T1 二值分割。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import yaml

from common import (
    DEFAULT_SETTINGS_PATH,
    REPO_ROOT,
    SettingsError,
    atomic_write_image,
    load_settings,
    parse_mask_refine_lightweight,
    require_mapping,
    resolve_config_path,
)
from inference import PaddleSegPredictor


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.surface_mask import MaskRefineConfig,refine_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练后的 PP-LiteSeg 实时提取 config 指定相机的 surface mask")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--model-dir", help="覆盖 realtime.model_dir")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--tensorrt", action="store_true", help="强制启用 TensorRT")
    parser.add_argument("--no-tensorrt", action="store_true", help="强制禁用 TensorRT")
    parser.add_argument("--video", help="用录像代替相机，便于离线测试")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-dir", help="按 S 保存截图的目录")
    return parser.parse_args()


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SettingsError(f"{name} 必须是大于等于 {minimum} 的整数")
    return value


def alpha_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError("realtime.overlay_alpha 必须是数字")
    result = float(value)
    if not 0 <= result <= 1:
        raise SettingsError("realtime.overlay_alpha 必须在 0..1")
    return result


def render_realtime_panel(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    fps: float,
    inference_ms: float,
    total_ms: float,
) -> np.ndarray:
    overlay = frame.copy()
    color = np.zeros_like(frame)
    color[..., 1] = 255
    overlay[mask] = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    text = f"FPS {fps:5.1f} | infer {inference_ms:5.1f} ms | total {total_ms:5.1f} ms"
    cv2.putText(
        overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 255, 255), 2, cv2.LINE_AA)
    mask_bgr = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return np.hstack((overlay, mask_bgr))


def refine_prediction_mask(
    mask: np.ndarray,
    config: MaskRefineConfig,
) -> np.ndarray:
    """使独立预览显示的 mask 与 SAM2 监督和曲面重建完全一致。"""
    return np.ascontiguousarray(refine_mask(mask, config), dtype=np.bool_)


def open_input(args: argparse.Namespace, project_config: Path) -> cv2.VideoCapture:
    if args.video:
        capture = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开测试视频: {args.video}")
        return capture
    from utils.camera import open_camera
    from utils.config import load_config_sections, parse_camera_config

    (camera_section,) = load_config_sections(project_config, "camera")
    camera = parse_camera_config(camera_section)
    return open_camera(
        camera.device,
        camera.exposure,
        camera.white_balance_temperature,
        camera.width,
        camera.height,
    )


def main() -> None:
    args = parse_args()
    try:
        if args.tensorrt and args.no_tensorrt:
            raise SettingsError("--tensorrt 和 --no-tensorrt 不能同时使用")
        if args.max_frames < 0:
            raise SettingsError("--max-frames 必须是非负整数")
        settings_path, settings = load_settings(args.settings)
        base = settings_path.parent
        realtime = require_mapping(settings.get("realtime"), "realtime")
        project_config = resolve_config_path(
            settings.get("project_config"), base=base, name="project_config")
        try:
            project_root = require_mapping(
                yaml.safe_load(project_config.read_text(encoding="utf-8")),
                "主配置根节点")
        except FileNotFoundError as error:
            raise SettingsError(f"主配置不存在: {project_config}") from error
        except yaml.YAMLError as error:
            raise SettingsError(f"主配置格式错误: {error}") from error
        surface = require_mapping(project_root.get("get_surface"), "get_surface")
        mask_refine = parse_mask_refine_lightweight(surface.get("mask_refine"))
        model_dir = resolve_config_path(
            args.model_dir or realtime.get("model_dir"),
            base=Path.cwd() if args.model_dir else base,
            name="realtime.model_dir",
        )
        configured_trt = realtime.get("use_tensorrt", False)
        if not isinstance(configured_trt, bool):
            raise SettingsError("realtime.use_tensorrt 必须是 bool")
        use_tensorrt = args.tensorrt or (configured_trt and not args.no_tensorrt)
        precision = realtime.get("precision", "fp16")
        if precision not in {"fp16", "fp32"}:
            raise SettingsError("realtime.precision 必须是 fp16 或 fp32")
        foreground_class = integer(
            realtime.get("foreground_class", 1), "realtime.foreground_class")
        cpu_threads = integer(realtime.get("cpu_threads", 4), "realtime.cpu_threads", minimum=1)
        overlay_alpha = alpha_value(realtime.get("overlay_alpha", 0.45))
        if args.save_dir:
            save_dir = Path(args.save_dir).expanduser().resolve()
        else:
            save_dir = resolve_config_path(
                realtime.get("snapshot_dir", "realtime_snapshots"),
                base=base,
                name="realtime.snapshot_dir",
            )

        predictor = PaddleSegPredictor(
            model_dir,
            device=args.device,
            use_tensorrt=use_tensorrt,
            precision=str(precision),
            cpu_threads=cpu_threads,
            foreground_class=foreground_class,
        )
        capture = open_input(args, project_config)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"实时输入已打开: {width}x{height}, model={model_dir}, "
            f"device={args.device}, TensorRT={use_tensorrt}")
        print(
            "mask 后处理: "
            f"enabled={mask_refine.enabled}, "
            "最大外轮廓、孔洞填充（不修正圆角）")
        print("按 Q/ESC 退出，按 S 保存原图、叠加图和 mask")

        frame_count = 0
        smoothed_fps = 0.0
        last_frame_time = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if args.video:
                        break
                    raise RuntimeError("读取相机帧失败")
                prediction = predictor.predict(frame)
                refine_started = time.perf_counter()
                mask = refine_prediction_mask(prediction.mask, mask_refine)
                refine_ms = (time.perf_counter() - refine_started) * 1000.0
                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-9)
                last_frame_time = now
                smoothed_fps = (
                    instantaneous_fps if smoothed_fps == 0
                    else 0.9 * smoothed_fps + 0.1 * instantaneous_fps)
                panel = render_realtime_panel(
                    frame,
                    mask,
                    alpha=overlay_alpha,
                    fps=smoothed_fps,
                    inference_ms=prediction.inference_ms,
                    total_ms=prediction.total_ms + refine_ms,
                )
                frame_count += 1
                key = -1
                if not args.no_display:
                    cv2.imshow("PP-LiteSeg realtime | overlay + mask", panel)
                    key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    atomic_write_image(save_dir / f"{timestamp}_frame.png", frame)
                    atomic_write_image(save_dir / f"{timestamp}_panel.jpg", panel)
                    atomic_write_image(
                        save_dir / f"{timestamp}_mask.png",
                        mask.astype(np.uint8) * 255,
                    )
                    print(f"已保存截图: {save_dir / timestamp}")
                if args.max_frames and frame_count >= args.max_frames:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
        print(f"结束: frames={frame_count}, smoothed_fps={smoothed_fps:.2f}")
    except (SettingsError, RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
