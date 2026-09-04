#!/usr/bin/env python3
"""用仓库现有 SAM2 对视频关键帧生成 PP-LiteSeg 二分类监督数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from common import (
    DEFAULT_SETTINGS_PATH,
    REPO_ROOT,
    DatasetSample,
    SettingsError,
    append_manifest_sample,
    atomic_write_image,
    atomic_write_text,
    discover_videos,
    fraction,
    load_settings,
    parse_mask_refine_lightweight,
    positive_number,
    read_manifest,
    render_mask_preview,
    require_mapping,
    resolve_config_path,
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 config.yaml 中的 SAM2 点提示自动标注视频关键帧")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH)
    parser.add_argument(
        "--video",
        action="append",
        help="视频文件、目录或 glob；可重复，设置后覆盖 labeling.videos",
    )
    parser.add_argument("--dataset-dir", help="覆盖 labeling.dataset_dir")
    parser.add_argument("--sample-fps", type=float, help="覆盖 labeling.sample_fps")
    parser.add_argument("--max-samples-per-video", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 labeling.resume；续跑时仍顺序运行 SAM2，但跳过已有文件",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="不保存人工复核叠加图")
    parser.add_argument(
        "--no-compile", action="store_true", help="本次标注禁用 torch.compile")
    return parser.parse_args()


def load_project_surface_config(project_config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with project_config.open("r", encoding="utf-8") as stream:
            root = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise SettingsError(f"主配置不存在: {project_config}") from error
    except yaml.YAMLError as error:
        raise SettingsError(f"主配置格式错误: {error}") from error
    root_mapping = require_mapping(root, "主配置根节点")
    surface = require_mapping(root_mapping.get("get_surface"), "get_surface")
    prompts = require_mapping(surface.get("prompts"), "get_surface.prompts")
    return surface, prompts


def parse_prompt_labels(raw: Any, prompts: Mapping[str | int, Any]) -> list[str | int]:
    if raw is None:
        return list(prompts)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise SettingsError("labeling.prompt_labels 必须是非空列表或 null")
    result: list[str | int] = []
    for label in raw:
        if label not in prompts:
            raise SettingsError(
                f"labeling.prompt_labels 中的 {label!r} 不在 get_surface.prompts")
        result.append(label)
    return result


def parse_point_list(label: str | int, name: str, raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SettingsError(f"prompt {label!r}.{name} 必须是点列表")
    points: list[tuple[float, float]] = []
    for point in raw:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise SettingsError(f"prompt {label!r}.{name} 中的点必须是 [x, y]")
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError) as error:
            raise SettingsError(f"prompt {label!r}.{name} 坐标必须是数字") from error
    return points


def parse_prompts_lightweight(
    raw: Mapping[str | int, Any],
) -> dict[str | int, dict[str, list[tuple[float, float]]]]:
    result: dict[str | int, dict[str, list[tuple[float, float]]]] = {}
    for label, raw_group in raw.items():
        if not isinstance(label, (str, int)) or isinstance(label, bool):
            raise SettingsError("prompt label 必须是字符串或整数")
        group = require_mapping(raw_group, f"prompt {label!r}")
        unknown = set(group) - {"positive", "negative"}
        if unknown:
            raise SettingsError(f"prompt {label!r} 含未知字段: {sorted(unknown)}")
        positive = parse_point_list(label, "positive", group.get("positive", []))
        negative = parse_point_list(label, "negative", group.get("negative", []))
        if not positive:
            raise SettingsError(f"prompt {label!r} 至少需要一个 positive 点")
        result[label] = {"positive": positive, "negative": negative}
    if not result:
        raise SettingsError("get_surface.prompts 不能为空")
    return result


def video_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in path.stem)
    return f"{safe_stem}-{digest}"


def frame_sample_id(source_id: str, frame_index: int) -> str:
    return f"{source_id}-f{frame_index:08d}"


def selected_frame(frame_index: int, next_sample_position: float) -> bool:
    return frame_index + 1e-9 >= next_sample_position


def label_video(
    video_path: Path,
    *,
    dataset_dir: Path,
    segmenter: Any,
    prompts: Mapping[str | int, Any],
    prompt_labels: Sequence[str | int],
    sample_fps: float,
    minimum_area: float,
    maximum_area: float,
    save_previews: bool,
    existing_ids: set[str],
    resume: bool,
    max_samples: int,
) -> tuple[int, int]:
    source_id = video_id(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"视频帧率无效: {video_path}，fps={fps}")
    if sample_fps > fps:
        capture.release()
        raise SettingsError(
            f"sample_fps={sample_fps} 不能超过视频 {video_path.name} 的 {fps:.3f} FPS")

    interval = fps / sample_fps
    next_sample_position = 0.0
    frame_index = 0
    accepted = 0
    rejected = 0
    segmenter.reset()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # SAM2 的 video memory 必须逐帧推进，与教师预览保持相同的
            # 时序语义；sample_fps 只决定哪些结果写入训练集。
            results = segmenter.segment(frame, prompts)
            if not selected_frame(frame_index, next_sample_position):
                frame_index += 1
                continue
            next_sample_position += interval
            if max_samples > 0 and accepted + rejected >= max_samples:
                break

            sample_id = frame_sample_id(source_id, frame_index)
            selected_masks = [np.asarray(results[label], dtype=bool) for label in prompt_labels]
            mask = np.logical_or.reduce(selected_masks)
            area_ratio = float(mask.mean())
            if not minimum_area <= area_ratio <= maximum_area:
                rejected += 1
                print(
                    f"跳过 {video_path.name} frame={frame_index}: "
                    f"mask area={area_ratio:.3%} 不在 "
                    f"[{minimum_area:.1%}, {maximum_area:.1%}]")
                frame_index += 1
                continue

            relative_image = Path("images") / source_id / f"{sample_id}.png"
            relative_mask = Path("masks") / source_id / f"{sample_id}.png"
            relative_preview = Path("previews") / source_id / f"{sample_id}.jpg"
            if sample_id in existing_ids:
                if not resume:
                    raise SettingsError(
                        f"样本 {sample_id} 已存在；如需续跑请添加 --resume")
                print(f"已有，跳过写入: {sample_id}")
                frame_index += 1
                continue

            atomic_write_image(dataset_dir / relative_image, frame)
            atomic_write_image(dataset_dir / relative_mask, mask.astype(np.uint8))
            if save_previews:
                preview = render_mask_preview(frame, mask)
                cv2.putText(
                    preview,
                    f"{video_path.name}  frame={frame_index}  area={area_ratio:.1%}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                atomic_write_image(dataset_dir / relative_preview, preview)

            height, width = frame.shape[:2]
            sample = DatasetSample(
                sample_id=sample_id,
                source_video=str(video_path.resolve()),
                source_frame=frame_index,
                timestamp_seconds=frame_index / fps,
                image=relative_image.as_posix(),
                mask=relative_mask.as_posix(),
                width=width,
                height=height,
                mask_area_ratio=area_ratio,
            )
            append_manifest_sample(dataset_dir, sample)
            existing_ids.add(sample_id)
            accepted += 1
            elapsed = time.perf_counter() - started
            print(
                f"[{video_path.name}] accepted={accepted} rejected={rejected} "
                f"frame={frame_index} area={area_ratio:.1%} elapsed={elapsed:.1f}s")
            frame_index += 1
    finally:
        capture.release()
        segmenter.reset()
    return accepted, rejected


def main() -> None:
    args = parse_args()
    try:
        settings_path, settings = load_settings(args.settings)
        base = settings_path.parent
        labeling = require_mapping(settings.get("labeling"), "labeling")
        project_config = resolve_config_path(
            settings.get("project_config"), base=base, name="project_config")
        surface, prompts = load_project_surface_config(project_config)
        raw_videos = args.video if args.video else labeling.get("videos")
        if not isinstance(raw_videos, Sequence) or isinstance(raw_videos, (str, bytes)):
            raise SettingsError("labeling.videos 必须是路径列表")
        videos = discover_videos(list(raw_videos), base=base)
        if args.dataset_dir:
            dataset_dir = resolve_config_path(
                args.dataset_dir, base=Path.cwd(), name="--dataset-dir")
        else:
            dataset_dir = resolve_config_path(
                labeling.get("dataset_dir"), base=base, name="labeling.dataset_dir")
        sample_fps = positive_number(
            args.sample_fps if args.sample_fps is not None else labeling.get("sample_fps", 2.0),
            "labeling.sample_fps",
        )
        minimum_area = fraction(
            labeling.get("minimum_mask_area_ratio", 0.01),
            "labeling.minimum_mask_area_ratio",
        )
        maximum_area = fraction(
            labeling.get("maximum_mask_area_ratio", 0.99),
            "labeling.maximum_mask_area_ratio",
        )
        if minimum_area >= maximum_area:
            raise SettingsError("minimum_mask_area_ratio 必须小于 maximum_mask_area_ratio")
        save_previews = bool(labeling.get("save_previews", True)) and not args.no_preview
        prompt_labels = parse_prompt_labels(labeling.get("prompt_labels"), prompts)
        configured_resume = labeling.get("resume", False)
        if not isinstance(configured_resume, bool):
            raise SettingsError("labeling.resume 必须是 bool")
        resume = configured_resume if args.resume is None else args.resume
        if args.max_samples_per_video < 0:
            raise SettingsError("--max-samples-per-video 必须是非负整数")

        existing = read_manifest(dataset_dir)
        if existing and not resume:
            raise SettingsError(
                f"{dataset_dir} 已含 {len(existing)} 条样本；"
                "请换目录、在配置中设置 labeling.resume=true，或添加 --resume")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = {sample.sample_id for sample in existing}

        from utils.sam2_surface import SurfaceSegmenter

        parsed_prompts = parse_prompts_lightweight(prompts)
        parsed_mask_refine = parse_mask_refine_lightweight(surface.get("mask_refine"))
        segmentation = surface.get("segmentation", {})
        if not isinstance(segmentation, Mapping):
            raise SettingsError("get_surface.segmentation 必须是字典")
        sam2 = segmentation.get("sam2", {})
        if not isinstance(sam2, Mapping):
            raise SettingsError("get_surface.segmentation.sam2 必须是字典")
        model_id = sam2.get("model", surface.get("model"))
        if not isinstance(model_id, str) or not model_id:
            raise SettingsError("get_surface.model 必须是非空字符串")
        compile_model = sam2.get(
            "torch_compile", surface.get("torch_compile", False)) \
            and not args.no_compile
        memory_frames = sam2.get(
            "memory_frames", surface.get("sam_memory_frames"))
        if not isinstance(compile_model, bool):
            raise SettingsError("get_surface.torch_compile 必须是 bool")
        print(
            f"加载 SAM2: model={model_id}, compile={compile_model}, "
            f"memory_frames={memory_frames}")
        segmenter = SurfaceSegmenter(
            model_id=model_id,
            mask_refine=parsed_mask_refine,
            compile_model=compile_model,
            memory_frames=memory_frames,
        )

        total_accepted = 0
        total_rejected = 0
        for video in videos:
            accepted, rejected = label_video(
                video,
                dataset_dir=dataset_dir,
                segmenter=segmenter,
                prompts=parsed_prompts,
                prompt_labels=prompt_labels,
                sample_fps=sample_fps,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
                save_previews=save_previews,
                existing_ids=existing_ids,
                resume=resume,
                max_samples=args.max_samples_per_video,
            )
            total_accepted += accepted
            total_rejected += rejected

        all_samples = read_manifest(dataset_dir)
        metadata = {
            "format": "PaddleSeg Dataset binary class-index masks",
            "class_names": ["background", "surface"],
            "sample_count": len(all_samples),
            "source_video_count": len({sample.source_video for sample in all_samples}),
            "sample_fps": sample_fps,
            "sam2_model": model_id,
            "prompt_labels": list(prompt_labels),
            "project_config": str(project_config),
        }
        atomic_write_text(
            dataset_dir / "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(
            f"标注完成: 本次 accepted={total_accepted}, rejected={total_rejected}, "
            f"总样本={len(all_samples)}, 输出={dataset_dir}")
    except (SettingsError, ValueError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
