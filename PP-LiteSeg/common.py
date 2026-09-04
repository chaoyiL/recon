"""PP-LiteSeg 数据准备、划分和配置读取的共享函数。"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config.yaml"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm"}


class SettingsError(ValueError):
    """PP-LiteSeg 工具配置无效。"""


@dataclass(frozen=True)
class DatasetSample:
    """manifest.jsonl 中的一条监督样本记录。"""

    sample_id: str
    source_video: str
    source_frame: int
    timestamp_seconds: float
    image: str
    mask: str
    width: int
    height: int
    mask_area_ratio: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetSample":
        required = {
            "sample_id",
            "source_video",
            "source_frame",
            "timestamp_seconds",
            "image",
            "mask",
            "width",
            "height",
            "mask_area_ratio",
        }
        missing = required - set(value)
        if missing:
            raise SettingsError(f"manifest 样本缺少字段: {sorted(missing)}")
        return cls(
            sample_id=str(value["sample_id"]),
            source_video=str(value["source_video"]),
            source_frame=int(value["source_frame"]),
            timestamp_seconds=float(value["timestamp_seconds"]),
            image=str(value["image"]),
            mask=str(value["mask"]),
            width=int(value["width"]),
            height=int(value["height"]),
            mask_area_ratio=float(value["mask_area_ratio"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_video": self.source_video,
            "source_frame": self.source_frame,
            "timestamp_seconds": self.timestamp_seconds,
            "image": self.image,
            "mask": self.mask,
            "width": self.width,
            "height": self.height,
            "mask_area_ratio": self.mask_area_ratio,
        }


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SettingsError(f"{name} 必须是字典")
    return dict(value)


def parse_mask_refine_lightweight(raw: Any) -> Any:
    """读取主配置的公共 mask 后处理参数，不导入重建/JAX 模块。"""
    from utils.surface_mask import MaskRefineConfig

    if raw is None:
        return MaskRefineConfig()
    section = require_mapping(raw, "get_surface.mask_refine")
    known = {"enabled"}
    unknown = set(section) - known
    if unknown:
        raise SettingsError(f"get_surface.mask_refine 含未知字段: {sorted(unknown)}")
    enabled = section.get("enabled", MaskRefineConfig.enabled)
    if not isinstance(enabled, bool):
        raise SettingsError("mask_refine.enabled 必须是 bool")
    return MaskRefineConfig(enabled=enabled)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> tuple[Path, dict[str, Any]]:
    settings_path = Path(path).expanduser().resolve()
    try:
        with settings_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise SettingsError(f"配置文件不存在: {settings_path}") from error
    except yaml.YAMLError as error:
        raise SettingsError(f"配置文件格式错误: {error}") from error
    root = require_mapping(raw, "配置根节点")
    # 主 config.yaml 将 PP-LiteSeg 配置收在独立命名空间；仍兼容显式传入
    # 旧 PP-LiteSeg/config.yaml 的平铺结构。
    if "pp_liteseg" in root:
        return settings_path, require_mapping(root["pp_liteseg"], "pp_liteseg")
    return settings_path, root


def resolve_config_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"{name} 必须是非空路径字符串")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{name} 必须是数字")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "非负" if allow_zero else "正"
        raise SettingsError(f"{name} 必须是有限{qualifier}数")
    return result


def fraction(value: Any, name: str, *, inclusive: bool = False) -> float:
    result = positive_number(value, name, allow_zero=inclusive)
    valid = 0 <= result <= 1 if inclusive else 0 < result < 1
    if not valid:
        bounds = "0..1" if inclusive else "0..1 之间"
        raise SettingsError(f"{name} 必须在 {bounds}")
    return result


def discover_videos(patterns: Sequence[str], *, base: Path) -> list[Path]:
    """展开文件、目录和 glob，返回去重后的有序视频列表。"""
    if not patterns:
        raise SettingsError("至少需要配置一个 labeling.videos 或 --video")
    discovered: dict[str, Path] = {}
    for raw_pattern in patterns:
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise SettingsError("视频路径必须是非空字符串")
        expanded = os.path.expanduser(raw_pattern)
        if not Path(expanded).is_absolute():
            expanded = str(base / expanded)
        candidates = [Path(value) for value in glob.glob(expanded, recursive=True)]
        if not candidates and not glob.has_magic(expanded):
            candidates = [Path(expanded)]
        for candidate in candidates:
            if candidate.is_dir():
                children = sorted(
                    child for child in candidate.iterdir()
                    if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES
                )
            else:
                children = [candidate]
            for child in children:
                if child.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                resolved = child.resolve()
                if not resolved.is_file():
                    raise SettingsError(f"视频文件不存在: {resolved}")
                discovered[str(resolved)] = resolved
    if not discovered:
        raise SettingsError("配置的视频路径没有匹配到支持的视频文件")
    return [discovered[key] for key in sorted(discovered)]


def read_manifest(dataset_dir: str | Path) -> list[DatasetSample]:
    path = Path(dataset_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    samples: list[DatasetSample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SettingsError(
                    f"{path}:{line_number} 不是合法 JSON: {error}") from error
            sample = DatasetSample.from_mapping(require_mapping(value, "manifest 样本"))
            if sample.sample_id in seen:
                raise SettingsError(f"manifest 含重复 sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            samples.append(sample)
    return samples


def append_manifest_sample(dataset_dir: Path, sample: DatasetSample) -> None:
    path = dataset_dir / "manifest.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        json.dump(sample.to_mapping(), stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"OpenCV 无法写入图像: {temporary}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def split_samples_by_video(
    samples: Sequence[DatasetSample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample], str]:
    """优先按视频分组划分；只有一个视频时退化为末段验证集。"""
    if len(samples) < 2:
        raise SettingsError("至少需要 2 条样本才能划分训练集和验证集")
    val_ratio = fraction(val_ratio, "training.val_ratio")
    grouped: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_video, []).append(sample)

    if len(grouped) == 1:
        ordered = sorted(samples, key=lambda item: (item.timestamp_seconds, item.source_frame))
        val_count = max(1, min(len(ordered) - 1, round(len(ordered) * val_ratio)))
        return ordered[:-val_count], ordered[-val_count:], "temporal_tail"

    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(samples) * val_ratio))
    val_groups: set[str] = set()
    val_count = 0
    for group in groups:
        if len(val_groups) >= len(groups) - 1:
            break
        current_error = abs(target - val_count)
        added_error = abs(target - (val_count + len(grouped[group])))
        if not val_groups or added_error <= current_error:
            val_groups.add(group)
            val_count += len(grouped[group])
    if not val_groups:
        val_groups.add(groups[0])
    train = [sample for sample in samples if sample.source_video not in val_groups]
    val = [sample for sample in samples if sample.source_video in val_groups]
    if not train or not val:
        raise SettingsError("按视频分组后训练集或验证集为空")
    return train, val, "source_video"


def write_paddleseg_split(
    dataset_dir: Path,
    train: Sequence[DatasetSample],
    val: Sequence[DatasetSample],
    *,
    mode: str,
    seed: int,
) -> None:
    def lines(values: Iterable[DatasetSample]) -> str:
        return "".join(f"{item.image} {item.mask}\n" for item in values)

    atomic_write_text(dataset_dir / "train.txt", lines(train))
    atomic_write_text(dataset_dir / "val.txt", lines(val))
    payload = {
        "mode": mode,
        "seed": seed,
        "train_count": len(train),
        "val_count": len(val),
        "train_videos": sorted({item.source_video for item in train}),
        "val_videos": sorted({item.source_video for item in val}),
        "train_sample_ids": [item.sample_id for item in train],
        "val_sample_ids": [item.sample_id for item in val],
    }
    atomic_write_text(
        dataset_dir / "split.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def validate_dataset_samples(
    dataset_dir: Path,
    samples: Sequence[DatasetSample],
) -> tuple[int, int]:
    """验证文件、尺寸和类别值；返回统一的 (height, width)。"""
    expected_shape: tuple[int, int] | None = None
    for sample in samples:
        image_path = dataset_dir / sample.image
        mask_path = dataset_dir / sample.mask
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SettingsError(f"无法读取训练图像: {image_path}")
        if mask is None:
            raise SettingsError(f"无法读取训练 mask: {mask_path}")
        if mask.ndim != 2:
            raise SettingsError(f"训练 mask 必须为单通道: {mask_path}")
        if image.shape[:2] != mask.shape:
            raise SettingsError(f"图像和 mask 尺寸不一致: {sample.sample_id}")
        values = set(int(value) for value in np.unique(mask))
        if not values <= {0, 1}:
            raise SettingsError(
                f"mask 只允许类别 0/1，{mask_path} 实际包含 {sorted(values)}")
        shape = image.shape[:2]
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise SettingsError(
                "当前固定形状训练要求所有样本尺寸一致："
                f"期望 {expected_shape[::-1]}，{sample.sample_id} 为 {shape[::-1]}")
    if expected_shape is None:
        raise SettingsError("数据集为空")
    return expected_shape


def render_mask_preview(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = np.asarray(frame).copy()
    binary = np.asarray(mask, dtype=bool)
    green = np.zeros_like(preview)
    green[..., 1] = 255
    preview[binary] = cv2.addWeighted(preview, 0.55, green, 0.45, 0)[binary]
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return preview
