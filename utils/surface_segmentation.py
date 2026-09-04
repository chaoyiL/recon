"""PP-LiteSeg 曲面分割运行时配置和接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.config import ConfigError
from utils.surface_mask import MaskRefineConfig,Prompts,refine_mask


@dataclass(frozen=True)
class SurfaceSegmentationConfig:
    mode: str
    frame_interval: int
    liteseg_model_dir: Path | None = None
    liteseg_device: str = "gpu"
    liteseg_use_tensorrt: bool = False
    liteseg_precision: str = "fp16"
    liteseg_cpu_threads: int = 4
    liteseg_foreground_class: int = 1
    liteseg_label: str | int = "surface"

    @property
    def description(self) -> str:
        return f"PP-LiteSeg ({self.liteseg_model_dir})"


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "非负" if allow_zero else "正"
        raise ConfigError(f"{name} 必须是{qualifier}整数")
    return value


def parse_surface_segmentation_config(
    surface: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> SurfaceSegmentationConfig:
    """读取只允许 PP-LiteSeg 的重建运行时分割配置。"""
    raw = surface.get("segmentation")
    if raw is None:
        raise ConfigError(
            "get_surface.segmentation 必须配置为 PP-LiteSeg；"
            "SAM2 仅用于自动标注教师")
    if not isinstance(raw, Mapping):
        raise ConfigError("get_surface.segmentation 必须是字典")
    unknown = set(raw) - {"mode", "sam2", "liteseg"}
    if unknown:
        raise ConfigError(
            f"get_surface.segmentation 包含未知字段: {sorted(unknown)}")
    mode = raw.get("mode", "liteseg")
    if mode != "liteseg":
        raise ConfigError(
            "get_surface.segmentation.mode 只允许 liteseg；"
            "SAM2 仅用于自动标注教师")

    section = raw.get("liteseg", {})
    if not isinstance(section, Mapping):
        raise ConfigError("get_surface.segmentation.liteseg 必须是字典")
    unknown = set(section) - {
        "model_dir", "device", "use_tensorrt", "precision", "cpu_threads",
        "foreground_class", "label", "frame_interval"}
    if unknown:
        raise ConfigError(
            f"get_surface.segmentation.liteseg 包含未知字段: {sorted(unknown)}")
    raw_model_dir = section.get("model_dir")
    if not isinstance(raw_model_dir, str) or not raw_model_dir.strip():
        raise ConfigError(
            "get_surface.segmentation.liteseg.model_dir 必须是非空路径")
    model_dir = Path(raw_model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(config_path).expanduser().resolve().parent / model_dir
    device = section.get("device", "gpu")
    precision = section.get("precision", "fp16")
    use_tensorrt = section.get("use_tensorrt", False)
    label = section.get("label", "surface")
    foreground_class = _positive_integer(
        section.get("foreground_class", 1),
        "get_surface.segmentation.liteseg.foreground_class", allow_zero=True)
    if device not in {"cpu", "gpu"}:
        raise ConfigError("get_surface.segmentation.liteseg.device 必须是 cpu 或 gpu")
    if precision not in {"fp16", "fp32"}:
        raise ConfigError("get_surface.segmentation.liteseg.precision 必须是 fp16 或 fp32")
    if not isinstance(use_tensorrt, bool):
        raise ConfigError("get_surface.segmentation.liteseg.use_tensorrt 必须是布尔值")
    if not isinstance(label, (str, int)) or isinstance(label, bool):
        raise ConfigError("get_surface.segmentation.liteseg.label 必须是字符串或整数")
    return SurfaceSegmentationConfig(
        mode="liteseg",
        frame_interval=_positive_integer(
            section.get("frame_interval", 1),
            "get_surface.segmentation.liteseg.frame_interval"),
        liteseg_model_dir=model_dir.resolve(),
        liteseg_device=str(device),
        liteseg_use_tensorrt=use_tensorrt,
        liteseg_precision=str(precision),
        liteseg_cpu_threads=_positive_integer(
            section.get("cpu_threads", 4),
            "get_surface.segmentation.liteseg.cpu_threads"),
        liteseg_foreground_class=foreground_class,
        liteseg_label=label,
    )


class SurfaceSegmentationBackend:
    """运行 PP-LiteSeg 并返回 ``(labels, KxHxW masks)``。"""

    def __init__(
        self,
        config: SurfaceSegmentationConfig,
        *,
        prompts: Prompts,
        mask_refine: MaskRefineConfig,
    ) -> None:
        self.config = config
        self.prompts = prompts
        from utils.liteseg_surface import PaddleSegPredictor
        self._backend: Any = PaddleSegPredictor(
            config.liteseg_model_dir,
            device=config.liteseg_device,
            use_tensorrt=config.liteseg_use_tensorrt,
            precision=config.liteseg_precision,
            cpu_threads=config.liteseg_cpu_threads,
            foreground_class=config.liteseg_foreground_class)
        self.device = config.liteseg_device
        self.history_frames = 1

    def reset(self) -> None:
        reset = getattr(self._backend, "reset", None)
        if reset is not None:
            reset()

    def segment_tensors(self, frame: np.ndarray) -> tuple[tuple[str | int, ...], Any]:
        prediction = self._backend.predict(frame)
        return (self.config.liteseg_label,), prediction.mask[None]


def masks_to_numpy(masks: Any) -> np.ndarray:
    """将统一后端的 mask 转成连续 KxHxW bool NumPy 数组。"""
    detach = getattr(masks, "detach", None)
    if detach is not None:
        masks = detach().cpu()
        to_numpy = getattr(masks, "numpy", None)
        if to_numpy is not None:
            masks = to_numpy()
    values = np.asarray(masks, dtype=np.bool_)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise RuntimeError(f"分割 mask 必须是 KxHxW，实际 {values.shape}")
    return np.ascontiguousarray(values)


def refine_masks_numpy(
    masks: Any,
    config: MaskRefineConfig,
) -> np.ndarray:
    """用统一的最大区域和填洞逻辑整理 KxHxW mask。"""
    raw = masks_to_numpy(masks)
    return np.stack([
        np.ascontiguousarray(
            refine_mask(mask,config),dtype=np.bool_)
        for mask in raw
    ])
