"""PP-LiteSeg 导出模型的 Paddle Inference 封装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from utils.config import ConfigError


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是字典")
    return dict(value)


@dataclass(frozen=True)
class Prediction:
    mask: np.ndarray
    label_map: np.ndarray
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


def load_deploy_config(model_dir: Path) -> dict[str, Any]:
    deploy_path = model_dir / "deploy.yaml"
    try:
        with deploy_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"推理模型缺少 deploy.yaml: {deploy_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"deploy.yaml 格式错误: {error}") from error
    root = _mapping(payload, "deploy.yaml")
    return _mapping(root.get("Deploy"), "deploy.yaml.Deploy")


def locate_model_files(model_dir: Path, deploy: Mapping[str, Any]) -> tuple[Path, Path]:
    configured_model = deploy.get("model")
    configured_params = deploy.get("params")
    model_candidates: list[Path] = []
    if isinstance(configured_model, str):
        model_candidates.append(model_dir / configured_model)
    model_candidates.extend(
        model_dir / name
        for name in ("model.json", "model.pdmodel", "inference.json", "inference.pdmodel")
    )
    params_candidates: list[Path] = []
    if isinstance(configured_params, str):
        params_candidates.append(model_dir / configured_params)
    params_candidates.extend(
        model_dir / name for name in ("model.pdiparams", "inference.pdiparams"))
    model_path = next((path for path in model_candidates if path.is_file()), None)
    params_path = next((path for path in params_candidates if path.is_file()), None)
    if model_path is None or params_path is None:
        raise ConfigError(
            f"{model_dir} 中找不到静态图模型和参数文件；"
            "期望 model.json/model.pdmodel + model.pdiparams")
    return model_path, params_path


def parse_fixed_input_shape(deploy: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = deploy.get("input_shape")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        height = int(raw[2])
        width = int(raw[3])
    except (TypeError, ValueError):
        return None
    return (height, width) if height > 0 and width > 0 else None


def parse_normalize(deploy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean: Any = [0.5, 0.5, 0.5]
    std: Any = [0.5, 0.5, 0.5]
    transforms = deploy.get("transforms", [])
    if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise ConfigError("deploy.yaml.Deploy.transforms 必须是列表")
    for raw_transform in transforms:
        transform = _mapping(raw_transform, "Deploy.transforms[]")
        transform_type = transform.get("type")
        if transform_type == "Normalize":
            mean = transform.get("mean", mean)
            std = transform.get("std", std)
        elif transform_type not in {"Resize", "ResizeByLong", "ResizeByShort"}:
            raise ConfigError(
                f"实时推理暂不支持导出预处理 {transform_type!r}；"
                "本项目 train.py 生成的模型只包含 Normalize")
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_array.size == 1:
        mean_array = np.repeat(mean_array, 3)
    if std_array.size == 1:
        std_array = np.repeat(std_array, 3)
    if mean_array.size != 3 or std_array.size != 3 or np.any(std_array == 0):
        raise ConfigError("Normalize mean/std 必须各含 1 或 3 个值，且 std 非零")
    return mean_array.reshape(1, 1, 3), std_array.reshape(1, 1, 3)


def preprocess_frame(
    frame_bgr: np.ndarray,
    *,
    input_shape: tuple[int, int] | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 \
            or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame 必须是 HxWx3 uint8 BGR 图像")
    original_shape = frame_bgr.shape[:2]
    target_shape = input_shape or original_shape
    # 当前训练/导出固定使用 mean=std=0.5。OpenCV 的融合 resize、BGR->RGB、
    # uint8->NCHW float32 比逐步生成多个 640x480 临时数组明显更快。
    if np.all(mean==np.float32(.5)) and np.all(std==np.float32(.5)):
        tensor=cv2.dnn.blobFromImage(
            frame_bgr,scalefactor=1/127.5,
            size=(target_shape[1],target_shape[0]),mean=(127.5,127.5,127.5),
            swapRB=True,crop=False,ddepth=cv2.CV_32F)
        return np.ascontiguousarray(tensor,dtype=np.float32),original_shape
    image = frame_bgr
    if original_shape != target_shape:
        image = cv2.resize(
            image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - mean) / std
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, original_shape


def output_to_label_map(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output)
    if values.ndim == 4:
        values = values[:, 0] if values.shape[1] == 1 else np.argmax(values, axis=1)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise RuntimeError(f"实时推理只支持 batch=1，实际输出 {values.shape}")
        values = values[0]
    if values.ndim != 2:
        raise RuntimeError(f"无法解释 PP-LiteSeg 输出形状: {values.shape}")
    return np.ascontiguousarray(values, dtype=np.int32)


class PaddleSegPredictor:
    """加载 PP-LiteSeg 导出的固定形状二分类 Paddle 静态模型。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "gpu",
        use_tensorrt: bool = False,
        precision: str = "fp16",
        cpu_threads: int = 4,
        foreground_class: int = 1,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.deploy = load_deploy_config(self.model_dir)
        model_path, params_path = locate_model_files(self.model_dir, self.deploy)
        self.input_shape = parse_fixed_input_shape(self.deploy)
        self.mean, self.std = parse_normalize(self.deploy)
        if device not in {"cpu", "gpu"}:
            raise ConfigError("device 必须是 cpu 或 gpu")
        if precision not in {"fp32", "fp16"}:
            raise ConfigError("precision 必须是 fp32 或 fp16")
        if not isinstance(foreground_class, int) or isinstance(foreground_class, bool) \
                or foreground_class < 0:
            raise ConfigError("foreground_class 必须是非负整数")
        self.device = device
        self.foreground_class = foreground_class

        try:
            import paddle.inference as paddle_infer
        except ImportError as error:
            raise RuntimeError(
                "PP-LiteSeg 推理需要安装 paddlepaddle 或 paddlepaddle-gpu") from error
        config = paddle_infer.Config(str(model_path), str(params_path))
        if device == "gpu":
            config.enable_use_gpu(1024, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(int(cpu_threads))
            if hasattr(config, "disable_onednn"):
                config.disable_onednn()
            elif hasattr(config, "disable_mkldnn"):
                config.disable_mkldnn()
        if model_path.suffix == ".pdmodel":
            config.enable_memory_optim()
        config.switch_ir_optim(True)
        if use_tensorrt:
            if device != "gpu":
                raise ConfigError("TensorRT 只能在 GPU 推理时启用")
            precision_mode = (
                paddle_infer.PrecisionType.Half
                if precision == "fp16" else paddle_infer.PrecisionType.Float32)
            config.enable_tensorrt_engine(
                workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3,
                precision_mode=precision_mode, use_static=True, use_calib_mode=False)
            config.set_optim_cache_dir(str(self.model_dir / ".trt_cache"))
        config.disable_glog_info()
        self.predictor = paddle_infer.create_predictor(config)
        input_names = self.predictor.get_input_names()
        output_names = self.predictor.get_output_names()
        if len(input_names) != 1 or not output_names:
            raise RuntimeError(
                f"期望单输入且至少一个输出，实际 inputs={input_names}, outputs={output_names}")
        self.input_handle = self.predictor.get_input_handle(input_names[0])
        self.output_handle = self.predictor.get_output_handle(output_names[0])

    def predict(self, frame_bgr: np.ndarray) -> Prediction:
        started = time.perf_counter()
        tensor, original_shape = preprocess_frame(
            frame_bgr, input_shape=self.input_shape, mean=self.mean, std=self.std)
        after_preprocess = time.perf_counter()
        self.input_handle.reshape(tensor.shape)
        self.input_handle.copy_from_cpu(tensor)
        self.predictor.run()
        output = self.output_handle.copy_to_cpu()
        after_inference = time.perf_counter()
        label_map = output_to_label_map(output)
        if label_map.shape != original_shape:
            label_map = cv2.resize(
                label_map, (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST)
        mask = np.ascontiguousarray(label_map == self.foreground_class)
        finished = time.perf_counter()
        return Prediction(
            mask=mask, label_map=label_map,
            preprocess_ms=(after_preprocess - started) * 1000.0,
            inference_ms=(after_inference - after_preprocess) * 1000.0,
            postprocess_ms=(finished - after_inference) * 1000.0)
