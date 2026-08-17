"""基于 SAM2 的多对象点提示分割。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from threading import Lock
from typing import Mapping, Sequence, TypeAlias

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

Point: TypeAlias = tuple[float, float]
PromptGroup: TypeAlias = Mapping[str, Sequence[Point]]
Prompts: TypeAlias = Mapping[str | int, PromptGroup]
DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-tiny"


@dataclass(frozen=True)
class MaskRefineConfig:
    """二值 mask 后处理：填洞、去噪、平滑边缘。"""

    enabled: bool = True
    close_kernel: int = 9
    open_kernel: int = 3
    blur_kernel: int = 5
    keep_largest: bool = True


class SurfaceSegmenter:
    """SAM2 流式视频分割；首帧提示，后续帧复用历史 mask memory。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | torch.device | None = None,
        mask_refine: MaskRefineConfig | None = None,
        compile_model: bool = False,
        memory_frames: int | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")

        self.model_id = model_id
        self.mask_refine = mask_refine if mask_refine is not None else MaskRefineConfig()
        try:
            self.processor = Sam2VideoProcessor.from_pretrained(
                model_id,
                local_files_only=True,
            )
            model = Sam2VideoModel.from_pretrained(
                model_id,
                local_files_only=True,
            )
        except OSError:
            self.processor = Sam2VideoProcessor.from_pretrained(model_id)
            model = Sam2VideoModel.from_pretrained(model_id)

        model = model.to(self.device).eval()
        model_memory_frames=max(1,int(model.config.num_maskmem))
        if memory_frames is None:
            memory_frames=model_memory_frames
        if not isinstance(memory_frames,int) or isinstance(memory_frames,bool) \
                or not 1<=memory_frames<=model_memory_frames:
            raise ValueError(
                f"memory_frames 必须是 1..{model_memory_frames} 的整数")
        # SAM2 同时用 num_maskmem 和 object pointer 追溯历史。实时曲面
        # 外轮廓变化缓慢时，限制两者可显著缩短 memory attention。
        model.num_maskmem=memory_frames
        model.config.num_maskmem=memory_frames
        if hasattr(model.config,"max_object_pointers_in_encoder"):
            model.config.max_object_pointers_in_encoder=min(
                int(model.config.max_object_pointers_in_encoder),memory_frames)
        self.history_frames=memory_frames
        # 视频会话本身会修改 Python 字典并使用递增 frame_idx，不能作为单一
        # torch.compile 图稳定缓存；只编译固定输入形状且占主要算力的视觉编码器。
        if compile_model and self.device.type=="cuda":
            model.vision_encoder=torch.compile(
                model.vision_encoder,mode="reduce-overhead",
                fullgraph=False,dynamic=False)
            # 1--2 帧短历史只会产生很少的 tracking 张量形状，
            # 这两个模块适合使用 CUDA graphs。更长历史在填满前
            # 会引起更多次重编译。
            if memory_frames<=2:
                model.memory_attention=torch.compile(
                    model.memory_attention,mode="reduce-overhead",
                    fullgraph=False,dynamic=False)
                model.mask_decoder=torch.compile(
                    model.mask_decoder,mode="reduce-overhead",
                    fullgraph=False,dynamic=False)
        self.model=model
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self._inference_lock = Lock()
        self._inference_session: object | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._prompt_signature: tuple[object, ...] | None = None
        self._object_labels: dict[int, str | int] = {}
        self._next_frame_index = 0

    def upload_frame(self,frame: np.ndarray) -> torch.Tensor:
        """将 BGR 帧上传到推理设备。

        跳过 SAM 的帧仍通过这个轻量路径进入 JAX，避免为了复用
        mask 而重新运行整个分割器。
        """
        _validate_frame(frame)
        return torch.from_numpy(np.ascontiguousarray(frame)).to(
            self.device,non_blocking=self.device.type=="cuda").contiguous()

    def reset(self) -> None:
        """清空视频记忆；下一帧将重新注入提示。"""
        with self._inference_lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        session = self._inference_session
        if session is not None and hasattr(session, "reset_inference_session"):
            session.reset_inference_session()
        self._inference_session = None
        self._frame_shape = None
        self._prompt_signature = None
        self._object_labels = {}
        self._next_frame_index = 0

    def _start_session(
        self,
        labels: list[str | int],
        input_points: list[list[list[float]]],
        input_point_labels: list[list[int]],
        original_size: tuple[int, int],
    ) -> None:
        self._inference_session = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device=self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        object_ids = list(range(1, len(labels) + 1))
        self._object_labels = dict(zip(object_ids, labels))
        self.processor.add_inputs_to_inference_session(
            inference_session=self._inference_session,
            frame_idx=0,
            obj_ids=object_ids,
            input_points=[input_points],
            input_labels=[input_point_labels],
            original_size=original_size,
        )
        self._frame_shape = original_size

    def _prune_history(self, current_frame_index: int) -> None:
        """流式相机只保留模型会使用的最近若干帧，避免 session 无限增长。"""
        session = self._inference_session
        if session is None:
            return
        cutoff = current_frame_index - self.history_frames + 1
        processed_frames = getattr(session, "processed_frames", None)
        if isinstance(processed_frames, dict):
            for frame_index in tuple(processed_frames):
                if frame_index < cutoff:
                    processed_frames.pop(frame_index, None)
        for object_outputs in getattr(session, "output_dict_per_obj", {}).values():
            non_conditioning = object_outputs.get("non_cond_frame_outputs", {})
            for frame_index in tuple(non_conditioning):
                if frame_index < cutoff:
                    non_conditioning.pop(frame_index, None)
        for tracked_frames in getattr(session, "frames_tracked_per_obj", {}).values():
            for frame_index in tuple(tracked_frames):
                if frame_index < cutoff:
                    tracked_frames.pop(frame_index, None)

    @staticmethod
    def _make_prompt_signature(
        labels: Sequence[str | int],
        input_points: Sequence[Sequence[Sequence[float]]],
        input_point_labels: Sequence[Sequence[int]],
    ) -> tuple[object, ...]:
        return tuple(
            (
                label,
                tuple((float(point[0]), float(point[1])) for point in points),
                tuple(int(point_label) for point_label in point_labels),
            )
            for label, points, point_labels in zip(
                labels,
                input_points,
                input_point_labels,
            )
        )

    def segment_tensors(
        self,
        frame: np.ndarray,
        prompts: Prompts,
    ) -> tuple[tuple[str | int,...],torch.Tensor,torch.Tensor]:
        """分割一帧并将 BGR 帧和二值 mask 保留在推理设备上。"""
        labels, input_points, input_point_labels = _validate_prompts(frame, prompts)
        frame_bgr=self.upload_frame(frame)
        if self.device.type == "cuda":
            # TorchvisionBackend 可直接处理 CUDA HWC uint8；避免 OpenCV 转色、PIL
            # 封装以及随后再次把整帧从 CPU 上传到 GPU。
            image=frame_bgr.flip(-1)
        else:
            rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(rgb_image)

        model_inputs = self.processor(
            images=image,
            device=self.device,
            return_tensors="pt",
        ).to(self.device)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        frame_shape = frame.shape[:2]
        prompt_signature = self._make_prompt_signature(
            labels,
            input_points,
            input_point_labels,
        )
        with self._inference_lock, torch.inference_mode(), autocast_context:
            if (
                self._inference_session is None
                or self._frame_shape != frame_shape
                or self._prompt_signature != prompt_signature
            ):
                self._reset_unlocked()
                self._start_session(
                    labels,
                    input_points,
                    input_point_labels,
                    frame_shape,
                )
                self._prompt_signature = prompt_signature

            if self.device.type=="cuda" and hasattr(
                    torch.compiler,"cudagraph_mark_step_begin"):
                # 编译后的 vision/memory/mask 子图会复用输出缓冲区；
                # 显式标记视频帧边界，防止后续子图过早覆写。
                torch.compiler.cudagraph_mark_step_begin()
            model_arguments={
                "inference_session":self._inference_session,
                "frame":model_inputs.pixel_values[0],
                "frame_idx":self._next_frame_index,
            }
            if self.history_frames==1:
                # 只使用首帧 conditioning memory 时，后续帧的 memory
                # encoder 输出绝不会被消费，可直接跳过。
                model_arguments["run_mem_encoder"]=self._next_frame_index==0
            outputs = self.model(**model_arguments)
            self._next_frame_index += 1
            masks = self.processor.post_process_masks(
                [outputs.pred_masks],
                original_sizes=model_inputs.original_sizes,
                binarize=False,
            )[0]
            self._prune_history(int(outputs.frame_idx))
            object_labels = self._object_labels.copy()

        ordered_labels=tuple(object_labels[int(object_id)]
                             for object_id in outputs.object_ids)
        tensor_masks=masks.detach().to(
            self.device,non_blocking=self.device.type=="cuda")
        if tensor_masks.ndim==4 and tensor_masks.shape[1]==1:
            tensor_masks=tensor_masks[:,0]
        if tensor_masks.ndim==2:
            tensor_masks=tensor_masks[None]
        if tensor_masks.ndim!=3 or tensor_masks.shape[0]!=len(ordered_labels):
            raise RuntimeError("SAM2 后处理 mask 必须是 KxHxW")
        return ordered_labels,(tensor_masks>0).contiguous(),frame_bgr.contiguous()

    def segment(
        self,
        frame: np.ndarray,
        prompts: Prompts,
    ) -> dict[str | int, np.ndarray]:
        """兼容 CPU 调用方：按 label 返回经 OpenCV 整理的 NumPy mask。"""
        labels,masks,_=self.segment_tensors(frame,prompts)
        results: dict[str | int, np.ndarray] = {}
        for object_index,label in enumerate(labels):
            mask = masks[object_index].detach().cpu().numpy()
            mask = refine_mask(mask, self.mask_refine)
            mask = np.ascontiguousarray(mask, dtype=np.bool_)

            results[label] = mask

        return results

    def __call__(
        self,
        frame: np.ndarray,
        prompts: Prompts,
    ) -> dict[str | int, np.ndarray]:
        return self.segment(frame, prompts)


def refine_mask(mask: np.ndarray, config: MaskRefineConfig) -> np.ndarray:
    """对 SAM 二值 mask 做形态学整理，填充空洞并平滑锯齿边缘。"""
    if not config.enabled:
        return np.asarray(mask, dtype=np.bool_)

    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return binary.astype(bool)

    close_k = _odd_kernel(config.close_kernel)
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    open_k = _odd_kernel(config.open_kernel)
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 只保留外轮廓并填充，可去掉内部空洞。
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary, dtype=bool)

    if config.keep_largest:
        contours = (max(contours, key=cv2.contourArea),)

    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    blur_k = _odd_kernel(config.blur_kernel)
    if blur_k > 1:
        filled = cv2.GaussianBlur(filled, (blur_k, blur_k), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)

    return filled.astype(bool)


def _odd_kernel(size: int) -> int:
    if size <= 0:
        return 0
    return size if size % 2 == 1 else size + 1


def _validate_prompts(
    frame: np.ndarray,
    prompts: Prompts,
) -> tuple[list[str | int], list[list[list[float]]], list[list[int]]]:
    _validate_frame(frame)
    if not isinstance(prompts, Mapping) or not prompts:
        raise ValueError("prompts 必须是非空字典")

    height, width = frame.shape[:2]
    labels: list[str | int] = []
    all_points: list[list[list[float]]] = []
    all_point_labels: list[list[int]] = []

    for label, group in prompts.items():
        if not isinstance(label, (str, int)) or isinstance(label, bool):
            raise TypeError("label 必须是字符串或整数")
        if isinstance(label, str) and not label:
            raise ValueError("字符串 label 不能为空")
        if not isinstance(group, Mapping):
            raise TypeError(f"label {label!r} 的值必须是字典")

        unknown_keys = set(group) - {"positive", "negative"}
        if unknown_keys:
            raise ValueError(f"label {label!r} 包含未知点类型: {sorted(unknown_keys)}")

        positive = group.get("positive", ())
        negative = group.get("negative", ())
        if not positive:
            raise ValueError(f"label {label!r} 至少需要一个前景点")

        object_points: list[list[float]] = []
        object_point_labels: list[int] = []
        for point_type, points, sam_label in (
            ("positive", positive, 1),
            ("negative", negative, 0),
        ):
            if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
                raise TypeError(f"label {label!r} 的 {point_type} 必须是点序列")
            for point in points:
                x, y = _validate_point(point, label, point_type, width, height)
                object_points.append([x, y])
                object_point_labels.append(sam_label)

        labels.append(label)
        all_points.append(object_points)
        all_point_labels.append(object_point_labels)

    return labels, all_points, all_point_labels


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame,np.ndarray):
        raise TypeError("frame 必须是 numpy.ndarray")
    if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
        raise ValueError("frame 必须是 HxWx3 的 uint8 BGR 图像")


def _validate_point(
    point: Point,
    label: str | int,
    point_type: str,
    width: int,
    height: int,
) -> tuple[float, float]:
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
        raise ValueError(f"label {label!r} 的 {point_type} 点必须是 (x, y)")

    x, y = point
    if (
        not isinstance(x, Real)
        or isinstance(x, bool)
        or not isinstance(y, Real)
        or isinstance(y, bool)
        or not np.isfinite(x)
        or not np.isfinite(y)
    ):
        raise ValueError(f"label {label!r} 的坐标必须是有限数字")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"label {label!r} 的点 ({x}, {y}) 超出图像范围 "
            f"[0, {width}) x [0, {height})"
        )

    return float(x), float(y)
