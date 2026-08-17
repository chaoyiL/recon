"""从固定曝光相机实时读取帧，并使用 SAM2 做多对象点提示分割。"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Mapping

# 与 recon.py 的实时 JAX 路径一致，避免几何程序启动时预占大部分显存。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cv2
import jax
from jax import dlpack as jax_dlpack
import numpy as np

from utils.camera import open_camera
from utils.config import (
    ConfigError,
    load_config_sections,
    parse_camera_config,
    parse_reconstruction_config,
    require_keys,
)
from utils.jax_reconstruction import (
    prepare_edge_curves_from_masks_jax,
    reconstruct_surface_from_masks_jax,
)
from utils.lightfield import choose_device
from utils.process import (
    EdgePointCloudVisualizer,
    EdgeReconstructor,
    ReconstructionPointSet,
    build_reconstruction_point_set,
    concatenate_point_sets,
    contour_center_u,
    filter_contour_by_u_band,
)
from utils.sam2_surface import (
    MaskRefineConfig,
    Prompts,
    SurfaceSegmenter,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def torch_tensor_to_jax(tensor: object) -> jax.Array:
    """通过 DLPack 共享 Torch Tensor，不经过 NumPy/CPU 拷贝。"""
    contiguous = getattr(tensor, "contiguous", None)
    if contiguous is None:
        raise TypeError("Torch→JAX 输入必须是 Tensor")
    return jax_dlpack.from_dlpack(contiguous())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 多点提示实时相机分割")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def parse_mask_refine(raw_refine: Any) -> MaskRefineConfig:
    if raw_refine is None:
        return MaskRefineConfig()
    if not isinstance(raw_refine, Mapping):
        raise ConfigError("get_surface.mask_refine 必须是字典或 null")

    known = {"enabled", "close_kernel", "open_kernel", "blur_kernel", "keep_largest"}
    unknown = set(raw_refine) - known
    if unknown:
        raise ConfigError(f"get_surface.mask_refine 包含未知字段: {sorted(unknown)}")

    defaults = MaskRefineConfig()
    enabled = raw_refine.get("enabled", defaults.enabled)
    close_kernel = raw_refine.get("close_kernel", defaults.close_kernel)
    open_kernel = raw_refine.get("open_kernel", defaults.open_kernel)
    blur_kernel = raw_refine.get("blur_kernel", defaults.blur_kernel)
    keep_largest = raw_refine.get("keep_largest", defaults.keep_largest)

    if not isinstance(enabled, bool):
        raise ConfigError("get_surface.mask_refine.enabled 必须是 true 或 false")
    if not isinstance(keep_largest, bool):
        raise ConfigError("get_surface.mask_refine.keep_largest 必须是 true 或 false")
    for name, value in (
        ("close_kernel", close_kernel),
        ("open_kernel", open_kernel),
        ("blur_kernel", blur_kernel),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"get_surface.mask_refine.{name} 必须是非负整数")

    return MaskRefineConfig(
        enabled=enabled,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
        blur_kernel=blur_kernel,
        keep_largest=keep_largest,
    )


def parse_prompts(raw_prompts: Any) -> Prompts:
    if not isinstance(raw_prompts, Mapping) or not raw_prompts:
        raise ConfigError("get_surface.prompts 必须是非空字典")

    prompts: dict[str | int, dict[str, list[tuple[float, float]]]] = {}
    for label, raw_group in raw_prompts.items():
        if not isinstance(label, (str, int)) or isinstance(label, bool):
            raise ConfigError("prompt label 必须是字符串或整数")
        if not isinstance(raw_group, Mapping):
            raise ConfigError(f"label {label!r} 的配置必须是字典")

        unknown_keys = set(raw_group) - {"positive", "negative"}
        if unknown_keys:
            raise ConfigError(f"label {label!r} 包含未知字段: {sorted(unknown_keys)}")

        positive = _parse_point_list(label, "positive", raw_group.get("positive", []))
        negative = _parse_point_list(label, "negative", raw_group.get("negative", []))
        if not positive:
            raise ConfigError(f"label {label!r} 至少需要一个 positive 点")
        prompts[label] = {"positive": positive, "negative": negative}

    return prompts


def _parse_point_list(
    label: str | int,
    point_type: str,
    raw_points: Any,
) -> list[tuple[float, float]]:
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise ConfigError(f"label {label!r} 的 {point_type} 必须是点列表")

    points: list[tuple[float, float]] = []
    for point in raw_points:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise ConfigError(f"label {label!r} 的点必须是 [x, y]")
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError) as error:
            raise ConfigError(f"label {label!r} 的点坐标必须是数字") from error

    return points


def compose_masks(
    results: dict[str | int, np.ndarray],
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """将所有 mask 合成一张彩色图，每个 label 使用独立颜色。"""
    if not results:
        if shape is None:
            raise ValueError("results 为空时必须提供 shape=(height, width)")
        return np.zeros((*shape, 3), dtype=np.uint8)

    first_mask = next(iter(results.values()))
    height, width = first_mask.shape[:2]
    if shape is not None and shape != (height, width):
        raise ValueError(f"shape={shape} 与 mask 尺寸 {(height, width)} 不一致")

    composed = np.zeros((height, width, 3), dtype=np.uint8)
    for label, mask in results.items():
        if mask.shape[:2] != (height, width):
            raise ValueError(f"label {label!r} 的 mask 尺寸不一致")
        composed[mask] = _label_color(label)
    return composed


def draw_center_u_line(
    image: np.ndarray,
    u0: float,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    """在图像上绘制竖直中轴线 u = u0。"""
    height = image.shape[0]
    x = int(round(u0))
    cv2.line(image, (x, 0), (x, height - 1), color, thickness, cv2.LINE_AA)
    cv2.putText(
        image,
        f"u0={u0:.1f}",
        (x + 6, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def extract_filtered_edges(
    results: dict[str | int, np.ndarray],
    center_band_d: float,
) -> list[tuple[str | int, float, list[np.ndarray]]]:
    """从分割结果提取最大轮廓的中轴线与侧边缘折线。"""
    edges: list[tuple[str | int, float, list[np.ndarray]]] = []
    for label, mask in results.items():
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        u0 = contour_center_u(contour)
        filtered = filter_contour_by_u_band(contour, u0, center_band_d)
        edges.append((label, u0, filtered))
    return edges


def _uv_points_to_polyline(points: np.ndarray) -> np.ndarray | None:
    """将 (N,2) 图像点转为 cv2.polylines 可用的折线。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return None
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def draw_results(
    frame: np.ndarray,
    prompts: Prompts,
    results: dict[str | int, np.ndarray],
    fps: float,
    *,
    center_band_d: float = 40.0,
    edges: list[tuple[str | int, float, list[np.ndarray]]] | None = None,
    repaired_edges: list[np.ndarray] | None = None,
    pose: tuple[np.ndarray, float, float] | None = None,
) -> np.ndarray:
    visualization = frame.copy()
    edge_map = {
        label: (u0, filtered)
        for label, u0, filtered in (edges or extract_filtered_edges(results, center_band_d))
    }

    for label, mask in results.items():
        color = _label_color(label)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(visualization, [contour], -1, color, 2, cv2.LINE_AA)

        if label in edge_map:
            u0, filtered = edge_map[label]
        else:
            u0 = contour_center_u(contour)
            filtered = filter_contour_by_u_band(contour, u0, center_band_d)

        draw_center_u_line(visualization, u0)
        cv2.polylines(
            visualization,
            filtered,
            isClosed=False,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if repaired_edges:
        cv2.polylines(
            visualization,
            repaired_edges,
            isClosed=False,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    for label, group in prompts.items():
        color = _label_color(label)
        positive = group.get("positive", ())
        negative = group.get("negative", ())

        for x, y in positive:
            cv2.circle(visualization, (round(x), round(y)), 6, color, -1, cv2.LINE_AA)
            cv2.circle(visualization, (round(x), round(y)), 7, (255, 255, 255), 1)
        for x, y in negative:
            point = (round(x), round(y))
            cv2.drawMarker(
                visualization,
                point,
                color,
                cv2.MARKER_TILTED_CROSS,
                14,
                2,
                cv2.LINE_AA,
            )

        first_x, first_y = positive[0]
        cv2.putText(
            visualization,
            str(label),
            (round(first_x) + 10, round(first_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        visualization,
        f"FPS: {fps:.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if pose is not None:
        rotation_vector, tx, reprojection_rms = pose
        rotation_degrees = np.rad2deg(rotation_vector)
        cv2.putText(
            visualization,
            f"r=({rotation_degrees[0]:.2f},{rotation_degrees[1]:.2f},"
            f"{rotation_degrees[2]:.2f}) tx={tx:.3f} rms={reprojection_rms:.2f}px",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return visualization


def reconstruct_edges(
    edges: list[tuple[str | int, float, list[np.ndarray]]],
    reconstructor: EdgeReconstructor,
    *,
    pair_fill_count: int = 10,
    uv_boundary_smooth_lambda: float = 10.0,
    uv_boundary_huber_delta_px: float = 2.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, float, float] | None,
    list[np.ndarray],
    ReconstructionPointSet,
    np.ndarray,
    np.ndarray,
]:
    """对所有 label 的侧边缘做空间重建，合并左右点云；并返回处理后的 2D 绿线折线。

    所有边缘点和三维补全点统一返回为逐行对应的 UV-XYZ 点集。
    """
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    point_sets: list[ReconstructionPointSet] = []
    line_point_parts: list[np.ndarray] = []
    line_index_parts: list[np.ndarray] = []
    repaired_edges: list[np.ndarray] = []
    pose: tuple[np.ndarray, float, float] | None = None
    line_point_offset = 0
    cross_section_offset = 0
    for source_index, (_, u0, filtered) in enumerate(edges):
        if not filtered:
            continue
        try:
            result = reconstructor.process(filtered, u0)
        except ValueError:
            continue
        left_xyz = result.left_xyz
        right_xyz = result.right_xyz
        left_uv = result.left_uv
        right_uv = result.right_uv
        pose = (result.rotation_vector, result.tx, result.reprojection_rms_px)
        try:
            point_set, line_points, line_indices = build_reconstruction_point_set(
                left_xyz,
                right_xyz,
                reconstructor.K,
                reconstructor.distortion,
                result.rotation_vector,
                result.tx,
                n_fill=pair_fill_count,
                source_index=source_index,
                cross_section_offset=cross_section_offset,
                observed_left_uv=left_uv,
                observed_right_uv=right_uv,
                uv_boundary_smooth_lambda=uv_boundary_smooth_lambda,
                uv_boundary_huber_delta_px=uv_boundary_huber_delta_px,
            )
        except ValueError:
            continue
        left_parts.append(left_xyz)
        right_parts.append(right_xyz)
        point_sets.append(point_set)
        cross_section_offset += left_xyz.shape[0]
        if line_points.shape[0]:
            line_point_parts.append(line_points)
            line_index_parts.append(line_indices + line_point_offset)
            line_point_offset += line_points.shape[0]
        for side_uv in (left_uv, right_uv):
            poly = _uv_points_to_polyline(side_uv)
            if poly is not None:
                repaired_edges.append(poly)

    left_xyz = (
        np.concatenate(left_parts, axis=0)
        if left_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    right_xyz = (
        np.concatenate(right_parts, axis=0)
        if right_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    point_set = concatenate_point_sets(point_sets)
    line_points = (
        np.concatenate(line_point_parts, axis=0)
        if line_point_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    line_indices = (
        np.concatenate(line_index_parts, axis=0)
        if line_index_parts
        else np.zeros((0, 2), dtype=np.int32)
    )
    return left_xyz, right_xyz, pose, repaired_edges, point_set, line_points, line_indices


def point_set_from_surface_grids(
    xyz: np.ndarray,
    uv: np.ndarray,
    st: np.ndarray,
    camera_depth: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    surface_count: int,
    surface_rows: int,
) -> ReconstructionPointSet:
    """把实时 JAX 规则网格转换为既有保存/显示接口使用的逐点结构。"""
    points = np.asarray(xyz)
    pixels = np.asarray(uv)
    coordinates = np.asarray(st)
    depth = np.asarray(camera_depth)
    if not isinstance(surface_count, int) or surface_count < 1 \
            or not isinstance(surface_rows, int) or surface_rows < 1:
        raise ValueError("surface_count 和 surface_rows 必须是正整数")
    expected_rows = surface_count * surface_rows
    if points.ndim != 3 or points.shape[0] != expected_rows \
            or points.shape[-1] != 3:
        raise ValueError("xyz 尺寸与曲面数量/行数不一致")
    rows, columns = points.shape[:2]
    if pixels.shape != (rows, columns, 2) \
            or coordinates.shape != (rows, columns, 2) \
            or depth.shape != (rows, columns):
        raise ValueError("JAX 曲面网格的 XYZ/UV/ST/depth 尺寸不一致")
    if not np.isfinite(points).all() or not np.isfinite(pixels).all() \
            or not np.isfinite(coordinates).all() \
            or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("JAX 曲面网格包含无效坐标或非正深度")

    flat_uv = pixels.reshape(-1, 2)
    undistorted = cv2.undistortPoints(
        flat_uv.reshape(-1, 1, 2),
        np.asarray(camera_matrix, np.float64).reshape(3, 3),
        np.asarray(distortion, np.float64).reshape(-1),
        P=np.asarray(camera_matrix, np.float64).reshape(3, 3),
    ).reshape(-1, 2)
    edge_columns = np.zeros(columns, dtype=np.bool_)
    edge_columns[[0, -1]] = True
    return ReconstructionPointSet(
        xyz=points.reshape(-1, 3),
        uv=flat_uv,
        st=coordinates.reshape(-1, 2),
        undistorted_uv=undistorted,
        camera_depth=depth.reshape(-1),
        is_edge=np.tile(edge_columns, rows),
        cross_section_index=np.repeat(
            np.arange(rows, dtype=np.int32), columns),
        cross_section_alpha=coordinates[..., 1].reshape(-1),
        source_index=np.repeat(
            np.arange(surface_count, dtype=np.int32), surface_rows * columns),
    )


def save_uv_xyz_map(
    image_path: str | Path,
    point_set: ReconstructionPointSet,
) -> Path:
    """把全局逐点 XYZ/UV/ST 映射保存为与图像同名的压缩 NPZ。"""
    path = Path(image_path).expanduser()
    map_path = path.with_name(f"{path.stem}_uv_xyz.npz")
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        map_path,
        xyz=point_set.xyz,
        uv=point_set.uv,
        st=point_set.st,
        undistorted_uv=point_set.undistorted_uv,
        camera_depth=point_set.camera_depth,
        is_edge=point_set.is_edge,
        cross_section_index=point_set.cross_section_index,
        cross_section_alpha=point_set.cross_section_alpha,
        source_index=point_set.source_index,
    )
    return map_path


def save_generated_uv_xyz_map(
    map_path: str | Path,
    point_set: ReconstructionPointSet,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """原子保存离线生成的曲面映射；实时渲染不调用此函数。"""
    path = Path(map_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: dict[str, object] = {
        "xyz": point_set.xyz,
        "uv": point_set.uv,
        "st": point_set.st,
        "undistorted_uv": point_set.undistorted_uv,
        "camera_depth": point_set.camera_depth,
        "is_edge": point_set.is_edge,
        "cross_section_index": point_set.cross_section_index,
        "cross_section_alpha": point_set.cross_section_alpha,
        "source_index": point_set.source_index,
    }
    if metadata is not None:
        overlap = fields.keys() & metadata.keys()
        if overlap:
            raise ValueError(
                f"曲面映射元数据不能覆盖数据字段: {sorted(overlap)}")
        fields.update(metadata)
    with temporary.open("wb") as stream:
        np.savez(stream, **fields)
    temporary.replace(path)
    return path


def main() -> None:
    args = parse_args()
    try:
        camera_section, surface_section, lightfield_section = load_config_sections(
            args.config,
            "camera",
            "get_surface",
            "lightfield",
        )
        camera = parse_camera_config(camera_section)
        require_keys(
            surface_section,
            "get_surface",
            "model",
            "save",
            "no_display",
            "max_frames",
            "prompts",
        )

        model_id = surface_section["model"]
        save_path = surface_section["save"]
        no_display = surface_section["no_display"]
        max_frames = surface_section["max_frames"]
        if not isinstance(model_id, str) or not model_id:
            raise ConfigError("get_surface.model 必须是非空字符串")
        if save_path is not None and not isinstance(save_path, str):
            raise ConfigError("get_surface.save 必须是字符串或 null")
        if not isinstance(no_display, bool):
            raise ConfigError("get_surface.no_display 必须是 true 或 false")
        if (
            not isinstance(max_frames, int)
            or isinstance(max_frames, bool)
            or max_frames < 0
        ):
            raise ConfigError("get_surface.max_frames 必须是非负整数")
        prompts = parse_prompts(surface_section["prompts"])
        mask_refine = parse_mask_refine(surface_section.get("mask_refine"))
        compile_sam = surface_section.get("torch_compile", True)
        sam_frame_interval = surface_section.get("sam_frame_interval", 1)
        sam_memory_frames = surface_section.get("sam_memory_frames", 7)
        if not isinstance(compile_sam, bool):
            raise ConfigError("get_surface.torch_compile 必须是布尔值")
        if not isinstance(sam_frame_interval, int) \
                or isinstance(sam_frame_interval, bool) \
                or sam_frame_interval < 1:
            raise ConfigError("get_surface.sam_frame_interval 必须为正整数")
        if not isinstance(sam_memory_frames, int) \
                or isinstance(sam_memory_frames, bool) \
                or sam_memory_frames < 1:
            raise ConfigError("get_surface.sam_memory_frames 必须为正整数")
        jax_device_name = lightfield_section.get("device", "gpu")
        center_band_d = surface_section.get("center_band_d", 40)
        if (
            not isinstance(center_band_d, (int, float))
            or isinstance(center_band_d, bool)
            or center_band_d < 0
        ):
            raise ConfigError("get_surface.center_band_d 必须是非负数")
        center_band_d = float(center_band_d)
        calibration_output = None
        try:
            calibration_section = load_config_sections(args.config, "calibration")[0]
            raw_output = calibration_section.get("output")
            if isinstance(raw_output, str) and raw_output.strip():
                calibration_output = raw_output
        except ConfigError:
            calibration_output = None

        reconstruction = parse_reconstruction_config(
            surface_section.get("reconstruction"),
            config_path=args.config,
            calibration_output=calibration_output,
        )
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(2)

    print(f"正在加载模型: {model_id}")
    segmenter = SurfaceSegmenter(
        model_id=model_id,
        mask_refine=mask_refine,
        compile_model=compile_sam,
        memory_frames=sam_memory_frames,
    )
    print(f"模型已加载到: {segmenter.device}")
    print(
        "SAM2 视频记忆已启用: "
        f"history_frames={segmenter.history_frames}, "
        f"torch.compile={'on' if compile_sam else 'off'}, "
        f"每 {sam_frame_interval} 帧更新一次"
    )
    if mask_refine.enabled:
        print(
            "mask 后处理已启用: "
            f"close={mask_refine.close_kernel}, "
            f"open={mask_refine.open_kernel}, "
            f"blur={mask_refine.blur_kernel}, "
            f"keep_largest={mask_refine.keep_largest}"
        )
    print(f"中轴线过滤带宽: center_band_d={center_band_d}")
    K = reconstruction.K
    print(
        "边缘重建: "
        f"calibration={reconstruction.calibration_file}, "
        f"fx={K[0, 0]:.3f}, fy={K[1, 1]:.3f}, "
        f"cx={K[0, 2]:.3f}, cy={K[1, 2]:.3f}, "
        f"s1={reconstruction.s1}, s2={reconstruction.s2}, "
        f"geometry_grid={reconstruction.geometry_rows}x"
        f"{reconstruction.geometry_columns}, "
        f"lightfield_grid={reconstruction.lightfield_rows}x"
        f"{reconstruction.lightfield_columns}, "
        f"observation_grid={reconstruction.observation_rows}x"
        f"{reconstruction.observation_columns}, "
        f"residual_coefficient_grid={reconstruction.residual_coefficient_rows}x"
        f"{reconstruction.residual_coefficient_columns}, "
        f"residual_texture_grid={reconstruction.residual_texture_rows}x"
        f"{reconstruction.residual_texture_columns}, "
        f"uv_boundary_lambda={reconstruction.uv_boundary_smooth_lambda}, "
        f"uv_boundary_huber={reconstruction.uv_boundary_huber_delta_px}px, "
        f"show_point_cloud={reconstruction.show_point_cloud}"
    )
    print("整体重建仅使用当前 SAM 边界：无时间先验、无历史状态拒绝门控")
    print("外参策略: 首个有效边缘帧标定一次 R/tx，后续帧锁定外参并线性重建")
    device = choose_device(jax_device_name)
    if reconstruction.distortion_coefficients.size != 5:
        raise ValueError("JAX GPU 实时重建要求 OpenCV 五参数畸变模型")
    print(f"整体曲面重建已切换到与 recon.py 相同的 JAX/{device.platform} 路径")

    cap = open_camera(
        camera.device,
        camera.exposure,
        camera.white_balance_temperature,
        camera.width,
        camera.height,
    )
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_white_balance = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    print(
        f"相机已打开: device={camera.device}, "
        f"分辨率={frame_width}x{frame_height}, 曝光={actual_exposure}, "
        f"白平衡={actual_white_balance} K"
    )
    if not no_display:
        print("按 q 或 Esc 退出；按 s 保存当前叠加结果")
        cv2.namedWindow("SAM2 surface", cv2.WINDOW_NORMAL)
        cv2.namedWindow("SAM2 masks", cv2.WINDOW_NORMAL)

    reconstructor = EdgeReconstructor(
        K,
        reconstruction.distortion_coefficients,
        reconstruction.s1,
        reconstruction.s2,
        sample_count=reconstruction.sample_count,
    )
    camera_matrix_gpu = jax.device_put(np.asarray(K, np.float32), device)
    distortion_gpu = jax.device_put(
        np.asarray(reconstruction.distortion_coefficients, np.float32), device)
    inverse_camera_gpu = jax.device_put(
        np.asarray(np.linalg.inv(K), np.float32), device)

    def gpu_mask_kernel(size: int) -> int:
        if not mask_refine.enabled or size <= 0:
            return 0
        return size if size % 2 else size + 1

    gpu_mask_close_kernel = gpu_mask_kernel(mask_refine.close_kernel)
    gpu_mask_open_kernel = gpu_mask_kernel(mask_refine.open_kernel)
    gpu_mask_blur_kernel = gpu_mask_kernel(mask_refine.blur_kernel)
    prepare_curves_gpu = jax.jit(
        lambda masks: prepare_edge_curves_from_masks_jax(
            masks,
            camera_matrix_gpu,
            distortion_gpu,
            reconstructor.sample_count,
            center_band_d,
            close_kernel=gpu_mask_close_kernel,
            open_kernel=gpu_mask_open_kernel,
            blur_kernel=gpu_mask_blur_kernel,
        )[1:]
    )

    def reconstruct_geometry_gpu_impl(raw_masks,rotation,tx):
        return reconstruct_surface_from_masks_jax(
            raw_masks,
            camera_matrix_gpu,
            distortion_gpu,
            inverse_camera_gpu,
            rotation,
            reconstruction.s1,
            reconstruction.s2,
            tx,
            reconstructor.sample_count,
            center_band_d,
            reconstruction.pair_fill_count,
            reconstruction.uv_boundary_smooth_lambda,
            reconstruction.uv_boundary_huber_delta_px,
            curve_convexity=reconstruction.curve_convexity,
            close_kernel=gpu_mask_close_kernel,
            open_kernel=gpu_mask_open_kernel,
            blur_kernel=gpu_mask_blur_kernel,
        )

    reconstruct_geometry_gpu = jax.jit(reconstruct_geometry_gpu_impl)
    point_cloud_vis: EdgePointCloudVisualizer | None = None
    if reconstruction.show_point_cloud and not no_display:
        point_cloud_vis = EdgePointCloudVisualizer()

    frame_count = 0
    fps = 0.0
    last_visualization: np.ndarray | None = None
    last_masks: np.ndarray | None = None
    last_point_set = ReconstructionPointSet.empty()
    mask_gpu: jax.Array | None = None
    mask_labels: tuple[str | int, ...] = ()
    rotation_gpu: jax.Array | None = None
    tx_gpu: jax.Array | None = None
    geometry_state_gpu: tuple[jax.Array, ...] | None = None

    try:
        while max_frames <= 0 or frame_count < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("读取相机帧失败")

            started_at = time.perf_counter()
            current_frame_number = frame_count
            frame_count += 1
            update_sam = (
                mask_gpu is None
                or current_frame_number % sam_frame_interval == 0
            )
            if update_sam:
                mask_labels, mask_tensor, _ = segmenter.segment_tensors(
                    frame, prompts)
                mask_gpu = torch_tensor_to_jax(mask_tensor)
                geometry_state_gpu = None
            assert mask_gpu is not None

            if not reconstructor.calibrated:
                initial_left, initial_right, initial_valid = jax.device_get(
                    prepare_curves_gpu(mask_gpu))
                for label_index in np.flatnonzero(initial_valid):
                    try:
                        reconstructor.process_curves(
                            initial_left[label_index], initial_right[label_index])
                    except ValueError:
                        continue
                    break

            if reconstructor.calibrated and rotation_gpu is None:
                rotation = cv2.Rodrigues(
                    reconstructor.rotation_vector)[0].astype(np.float32)
                rotation_gpu, tx_gpu = jax.device_put(
                    (rotation, np.asarray(reconstructor.tx, np.float32)), device)

            if reconstructor.calibrated:
                assert rotation_gpu is not None and tx_gpu is not None
                if geometry_state_gpu is None:
                    geometry_state_gpu=reconstruct_geometry_gpu(
                        mask_gpu,rotation_gpu,tx_gpu)
                (
                    refined_masks,
                    xyz_grid,
                    uv_grid,
                    st_grid,
                    depth_grid,
                    rms_values,
                    reconstruction_valid,
                ) = jax.device_get(geometry_state_gpu)
                geometry_valid = bool(np.all(reconstruction_valid))
                masks_host = np.asarray(refined_masks, np.bool_)
            else:
                geometry_valid = False
                masks_host = np.asarray(jax.device_get(mask_gpu), np.bool_)
                xyz_grid = uv_grid = st_grid = depth_grid = None
                rms_values = np.zeros(len(mask_labels), np.float32)

            results = {
                label: np.ascontiguousarray(masks_host[index], dtype=np.bool_)
                for index, label in enumerate(mask_labels)
            }
            edges = extract_filtered_edges(results, center_band_d)
            point_set = ReconstructionPointSet.empty()
            left_xyz = np.zeros((0, 3), np.float32)
            right_xyz = np.zeros((0, 3), np.float32)
            line_points = np.zeros((0, 3), np.float32)
            line_indices = np.zeros((0, 2), np.int32)
            repaired_edges: list[np.ndarray] = []
            pose: tuple[np.ndarray, float, float] | None = None
            if geometry_valid:
                assert xyz_grid is not None and uv_grid is not None \
                    and st_grid is not None and depth_grid is not None
                surface_count = len(mask_labels)
                point_set = point_set_from_surface_grids(
                    xyz_grid,
                    uv_grid,
                    st_grid,
                    depth_grid,
                    K,
                    reconstruction.distortion_coefficients,
                    surface_count=surface_count,
                    surface_rows=reconstruction.geometry_rows,
                )
                left_xyz = np.asarray(xyz_grid[:, 0])
                right_xyz = np.asarray(xyz_grid[:, -1])
                line_points = np.asarray(
                    xyz_grid[:, [0, -1], :]).reshape(-1, 3)
                line_indices = np.arange(
                    line_points.shape[0], dtype=np.int32).reshape(-1, 2)
                for surface_index in range(surface_count):
                    start = surface_index * reconstruction.geometry_rows
                    stop = start + reconstruction.geometry_rows
                    for side in (0, -1):
                        polyline = _uv_points_to_polyline(
                            np.asarray(uv_grid[start:stop, side]))
                        if polyline is not None:
                            repaired_edges.append(polyline)
                pose = (
                    reconstructor.rotation_vector.copy(),
                    reconstructor.tx,
                    float(np.asarray(rms_values)[-1]),
                )

            elapsed = time.perf_counter() - started_at
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps = current_fps if fps == 0 else fps * 0.85 + current_fps * 0.15
            fill_xyz = point_set.fill_xyz
            last_point_set = point_set
            last_visualization = draw_results(
                frame,
                prompts,
                results,
                fps,
                center_band_d=center_band_d,
                edges=edges,
                repaired_edges=repaired_edges,
                pose=pose,
            )
            if not geometry_valid:
                cv2.putText(
                    last_visualization,
                    "surface reconstruction failed",
                    (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            last_masks = compose_masks(results, shape=frame.shape[:2])

            if point_cloud_vis is not None and (
                left_xyz.shape[0] or right_xyz.shape[0] or fill_xyz.shape[0]
            ):
                if not point_cloud_vis.update(
                    left_xyz,
                    right_xyz,
                    fill_xyz=fill_xyz,
                    line_points=line_points,
                    line_indices=line_indices,
                ):
                    point_cloud_vis = None

            if no_display:
                print(f"\r已处理 {frame_count} 帧，FPS={fps:.1f}", end="", flush=True)
                continue

            cv2.imshow("SAM2 surface", last_visualization)
            cv2.imshow("SAM2 masks", last_masks)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if save_path:
                    if not cv2.imwrite(save_path, last_visualization):
                        print(f"保存失败: {save_path}", file=sys.stderr)
                    else:
                        print(f"已保存到 {save_path}")
                        if point_set.xyz.shape[0]:
                            map_path = save_uv_xyz_map(save_path, point_set)
                            print(f"已保存 UV-XYZ 映射到 {map_path}")
                else:
                    print("请在 config.yaml 中配置 get_surface.save")
    finally:
        if point_cloud_vis is not None:
            point_cloud_vis.close()
        cap.release()
        cv2.destroyAllWindows()
        if no_display and frame_count:
            print()

    if no_display and save_path and last_visualization is not None:
        if cv2.imwrite(save_path, last_visualization):
            print(f"已保存最后一帧到 {save_path}")
        else:
            raise RuntimeError(f"保存失败: {save_path}")
        if last_masks is not None:
            mask_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_masks{Path(save_path).suffix}"
            )
            if cv2.imwrite(str(mask_path), last_masks):
                print(f"已保存 mask 图到 {mask_path}")
        if last_point_set.xyz.shape[0]:
            map_path = save_uv_xyz_map(save_path, last_point_set)
            print(f"已保存 UV-XYZ 映射到 {map_path}")


def _label_color(label: str | int) -> tuple[int, int, int]:
    digest = hashlib.sha256(repr(label).encode("utf-8")).digest()
    return tuple(80 + channel % 176 for channel in digest[:3])


if __name__ == "__main__":
    main()
