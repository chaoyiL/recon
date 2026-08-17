"""轮廓点加工：中轴线估计、按 u 带宽过滤、侧边缘空间重建等。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import hstack, lil_matrix, vstack
from scipy.sparse.linalg import lsmr


def contour_center_u(contour: np.ndarray) -> float:
    """边缘轮廓中轴线 u0：所有轮廓点 u 坐标的平均值。"""
    points = np.asarray(contour).reshape(-1, 2)
    if points.size == 0:
        raise ValueError("contour 为空，无法计算中轴线")
    return float(np.mean(points[:, 0]))


def filter_contour_by_u_band(
    contour: np.ndarray,
    u0: float,
    d: float,
    *,
    min_v: float = 5.0,
) -> list[np.ndarray]:
    """删除 contour 中 u ∈ [u0-d, u0+d] 或 v < min_v 的点，返回剩余连通折线段。"""
    if d < 0:
        raise ValueError("d 必须是非负数")

    points = np.asarray(contour).reshape(-1, 2)
    if points.size == 0:
        return []

    outside_u_band = (points[:, 0] < u0 - d) | (points[:, 0] > u0 + d)
    keep = outside_u_band & (points[:, 1] >= min_v)
    if not np.any(keep):
        return []

    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for point, retained in zip(points, keep):
        if retained:
            current.append(point)
            continue
        if current:
            segments.append(np.asarray(current, dtype=np.int32).reshape(-1, 1, 2))
            current = []
    if current:
        segments.append(np.asarray(current, dtype=np.int32).reshape(-1, 1, 2))

    # 闭合轮廓首尾都保留时，合并为同一条折线。
    if (
        len(segments) >= 2
        and keep[0]
        and keep[-1]
        and not np.array_equal(segments[0], segments[-1])
    ):
        merged = np.concatenate([segments[-1], segments[0]], axis=0)
        segments = [merged, *segments[1:-1]]

    return [segment for segment in segments if len(segment) >= 2]


def _segments_to_points(segments: Sequence[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros((0, 2), dtype=np.float64)
    parts = [np.asarray(segment).reshape(-1, 2) for segment in segments]
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def split_edge_segments(
    segments: Sequence[np.ndarray],
    u0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按相对中轴线位置合并为左/右边缘点集。左: u < u0，右: u > u0。"""
    points = _segments_to_points(segments)
    if points.size == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    left = points[points[:, 0] < u0]
    right = points[points[:, 0] > u0]
    return left, right


def _aggregate_edge_by_v(points: np.ndarray) -> np.ndarray:
    """按 v 整理单侧折线；这里只去重排序，不建立左右对应。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    v_keys = np.rint(points[:, 1]).astype(np.int64)
    order = np.argsort(v_keys, kind="mergesort")
    v_sorted = v_keys[order]
    u_sorted = points[order, 0]
    unique_v, start_idx, counts = np.unique(v_sorted, return_index=True, return_counts=True)
    u_agg = np.empty(unique_v.shape[0], dtype=np.float64)
    for i, (start, count) in enumerate(zip(start_idx, counts)):
        u_agg[i] = float(np.median(u_sorted[start : start + count]))
    return np.column_stack([u_agg, unique_v.astype(np.float64)])


@dataclass(frozen=True)
class ReconstructionPointSet:
    """同一行索引下严格对应的 XYZ、图像 UV 和规范曲面 ST 坐标。"""

    xyz: np.ndarray
    uv: np.ndarray
    st: np.ndarray
    undistorted_uv: np.ndarray
    camera_depth: np.ndarray
    is_edge: np.ndarray
    cross_section_index: np.ndarray
    cross_section_alpha: np.ndarray # 用于区分左右边缘，0左1右，中间值为补全点
    source_index: np.ndarray

    def __post_init__(self) -> None:
        count = self.xyz.shape[0]
        if self.xyz.shape != (count, 3):
            raise ValueError("ReconstructionPointSet.xyz 必须是 (N,3)")
        if (
            self.uv.shape != (count, 2)
            or self.st.shape != (count, 2)
            or self.undistorted_uv.shape != (count, 2)
        ):
            raise ValueError("ReconstructionPointSet 的 UV/ST 必须是 (N,2)")
        for name, values in (
            ("camera_depth", self.camera_depth),
            ("is_edge", self.is_edge),
            ("cross_section_index", self.cross_section_index),
            ("cross_section_alpha", self.cross_section_alpha),
            ("source_index", self.source_index),
        ):
            if values.shape != (count,):
                raise ValueError(f"ReconstructionPointSet.{name} 必须是 (N,)")
        if not (
            np.isfinite(self.xyz).all()
            and np.isfinite(self.uv).all()
            and np.isfinite(self.st).all()
            and np.isfinite(self.undistorted_uv).all()
            and np.isfinite(self.camera_depth).all()
        ):
            raise ValueError("ReconstructionPointSet 包含非有限坐标")
        if np.any(self.st < 0) or np.any(self.st > 1):
            raise ValueError("ReconstructionPointSet.st 必须位于 [0,1]")
        if np.any(self.camera_depth <= 0):
            raise ValueError("ReconstructionPointSet.camera_depth 必须为正")

    @classmethod
    def empty(cls) -> "ReconstructionPointSet":
        return cls(
            xyz=np.zeros((0, 3), dtype=np.float64),
            uv=np.zeros((0, 2), dtype=np.float64),
            st=np.zeros((0, 2), dtype=np.float64),
            undistorted_uv=np.zeros((0, 2), dtype=np.float64),
            camera_depth=np.zeros(0, dtype=np.float64),
            is_edge=np.zeros(0, dtype=bool),
            cross_section_index=np.zeros(0, dtype=np.int32),
            cross_section_alpha=np.zeros(0, dtype=np.float64),
            source_index=np.zeros(0, dtype=np.int32),
        )

    @property
    def fill_xyz(self) -> np.ndarray:
        return self.xyz[~self.is_edge]


def concatenate_point_sets(
    point_sets: Sequence[ReconstructionPointSet],
) -> ReconstructionPointSet:
    if not point_sets:
        return ReconstructionPointSet.empty()
    return ReconstructionPointSet(
        xyz=np.concatenate([point_set.xyz for point_set in point_sets]),
        uv=np.concatenate([point_set.uv for point_set in point_sets]),
        st=np.concatenate([point_set.st for point_set in point_sets]),
        undistorted_uv=np.concatenate(
            [point_set.undistorted_uv for point_set in point_sets]
        ),
        camera_depth=np.concatenate(
            [point_set.camera_depth for point_set in point_sets]
        ),
        is_edge=np.concatenate([point_set.is_edge for point_set in point_sets]),
        cross_section_index=np.concatenate(
            [point_set.cross_section_index for point_set in point_sets]
        ),
        cross_section_alpha=np.concatenate(
            [point_set.cross_section_alpha for point_set in point_sets]
        ),
        source_index=np.concatenate(
            [point_set.source_index for point_set in point_sets]
        ),
    )


@dataclass(frozen=True)
class ReconstructionResult:
    left_xyz: np.ndarray
    right_xyz: np.ndarray
    left_uv: np.ndarray
    right_uv: np.ndarray
    rotation_vector: np.ndarray
    tx: float
    reprojection_rms_px: float


def _undistort_pixels(
    points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return cv2.undistortPoints(points, K, distortion, P=K).reshape(-1, 2)


def _distort_pixels(
    points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """把无畸变像素坐标变回原始图像像素坐标。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normalized = np.column_stack(
        [
            (points[:, 0] - K[0, 2]) / K[0, 0],
            (points[:, 1] - K[1, 2]) / K[1, 1],
            np.ones(points.shape[0]),
        ]
    )
    projected, _ = cv2.projectPoints(
        normalized,
        np.zeros(3),
        np.zeros(3),
        K,
        distortion,
    )
    return projected.reshape(-1, 2)


def _smooth_boundary_uv_error(
    error: np.ndarray,
    smooth_lambda: float,
    huber_delta_px: float,
    *,
    iterations: int = 4,
) -> np.ndarray:
    """用二阶正则和二维 Huber IRLS 平滑一侧边界的 (du,dv) 误差。"""
    values = np.asarray(error, dtype=np.float64).reshape(-1, 2)
    count = values.shape[0]
    if count < 3 or smooth_lambda == 0:
        return values.copy()

    second = np.zeros((count - 2, count), dtype=np.float64)
    row = np.arange(count - 2)
    second[row, row] = 1.0
    second[row, row + 1] = -2.0
    second[row, row + 2] = 1.0
    regularizer = float(smooth_lambda) * (second.T @ second)

    estimate = np.linalg.solve(np.eye(count) + regularizer, values)
    for _ in range(iterations):
        residual_norm = np.linalg.norm(estimate - values, axis=1)
        weights = np.minimum(
            1.0,
            float(huber_delta_px) / np.maximum(residual_norm, 1e-12),
        )
        normal = np.diag(weights) + regularizer
        estimate = np.linalg.solve(normal, weights[:, None] * values)
    return estimate


def build_reconstruction_point_set(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
    rotation_vector: np.ndarray,
    tx: float,
    *,
    n_fill: int = 10,
    source_index: int = 0,
    cross_section_offset: int = 0,
    observed_left_uv: np.ndarray | None = None,
    observed_right_uv: np.ndarray | None = None,
    uv_boundary_smooth_lambda: float = 10.0,
    uv_boundary_huber_delta_px: float = 2.0,
) -> tuple[ReconstructionPointSet, np.ndarray, np.ndarray]:
    """在三维截面上补点，并用实测左右边界约束最终 UV-XYZ 映射。"""
    left_xyz = np.asarray(left_xyz, dtype=np.float64).reshape(-1, 3)
    right_xyz = np.asarray(right_xyz, dtype=np.float64).reshape(-1, 3)
    if left_xyz.shape != right_xyz.shape or left_xyz.shape[0] == 0:
        raise ValueError("左右边缘必须包含数量相同的三维点")
    if n_fill < 0:
        raise ValueError("n_fill 必须是非负整数")
    if uv_boundary_smooth_lambda < 0:
        raise ValueError("uv_boundary_smooth_lambda 必须是非负数")
    if uv_boundary_huber_delta_px <= 0:
        raise ValueError("uv_boundary_huber_delta_px 必须是正数")
    if (observed_left_uv is None) != (observed_right_uv is None):
        raise ValueError("observed_left_uv 和 observed_right_uv 必须同时提供")

    section_count = left_xyz.shape[0]
    alpha = np.linspace(0.0, 1.0, n_fill + 2, dtype=np.float64)
    xyz_grid = (
        (1.0 - alpha)[None, :, None] * left_xyz[:, None, :]
        + alpha[None, :, None] * right_xyz[:, None, :]
    )
    xyz = xyz_grid.reshape(-1, 3)
    undistorted_uv, depth = _project_world_points(
        xyz,
        K,
        rotation_vector,
        tx,
    )
    if np.any(depth <= 0):
        raise ValueError("补全点中存在相机后方的点，无法建立 UV 映射")
    if observed_left_uv is not None:
        left_observed = np.asarray(observed_left_uv, dtype=np.float64).reshape(-1, 2)
        right_observed = np.asarray(observed_right_uv, dtype=np.float64).reshape(-1, 2)
        expected_shape = (section_count, 2)
        if left_observed.shape != expected_shape or right_observed.shape != expected_shape:
            raise ValueError(f"实测左右边界 UV 必须都是 {expected_shape}")
        if not np.isfinite(left_observed).all() or not np.isfinite(right_observed).all():
            raise ValueError("实测左右边界 UV 包含非有限坐标")

        projected_grid = undistorted_uv.reshape(section_count, n_fill + 2, 2)
        left_observed = _undistort_pixels(left_observed, K, distortion)
        right_observed = _undistort_pixels(right_observed, K, distortion)
        left_error = _smooth_boundary_uv_error(
            left_observed - projected_grid[:, 0],
            uv_boundary_smooth_lambda,
            uv_boundary_huber_delta_px,
        )
        right_error = _smooth_boundary_uv_error(
            right_observed - projected_grid[:, -1],
            uv_boundary_smooth_lambda,
            uv_boundary_huber_delta_px,
        )
        correction = (
            (1.0 - alpha)[None, :, None] * left_error[:, None, :]
            + alpha[None, :, None] * right_error[:, None, :]
        )
        undistorted_uv = (projected_grid + correction).reshape(-1, 2)
    uv = _distort_pixels(undistorted_uv, K, distortion)
    is_edge_per_section = np.zeros(n_fill + 2, dtype=bool)
    is_edge_per_section[[0, -1]] = True
    point_set = ReconstructionPointSet(
        xyz=xyz,
        uv=uv,
        # s 在每个独立 source 内沿截面方向归一化；t 沿左右边界归一化。
        # concatenate_point_sets 只拼接行，不会改变各 source 自身的规范坐标。
        st=np.stack(
            np.meshgrid(
                np.linspace(0.0, 1.0, section_count, dtype=np.float64),
                alpha,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2),
        undistorted_uv=undistorted_uv,
        camera_depth=depth,
        is_edge=np.tile(is_edge_per_section, section_count),
        cross_section_index=np.repeat(
            np.arange(
                cross_section_offset,
                cross_section_offset + section_count,
                dtype=np.int32,
            ),
            n_fill + 2,
        ),
        cross_section_alpha=np.tile(alpha, section_count),
        source_index=np.full(xyz.shape[0], int(source_index), dtype=np.int32),
    )

    line_points = xyz_grid[:, [0, -1], :].reshape(-1, 3)
    line_indices = np.column_stack(
        [
            np.arange(0, 2 * section_count, 2),
            np.arange(1, 2 * section_count, 2),
        ]
    ).astype(np.int32)
    return point_set, line_points, line_indices


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """按二维累计弧长等距重采样，返回亚像素坐标。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2 or count < 2:
        raise ValueError("边缘点或重采样数量不足")
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-9])
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("边缘折线长度为零")
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    targets = np.linspace(0.0, float(length[-1]), count)
    return np.column_stack(
        [np.interp(targets, length, points[:, axis]) for axis in range(2)]
    )


def _rotation(rotation_vector: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64).reshape(3))[0]


def _intersect_pixels_with_x_plane(
    pixels: np.ndarray,
    x_plane: float,
    K: np.ndarray,
    rotation_vector: np.ndarray,
    tx: float,
) -> np.ndarray:
    """把无畸变像素射线与世界平面 X=x_plane 求交。"""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    rays_camera = (
        np.linalg.inv(K)
        @ np.column_stack([pixels, np.ones(pixels.shape[0])]).T
    ).T
    R = _rotation(rotation_vector)
    t = np.asarray([tx, 0.0, 0.0], dtype=np.float64)
    camera_center = -R.T @ t
    rays_world = (R.T @ rays_camera.T).T
    dx = rays_world[:, 0]
    if np.any(np.abs(dx) < 1e-9):
        raise ValueError("存在与侧平面近乎平行的相机射线")
    scale = (float(x_plane) - camera_center[0]) / dx
    points = camera_center + scale[:, None] * rays_world
    if not np.isfinite(points).all():
        raise ValueError("射线与侧平面求交失败")
    points[:, 0] = float(x_plane)
    return points


def _project_world_points(
    points: np.ndarray,
    K: np.ndarray,
    rotation_vector: np.ndarray,
    tx: float,
) -> tuple[np.ndarray, np.ndarray]:
    R = _rotation(rotation_vector)
    camera = (R @ np.asarray(points, dtype=np.float64).T).T
    camera[:, 0] += float(tx)
    depth = camera[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    normalized = camera[:, :2] / safe_depth[:, None]
    pixels = np.column_stack(
        [
            K[0, 0] * normalized[:, 0] + K[0, 2],
            K[1, 1] * normalized[:, 1] + K[1, 2],
        ]
    )
    return pixels, depth


def _initial_shared_curve(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    K: np.ndarray,
    s1: float,
    s2: float,
    rotation_vector: np.ndarray,
    tx: float,
) -> tuple[np.ndarray, np.ndarray]:
    # 首次求解严格从 R=I、tx=0 开始。用左右视差初始化共享深度，
    # 只生成曲线初值，不再反解或预标定 tx。
    if np.linalg.norm(rotation_vector) < 1e-12 and abs(tx) < 1e-12:
        disparity = right_uv[:, 0] - left_uv[:, 0]
        if np.any(np.abs(disparity) < 1e-6):
            raise ValueError("左右初始对应点视差过小")
        z = K[0, 0] * (s1 - s2) / disparity
        h_left = (left_uv[:, 1] - K[1, 2]) * z / K[1, 1]
        h_right = (right_uv[:, 1] - K[1, 2]) * z / K[1, 1]
        return 0.5 * (h_left + h_right), z

    left = _intersect_pixels_with_x_plane(left_uv, s2, K, rotation_vector, tx)
    right = _intersect_pixels_with_x_plane(right_uv, s1, K, rotation_vector, tx)
    h = 0.5 * (left[:, 1] + right[:, 1])
    z = 0.5 * (left[:, 2] + right[:, 2])
    return h, z


def _optimize_shared_curve(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    K: np.ndarray,
    s1: float,
    s2: float,
    initial_rotation: np.ndarray,
    initial_tx: float,
    initial_h: np.ndarray,
    initial_z: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """联合优化 R、tx 和共享的 Y=h、Z=z 曲线。"""
    count = left_uv.shape[0]
    x0 = np.concatenate(
        [
            np.asarray(initial_rotation, dtype=np.float64).reshape(3),
            [float(initial_tx)],
            np.asarray(initial_h, dtype=np.float64).reshape(count),
            np.asarray(initial_z, dtype=np.float64).reshape(count),
        ]
    )
    angle_bound = np.deg2rad(15.0)
    lower = np.concatenate(
        [np.full(3, -angle_bound), [-10.0], np.full(2 * count, -np.inf)]
    )
    upper = np.concatenate(
        [np.full(3, angle_bound), [10.0], np.full(2 * count, np.inf)]
    )

    residual_count = 4 * count + 2 * (count - 2) + 3 + 2 * count
    sparsity = lil_matrix((residual_count, 4 + 2 * count), dtype=np.int8)
    for k in range(count):
        variable_columns = [0, 1, 2, 3, 4 + k, 4 + count + k]
        for row in (2 * k, 2 * k + 1, 2 * count + 2 * k, 2 * count + 2 * k + 1):
            sparsity[row, variable_columns] = 1
    smooth_h_start = 4 * count
    smooth_z_start = smooth_h_start + count - 2
    for k in range(count - 2):
        sparsity[smooth_h_start + k, 4 + k : 4 + k + 3] = 1
        sparsity[
            smooth_z_start + k,
            4 + count + k : 4 + count + k + 3,
        ] = 1
    rotation_start = smooth_z_start + count - 2
    sparsity[rotation_start : rotation_start + 3, :3] = np.eye(3, dtype=np.int8)
    depth_start = rotation_start + 3
    for k in range(count):
        variable_columns = [0, 1, 2, 3, 4 + k, 4 + count + k]
        sparsity[depth_start + k, variable_columns] = 1
        sparsity[depth_start + count + k, variable_columns] = 1

    def residual(values: np.ndarray) -> np.ndarray:
        rotation_vector = values[:3]
        tx = float(values[3])
        h = values[4 : 4 + count]
        z = values[4 + count :]
        left_xyz = np.column_stack([np.full(count, s2), h, z])
        right_xyz = np.column_stack([np.full(count, s1), h, z])
        left_projection, left_depth = _project_world_points(
            left_xyz, K, rotation_vector, tx
        )
        right_projection, right_depth = _project_world_points(
            right_xyz, K, rotation_vector, tx
        )
        reprojection = np.concatenate(
            [
                ((left_projection - left_uv) / 1.5).reshape(-1),
                ((right_projection - right_uv) / 1.5).reshape(-1),
            ]
        )
        smooth_h = np.diff(h, n=2) / 1.0
        smooth_z = np.diff(z, n=2) / 1.0
        rotation_prior = rotation_vector / np.deg2rad(5.0)
        # 始终保留固定数量的正深度残差。
        depth_penalty = np.concatenate(
            [np.minimum(left_depth - 1e-3, 0.0), np.minimum(right_depth - 1e-3, 0.0)]
        )
        return np.concatenate(
            [reprojection, smooth_h, smooth_z, rotation_prior, depth_penalty]
        )

    optimized = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        jac_sparsity=sparsity.tocsr(),
        max_nfev=200,
    )
    values = optimized.x
    rotation_vector = values[:3]
    tx = float(values[3])
    h = values[4 : 4 + count]
    z = values[4 + count :]
    left_xyz = np.column_stack([np.full(count, s2), h, z])
    right_xyz = np.column_stack([np.full(count, s1), h, z])
    left_projection, _ = _project_world_points(left_xyz, K, rotation_vector, tx)
    right_projection, _ = _project_world_points(right_xyz, K, rotation_vector, tx)
    squared = np.concatenate(
        [
            np.sum((left_projection - left_uv) ** 2, axis=1),
            np.sum((right_projection - right_uv) ** 2, axis=1),
        ]
    )
    rms = float(np.sqrt(np.mean(squared)))
    return rotation_vector, tx, h, z, rms


def _monotone_right_matches(
    left_uv: np.ndarray,
    right_dense_uv: np.ndarray,
    K: np.ndarray,
    s1: float,
    s2: float,
    rotation_vector: np.ndarray,
    tx: float,
) -> np.ndarray:
    """动态规划选择顺序一致、Y/Z 接近且弧长偏移有限的右侧点。"""
    left_xyz = _intersect_pixels_with_x_plane(
        left_uv, s2, K, rotation_vector, tx
    )
    right_xyz = _intersect_pixels_with_x_plane(
        right_dense_uv, s1, K, rotation_vector, tx
    )
    n = left_uv.shape[0]
    m = right_dense_uv.shape[0]
    tau = np.linspace(0.0, 1.0, n)
    xi = np.linspace(0.0, 1.0, m)
    dy = (left_xyz[:, None, 1] - right_xyz[None, :, 1]) / 2.0
    dz = (left_xyz[:, None, 2] - right_xyz[None, :, 2]) / 2.0
    dt = (tau[:, None] - xi[None, :]) / 0.05
    cost = dy * dy + dz * dz + dt * dt
    cost[np.abs(tau[:, None] - xi[None, :]) > 0.08] = np.inf

    accumulated = np.full((n, m), np.inf, dtype=np.float64)
    previous = np.full((n, m), -1, dtype=np.int32)
    accumulated[0, 0] = cost[0, 0]
    ideal_step = (m - 1) / (n - 1)
    max_step = max(2, int(np.ceil(2.0 * ideal_step)))

    for k in range(1, n):
        candidates = range(m) if k < n - 1 else (m - 1,)
        for j in candidates:
            if not np.isfinite(cost[k, j]):
                continue
            start = max(0, j - max_step)
            for prior_j in range(start, j):
                if not np.isfinite(accumulated[k - 1, prior_j]):
                    continue
                step_penalty = 0.25 * (((j - prior_j) - ideal_step) / ideal_step) ** 2
                value = accumulated[k - 1, prior_j] + cost[k, j] + step_penalty
                if value < accumulated[k, j]:
                    accumulated[k, j] = value
                    previous[k, j] = prior_j

    if not np.isfinite(accumulated[-1, -1]):
        raise ValueError("动态规划未找到有效的单调左右对应关系")
    indices = np.empty(n, dtype=np.int32)
    indices[-1] = m - 1
    for k in range(n - 1, 0, -1):
        indices[k - 1] = previous[k, indices[k]]
    return indices


def _solve_smoothed_shared_curve(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    K: np.ndarray,
    s1: float,
    s2: float,
    rotation_vector: np.ndarray,
    tx: float,
    *,
    smooth_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """固定外参，线性求解共享 h/z，并惩罚两条序列的二阶差分。"""
    left_uv = np.asarray(left_uv, dtype=np.float64).reshape(-1, 2)
    right_uv = np.asarray(right_uv, dtype=np.float64).reshape(-1, 2)
    if left_uv.shape != right_uv.shape or left_uv.shape[0] < 3:
        raise ValueError("线性重建要求至少三对左右对应点")
    if smooth_lambda < 0:
        raise ValueError("smooth_lambda 必须是非负数")

    count = left_uv.shape[0]
    R = _rotation(rotation_vector)
    observations = lil_matrix((4 * count, 2 * count), dtype=np.float64)
    target = np.empty(4 * count, dtype=np.float64)

    def add_side_rows(
        row: int,
        index: int,
        uv: np.ndarray,
        x_plane: float,
    ) -> None:
        x = (uv[0] - K[0, 2]) / K[0, 0]
        y = (uv[1] - K[1, 2]) / K[1, 1]
        observations[row, index] = R[0, 1] - x * R[2, 1]
        observations[row, count + index] = R[0, 2] - x * R[2, 2]
        target[row] = -(R[0, 0] - x * R[2, 0]) * x_plane - tx
        observations[row + 1, index] = R[1, 1] - y * R[2, 1]
        observations[row + 1, count + index] = R[1, 2] - y * R[2, 2]
        target[row + 1] = -(R[1, 0] - y * R[2, 0]) * x_plane

    for k in range(count):
        add_side_rows(4 * k, k, left_uv[k], s2)
        add_side_rows(4 * k + 2, k, right_uv[k], s1)

    second_difference = lil_matrix((count - 2, count), dtype=np.float64)
    rows = np.arange(count - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    zero = lil_matrix((count - 2, count), dtype=np.float64)
    smooth_h = hstack([second_difference, zero], format="csr")
    smooth_z = hstack([zero, second_difference], format="csr")
    smooth_scale = np.sqrt(float(smooth_lambda))
    augmented_matrix = vstack(
        [
            observations.tocsr(),
            smooth_scale * smooth_h,
            smooth_scale * smooth_z,
        ],
        format="csr",
    )
    augmented_target = np.concatenate(
        [target, np.zeros(2 * (count - 2), dtype=np.float64)]
    )
    solution = lsmr(
        augmented_matrix,
        augmented_target,
        atol=1e-10,
        btol=1e-10,
        maxiter=max(200, 4 * count),
    )[0]
    if not np.isfinite(solution).all():
        raise ValueError("带平滑项的线性最小二乘求解失败")

    h = solution[:count]
    z = solution[count:]
    left_xyz = np.column_stack([np.full(count, s2), h, z])
    right_xyz = np.column_stack([np.full(count, s1), h, z])
    left_projection, left_depth = _project_world_points(
        left_xyz, K, rotation_vector, tx
    )
    right_projection, right_depth = _project_world_points(
        right_xyz, K, rotation_vector, tx
    )
    if np.any(left_depth <= 0) or np.any(right_depth <= 0):
        raise ValueError("线性最小二乘产生了相机后方的点")
    squared = np.concatenate(
        [
            np.sum((left_projection - left_uv) ** 2, axis=1),
            np.sum((right_projection - right_uv) ** 2, axis=1),
        ]
    )
    return h, z, float(np.sqrt(np.mean(squared)))


class EdgeReconstructor:
    """首次联合标定四自由度外参，之后固定外参做线性重建。"""

    def __init__(
        self,
        K: np.ndarray,
        distortion: np.ndarray,
        s1: float,
        s2: float,
        *,
        sample_count: int = 100,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.sample_count = int(sample_count)
        self.rotation_vector = np.zeros(3, dtype=np.float64)
        self.tx = 0.0
        self.calibrated = False

    def process(
        self,
        segments: Sequence[np.ndarray],
        u0: float,
    ) -> ReconstructionResult:
        left_raw, right_raw = split_edge_segments(segments, u0)
        left_ordered = _aggregate_edge_by_v(left_raw)
        right_ordered = _aggregate_edge_by_v(right_raw)
        if left_ordered.shape[0] < 4 or right_ordered.shape[0] < 4:
            raise ValueError("左右边缘点不足，无法进行联合优化")

        left_curve = _undistort_pixels(left_ordered, self.K, self.distortion)
        right_curve = _undistort_pixels(right_ordered, self.K, self.distortion)
        left_uv = _resample_polyline(left_curve, self.sample_count)
        right_dense_uv = _resample_polyline(right_curve, 4 * self.sample_count)
        return self.process_curves(left_uv,right_dense_uv)

    def process_curves(
        self,
        left_uv: np.ndarray,
        right_dense_uv: np.ndarray,
    ) -> ReconstructionResult:
        """处理已经无畸变并固定重采样的左右边缘曲线。"""
        left_uv=np.asarray(left_uv,dtype=np.float64).reshape(-1,2)
        right_dense_uv=np.asarray(right_dense_uv,dtype=np.float64).reshape(-1,2)
        if left_uv.shape!=(self.sample_count,2) \
                or right_dense_uv.shape!=(4*self.sample_count,2) \
                or not np.isfinite(left_uv).all() \
                or not np.isfinite(right_dense_uv).all():
            raise ValueError("GPU 左右边缘曲线尺寸或数值无效")
        if not self.calibrated:
            right_uv = _resample_polyline(right_dense_uv, self.sample_count)
            initial_h, initial_z = _initial_shared_curve(
                left_uv,
                right_uv,
                self.K,
                self.s1,
                self.s2,
                np.zeros(3, dtype=np.float64),
                0.0,
            )
            rotation, tx, h, z, first_rms = _optimize_shared_curve(
                left_uv,
                right_uv,
                self.K,
                self.s1,
                self.s2,
                np.zeros(3, dtype=np.float64),
                0.0,
                initial_h,
                initial_z,
            )
            right_indices = _monotone_right_matches(
                left_uv,
                right_dense_uv,
                self.K,
                self.s1,
                self.s2,
                rotation,
                tx,
            )
            matched_right_uv = right_dense_uv[right_indices]
            second_rotation, second_tx, second_h, second_z, second_rms = (
                _optimize_shared_curve(
                    left_uv,
                    matched_right_uv,
                    self.K,
                    self.s1,
                    self.s2,
                    rotation,
                    tx,
                    h,
                    z,
                )
            )
            if second_rms <= first_rms:
                right_uv = matched_right_uv
                rotation, tx, h, z, rms = (
                    second_rotation,
                    second_tx,
                    second_h,
                    second_z,
                    second_rms,
                )
            else:
                # 错误外参下的 Y/Z 最近邻也可能形成错误匹配；只接受能改善
                # 最终重投影一致性的动态规划结果。
                rms = first_rms
            self.rotation_vector = rotation.copy()
            self.tx = tx
            self.calibrated = True
        else:
            rotation = self.rotation_vector
            tx = self.tx
            right_indices = _monotone_right_matches(
                left_uv,
                right_dense_uv,
                self.K,
                self.s1,
                self.s2,
                rotation,
                tx,
            )
            right_uv = right_dense_uv[right_indices]
            h, z, rms = _solve_smoothed_shared_curve(
                left_uv,
                right_uv,
                self.K,
                self.s1,
                self.s2,
                rotation,
                tx,
                smooth_lambda=1.0,
            )
        left_xyz = np.column_stack([np.full(self.sample_count, self.s2), h, z])
        right_xyz = np.column_stack([np.full(self.sample_count, self.s1), h, z])
        return ReconstructionResult(
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            left_uv=_distort_pixels(left_uv, self.K, self.distortion),
            right_uv=_distort_pixels(right_uv, self.K, self.distortion),
            rotation_vector=rotation,
            tx=tx,
            reprojection_rms_px=rms,
        )


class EdgePointCloudVisualizer:
    """非阻塞 Open3D 点云窗口，供实时循环更新。"""

    def __init__(self, *, window_name: str = "Edge point cloud") -> None:
        self.window_name = window_name
        self._vis = None
        self._pcd: object | None = None
        self._pcd_fill: object | None = None
        self._lines: object | None = None
        self._frame: object | None = None
        self._closed = False
        self._geometry_added = False

    def _ensure_window(self) -> bool:
        if self._closed:
            return False
        if self._vis is not None:
            return True
        try:
            import open3d as o3d
        except ImportError:
            return False

        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(window_name=self.window_name, width=960, height=720):
            self._vis = None
            self._closed = True
            return False

        self._pcd = o3d.geometry.PointCloud()
        self._pcd_fill = o3d.geometry.PointCloud()
        self._lines = o3d.geometry.LineSet()
        # 坐标系稍后按点云尺度创建，避免空几何把相机锁在原点。
        self._frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        opt = self._vis.get_render_option()
        if opt is not None:
            # Legacy Visualizer 默认灯光会让坐标轴网格产生高光；点云颜色本身
            # 已携带语义，因此使用完全无光照的哑光显示。
            opt.light_on = False
            opt.point_size = 4.0
            opt.background_color = np.asarray([0.025, 0.027, 0.03])
            opt.line_width = 1.0
        return True

    @staticmethod
    def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        extent = np.maximum(maxs - mins, 1e-3)
        return center, extent

    def _sync_frame(self, center: np.ndarray, extent: np.ndarray) -> None:
        import open3d as o3d

        assert self._vis is not None and self._frame is not None
        size = max(float(np.max(extent)) * 0.25, 1.0)
        origin = (center - 0.5 * extent).astype(np.float64)
        new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
            origin=origin.tolist(),
        )
        self._frame.vertices = new_frame.vertices
        self._frame.triangles = new_frame.triangles
        self._frame.vertex_colors = new_frame.vertex_colors
        if self._geometry_added:
            self._vis.update_geometry(self._frame)

    def _fit_view(self, points: np.ndarray) -> None:
        """按点云包围盒重置相机，保证两侧边缘都落在视野内。"""
        if self._vis is None or points.shape[0] == 0:
            return

        center, extent = self._bounds(points)
        self._sync_frame(center, extent)

        # 先按全部几何重置包围盒，再微调到更利于观察左右边的视角。
        self._vis.reset_view_point(True)
        ctr = self._vis.get_view_control()
        ctr.set_lookat(center.tolist())
        # 从斜前方看向 -Z，同时带一点俯视，便于看到 x=±s 两条边。
        ctr.set_front([0.35, -0.45, -0.82])
        ctr.set_up([0.0, -1.0, 0.0])
        # Open3D zoom：数值越小越远；按对角线粗略归一到舒适尺度。
        ctr.set_zoom(0.55)

    def update(
        self,
        left_xyz: np.ndarray,
        right_xyz: np.ndarray,
        *,
        fill_xyz: np.ndarray | None = None,
        line_points: np.ndarray | None = None,
        line_indices: np.ndarray | None = None,
    ) -> bool:
        """更新点云与左右对应截面连线；窗口被关闭时返回 False。"""
        if not self._ensure_window():
            return False

        import open3d as o3d

        left_xyz = np.asarray(left_xyz, dtype=np.float64).reshape(-1, 3)
        right_xyz = np.asarray(right_xyz, dtype=np.float64).reshape(-1, 3)
        fill_xyz = (
            np.asarray(fill_xyz, dtype=np.float64).reshape(-1, 3)
            if fill_xyz is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        line_points_arr = (
            np.asarray(line_points, dtype=np.float64).reshape(-1, 3)
            if line_points is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        line_indices_arr = (
            np.asarray(line_indices, dtype=np.int32).reshape(-1, 2)
            if line_indices is not None
            else np.zeros((0, 2), dtype=np.int32)
        )

        edge_parts = [p for p in (left_xyz, right_xyz) if p.shape[0]]
        edge_points = (
            np.concatenate(edge_parts, axis=0)
            if edge_parts
            else np.zeros((0, 3), dtype=np.float64)
        )
        view_parts = [p for p in (edge_points, fill_xyz, line_points_arr) if p.shape[0]]
        view_points = (
            np.concatenate(view_parts, axis=0)
            if view_parts
            else np.zeros((0, 3), dtype=np.float64)
        )

        if view_points.shape[0] == 0:
            if not self._vis.poll_events():
                self.close()
                return False
            self._vis.update_renderer()
            return True

        colors = np.zeros((edge_points.shape[0], 3), dtype=np.float64)
        n_left = left_xyz.shape[0]
        if n_left:
            colors[:n_left] = (0.10, 0.32, 0.62)
        if right_xyz.shape[0]:
            colors[n_left:] = (0.68, 0.25, 0.10)

        assert (
            self._pcd is not None
            and self._pcd_fill is not None
            and self._lines is not None
            and self._vis is not None
        )
        self._pcd.points = o3d.utility.Vector3dVector(edge_points)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        self._pcd_fill.points = o3d.utility.Vector3dVector(fill_xyz)
        if fill_xyz.shape[0]:
            self._pcd_fill.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray([[0.34, 0.48, 0.22]]), (fill_xyz.shape[0], 1))
            )
        else:
            self._pcd_fill.colors = o3d.utility.Vector3dVector(
                np.zeros((0, 3), dtype=np.float64)
            )
        self._lines.points = o3d.utility.Vector3dVector(line_points_arr)
        self._lines.lines = o3d.utility.Vector2iVector(line_indices_arr)
        if line_indices_arr.shape[0]:
            self._lines.colors = o3d.utility.Vector3dVector(
                np.tile(
                    np.asarray([[0.52, 0.54, 0.56]]),
                    (line_indices_arr.shape[0], 1),
                )
            )
        else:
            self._lines.colors = o3d.utility.Vector3dVector(
                np.zeros((0, 3), dtype=np.float64)
            )

        if not self._geometry_added:
            center, extent = self._bounds(view_points)
            self._sync_frame(center, extent)
            self._vis.add_geometry(self._pcd, reset_bounding_box=True)
            self._vis.add_geometry(self._pcd_fill, reset_bounding_box=False)
            self._vis.add_geometry(self._lines, reset_bounding_box=False)
            self._vis.add_geometry(self._frame, reset_bounding_box=True)
            self._geometry_added = True
            self._fit_view(view_points)
        else:
            self._vis.update_geometry(self._pcd)
            self._vis.update_geometry(self._pcd_fill)
            self._vis.update_geometry(self._lines)

        if not self._vis.poll_events():
            self.close()
            return False
        self._vis.update_renderer()
        return True

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
        self._pcd = None
        self._pcd_fill = None
        self._lines = None
        self._frame = None
        self._closed = True
        self._geometry_added = False


def build_colored_surface_mesh(
    xyz_grid: np.ndarray,
    normal_displacement: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    depth_range_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把规则曲面三角化，并以 Turbo 色表编码有符号法向深度。"""
    grid = np.asarray(xyz_grid, dtype=np.float64)
    depth = np.asarray(normal_displacement, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 3 or min(grid.shape[:2]) < 2:
        raise ValueError("xyz_grid 必须是至少 2x2 的 rows x columns x 3 数组")
    if depth.shape != grid.shape[:2]:
        raise ValueError("normal_displacement 尺寸必须与 xyz_grid 前两维一致")
    if not np.isfinite(grid).all() or not np.isfinite(depth).all():
        raise ValueError("曲面网格或法向深度包含非有限数值")
    if not np.isfinite(depth_range_mm) or depth_range_mm <= 0:
        raise ValueError("depth_range_mm 必须是有限正数")
    valid = np.ones(grid.shape[:2], dtype=np.bool_)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=np.bool_)
        if supplied.shape != grid.shape[:2]:
            raise ValueError("valid_mask 尺寸必须与 xyz_grid 前两维一致")
        valid &= supplied

    rows, columns = grid.shape[:2]
    indices = np.arange(rows * columns, dtype=np.int32).reshape(rows, columns)
    first = np.stack(
        [indices[:-1, :-1], indices[1:, :-1], indices[1:, 1:]], axis=-1
    ).reshape(-1, 3)
    second = np.stack(
        [indices[:-1, :-1], indices[1:, 1:], indices[:-1, 1:]], axis=-1
    ).reshape(-1, 3)
    triangles = np.concatenate([first, second], axis=0)
    flat_valid = valid.reshape(-1)
    triangles = triangles[np.all(flat_valid[triangles], axis=1)]

    normalized = np.clip(
        0.5 + 0.5 * depth / float(depth_range_mm), 0.0, 1.0)
    color_indices = np.rint(255 * normalized).astype(np.uint8)
    # 恢复原先对比更强的 Turbo 顶点色。Open3D 灯光仍保持关闭，因此只有
    # 色表本身的亮度变化，不会出现镜面高光或随视角移动的反光。
    colors = cv2.applyColorMap(
        color_indices, cv2.COLORMAP_TURBO)[..., ::-1].reshape(-1, 3) / 255.0
    colors[~flat_valid] = (0.15, 0.15, 0.15)
    return grid.reshape(-1, 3), triangles, colors


class SurfaceMeshVisualizer:
    """非阻塞 Open3D 曲面窗口，几何表示形变、顶点颜色表示法向深度。"""

    def __init__(
        self,
        *,
        window_name: str = "Realtime deformation depth",
        depth_range_mm: float = 2.0,
        show_coordinate_frame: bool = False,
    ) -> None:
        if not np.isfinite(depth_range_mm) or depth_range_mm <= 0:
            raise ValueError("depth_range_mm 必须是有限正数")
        if not isinstance(show_coordinate_frame, bool):
            raise ValueError("show_coordinate_frame 必须是布尔值")
        self.window_name = window_name
        self.depth_range_mm = float(depth_range_mm)
        self.show_coordinate_frame = show_coordinate_frame
        self._vis = None
        self._mesh: object | None = None
        self._frame: object | None = None
        self._closed = False
        self._geometry_added = False

    def _ensure_window(self) -> bool:
        if self._closed:
            return False
        if self._vis is not None:
            return True
        try:
            import open3d as o3d
        except ImportError:
            return False
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(
            window_name=self.window_name, width=960, height=720
        ):
            self._vis = None
            self._closed = True
            return False
        self._mesh = o3d.geometry.TriangleMesh()
        self._frame = (o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                       if self.show_coordinate_frame else None)
        option = self._vis.get_render_option()
        if option is not None:
            # 顶点色直接表达位移；关闭 Open3D 默认灯光即可彻底去掉镜面高光和
            # 随观察角度移动的亮斑。
            option.light_on = False
            option.background_color = np.asarray([0.025, 0.027, 0.03])
            option.mesh_show_back_face = True
        return True

    @staticmethod
    def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        return 0.5 * (minimum + maximum), np.maximum(maximum - minimum, 1e-3)

    def _fit_view(self, points: np.ndarray) -> None:
        import open3d as o3d

        assert self._vis is not None
        center, extent = self._bounds(points)
        if self._frame is not None:
            size = max(float(np.max(extent)) * 0.25, 1.0)
            origin = (center - 0.5 * extent).astype(np.float64)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size, origin=origin.tolist()
            )
            self._frame.vertices = frame.vertices
            self._frame.triangles = frame.triangles
            self._frame.vertex_colors = frame.vertex_colors
        self._vis.reset_view_point(True)
        control = self._vis.get_view_control()
        control.set_lookat(center.tolist())
        control.set_front([0.35, -0.45, -0.82])
        control.set_up([0.0, -1.0, 0.0])
        control.set_zoom(0.55)

    def update(
        self,
        xyz_grid: np.ndarray,
        normal_displacement: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> bool:
        """更新三角曲面；窗口关闭或无法创建时返回 False。"""
        if not self._ensure_window():
            return False
        import open3d as o3d

        vertices, triangles, colors = build_colored_surface_mesh(
            xyz_grid,
            normal_displacement,
            valid_mask=valid_mask,
            depth_range_mm=self.depth_range_mm,
        )
        assert self._vis is not None and self._mesh is not None
        self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self._mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self._mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        if not self._geometry_added:
            self._vis.add_geometry(self._mesh, reset_bounding_box=True)
            if self._frame is not None:
                self._vis.add_geometry(self._frame, reset_bounding_box=False)
            self._geometry_added = True
            self._fit_view(vertices)
            if self._frame is not None:
                self._vis.update_geometry(self._frame)
        else:
            self._vis.update_geometry(self._mesh)
        if not self._vis.poll_events():
            self.close()
            return False
        self._vis.update_renderer()
        return True

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
        self._mesh = None
        self._frame = None
        self._closed = True
        self._geometry_added = False
