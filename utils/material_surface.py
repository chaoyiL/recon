"""配置驱动的固定材料曲面模板与运行时状态。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral,Real
from typing import Any

import jax
import numpy as np


MATERIAL_TEMPLATE_FORMAT_VERSION = 5
MATERIAL_COORDINATE_MODE = "configured_intrinsic_dimensions_v5"
MATERIAL_S_ZERO_ENDPOINTS = ("image_top","image_bottom")


def canonical_st(
    rows: int,
    columns: int,
    *,
    s_zero_endpoint: str = "image_top",
    dtype: np.dtype[Any] = np.float32,
) -> np.ndarray:
    """生成固定材料坐标；s=0 可配置为图像上端或下端。"""
    if rows<2 or columns<2:
        raise ValueError("材料网格至少需要 2x2")
    if s_zero_endpoint not in MATERIAL_S_ZERO_ENDPOINTS:
        raise ValueError("s_zero_endpoint 必须是 image_top 或 image_bottom")
    s_values=np.linspace(0.,1.,rows,dtype=dtype)
    if s_zero_endpoint=="image_bottom":
        s_values=s_values[::-1]
    s,t=np.meshgrid(
        s_values,np.linspace(0.,1.,columns,dtype=dtype),indexing="ij")
    return np.stack([s,t],axis=-1)


def array_sha256(values: np.ndarray) -> str:
    array=np.ascontiguousarray(values)
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape,np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def camera_calibration_sha256(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> str:
    values=np.concatenate([
        np.asarray(camera_matrix,np.float64).reshape(-1),
        np.asarray(distortion,np.float64).reshape(-1)])
    return array_sha256(values)


def material_reconstruction_sha256(
    *,
    s1: float,
    s2: float,
    geometry_rows: int,
    geometry_columns: int,
    bend_direction: str,
    uv_boundary_smooth_lambda: float,
    uv_boundary_huber_delta_px: float,
    side_edge_exclusion_ratio: float = 0.,
) -> str:
    payload={
        "global_reconstruction":"full_interval_without_endpoint_inference_v6",
        "s1":float(s1),"s2":float(s2),
        "geometry_rows":int(geometry_rows),
        "geometry_columns":int(geometry_columns),
        "bend_direction":str(bend_direction),
        "uv_boundary_smooth_lambda":float(uv_boundary_smooth_lambda),
        "uv_boundary_huber_delta_px":float(uv_boundary_huber_delta_px),
        "side_edge_exclusion_ratio":float(side_edge_exclusion_ratio),
    }
    encoded=json.dumps(
        payload,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MaterialSurfaceTemplate:
    """由配置尺寸确定的材料拓扑；不包含任何相机采集结果。"""

    width_mm: float
    length_mm: float
    rows: int
    columns: int
    s_zero_endpoint: str
    camera_sha256: str
    reconstruction_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value,Real) or isinstance(value,bool)
            or not np.isfinite(float(value)) or float(value)<=0
            for value in (self.width_mm,self.length_mm)
        ):
            raise ValueError("材料模板宽度和长度必须是有限正数")
        if any(
            not isinstance(value,Integral) or isinstance(value,bool)
            or int(value)<2 for value in (self.rows,self.columns)
        ):
            raise ValueError("材料模板网格至少需要 2x2")
        if self.s_zero_endpoint not in MATERIAL_S_ZERO_ENDPOINTS:
            raise ValueError("s_zero_endpoint 必须是 image_top 或 image_bottom")
        for name,digest in (
            ("camera_sha256",self.camera_sha256),
            ("reconstruction_sha256",self.reconstruction_sha256),
        ):
            if len(digest)!=64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} 必须是 SHA-256")

    @property
    def st(self) -> np.ndarray:
        return canonical_st(
            int(self.rows),int(self.columns),
            s_zero_endpoint=self.s_zero_endpoint)

    @property
    def segment_lengths_mm(self) -> np.ndarray:
        return np.full(
            int(self.rows)-1,float(self.length_mm)/(int(self.rows)-1),
            np.float32)

    @property
    def x_coordinates_mm(self) -> np.ndarray:
        half_width=float(self.width_mm)/2
        return np.linspace(
            -half_width,half_width,int(self.columns),dtype=np.float32)

    @property
    def reference_curve_yz(self) -> np.ndarray:
        """仅用于首帧前的无效占位；首个有效几何观测会整体替换它。"""
        y=np.linspace(
            -float(self.length_mm)/2,float(self.length_mm)/2,
            int(self.rows),dtype=np.float32)
        return np.stack([y,np.full_like(y,100.)],axis=-1)

    @property
    def reference_angles_rad(self) -> np.ndarray:
        return np.zeros(int(self.rows)-1,np.float32)

    @property
    def total_length_mm(self) -> float:
        return float(self.length_mm)

    @property
    def sha256(self) -> str:
        payload={
            "format_version":MATERIAL_TEMPLATE_FORMAT_VERSION,
            "coordinate_mode":MATERIAL_COORDINATE_MODE,
            "width_mm":float(self.width_mm),
            "length_mm":float(self.length_mm),
            "rows":int(self.rows),
            "columns":int(self.columns),
            "s_zero_endpoint":self.s_zero_endpoint,
            "camera_sha256":self.camera_sha256,
            "reconstruction_sha256":self.reconstruction_sha256,
        }
        encoded=json.dumps(
            payload,sort_keys=True,separators=(",",":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaterialSurfaceState:
    """JAX 可传递的固定拓扑运行时状态。"""

    curve_yz: Any
    angles_rad: Any
    xyz: Any
    uv: Any
    camera_depth: Any
    visible: Any
    observable_rows: Any
    left_uv_error: Any
    right_uv_error: Any
    valid: Any
    tracking_accepted: Any
    reprojection_rms_px: Any
    visible_fraction: Any
    matching_confidence: Any

    def tree_flatten(self):
        return (
            self.curve_yz,self.angles_rad,self.xyz,self.uv,
            self.camera_depth,self.visible,self.observable_rows,
            self.left_uv_error,self.right_uv_error,self.valid,
            self.tracking_accepted,self.reprojection_rms_px,
            self.visible_fraction,self.matching_confidence,
        ),None

    @classmethod
    def tree_unflatten(cls,auxiliary,children):
        del auxiliary
        return cls(*children)
