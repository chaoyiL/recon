"""曲率感知的视触觉局部形变重建。

数值求解始终在整体重建产生的原生规则曲面网格上执行。颜色残差可以直接以
同形网格传入，也可以保留为相机全分辨率图像，并通过每个曲面顶点的 UV 坐标
双线性采样。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from utils.config import (BACKGROUND_METHODS,parse_background_method,
                          resolve_method_path)
from utils.jax_reconstruction import SURFACE_RECONSTRUCTION_PIPELINE_VERSION


@dataclass(frozen=True)
class NormalCalibration:
    """颜色残差到局部坡度的完整查找表。"""

    slopes: np.ndarray
    variances: np.ndarray
    color_min: np.ndarray
    color_max: np.ndarray
    sigma_ref2: float
    original_valid: np.ndarray | None = None
    sample_counts: np.ndarray | None = None
    residual_method: str | None = None
    background_method: str | None = None
    background_model_sha256: str | None = None
    reconstruction_pipeline: str = SURFACE_RECONSTRUCTION_PIPELINE_VERSION
    curve_convexity: str = "none"

    def __post_init__(self) -> None:
        slopes=np.asarray(self.slopes,np.float32)
        variances=np.asarray(self.variances,np.float32)
        color_min=np.asarray(self.color_min,np.float32)
        color_max=np.asarray(self.color_max,np.float32)
        if slopes.ndim!=4 or slopes.shape[-1]!=2 \
                or slopes.shape[:3]!=(slopes.shape[0],)*3:
            raise ValueError("slopes 必须是 NxNxNx2 立方查找表")
        if variances.shape!=slopes.shape[:3]:
            raise ValueError("variances 尺寸必须与 slopes 的前三维一致")
        if color_min.shape!=(3,) or color_max.shape!=(3,) \
                or np.any(color_max<=color_min) or np.any(color_min>0) \
                or np.any(color_max<0):
            raise ValueError(
                "color_min/color_max 必须是包含零点的有效 RGB 三通道范围")
        if not np.isfinite(slopes).all() or not np.isfinite(variances).all() \
                or np.any(variances<0) or not np.isfinite(self.sigma_ref2) \
                or self.sigma_ref2<=0:
            raise ValueError("法向标定模型包含无效数值")
        if self.original_valid is not None \
                and np.asarray(self.original_valid).shape!=variances.shape:
            raise ValueError("original_valid 尺寸无效")
        if self.sample_counts is not None \
                and np.asarray(self.sample_counts).shape!=variances.shape:
            raise ValueError("sample_counts 尺寸无效")
        if self.residual_method is not None \
                and (not isinstance(self.residual_method,str)
                     or not self.residual_method):
            raise ValueError("residual_method 必须是非空字符串或 None")
        if self.background_method is not None \
                and self.background_method not in BACKGROUND_METHODS:
            raise ValueError("background_method 必须是有效背景方法或 None")
        if self.background_model_sha256 is not None:
            digest=self.background_model_sha256
            if not isinstance(digest,str) or len(digest)!=64 \
                    or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(
                    "background_model_sha256 必须是 64 位小写十六进制字符串或 None")
        if self.reconstruction_pipeline!=SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
            raise ValueError("法向 LUT 的整体重建链元数据无效")
        if self.curve_convexity not in {"none","increasing","decreasing"}:
            raise ValueError("法向 LUT 的曲线凸性元数据无效")
        object.__setattr__(self,"slopes",slopes)
        object.__setattr__(self,"variances",variances)
        object.__setattr__(self,"color_min",color_min)
        object.__setattr__(self,"color_max",color_max)
        object.__setattr__(self,"sigma_ref2",float(self.sigma_ref2))

    @property
    def size(self) -> int:
        return int(self.slopes.shape[0])

    def save(self,path: str | Path,**metadata: Any) -> Path:
        output=Path(path).expanduser()
        output.parent.mkdir(parents=True,exist_ok=True)
        values: dict[str,Any]={
            "format_version":np.asarray(3,np.int32),
            "color_residual_mode":np.asarray("signed"),
            "slopes":self.slopes,
            "variances":self.variances,
            "color_min":self.color_min,
            "color_max":self.color_max,
            "sigma_ref2":np.asarray(self.sigma_ref2,np.float32),
            "reconstruction_pipeline":np.asarray(self.reconstruction_pipeline),
            "curve_convexity":np.asarray(self.curve_convexity),
        }
        if self.original_valid is not None:
            values["original_valid"]=np.asarray(self.original_valid,np.bool_)
        if self.sample_counts is not None:
            values["sample_counts"]=np.asarray(self.sample_counts,np.int32)
        if self.residual_method is not None:
            values["residual_method"]=np.asarray(self.residual_method)
        if self.background_method is not None:
            values["background_method"]=np.asarray(self.background_method)
        if self.background_model_sha256 is not None:
            values["background_model_sha256"]=np.asarray(
                self.background_model_sha256)
        values.update(metadata)
        temporary=output.with_suffix(output.suffix+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**values)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls,path: str | Path) -> "NormalCalibration":
        source=Path(path).expanduser()
        with np.load(source,allow_pickle=False) as data:
            version=int(data["format_version"]) if "format_version" in data else 0
            if version==1:
                raise ValueError(
                    "旧法向 LUT 使用单边正色差；请重新运行 calibrate-norm")
            if version==2:
                raise ValueError(
                    "旧法向 LUT 未记录完整实时 JAX/凸性重建语义；"
                    "请重新运行 calibrate-norm")
            if version!=3:
                raise ValueError(f"不支持的法向标定模型版本: {version}")
            if "color_residual_mode" not in data \
                    or str(data["color_residual_mode"])!="signed":
                raise ValueError("法向 LUT 缺少有符号色差语义；请重新运行 calibrate-norm")
            return cls(
                slopes=data["slopes"],variances=data["variances"],
                color_min=data["color_min"],color_max=data["color_max"],
                sigma_ref2=float(data["sigma_ref2"]),
                original_valid=(data["original_valid"]
                                if "original_valid" in data else None),
                sample_counts=(data["sample_counts"]
                               if "sample_counts" in data else None),
                residual_method=(str(data["residual_method"])
                                 if "residual_method" in data else None),
                background_method=(str(data["background_method"])
                                   if "background_method" in data else None),
                background_model_sha256=(str(data["background_model_sha256"])
                                         if "background_model_sha256" in data
                                         else None),
                reconstruction_pipeline=str(data["reconstruction_pipeline"]),
                curve_convexity=str(data["curve_convexity"]),
            )


@dataclass(frozen=True)
class LocalReconstructionResult:
    xyz_out: np.ndarray
    displacement: np.ndarray
    displacement_vectors: np.ndarray
    reference_normals: np.ndarray
    slopes: np.ndarray
    observed_normals: np.ndarray
    curvature_correction: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    boundary_mask: np.ndarray
    solver_istop: int
    solver_iterations: int
    solver_residual_norm: float
    trusted_no_contact_mask: np.ndarray | None = None

    def save(self,path: str | Path) -> Path:
        output=Path(path).expanduser()
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(output.suffix+".tmp")
        values={
            "xyz_out":self.xyz_out,"displacement":self.displacement,
            "displacement_vectors":self.displacement_vectors,
            "reference_normals":self.reference_normals,"slopes":self.slopes,
            "observed_normals":self.observed_normals,
            "curvature_correction":self.curvature_correction,
            "confidence":self.confidence,"valid_mask":self.valid_mask,
            "boundary_mask":self.boundary_mask,
            "solver_istop":np.asarray(self.solver_istop,np.int32),
            "solver_iterations":np.asarray(self.solver_iterations,np.int32),
            "solver_residual_norm":np.asarray(
                self.solver_residual_norm,np.float64),
        }
        if self.trusted_no_contact_mask is not None:
            values["trusted_no_contact_mask"]=np.asarray(
                self.trusted_no_contact_mask,np.bool_)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**values)
        temporary.replace(output)
        return output


@dataclass(frozen=True)
class LocalReconstructionSettings:
    input_file: Path
    calibration_file: Path
    output_file: Path
    lsmr_atol: float
    lsmr_btol: float
    lsmr_max_iterations: int | None
    zero_color_inner_radius: float
    zero_color_outer_radius: float
    no_contact_constraints_enabled: bool
    trusted_no_contact_confidence: float
    displacement_zero_lambda_per_mm2: float


@dataclass(frozen=True)
class NoContactResidualModel:
    """启动无触碰帧得到的位置相关残差中心和鲁棒通道尺度。"""

    center: np.ndarray
    channel_scale: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        center=np.asarray(self.center,np.float32)
        scale=np.asarray(self.channel_scale,np.float32)
        valid=np.asarray(self.valid_mask,np.bool_)
        if center.ndim!=3 or center.shape[-1]!=3 \
                or valid.shape!=center.shape[:2] or scale.shape!=(3,) \
                or not np.isfinite(center).all() or not np.isfinite(scale).all() \
                or np.any(scale<=0):
            raise ValueError("无接触残差模型尺寸或数值无效")
        object.__setattr__(self,"center",center)
        object.__setattr__(self,"channel_scale",scale)
        object.__setattr__(self,"valid_mask",valid)


@dataclass(frozen=True)
class NoContactConstraintSettings:
    enabled: bool
    minimum_startup_valid_fraction: float
    minimum_channel_scale: float
    trusted_score_threshold: float
    contact_guard_score_threshold: float
    contact_guard_radius_pixels: int
    surface_edge_margin_pixels: int
    slope_confidence: float
    displacement_zero_lambda_per_mm2: float


def parse_no_contact_constraints(config: object) -> NoContactConstraintSettings:
    """解析可信无接触门控和绝对零位移软约束。"""
    if not isinstance(config,dict):
        raise ValueError("local_reconstruction 必须是映射")
    raw=config.get("no_contact_constraints",{})
    if raw is None:
        raw={}
    if not isinstance(raw,dict):
        raise ValueError(
            "local_reconstruction.no_contact_constraints 必须是映射")
    fields={
        "enabled","minimum_startup_valid_fraction","minimum_channel_scale",
        "trusted_score_threshold","contact_guard_score_threshold",
        "contact_guard_radius_pixels","surface_edge_margin_pixels",
        "slope_confidence","displacement_zero_lambda_per_mm2",
    }
    unknown=set(raw)-fields
    if unknown:
        raise ValueError(
            "local_reconstruction.no_contact_constraints 包含未知字段: "+
            ", ".join(sorted(unknown)))
    enabled=raw.get("enabled",False)
    minimum_fraction=float(raw.get("minimum_startup_valid_fraction",.8))
    minimum_scale=float(raw.get("minimum_channel_scale",.005))
    trusted=float(raw.get("trusted_score_threshold",2.5))
    guard=float(raw.get("contact_guard_score_threshold",5.))
    guard_radius=raw.get("contact_guard_radius_pixels",20)
    edge_margin=raw.get("surface_edge_margin_pixels",4)
    slope_confidence=float(raw.get("slope_confidence",1.))
    zero_lambda=float(raw.get("displacement_zero_lambda_per_mm2",.25))
    if not isinstance(enabled,bool) \
            or not 0<minimum_fraction<=1 or minimum_scale<=0 \
            or trusted<=0 or guard<=trusted \
            or not isinstance(guard_radius,int) or isinstance(guard_radius,bool) \
            or guard_radius<0 \
            or not isinstance(edge_margin,int) or isinstance(edge_margin,bool) \
            or edge_margin<0 or not 0<slope_confidence<=1 or zero_lambda<0 \
            or not np.isfinite([
                minimum_fraction,minimum_scale,trusted,guard,
                slope_confidence,zero_lambda]).all():
        raise ValueError("no_contact_constraints 参数无效")
    return NoContactConstraintSettings(
        enabled=enabled,
        minimum_startup_valid_fraction=minimum_fraction,
        minimum_channel_scale=minimum_scale,
        trusted_score_threshold=trusted,
        contact_guard_score_threshold=guard,
        contact_guard_radius_pixels=guard_radius,
        surface_edge_margin_pixels=edge_margin,
        slope_confidence=slope_confidence,
        displacement_zero_lambda_per_mm2=zero_lambda)


def fit_no_contact_residual_model(
    residual_samples: np.ndarray,
    valid_masks: np.ndarray,
    *,
    minimum_valid_fraction: float = .8,
    minimum_channel_scale: float = .005,
) -> NoContactResidualModel:
    """从启动无触碰帧拟合规范曲面残差中心及全局鲁棒通道尺度。"""
    samples=np.asarray(residual_samples,np.float64)
    masks=np.asarray(valid_masks,np.bool_)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3] or samples.shape[0]<2 \
            or not np.isfinite(samples).all() \
            or not 0<minimum_valid_fraction<=1 \
            or not np.isfinite(minimum_channel_scale) \
            or minimum_channel_scale<=0:
        raise ValueError("无接触残差样本、有效域或拟合参数无效")
    counts=np.sum(masks,axis=0)
    required=int(np.ceil(minimum_valid_fraction*samples.shape[0]))
    model_valid=counts>=required
    masked=np.ma.array(
        samples,mask=np.broadcast_to(~masks[...,None],samples.shape))
    center=np.ma.median(masked,axis=0).filled(0.)
    deviation=samples-center[None]
    scale=np.asarray([
        1.4826*np.median(np.abs(deviation[...,channel][masks]))
        if np.any(masks) else 0.
        for channel in range(3)],np.float64)
    scale=np.maximum(scale,float(minimum_channel_scale))
    center=np.where(model_valid[...,None],center,0.)
    return NoContactResidualModel(
        center.astype(np.float32),scale.astype(np.float32),model_valid)


def classify_trusted_no_contact(
    color_residual: np.ndarray,
    valid_mask: np.ndarray,
    model: NoContactResidualModel,
    *,
    trusted_score_threshold: float,
    contact_guard_score_threshold: float,
    contact_guard_radius_pixels: int = 0,
    surface_edge_margin_pixels: int = 0,
) -> tuple[np.ndarray,np.ndarray]:
    """以无接触统计分数和强接触邻域保护生成保守可信无接触掩膜。"""
    colors=np.asarray(color_residual,np.float32)
    valid=np.asarray(valid_mask,np.bool_)
    if colors.shape!=model.center.shape or valid.shape!=colors.shape[:2] \
            or trusted_score_threshold<=0 \
            or contact_guard_score_threshold<=trusted_score_threshold \
            or contact_guard_radius_pixels<0 or surface_edge_margin_pixels<0:
        raise ValueError("无接触分类输入尺寸或阈值无效")
    score=np.sqrt(np.sum(
        ((colors-model.center)/model.channel_scale)**2,axis=-1))
    input_valid=valid&np.isfinite(score)
    usable=input_valid&model.valid_mask
    if surface_edge_margin_pixels:
        radius=surface_edge_margin_pixels
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        usable=cv2.erode(
            usable.astype(np.uint8),kernel,iterations=1,
            borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    guarded=input_valid&(score>=contact_guard_score_threshold)
    if contact_guard_radius_pixels:
        radius=contact_guard_radius_pixels
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        guarded=cv2.dilate(
            guarded.astype(np.uint8),kernel,iterations=1,
            borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    trusted=usable&(score<=trusted_score_threshold)&~guarded
    return trusted,np.where(input_valid,score,0).astype(np.float32)


def parse_zero_color_protection(config: object) -> tuple[float,float]:
    """解析线性 dRGB 零点附近的坡度保护半径。"""
    if not isinstance(config,dict):
        raise ValueError("local_reconstruction 必须是映射")
    raw=config.get("zero_color_protection",{})
    if raw is None:
        raw={}
    if not isinstance(raw,dict):
        raise ValueError(
            "local_reconstruction.zero_color_protection 必须是映射")
    unknown=set(raw)-{"inner_radius","outer_radius"}
    if unknown:
        raise ValueError(
            "local_reconstruction.zero_color_protection 包含未知字段: "+
            ", ".join(sorted(unknown)))
    inner=float(raw.get("inner_radius",0.))
    outer=float(raw.get("outer_radius",0.))
    if not np.isfinite(inner) or not np.isfinite(outer) or inner<0 \
            or outer<0 or (outer==0 and inner!=0) \
            or (outer>0 and inner>=outer):
        raise ValueError(
            "zero_color_protection 要求 0 <= inner_radius < outer_radius；"
            "inner_radius=outer_radius=0 表示关闭")
    return inner,outer


def _configured_path(
    cli_value: str | None,
    configured_value: object,
    *,
    config_base: Path,
    name: str,
) -> Path:
    if cli_value is not None:
        if not cli_value:
            raise ValueError(f"{name} 命令行路径不能为空")
        return Path(cli_value).expanduser()
    if not isinstance(configured_value,str) or not configured_value:
        raise ValueError(
            f"缺少 local_reconstruction.{name}；请在 config.yaml 配置或用命令行覆盖")
    path=Path(configured_value).expanduser()
    return path if path.is_absolute() else config_base/path


def _optional_configured_path(
    cli_value: str | None,
    configured_value: object,
    *,
    config_base: Path,
    name: str,
) -> Path | None:
    """解析可选路径；显式传入空字符串仍视为配置错误。"""
    if cli_value is not None:
        if not cli_value:
            raise ValueError(f"{name} 命令行路径不能为空")
        return Path(cli_value).expanduser()
    if configured_value is None:
        return None
    if not isinstance(configured_value,str) or not configured_value:
        raise ValueError(f"local_reconstruction.{name} 必须是非空路径或 null")
    path=Path(configured_value).expanduser()
    return path if path.is_absolute() else config_base/path


def load_local_reconstruction_settings(
    config_path: str | Path,
    *,
    input_override: str | None = None,
    calibration_override: str | None = None,
    output_override: str | None = None,
    lsmr_atol_override: float | None = None,
    lsmr_btol_override: float | None = None,
    lsmr_max_iterations_override: int | None = None,
) -> LocalReconstructionSettings:
    """读取独立局部重建配置；命令行值优先于 YAML。"""
    source=Path(config_path).expanduser()
    with source.open("r",encoding="utf-8") as stream:
        all_config=yaml.safe_load(stream)
    if not isinstance(all_config,dict):
        raise ValueError("config.yaml 顶层必须是映射")
    raw=all_config.get("local_reconstruction")
    if not isinstance(raw,dict):
        raise ValueError("config.yaml 缺少 local_reconstruction 配置段")
    input_file=_optional_configured_path(
        input_override,raw.get("input_file"),config_base=source.parent,
        name="input_file")
    if input_file is None:
        raise ValueError(
            "recon-local 是离线 NPZ 求解器，必须用 --input 指定输入文件；"
            "实时重建请运行 uv run recon.py")
    if calibration_override is not None:
        calibration_file=_configured_path(
            calibration_override,None,config_base=source.parent,
            name="calibration_file")
    else:
        lightfield=all_config.get("lightfield",{})
        method=(parse_background_method(lightfield)
                if isinstance(lightfield,dict) else "physical_residual")
        calibration_file=resolve_method_path(
            raw,method=method,mapping_key="calibration_files",
            legacy_key="calibration_file",base=source.parent,
            section_name="local_reconstruction")
    output_file=_optional_configured_path(
        output_override,raw.get("output_file"),config_base=source.parent,
        name="output_file")
    if output_file is None:
        output_file=input_file.with_name(
            input_file.stem+"_local_reconstruction.npz")
    atol=(float(raw.get("lsmr_atol",1e-7)) if lsmr_atol_override is None
          else float(lsmr_atol_override))
    btol=(float(raw.get("lsmr_btol",1e-7)) if lsmr_btol_override is None
          else float(lsmr_btol_override))
    maximum=(raw.get("lsmr_max_iterations")
             if lsmr_max_iterations_override is None
             else lsmr_max_iterations_override)
    if atol<=0 or btol<=0 or not np.isfinite(atol) or not np.isfinite(btol) \
            or (maximum is not None and
                (not isinstance(maximum,int) or isinstance(maximum,bool)
                 or maximum<1)):
        raise ValueError("local_reconstruction 的 LSMR 参数无效")
    zero_inner,zero_outer=parse_zero_color_protection(raw)
    no_contact=parse_no_contact_constraints(raw)
    return LocalReconstructionSettings(
        input_file=input_file,calibration_file=calibration_file,
        output_file=output_file,lsmr_atol=atol,lsmr_btol=btol,
        lsmr_max_iterations=maximum,
        zero_color_inner_radius=zero_inner,
        zero_color_outer_radius=zero_outer,
        no_contact_constraints_enabled=no_contact.enabled,
        trusted_no_contact_confidence=no_contact.slope_confidence,
        displacement_zero_lambda_per_mm2=(
            no_contact.displacement_zero_lambda_per_mm2))


def sample_full_resolution_residual(
    residual_linear_rgb: np.ndarray,
    uv: np.ndarray,
    valid_image: np.ndarray | None = None,
) -> tuple[np.ndarray,np.ndarray]:
    """按原生曲面 UV 从全分辨率线性 RGB 色差图双线性采样。"""
    residual=np.asarray(residual_linear_rgb,np.float32)
    coordinates=np.asarray(uv,np.float32)
    if residual.ndim!=3 or residual.shape[-1]!=3:
        raise ValueError("residual_linear_rgb 必须是 HxWx3")
    if coordinates.ndim!=3 or coordinates.shape[-1]!=2:
        raise ValueError("uv 必须是 rows x columns x 2")
    sampled=cv2.remap(
        residual,coordinates[...,0],coordinates[...,1],cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    height,width=residual.shape[:2]
    inside=((coordinates[...,0]>=0)&(coordinates[...,0]<=width-1)&
            (coordinates[...,1]>=0)&(coordinates[...,1]<=height-1))
    if valid_image is not None:
        source=np.asarray(valid_image)
        if source.shape!=residual.shape[:2]:
            raise ValueError("valid_image 尺寸必须与全分辨率色差图一致")
        sampled_valid=cv2.remap(
            (source>0).astype(np.float32),coordinates[...,0],coordinates[...,1],
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        # 双线性邻域必须全部有效，避免把边界外的零值混入颜色。
        inside &= sampled_valid>=1-1e-6
    sampled=np.where(inside[...,None],sampled,0).astype(np.float32)
    return sampled,inside


def lookup_slopes(
    color_residual: np.ndarray,
    calibration: NormalCalibration,
    valid_mask: np.ndarray | None = None,
    *,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    trusted_no_contact_mask: np.ndarray | None = None,
    trusted_no_contact_confidence: float = 1.,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """查询完整 LUT；零色差及可信无接触点的坡度归零。"""
    colors=np.asarray(color_residual,np.float32)
    if colors.ndim<1 or colors.shape[-1]!=3:
        raise ValueError("color_residual 最后一维必须是 RGB")
    inner=float(zero_color_inner_radius); outer=float(zero_color_outer_radius)
    if not np.isfinite(inner) or not np.isfinite(outer) or inner<0 \
            or outer<0 or (outer==0 and inner!=0) \
            or (outer>0 and inner>=outer) \
            or not np.isfinite(trusted_no_contact_confidence) \
            or not 0<trusted_no_contact_confidence<=1:
        raise ValueError(
            "零色差保护要求 0 <= inner_radius < outer_radius；"
            "二者均为 0 表示关闭")
    valid=np.all(np.isfinite(colors),axis=-1)
    if valid_mask is not None:
        supplied=np.asarray(valid_mask,dtype=np.bool_)
        if supplied.shape!=colors.shape[:-1]:
            raise ValueError("valid_mask 尺寸与颜色残差不一致")
        valid &= supplied
    if trusted_no_contact_mask is None:
        trusted=np.zeros(colors.shape[:-1],np.bool_)
    else:
        trusted=np.asarray(trusted_no_contact_mask,np.bool_)
        if trusted.shape!=colors.shape[:-1]:
            raise ValueError("trusted_no_contact_mask 尺寸与颜色残差不一致")
        trusted &= valid
    scale=calibration.color_max-calibration.color_min
    normalized=np.clip((colors-calibration.color_min)/scale,0,1)
    coordinate=normalized*(calibration.size-1)
    lower=np.floor(coordinate).astype(np.int32)
    upper=np.minimum(lower+1,calibration.size-1)
    fraction=coordinate-lower
    slopes=np.zeros((*colors.shape[:-1],2),np.float32)
    variance=np.zeros(colors.shape[:-1],np.float32)
    corner_values=[]
    corner_variances=[]
    corner_weights=[]
    for ar in (0,1):
        ir=np.where(ar,upper[...,0],lower[...,0])
        wr=np.where(ar,fraction[...,0],1-fraction[...,0])
        for ag in (0,1):
            ig=np.where(ag,upper[...,1],lower[...,1])
            wg=np.where(ag,fraction[...,1],1-fraction[...,1])
            for ab in (0,1):
                ib=np.where(ab,upper[...,2],lower[...,2])
                wb=np.where(ab,fraction[...,2],1-fraction[...,2])
                weight=(wr*wg*wb).astype(np.float32)
                value=calibration.slopes[ir,ig,ib]
                corner_values.append(value)
                corner_variances.append(calibration.variances[ir,ig,ib])
                corner_weights.append(weight)
                slopes += weight[...,None]*value
    for weight,value,node_variance in zip(
            corner_weights,corner_values,corner_variances,strict=True):
        variance += weight*(node_variance+np.sum((value-slopes)**2,axis=-1))
    confidence=1/(1+variance/calibration.sigma_ref2)
    if outer>0:
        # max(abs(dRGB)) 与强色差过滤的逐通道幅值定义一致；smoothstep 在
        # inner/outer 两端的一阶导数均为零，避免保护边界产生坡度折痕。
        intensity=np.max(np.abs(colors),axis=-1)
        transition=np.clip((intensity-inner)/(outer-inner),0,1)
        attenuation=transition**2*(3-2*transition)
        slopes *= attenuation[...,None]
    slopes=np.where(trusted[...,None],0,slopes)
    confidence=np.where(
        trusted,np.maximum(confidence,trusted_no_contact_confidence),confidence)
    slopes=np.where(valid[...,None],slopes,0).astype(np.float32)
    variance=np.where(valid,variance,0).astype(np.float32)
    confidence=np.where(valid,confidence,0).astype(np.float32)
    return slopes,variance,confidence


def _surface_geometry(
    xyz: np.ndarray,valid_mask: np.ndarray,
) -> dict[str,np.ndarray]:
    points=np.asarray(xyz,np.float64)
    valid=np.asarray(valid_mask,dtype=np.bool_)&np.all(np.isfinite(points),axis=-1)
    if points.ndim!=3 or points.shape[-1]!=3 or min(points.shape[:2])<3:
        raise ValueError("xyz 必须是至少 3x3 的规则曲面点阵")
    kernel=np.ones((3,3),np.uint8)
    interior=cv2.erode(
        valid.astype(np.uint8),kernel,iterations=1,
        borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    xu=np.zeros_like(points); xv=np.zeros_like(points)
    xuu=np.zeros_like(points); xvv=np.zeros_like(points); xuv=np.zeros_like(points)
    xu[1:-1,1:-1]=(points[1:-1,2:]-points[1:-1,:-2])/2
    xv[1:-1,1:-1]=(points[2:,1:-1]-points[:-2,1:-1])/2
    xuu[1:-1,1:-1]=(points[1:-1,2:]-2*points[1:-1,1:-1]
                     +points[1:-1,:-2])
    xvv[1:-1,1:-1]=(points[2:,1:-1]-2*points[1:-1,1:-1]
                     +points[:-2,1:-1])
    xuv[1:-1,1:-1]=(points[2:,2:]-points[2:,:-2]
                     -points[:-2,2:]+points[:-2,:-2])/4
    xu_norm=np.linalg.norm(xu,axis=-1)
    cross=np.cross(xu,xv); cross_norm=np.linalg.norm(cross,axis=-1)
    nonsingular=interior&(xu_norm>1e-9)&(cross_norm>1e-12)
    e1=xu/np.maximum(xu_norm[...,None],1e-12)
    normal=cross/np.maximum(cross_norm[...,None],1e-12)
    e2=np.cross(normal,e1)
    reverse=np.sum(e2*xv,axis=-1)<0
    e2=np.where(reverse[...,None],-e2,e2)
    normal=np.where(reverse[...,None],-normal,normal)
    g=np.empty((*points.shape[:2],2,2),np.float64)
    g[...,0,0]=np.sum(xu*xu,axis=-1)
    g[...,0,1]=g[...,1,0]=np.sum(xu*xv,axis=-1)
    g[...,1,1]=np.sum(xv*xv,axis=-1)
    det=g[...,0,0]*g[...,1,1]-g[...,0,1]**2
    nonsingular &= det>1e-14
    g_inverse=np.zeros_like(g)
    safe=np.maximum(det,1e-14)
    g_inverse[...,0,0]=g[...,1,1]/safe
    g_inverse[...,1,1]=g[...,0,0]/safe
    g_inverse[...,0,1]=g_inverse[...,1,0]=-g[...,0,1]/safe
    second=np.empty_like(g)
    second[...,0,0]=np.sum(normal*xuu,axis=-1)
    second[...,0,1]=second[...,1,0]=np.sum(normal*xuv,axis=-1)
    second[...,1,1]=np.sum(normal*xvv,axis=-1)
    jacobian=np.stack([xu,xv],axis=-1)
    frame=np.stack([e1,e2],axis=-1)
    # K = E^T J G^-1
    k_matrix=np.einsum(
        "...ca,...cb,...bd->...ad",frame,jacobian,g_inverse,optimize=True)
    return {
        "valid":nonsingular,"interior":interior,"boundary":valid&~interior,
        "normal":normal,"frame":frame,"jacobian":jacobian,"metric":g,
        "metric_inverse":g_inverse,"second":second,"k":k_matrix,
        "area":np.sqrt(np.maximum(det,0)),
    }


def reconstruct_local_surface(
    xyz: np.ndarray,
    color_residual_grid: np.ndarray,
    calibration: NormalCalibration,
    *,
    valid_mask: np.ndarray | None = None,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    trusted_no_contact_mask: np.ndarray | None = None,
    trusted_no_contact_confidence: float = 1.,
    displacement_zero_lambda_per_mm2: float = 0.,
    lsmr_atol: float = 1e-7,
    lsmr_btol: float = 1e-7,
    lsmr_max_iterations: int | None = None,
) -> LocalReconstructionResult:
    """在原生规则曲面网格上求解法向局部位移。"""
    points=np.asarray(xyz,np.float64)
    colors=np.asarray(color_residual_grid,np.float32)
    if colors.shape!=points.shape:
        raise ValueError("color_residual_grid 必须与 xyz 同为 rows x columns x 3")
    valid=np.all(np.isfinite(points),axis=-1)&np.all(np.isfinite(colors),axis=-1)
    if valid_mask is not None:
        supplied=np.asarray(valid_mask,dtype=np.bool_)
        if supplied.shape!=points.shape[:2]:
            raise ValueError("valid_mask 尺寸必须与 xyz 前两维一致")
        valid &= supplied
    if trusted_no_contact_mask is None:
        trusted_no_contact=np.zeros(points.shape[:2],np.bool_)
    else:
        trusted_no_contact=np.asarray(trusted_no_contact_mask,np.bool_)
        if trusted_no_contact.shape!=points.shape[:2]:
            raise ValueError("trusted_no_contact_mask 尺寸与 xyz 前两维不一致")
        trusted_no_contact &= valid
    if not np.isfinite(displacement_zero_lambda_per_mm2) \
            or displacement_zero_lambda_per_mm2<0:
        raise ValueError("displacement_zero_lambda_per_mm2 必须为非负有限数")
    slopes,_,confidence=lookup_slopes(
        colors,calibration,valid,
        zero_color_inner_radius=zero_color_inner_radius,
        zero_color_outer_radius=zero_color_outer_radius,
        trusted_no_contact_mask=trusted_no_contact,
        trusted_no_contact_confidence=trusted_no_contact_confidence)
    geometry=_surface_geometry(points,valid)
    equation_mask=geometry["valid"]&(confidence>0)
    unknown_mask=geometry["interior"]
    unknown_index=np.full(points.shape[:2],-1,np.int32)
    unknown_index[unknown_mask]=np.arange(np.count_nonzero(unknown_mask),dtype=np.int32)
    if not np.any(unknown_mask) or not np.any(equation_mask):
        normal=geometry["normal"].astype(np.float32)
        zeros=np.zeros(points.shape[:2],np.float32)
        zero_vectors=np.zeros_like(points,np.float32)
        return LocalReconstructionResult(
            xyz_out=points.astype(np.float32),displacement=zeros,
            displacement_vectors=zero_vectors,reference_normals=normal,
            slopes=np.zeros((*points.shape[:2],2),np.float32),
            observed_normals=normal,
            curvature_correction=zero_vectors,
            confidence=np.zeros(points.shape[:2],np.float32),valid_mask=valid,
            boundary_mask=valid.copy(),solver_istop=0,solver_iterations=0,
            solver_residual_norm=0.,
            trusted_no_contact_mask=trusted_no_contact)

    frame=geometry["frame"]
    tangent=np.einsum("...ca,...a->...c",frame,slopes,optimize=True)
    jacobian=geometry["jacobian"]
    g_inverse=geometry["metric_inverse"]
    second=geometry["second"]
    parameter=np.einsum(
        "...ab,...cb,...c->...a",g_inverse,jacobian,tangent,optimize=True)
    shape_parameter=np.einsum(
        "...ab,...b->...a",second,parameter,optimize=True)
    curvature=np.einsum(
        "...ca,...ab,...b->...c",jacobian,g_inverse,shape_parameter,
        optimize=True)
    gamma=np.einsum("...ca,...c->...a",frame,curvature,optimize=True)
    k_matrix=geometry["k"]

    matrix_rows=[]; matrix_columns=[]; matrix_values=[]; targets=[]
    equation_weights=[]; row_number=0
    rows,columns=points.shape[:2]
    for i,j in np.argwhere(equation_mask):
        center=unknown_index[i,j]
        if center<0:
            continue
        weight=float(geometry["area"][i,j]*confidence[i,j])
        if not np.isfinite(weight) or weight<=0:
            continue
        for component in range(2):
            entries=((i,j,float(gamma[i,j,component])),
                     (i,j+1,float(.5*k_matrix[i,j,component,0])),
                     (i,j-1,float(-.5*k_matrix[i,j,component,0])),
                     (i+1,j,float(.5*k_matrix[i,j,component,1])),
                     (i-1,j,float(-.5*k_matrix[i,j,component,1])))
            for ni,nj,value in entries:
                index=unknown_index[ni,nj]
                if index>=0 and value!=0:
                    matrix_rows.append(row_number)
                    matrix_columns.append(int(index))
                    matrix_values.append(value)
            targets.append(float(slopes[i,j,component]))
            equation_weights.append(np.sqrt(weight))
            row_number+=1
    if row_number==0:
        raise ValueError("局部重建没有可用的加权方程")
    # Dprior0：可信无接触点只增加面积归一化的软 d=0 行。它不改变未知量域，
    # 因而不会把统计掩膜的方形形态硬刻进曲面。
    if displacement_zero_lambda_per_mm2>0:
        for i,j in np.argwhere(trusted_no_contact&unknown_mask):
            weight=float(
                geometry["area"][i,j]*displacement_zero_lambda_per_mm2)
            if not np.isfinite(weight) or weight<=0:
                continue
            matrix_rows.append(row_number)
            matrix_columns.append(int(unknown_index[i,j]))
            matrix_values.append(1.)
            targets.append(0.)
            equation_weights.append(np.sqrt(weight))
            row_number+=1
    # 中心差分在规则网格上存在四个奇偶子格常量零空间。连续问题的零 Dirichlet
    # 边界本应消除它们，但若边界点直接从未知量中删去，离散矩阵仍会保留这些
    # 棋盘格自由度。对四个子格分别用最靠近左上边界的 2x2 同奇偶点外推 d(0,0)=0；
    # 这只是离散 Dirichlet 闭合，不是平滑或拉普拉斯正则项。
    closure_weight=float(np.median(equation_weights))
    grid_i,grid_j=np.indices(points.shape[:2])
    for parity_i in (0,1):
        for parity_j in (0,1):
            submask=(unknown_mask&(grid_i%2==parity_i)&(grid_j%2==parity_j))
            coordinates=np.argwhere(submask)
            if coordinates.size==0:
                continue
            unique_i=np.unique(coordinates[:,0]); unique_j=np.unique(coordinates[:,1])
            added=False
            if unique_i.size>=2 and unique_j.size>=2:
                i0,i1=unique_i[:2]; j0,j1=unique_j[:2]
                corners=((i0,j0),(i0,j1),(i1,j0),(i1,j1))
                if all(unknown_index[i,j]>=0 for i,j in corners):
                    wi=np.asarray([i1,-i0],np.float64)/(i1-i0)
                    wj=np.asarray([j1,-j0],np.float64)/(j1-j0)
                    for ai,i in enumerate((i0,i1)):
                        for aj,j in enumerate((j0,j1)):
                            matrix_rows.append(row_number)
                            matrix_columns.append(int(unknown_index[i,j]))
                            matrix_values.append(float(wi[ai]*wj[aj]))
                    added=True
            if not added:
                i,j=coordinates[0]
                matrix_rows.append(row_number)
                matrix_columns.append(int(unknown_index[i,j]))
                matrix_values.append(1.)
            targets.append(0.)
            equation_weights.append(closure_weight)
            row_number+=1
    matrix=coo_matrix(
        (matrix_values,(matrix_rows,matrix_columns)),
        shape=(row_number,int(np.count_nonzero(unknown_mask)))).tocsr()
    scale=np.asarray(equation_weights,np.float64)
    weighted_matrix=matrix.multiply(scale[:,None])
    weighted_target=np.asarray(targets,np.float64)*scale
    maximum=(max(200,4*weighted_matrix.shape[1])
             if lsmr_max_iterations is None else int(lsmr_max_iterations))
    solution=lsmr(
        weighted_matrix,weighted_target,atol=float(lsmr_atol),
        btol=float(lsmr_btol),maxiter=maximum)
    displacement=np.zeros(points.shape[:2],np.float64)
    displacement[unknown_mask]=solution[0]
    normal=geometry["normal"]
    displacement_vectors=displacement[...,None]*normal
    observed=(normal-tangent)/np.sqrt(
        1+np.sum(tangent*tangent,axis=-1,keepdims=True))
    output=points+displacement_vectors
    return LocalReconstructionResult(
        xyz_out=output.astype(np.float32),
        displacement=displacement.astype(np.float32),
        displacement_vectors=displacement_vectors.astype(np.float32),
        reference_normals=normal.astype(np.float32),
        slopes=slopes.astype(np.float32),
        observed_normals=observed.astype(np.float32),
        curvature_correction=curvature.astype(np.float32),
        confidence=confidence.astype(np.float32),valid_mask=valid,
        boundary_mask=valid&~unknown_mask,solver_istop=int(solution[1]),
        solver_iterations=int(solution[2]),solver_residual_norm=float(solution[3]),
        trusted_no_contact_mask=trusted_no_contact)


def _load_reconstruction_input(
    path: Path,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray | None]:
    with np.load(path,allow_pickle=False) as data:
        xyz=data["xyz"]
        if "color_residual_grid" in data:
            colors=np.asarray(data["color_residual_grid"],np.float32)
            valid=(data["valid_mask"] if "valid_mask" in data
                   else np.ones(xyz.shape[:2],np.bool_))
        else:
            if "uv" not in data or "residual_linear_rgb" not in data:
                raise ValueError(
                    "输入 NPZ 必须提供 color_residual_grid，或同时提供 uv 和 "
                    "residual_linear_rgb")
            full_valid=data["residual_valid"] if "residual_valid" in data else None
            colors,valid=sample_full_resolution_residual(
                data["residual_linear_rgb"],data["uv"],full_valid)
        trusted=(np.asarray(data["trusted_no_contact_mask"],np.bool_)
                 if "trusted_no_contact_mask" in data else None)
        if trusted is not None and trusted.shape!=xyz.shape[:2]:
            raise ValueError("输入 trusted_no_contact_mask 尺寸与 xyz 不一致")
        return xyz,colors,valid,trusted


def local_main() -> None:
    """独立处理已导出的 NPZ；实时路径由 main() 启动。"""
    parser=argparse.ArgumentParser(description="离线 NPZ 局部接触形变重建")
    parser.add_argument(
        "--config",default=Path(__file__).with_name("config.yaml"),
        help="配置文件；默认读取 recon.py 同目录的 config.yaml")
    parser.add_argument(
        "--input",required=True,help="输入 NPZ（实时重建不使用这个参数）")
    parser.add_argument(
        "--calibration",help="覆盖 local_reconstruction.calibration_file")
    parser.add_argument(
        "--output",help="输出 NPZ；默认在输入文件旁自动添加 _local_reconstruction")
    parser.add_argument("--lsmr-atol",type=float,help="覆盖 LSMR atol")
    parser.add_argument("--lsmr-btol",type=float,help="覆盖 LSMR btol")
    parser.add_argument(
        "--lsmr-max-iterations",type=int,help="覆盖 LSMR 最大迭代次数")
    args=parser.parse_args()
    settings=load_local_reconstruction_settings(
        args.config,input_override=args.input,
        calibration_override=args.calibration,output_override=args.output,
        lsmr_atol_override=args.lsmr_atol,lsmr_btol_override=args.lsmr_btol,
        lsmr_max_iterations_override=args.lsmr_max_iterations)
    xyz,colors,valid,trusted=_load_reconstruction_input(settings.input_file)
    calibration=NormalCalibration.load(settings.calibration_file)
    result=reconstruct_local_surface(
        xyz,colors,calibration,valid_mask=valid,
        zero_color_inner_radius=settings.zero_color_inner_radius,
        zero_color_outer_radius=settings.zero_color_outer_radius,
        trusted_no_contact_mask=(
            trusted if settings.no_contact_constraints_enabled else None),
        trusted_no_contact_confidence=settings.trusted_no_contact_confidence,
        displacement_zero_lambda_per_mm2=(
            settings.displacement_zero_lambda_per_mm2
            if settings.no_contact_constraints_enabled else 0.),
        lsmr_atol=settings.lsmr_atol,lsmr_btol=settings.lsmr_btol,
        lsmr_max_iterations=settings.lsmr_max_iterations)
    output=result.save(settings.output_file)
    print(f"局部重建完成: {output}；unknown={int(np.count_nonzero(~result.boundary_mask & result.valid_mask))} "
          f"iterations={result.solver_iterations} residual={result.solver_residual_norm:.6g}")


def main() -> None:
    """启动相机实时重建；保留 recon.py 作为项目的直观主入口。"""
    from render_lightfield import main as render_main
    render_main()


if __name__=="__main__":
    main()
