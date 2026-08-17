"""无局部形变近场线光源模型（JAX/XLA）。"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .jax_reconstruction import SURFACE_RECONSTRUCTION_PIPELINE_VERSION

CHANNELS = ("R", "G", "B")
DIRECT_BACKGROUND_METHODS = ("direct_fit", "direct_fit_3")
GEOMETRY_BACKGROUND_METHODS = (
    "direct_fit", "direct_fit_3", "geometry_cache")
VALID_LIGHT_SOURCE_SIDES = ("left", "right", "top", "bottom")
LightSourceLayout = tuple[tuple[str,...],tuple[str,...],tuple[str,...]]
DEFAULT_LIGHT_SOURCE_LAYOUT: LightSourceLayout = (
    ("right",),("left",),("bottom",))
Array = jax.Array

def parse_light_source_layout(value: Mapping[str, Any]) -> LightSourceLayout:
    """把 RGB->一个或多个边界转换为固定顺序的不可变灯带布局。"""
    if not isinstance(value,Mapping) or set(value) != set(CHANNELS):
        raise ValueError("light_source_layout 必须且只能包含 R、G、B 三个键")
    sides_by_channel=[]
    for channel in CHANNELS:
        configured=value[channel]
        if isinstance(configured,str):
            sides=(configured.lower(),)
        elif isinstance(configured,(list,tuple)):
            sides=tuple(str(side).lower() for side in configured)
        else:
            raise ValueError("每个颜色的灯带位置必须是边名或边名列表")
        if not sides:
            raise ValueError("每个颜色必须至少配置一条灯带")
        if len(set(sides))!=len(sides):
            raise ValueError(f"{channel} 不能在同一条边重复配置灯带")
        sides_by_channel.append(sides)
    layout=tuple(sides_by_channel)
    invalid=[side for sides in layout for side in sides
             if side not in VALID_LIGHT_SOURCE_SIDES]
    if invalid:
        raise ValueError("灯带位置只能是 left、right、top、bottom")
    return layout

def light_source_layout_mapping(layout: LightSourceLayout) -> dict[str,Any]:
    """转为 YAML 友好映射；单边保留旧标量写法，多边写为列表。"""
    return {
        channel:(sides[0] if len(sides)==1 else list(sides))
        for channel,sides in zip(CHANNELS,layout,strict=True)
    }

def light_source_specs(
    layout: LightSourceLayout,
) -> tuple[tuple[int,str],...]:
    """按 RGB、再按配置顺序展开为 (颜色索引, 边) 灯带实例。"""
    return tuple(
        (channel,side)
        for channel,sides in enumerate(layout)
        for side in sides)

def bounded_mixing_matrix(raw: Array,max_offdiagonal_sum: float) -> Array:
    """把无约束 3x3 参数变为逐行归一、且总串扰受限的光谱混合矩阵。"""
    eye=jnp.eye(3,dtype=raw.dtype); offdiagonal=1-eye
    leakage=max_offdiagonal_sum*jax.nn.sigmoid(jnp.diag(raw))
    distribution=jax.nn.softmax(jnp.where(offdiagonal>0,raw,-1e9),axis=1)
    return eye*(1-leakage[:,None])+offdiagonal*distribution*leakage[:,None]

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LightFieldModel:
    delta: Array
    beta: Array
    bias: Array
    scatter_ratio: Array
    scatter_length: Array
    mixing_matrix: Array
    residual_b_coefficients: Array
    residual_m_coefficients: Array
    source_layout: LightSourceLayout = DEFAULT_LIGHT_SOURCE_LAYOUT
    background_method: str = "physical_residual"
    direct_base_texture: Array | None = None
    direct_coordinate_frequencies: Array | None = None
    direct_geometry_feature_mean: Array | None = None
    direct_geometry_feature_scale: Array | None = None
    direct_geometry_pca_components: Array | None = None
    direct_geometry_pca_scale: Array | None = None
    direct_local_geometry_feature_mean: Array | None = None
    direct_local_geometry_feature_scale: Array | None = None
    direct_geometry_encoder_weights: tuple[Array,...] | None = None
    direct_geometry_encoder_biases: tuple[Array,...] | None = None
    direct_decoder_weights: tuple[Array,...] | None = None
    direct_decoder_biases: tuple[Array,...] | None = None
    direct_channel_decoder_weights: tuple[tuple[Array,...],...] | None = None
    direct_channel_decoder_biases: tuple[tuple[Array,...],...] | None = None
    direct_geometry_descriptor_rows: int = 24
    direct_curve_convexity: str = "none"
    direct_reconstruction_pipeline: str = SURFACE_RECONSTRUCTION_PIPELINE_VERSION
    geometry_cache_base_texture: Array | None = None
    geometry_cache_anchor_coefficients: Array | None = None
    geometry_cache_descriptor_mean: Array | None = None
    geometry_cache_descriptor_scale: Array | None = None
    geometry_cache_pca_components: Array | None = None
    geometry_cache_pca_scale: Array | None = None
    geometry_cache_anchor_keys: Array | None = None
    geometry_cache_curve_coefficients: int = 12
    geometry_cache_descriptor_huber_delta: float = .5
    geometry_cache_interpolation_neighbors: int = 4
    geometry_cache_distance_power: float = 2.
    geometry_cache_distance_epsilon: float = 1e-3

    def tree_flatten(self):
        return (self.delta, self.beta, self.bias, self.scatter_ratio,
                self.scatter_length, self.mixing_matrix,
                self.residual_b_coefficients,
                self.residual_m_coefficients,
                self.direct_base_texture,
                self.direct_coordinate_frequencies,
                self.direct_geometry_feature_mean,
                self.direct_geometry_feature_scale,
                self.direct_geometry_pca_components,
                self.direct_geometry_pca_scale,
                self.direct_local_geometry_feature_mean,
                self.direct_local_geometry_feature_scale,
                self.direct_geometry_encoder_weights,
                self.direct_geometry_encoder_biases,
                self.direct_decoder_weights,
                self.direct_decoder_biases,
                self.direct_channel_decoder_weights,
                self.direct_channel_decoder_biases,
                self.geometry_cache_base_texture,
                self.geometry_cache_anchor_coefficients,
                self.geometry_cache_descriptor_mean,
                self.geometry_cache_descriptor_scale,
                self.geometry_cache_pca_components,
                self.geometry_cache_pca_scale,
                self.geometry_cache_anchor_keys),(
                    self.source_layout,self.background_method,
                    self.direct_geometry_descriptor_rows,
                    self.direct_curve_convexity,
                    self.direct_reconstruction_pipeline,
                    self.geometry_cache_curve_coefficients,
                    self.geometry_cache_descriptor_huber_delta,
                    self.geometry_cache_interpolation_neighbors,
                    self.geometry_cache_distance_power,
                    self.geometry_cache_distance_epsilon)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        (source_layout,background_method,descriptor_rows,
         curve_convexity,reconstruction_pipeline,cache_curve_coefficients,
         cache_huber_delta,cache_interpolation_neighbors,
         cache_distance_power,cache_distance_epsilon)=auxiliary
        core=children[:8]
        direct=children[8:]
        return cls(
            *core,source_layout=source_layout,
            background_method=background_method,
            direct_base_texture=direct[0],
            direct_coordinate_frequencies=direct[1],
            direct_geometry_feature_mean=direct[2],
            direct_geometry_feature_scale=direct[3],
            direct_geometry_pca_components=direct[4],
            direct_geometry_pca_scale=direct[5],
            direct_local_geometry_feature_mean=direct[6],
            direct_local_geometry_feature_scale=direct[7],
            direct_geometry_encoder_weights=direct[8],
            direct_geometry_encoder_biases=direct[9],
            direct_decoder_weights=direct[10],
            direct_decoder_biases=direct[11],
            direct_channel_decoder_weights=direct[12],
            direct_channel_decoder_biases=direct[13],
            direct_geometry_descriptor_rows=descriptor_rows,
            direct_curve_convexity=curve_convexity,
            direct_reconstruction_pipeline=reconstruction_pipeline,
            geometry_cache_base_texture=direct[14],
            geometry_cache_anchor_coefficients=direct[15],
            geometry_cache_descriptor_mean=direct[16],
            geometry_cache_descriptor_scale=direct[17],
            geometry_cache_pca_components=direct[18],
            geometry_cache_pca_scale=direct[19],
            geometry_cache_anchor_keys=direct[20],
            geometry_cache_curve_coefficients=cache_curve_coefficients,
            geometry_cache_descriptor_huber_delta=cache_huber_delta,
            geometry_cache_interpolation_neighbors=(
                cache_interpolation_neighbors),
            geometry_cache_distance_power=cache_distance_power,
            geometry_cache_distance_epsilon=cache_distance_epsilon)

    @classmethod
    def direct_fit(
        cls,session_b_coefficients: Array,
        *,base_texture: Array,coordinate_frequencies: Array,
        geometry_feature_mean: Array,
        geometry_feature_scale: Array,
        geometry_pca_components: Array,
        geometry_pca_scale: Array,
        local_geometry_feature_mean: Array,
        local_geometry_feature_scale: Array,
        geometry_encoder_weights: tuple[Array,...],
        geometry_encoder_biases: tuple[Array,...],
        decoder_weights: tuple[Array,...],
        decoder_biases: tuple[Array,...],
        geometry_descriptor_rows: int = 24,
        curve_convexity: str = "none",
        reconstruction_pipeline: str = SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
        source_layout: LightSourceLayout = DEFAULT_LIGHT_SOURCE_LAYOUT,
    ) -> "LightFieldModel":
        """建立静态 B、几何 delta B 和独立低频 Bsession 的 direct 模型。"""
        if curve_convexity not in {"none","increasing","decreasing"}:
            raise ValueError(
                "direct curve_convexity 必须是 none、increasing 或 decreasing")
        if reconstruction_pipeline!=SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
            raise ValueError("direct reconstruction pipeline 元数据无效")
        session=jnp.asarray(session_b_coefficients,jnp.float32)
        base=jnp.asarray(base_texture,jnp.float32)
        if base.ndim!=3 or base.shape[-1]!=3 or min(base.shape[:2])<2 \
                or not bool(jnp.all(jnp.isfinite(base))) \
                or not bool(jnp.all((base>=0)&(base<=1))):
            raise ValueError("direct base_texture 必须是 [0,1] 内有限的 HxWx3")
        source_count=len(light_source_specs(source_layout))
        return cls(
            jnp.zeros((source_count,2),jnp.float32),
            jnp.zeros((source_count,4),jnp.float32),
            jnp.zeros((3,),jnp.float32),
            jnp.zeros((source_count,),jnp.float32),
            jnp.ones((source_count,),jnp.float32),jnp.eye(3,dtype=jnp.float32),
            session,jnp.zeros((0,*session.shape),jnp.float32),source_layout,
            "direct_fit",base,jnp.asarray(coordinate_frequencies,jnp.float32),
            jnp.asarray(geometry_feature_mean,jnp.float32),
            jnp.asarray(geometry_feature_scale,jnp.float32),
            jnp.asarray(geometry_pca_components,jnp.float32),
            jnp.asarray(geometry_pca_scale,jnp.float32),
            jnp.asarray(local_geometry_feature_mean,jnp.float32),
            jnp.asarray(local_geometry_feature_scale,jnp.float32),
            tuple(jnp.asarray(value,jnp.float32)
                  for value in geometry_encoder_weights),
            tuple(jnp.asarray(value,jnp.float32)
                  for value in geometry_encoder_biases),
            tuple(jnp.asarray(value,jnp.float32) for value in decoder_weights),
            tuple(jnp.asarray(value,jnp.float32) for value in decoder_biases),
            None,None,geometry_descriptor_rows,curve_convexity,
            reconstruction_pipeline)

    @classmethod
    def direct_fit_3(
        cls,session_b_coefficients: Array,
        *,base_texture: Array,coordinate_frequencies: Array,
        geometry_feature_mean: Array,
        geometry_feature_scale: Array,
        geometry_pca_components: Array,
        geometry_pca_scale: Array,
        local_geometry_feature_mean: Array,
        local_geometry_feature_scale: Array,
        geometry_encoder_weights: tuple[Array,...],
        geometry_encoder_biases: tuple[Array,...],
        channel_decoder_weights: tuple[tuple[Array,...],...],
        channel_decoder_biases: tuple[tuple[Array,...],...],
        geometry_descriptor_rows: int = 24,
        curve_convexity: str = "none",
        reconstruction_pipeline: str = SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
        source_layout: LightSourceLayout = DEFAULT_LIGHT_SOURCE_LAYOUT,
    ) -> "LightFieldModel":
        """建立共享几何 encoder、三个标量 decoder 的 direct_fit_3 模型。"""
        if curve_convexity not in {"none","increasing","decreasing"}:
            raise ValueError(
                "direct curve_convexity 必须是 none、increasing 或 decreasing")
        if reconstruction_pipeline!=SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
            raise ValueError("direct reconstruction pipeline 元数据无效")
        session=jnp.asarray(session_b_coefficients,jnp.float32)
        base=jnp.asarray(base_texture,jnp.float32)
        if base.ndim!=3 or base.shape[-1]!=3 or min(base.shape[:2])<2 \
                or not bool(jnp.all(jnp.isfinite(base))) \
                or not bool(jnp.all((base>=0)&(base<=1))):
            raise ValueError("direct base_texture 必须是 [0,1] 内有限的 HxWx3")
        if len(channel_decoder_weights)!=3 or len(channel_decoder_biases)!=3:
            raise ValueError("direct_fit_3 必须包含 R/G/B 三个 decoder")
        if not geometry_encoder_weights \
                or len(geometry_encoder_weights)!=len(geometry_encoder_biases):
            raise ValueError("direct_fit_3 geometry encoder 网络层无效")
        source_count=len(light_source_specs(source_layout))
        return cls(
            jnp.zeros((source_count,2),jnp.float32),
            jnp.zeros((source_count,4),jnp.float32),
            jnp.zeros((3,),jnp.float32),
            jnp.zeros((source_count,),jnp.float32),
            jnp.ones((source_count,),jnp.float32),jnp.eye(3,dtype=jnp.float32),
            session,jnp.zeros((0,*session.shape),jnp.float32),source_layout,
            "direct_fit_3",base,jnp.asarray(coordinate_frequencies,jnp.float32),
            jnp.asarray(geometry_feature_mean,jnp.float32),
            jnp.asarray(geometry_feature_scale,jnp.float32),
            jnp.asarray(geometry_pca_components,jnp.float32),
            jnp.asarray(geometry_pca_scale,jnp.float32),
            jnp.asarray(local_geometry_feature_mean,jnp.float32),
            jnp.asarray(local_geometry_feature_scale,jnp.float32),
            tuple(jnp.asarray(value,jnp.float32)
                  for value in geometry_encoder_weights),
            tuple(jnp.asarray(value,jnp.float32)
                  for value in geometry_encoder_biases),
            None,None,
            tuple(tuple(jnp.asarray(value,jnp.float32) for value in decoder)
                  for decoder in channel_decoder_weights),
            tuple(tuple(jnp.asarray(value,jnp.float32) for value in decoder)
                  for decoder in channel_decoder_biases),
            geometry_descriptor_rows,curve_convexity,reconstruction_pipeline)

    @classmethod
    def geometry_cache(
        cls,session_b_coefficients: Array,*,base_texture: Array,
        anchor_coefficients: Array,descriptor_mean: Array,
        descriptor_scale: Array,pca_components: Array,pca_scale: Array,
        anchor_keys: Array,curve_coefficients: int = 12,
        descriptor_huber_delta: float = .5,
        interpolation_neighbors: int = 4,distance_power: float = 2.,
        distance_epsilon: float = 1e-3,curve_convexity: str = "none",
        reconstruction_pipeline: str = SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
        source_layout: LightSourceLayout = DEFAULT_LIGHT_SOURCE_LAYOUT,
    ) -> "LightFieldModel":
        """建立几何锚点缓存与 RGB B-spline 增量插值背景模型。"""
        session=np.asarray(session_b_coefficients,np.float32)
        base=np.asarray(base_texture,np.float32)
        anchors=np.asarray(anchor_coefficients,np.float32)
        mean=np.asarray(descriptor_mean,np.float32)
        scale=np.asarray(descriptor_scale,np.float32)
        components=np.asarray(pca_components,np.float32)
        component_scale=np.asarray(pca_scale,np.float32)
        keys=np.asarray(anchor_keys,np.float32)
        source_count=len(light_source_specs(source_layout))
        if session.ndim!=3 or session.shape[0]!=3 or min(session.shape[1:])<4:
            raise ValueError("geometry_cache Bsession 必须是 3xRxC")
        if base.ndim!=3 or base.shape[-1]!=3 or min(base.shape[:2])<2 \
                or not np.isfinite(base).all() or np.any((base<0)|(base>1)):
            raise ValueError("geometry_cache base texture 无效")
        if not isinstance(curve_coefficients,int) \
                or isinstance(curve_coefficients,bool) or curve_coefficients<4:
            raise ValueError("geometry_cache 几何描述参数无效")
        descriptor_count=4*curve_coefficients
        if mean.shape!=(descriptor_count,) or scale.shape!=mean.shape \
                or np.any(scale<=0) or not np.isfinite(mean).all() \
                or not np.isfinite(scale).all():
            raise ValueError("geometry_cache 几何描述参数无效")
        if components.ndim!=2 or components.shape[0]!=descriptor_count \
                or components.shape[1]<1 \
                or component_scale.shape!=(components.shape[1],) \
                or np.any(component_scale<=0) \
                or not np.isfinite(components).all() \
                or not np.isfinite(component_scale).all():
            raise ValueError("geometry_cache PCA 参数无效")
        anchor_count=anchors.shape[0] if anchors.ndim==4 else -1
        if anchor_count<1 or anchors.shape!=(anchor_count,3,*session.shape[1:]) \
                or keys.shape!=(anchor_count,components.shape[1]) \
                or not np.isfinite(anchors).all() or not np.isfinite(keys).all():
            raise ValueError("geometry_cache 锚点尺寸无效")
        if curve_convexity not in {"none","increasing","decreasing"} \
                or reconstruction_pipeline!=SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
            raise ValueError("geometry_cache 重建语义无效")
        if not isinstance(interpolation_neighbors,int) \
                or isinstance(interpolation_neighbors,bool) \
                or not 1<=interpolation_neighbors<=anchor_count \
                or descriptor_huber_delta<=0 or distance_power<=0 \
                or distance_epsilon<=0:
            raise ValueError("geometry_cache 插值参数无效")
        return cls(
            delta=jnp.zeros((source_count,2),jnp.float32),
            beta=jnp.zeros((source_count,4),jnp.float32),
            bias=jnp.zeros((3,),jnp.float32),
            scatter_ratio=jnp.zeros((source_count,),jnp.float32),
            scatter_length=jnp.ones((source_count,),jnp.float32),
            mixing_matrix=jnp.eye(3,dtype=jnp.float32),
            residual_b_coefficients=jnp.asarray(session),
            residual_m_coefficients=jnp.zeros((0,*session.shape),jnp.float32),
            source_layout=source_layout,background_method="geometry_cache",
            direct_curve_convexity=curve_convexity,
            direct_reconstruction_pipeline=reconstruction_pipeline,
            geometry_cache_base_texture=jnp.asarray(base),
            geometry_cache_anchor_coefficients=jnp.asarray(anchors),
            geometry_cache_descriptor_mean=jnp.asarray(mean),
            geometry_cache_descriptor_scale=jnp.asarray(scale),
            geometry_cache_pca_components=jnp.asarray(components),
            geometry_cache_pca_scale=jnp.asarray(component_scale),
            geometry_cache_anchor_keys=jnp.asarray(keys),
            geometry_cache_curve_coefficients=curve_coefficients,
            geometry_cache_descriptor_huber_delta=float(
                descriptor_huber_delta),
            geometry_cache_interpolation_neighbors=interpolation_neighbors,
            geometry_cache_distance_power=float(distance_power),
            geometry_cache_distance_epsilon=float(distance_epsilon))

    @classmethod
    def load(cls, path: str | Path, device: jax.Device | None = None) -> "LightFieldModel":
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        version=raw.get("format_version")
        if version not in {9,10,11,12,13,14,15,16,17,18,19,20,21}:
            raise ValueError("光场模型版本已经过期；请重新运行 calibrate-lightfield")
        background_method=("physical_residual" if version==9
                           else raw.get("background_method"))
        if background_method not in {
                "physical_residual","direct_fit","direct_fit_3",
                "geometry_cache"}:
            raise ValueError("光场模型包含无效 background_method")
        if background_method=="direct_fit" and version<16:
            raise ValueError(
                "旧 direct_fit 模型并非由完整实时 JAX XYZ/UV/depth 重建链训练；"
                "请重新运行 calibrate-lightfield")
        if background_method=="direct_fit" and version!=18:
            raise ValueError(
                "direct_fit 模型版本无效或尚未拆分静态 B 与几何 delta B；"
                "请重新运行 calibrate-lightfield")
        if background_method=="direct_fit_3" and version!=19:
            raise ValueError(
                "direct_fit_3 模型版本无效；"
                "请重新运行 calibrate-lightfield")
        if background_method=="geometry_cache" and version!=21:
            raise ValueError(
                "geometry_cache 模型版本无效；请重新运行 calibrate-lightfield")
        try:
            source_layout=parse_light_source_layout(raw.get("light_source_layout"))
        except ValueError as error:
            raise ValueError("光场模型中的 RGB 灯带布局无效；请重新运行 calibrate-lightfield") from error
        expected_mode=(
            "difference_only" if background_method=="physical_residual" else
            "geometry_background_plus_additive_session"
            if background_method=="geometry_cache" else
            "static_base_plus_geometry_delta_plus_additive_session")
        if raw.get("residual_correction_mode") != expected_mode:
            raise ValueError("光场模型的背景场语义与 background_method 不一致")
        residual_b=np.asarray(raw.get("residual_b_bspline_coefficients"),np.float32)
        if raw.get("residual_b_axis_order") != ["RGB","surface_row","surface_column"]:
            raise ValueError("residual_b_axis_order 必须为 [RGB, surface_row, surface_column]")
        if residual_b.ndim != 3 or residual_b.shape[0] != 3 \
                or min(residual_b.shape[1:]) < 4 or not np.isfinite(residual_b).all():
            raise ValueError("residual_b_bspline_coefficients 必须是有限的 3xRxC 数组")
        if background_method in DIRECT_BACKGROUND_METHODS:
            if raw.get("base_coefficient_mode")!="direct_static_texture":
                raise ValueError("direct_fit 模型缺少独立静态 B")
            if raw.get("direct_session_correction_mode")!="additive_bspline":
                raise ValueError("纯拟合模型缺少加性会话 B 修正语义")
            if raw.get("direct_base_mode")!="robust_full_resolution_texture" \
                    or raw.get("direct_delta_mode")!="additive_logit":
                raise ValueError("direct_fit 缺少 B + delta B 语义")
            if background_method=="direct_fit_3" \
                    and raw.get("direct_base_channel_mode")!="independent_huber":
                raise ValueError("direct_fit_3 缺少逐通道独立 B 拟合语义")
            base_texture=np.asarray(raw.get("direct_base_texture"),np.float32)
            if base_texture.ndim!=3 or base_texture.shape[-1]!=3 \
                    or min(base_texture.shape[:2])<2 \
                    or not np.isfinite(base_texture).all() \
                    or np.any((base_texture<0)|(base_texture>1)):
                raise ValueError("direct base texture 尺寸或数值无效")
            frequencies=np.asarray(
                raw.get("direct_coordinate_frequencies"),np.float32)
            feature_mean=np.asarray(
                raw.get("direct_geometry_feature_mean"),np.float32)
            feature_scale=np.asarray(
                raw.get("direct_geometry_feature_scale"),np.float32)
            pca_components=np.asarray(
                raw.get("direct_geometry_pca_components"),np.float32)
            pca_scale=np.asarray(
                raw.get("direct_geometry_pca_scale"),np.float32)
            local_feature_mean=np.asarray(
                raw.get("direct_local_geometry_feature_mean"),np.float32)
            local_feature_scale=np.asarray(
                raw.get("direct_local_geometry_feature_scale"),np.float32)
            encoder_weights=tuple(np.asarray(value,np.float32) for value in
                                  raw.get("direct_geometry_encoder_weights",[]))
            encoder_biases=tuple(np.asarray(value,np.float32) for value in
                                 raw.get("direct_geometry_encoder_biases",[]))
            decoder_weights=tuple(np.asarray(value,np.float32) for value in
                                  raw.get("direct_decoder_weights",[]))
            decoder_biases=tuple(np.asarray(value,np.float32) for value in
                                 raw.get("direct_decoder_biases",[]))
            channel_decoder_weights=tuple(
                tuple(np.asarray(value,np.float32) for value in decoder)
                for decoder in raw.get("direct_channel_decoder_weights",[]))
            channel_decoder_biases=tuple(
                tuple(np.asarray(value,np.float32) for value in decoder)
                for decoder in raw.get("direct_channel_decoder_biases",[]))
            descriptor_rows=raw.get("direct_geometry_descriptor_rows")
            curve_convexity=raw.get("direct_curve_convexity")
            reconstruction_pipeline=raw.get("direct_reconstruction_pipeline")
            if curve_convexity not in {"none","increasing","decreasing"}:
                raise ValueError("direct curve convexity 元数据无效")
            if reconstruction_pipeline!=SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
                raise ValueError("direct reconstruction pipeline 元数据无效")
            if frequencies.ndim!=1 or frequencies.size<1 \
                    or not np.isfinite(frequencies).all() \
                    or np.any(frequencies<=0):
                raise ValueError("direct coordinate frequencies 无效")
            if not isinstance(descriptor_rows,int) or isinstance(
                    descriptor_rows,bool) or descriptor_rows<4:
                raise ValueError("direct geometry descriptor rows 无效")
            feature_count=6+10*descriptor_rows
            if feature_mean.shape!=(feature_count,) \
                    or feature_scale.shape!=(feature_count,) \
                    or not np.isfinite(feature_mean).all() \
                    or not np.isfinite(feature_scale).all() \
                    or np.any(feature_scale<=0):
                raise ValueError("direct geometry 特征归一化参数无效")
            if pca_components.ndim!=2 \
                    or pca_components.shape[0]!=feature_count \
                    or pca_components.shape[1]<1 \
                    or pca_scale.shape!=(pca_components.shape[1],) \
                    or not np.isfinite(pca_components).all() \
                    or not np.isfinite(pca_scale).all() \
                    or np.any(pca_scale<=0):
                raise ValueError("direct geometry PCA 参数无效")
            if local_feature_mean.shape!=(15,) \
                    or local_feature_scale.shape!=(15,) \
                    or not np.isfinite(local_feature_mean).all() \
                    or not np.isfinite(local_feature_scale).all() \
                    or np.any(local_feature_scale<=0):
                raise ValueError("direct local geometry 特征归一化参数无效")

            def validate_network(weights,biases,input_count,output_count,name):
                if not weights or len(weights)!=len(biases):
                    raise ValueError(f"direct {name} 网络层无效")
                previous=input_count
                for weight,layer_bias in zip(weights,biases,strict=True):
                    if weight.ndim!=2 or weight.shape[0]!=previous \
                            or layer_bias.shape!=(weight.shape[1],) \
                            or not np.isfinite(weight).all() \
                            or not np.isfinite(layer_bias).all():
                        raise ValueError(f"direct {name} 网络权重尺寸无效")
                    previous=weight.shape[1]
                if previous!=output_count:
                    raise ValueError(f"direct {name} 网络输出尺寸无效")

            if not encoder_weights:
                raise ValueError("direct geometry encoder 网络层无效")
            latent_count=encoder_weights[-1].shape[1]
            validate_network(
                encoder_weights,encoder_biases,feature_count,latent_count,
                "geometry encoder")
            decoder_input_count=(2+4*frequencies.size+latent_count
                                 +pca_components.shape[1]+15)
            if raw.get("direct_decoder_skip_mode")!="input_every_layer":
                raise ValueError("direct decoder 跳连语义无效")

            def validate_decoder(weights,biases,output_count,name):
                if not weights or len(weights)!=len(biases):
                    raise ValueError(f"{name} 网络层无效")
                previous=decoder_input_count
                for index,(weight,layer_bias) in enumerate(zip(
                        weights,biases,strict=True)):
                    if weight.ndim!=2 or weight.shape[0]!=previous \
                            or layer_bias.shape!=(weight.shape[1],) \
                            or not np.isfinite(weight).all() \
                            or not np.isfinite(layer_bias).all():
                        raise ValueError(f"{name} 网络权重尺寸无效")
                    if index==len(weights)-1 and weight.shape[1]!=output_count:
                        raise ValueError(f"{name} 输出尺寸无效")
                    previous=weight.shape[1]+decoder_input_count

            if background_method=="direct_fit":
                validate_decoder(
                    decoder_weights,decoder_biases,3,"direct RGB decoder")
                if channel_decoder_weights or channel_decoder_biases:
                    raise ValueError("direct_fit 不应包含分通道 decoder")
                model=cls.direct_fit(
                    residual_b,base_texture=base_texture,
                    coordinate_frequencies=frequencies,
                    geometry_feature_mean=feature_mean,
                    geometry_feature_scale=feature_scale,
                    geometry_pca_components=pca_components,
                    geometry_pca_scale=pca_scale,
                    local_geometry_feature_mean=local_feature_mean,
                    local_geometry_feature_scale=local_feature_scale,
                    geometry_encoder_weights=encoder_weights,
                    geometry_encoder_biases=encoder_biases,
                    decoder_weights=decoder_weights,
                    decoder_biases=decoder_biases,
                    geometry_descriptor_rows=descriptor_rows,
                    curve_convexity=curve_convexity,
                    reconstruction_pipeline=reconstruction_pipeline,
                    source_layout=source_layout)
            else:
                if raw.get("direct_channel_decoder_order")!=list(CHANNELS) \
                        or len(channel_decoder_weights)!=3 \
                        or len(channel_decoder_biases)!=3:
                    raise ValueError(
                        "direct_fit_3 必须按 R/G/B 保存三个 decoder")
                if decoder_weights or decoder_biases:
                    raise ValueError("direct_fit_3 不应包含共享 RGB decoder")
                for channel,weights,biases in zip(
                        CHANNELS,channel_decoder_weights,
                        channel_decoder_biases,strict=True):
                    validate_decoder(
                        weights,biases,1,f"direct_fit_3 {channel} decoder")
                model=cls.direct_fit_3(
                    residual_b,base_texture=base_texture,
                    coordinate_frequencies=frequencies,
                    geometry_feature_mean=feature_mean,
                    geometry_feature_scale=feature_scale,
                    geometry_pca_components=pca_components,
                    geometry_pca_scale=pca_scale,
                    local_geometry_feature_mean=local_feature_mean,
                    local_geometry_feature_scale=local_feature_scale,
                    geometry_encoder_weights=encoder_weights,
                    geometry_encoder_biases=encoder_biases,
                    channel_decoder_weights=channel_decoder_weights,
                    channel_decoder_biases=channel_decoder_biases,
                    geometry_descriptor_rows=descriptor_rows,
                    curve_convexity=curve_convexity,
                    reconstruction_pipeline=reconstruction_pipeline,
                    source_layout=source_layout)
        elif background_method=="geometry_cache":
            if raw.get("base_coefficient_mode")!="geometry_cache_static_texture" \
                    or raw.get("geometry_cache_session_correction_mode") \
                    !="additive_bspline" \
                    or raw.get("geometry_cache_mode") \
                    !="nearest_anchor_convex_interpolation" \
                    or raw.get("geometry_cache_anchor_axis_order")!=[
                        "anchor","RGB","surface_row","surface_column"]:
                raise ValueError("geometry_cache 模型语义无效")
            base_texture=np.asarray(
                raw.get("geometry_cache_base_texture"),np.float32)
            anchor_coefficients=np.asarray(
                raw.get("geometry_cache_anchor_bspline_coefficients"),
                np.float32)
            descriptor_mean=np.asarray(
                raw.get("geometry_cache_descriptor_mean"),np.float32)
            descriptor_scale=np.asarray(
                raw.get("geometry_cache_descriptor_scale"),np.float32)
            pca_components=np.asarray(
                raw.get("geometry_cache_pca_components"),np.float32)
            pca_scale=np.asarray(
                raw.get("geometry_cache_pca_scale"),np.float32)
            anchor_keys=np.asarray(
                raw.get("geometry_cache_anchor_keys"),np.float32)
            model=cls.geometry_cache(
                residual_b,base_texture=base_texture,
                anchor_coefficients=anchor_coefficients,
                descriptor_mean=descriptor_mean,
                descriptor_scale=descriptor_scale,
                pca_components=pca_components,pca_scale=pca_scale,
                anchor_keys=anchor_keys,
                curve_coefficients=raw.get(
                    "geometry_cache_curve_coefficients"),
                descriptor_huber_delta=raw.get(
                    "geometry_cache_descriptor_huber_delta_mm"),
                interpolation_neighbors=raw.get(
                    "geometry_cache_interpolation_neighbors"),
                distance_power=raw.get(
                    "geometry_cache_distance_power"),
                distance_epsilon=raw.get(
                    "geometry_cache_distance_epsilon"),
                curve_convexity=raw.get("geometry_cache_curve_convexity"),
                reconstruction_pipeline=raw.get(
                    "geometry_cache_reconstruction_pipeline"),
                source_layout=source_layout)
        else:
            if raw.get("residual_m_basis") != "raw_offline":
                raise ValueError("离线 M 必须保存为未正交化的 raw_offline 基底；请重新运行 calibrate-lightfield")
            if raw.get("residual_m_axis_order") != ["M","RGB","surface_row","surface_column"]:
                raise ValueError("residual_m_axis_order 必须为 [M, RGB, surface_row, surface_column]")
            residual_ms=np.asarray(
                raw.get("residual_m_bspline_coefficients"),np.float32)
            if residual_ms.ndim != 4 or residual_ms.shape[0] < 1 \
                    or residual_ms.shape[1:] != residual_b.shape \
                    or not np.isfinite(residual_ms).all():
                raise ValueError("residual_m_bspline_coefficients 必须是有限的 Kx3xRxC 数组")
            if version==10 and raw.get("base_coefficient_mode")!="free":
                raise ValueError("物理残差模型必须允许逐帧拟合 B 系数")
            if raw.get("dark_bias_mode") != "fixed_zero":
                raise ValueError("光场模型仍使用旧的离线可学习暗场偏置；请重新运行 calibrate-lightfield")
            stored_bias=np.asarray(raw.get("dark_bias_linear_rgb"),np.float32)
            if stored_bias.shape!=(3,) or not np.array_equal(
                    stored_bias,np.zeros(3,np.float32)):
                raise ValueError("固定暗场基准 dark_bias_linear_rgb 必须严格为 [0, 0, 0]")
            if raw.get("delta_axes") != ["x","normal"]:
                raise ValueError("光场模型 delta_axes 必须为 [x, normal]；请重新运行 calibrate-lightfield")
            delta=np.asarray(raw["delta_mm"],np.float32)
            beta=np.asarray(raw["bspline_coefficients"],np.float32)
            scatter_ratio=np.asarray(raw.get("scatter_ratio"),np.float32)
            scatter_length=np.asarray(raw.get("scatter_length_mm"),np.float32)
            mixing_matrix=np.asarray(raw.get("mixing_matrix"),np.float32)
            source_count=len(light_source_specs(source_layout))
            if delta.shape!=(source_count,2):
                raise ValueError(
                    "光源几何偏移 delta_mm 必须是 Sx2（每条灯带 [x, normal]）")
            if beta.ndim!=2 or beta.shape[0]!=source_count or beta.shape[1]<4 \
                    or not np.isfinite(beta).all():
                raise ValueError("bspline_coefficients 必须是有限的 SxK 数组且 K>=4")
            if scatter_ratio.shape!=(source_count,) or np.any(
                    (scatter_ratio<0)|(scatter_ratio>1)):
                raise ValueError("scatter_ratio 必须是每条灯带一个 [0,1] 范围内的数")
            if scatter_length.shape!=(source_count,) or np.any(scatter_length<=0):
                raise ValueError("scatter_length_mm 必须是每条灯带一个正数")
            if mixing_matrix.shape!=(3,3) or np.any(mixing_matrix<0) \
                    or not np.allclose(mixing_matrix.sum(axis=1),1,atol=1e-5):
                raise ValueError("mixing_matrix 必须是非负且每行和为 1 的 3x3 矩阵")
            model=cls(
                jnp.asarray(delta,jnp.float32),
                jnp.asarray(beta,jnp.float32),
                jnp.zeros((3,),jnp.float32),
                jnp.asarray(scatter_ratio,jnp.float32),
                jnp.asarray(scatter_length,jnp.float32),
                jnp.asarray(mixing_matrix,jnp.float32),
                jnp.asarray(residual_b,jnp.float32),
                jnp.asarray(residual_ms,jnp.float32),source_layout,
                "physical_residual")
        return jax.device_put(model, device) if device is not None else model

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
        if self.background_method not in {
                "physical_residual","direct_fit","direct_fit_3",
                "geometry_cache"}:
            raise ValueError("background_method 无效")
        if not np.array_equal(np.asarray(self.bias), np.zeros(3,np.float32)):
            raise ValueError("离线模型的固定暗场基准必须为 [0, 0, 0]")
        source_layout=parse_light_source_layout(light_source_layout_mapping(self.source_layout))
        source_count=len(light_source_specs(source_layout))
        delta=np.asarray(self.delta)
        beta=np.asarray(self.beta)
        scatter_ratio=np.asarray(self.scatter_ratio)
        scatter_length=np.asarray(self.scatter_length)
        mixing_matrix=np.asarray(self.mixing_matrix)
        residual_b=np.asarray(self.residual_b_coefficients)
        residual_ms=np.asarray(self.residual_m_coefficients)
        if residual_b.ndim != 3 or residual_b.shape[0] != 3 \
                or min(residual_b.shape[1:]) < 4 or not np.isfinite(residual_b).all():
            raise ValueError("residual_b_coefficients 必须是有限的 3xRxC 数组")
        if self.background_method=="physical_residual" and (
                residual_ms.ndim != 4 or residual_ms.shape[0] < 1
                or residual_ms.shape[1:] != residual_b.shape
                or not np.isfinite(residual_ms).all()):
            raise ValueError("residual_m_coefficients 必须是有限的 Kx3xRxC 数组")
        if self.background_method in GEOMETRY_BACKGROUND_METHODS and (
                residual_ms.shape!=(0,*residual_b.shape)):
            raise ValueError("几何背景模式不应再包含显式 M 模式")
        format_version={
            "physical_residual":17,"direct_fit":18,"direct_fit_3":19,
            "geometry_cache":21,
        }[self.background_method]
        data = {"format_version":format_version,
                "background_method":self.background_method,
                "channel_order": list(CHANNELS),
                "light_source_layout": light_source_layout_mapping(source_layout),
                "bspline_degree":3,
                "residual_correction_mode": (
                    "difference_only" if self.background_method=="physical_residual"
                    else "geometry_background_plus_additive_session"
                    if self.background_method=="geometry_cache" else
                    "static_base_plus_geometry_delta_plus_additive_session"),
                "base_coefficient_mode": (
                    "free" if self.background_method=="physical_residual"
                    else "geometry_cache_static_texture"
                    if self.background_method=="geometry_cache" else
                    "direct_static_texture"),
                "residual_b_axis_order": ["RGB", "surface_row", "surface_column"],
                "residual_b_bspline_coefficients": residual_b.tolist()}
        if self.background_method=="physical_residual":
            if delta.shape!=(source_count,2):
                raise ValueError("光源几何偏移 delta 必须是 Sx2（每条灯带 [x, normal]）")
            if beta.ndim!=2 or beta.shape[0]!=source_count or beta.shape[1]<4 \
                    or not np.isfinite(beta).all():
                raise ValueError("beta 必须是有限的 SxK 数组且 K>=4")
            if scatter_ratio.shape!=(source_count,) or np.any(
                    (scatter_ratio<0)|(scatter_ratio>1)):
                raise ValueError("scatter_ratio 必须是每条灯带一个 [0,1] 范围内的数")
            if scatter_length.shape!=(source_count,) or np.any(scatter_length<=0):
                raise ValueError("scatter_length 必须是每条灯带一个正数")
            if mixing_matrix.shape!=(3,3) or np.any(mixing_matrix<0) \
                    or not np.allclose(mixing_matrix.sum(axis=1),1,atol=1e-5):
                raise ValueError("mixing_matrix 必须是非负且每行和为 1 的 3x3 矩阵")
            data.update({
                "residual_m_basis":"raw_offline",
                "residual_m_axis_order":[
                    "M","RGB","surface_row","surface_column"],
                "residual_m_bspline_coefficients":residual_ms.tolist(),
                "dark_bias_mode":"fixed_zero","delta_axes":["x","normal"],
                "delta_mm":delta.tolist(),
                "bspline_coefficients":beta.tolist(),
                "scatter_ratio":scatter_ratio.tolist(),
                "scatter_length_mm":scatter_length.tolist(),
                "mixing_matrix":mixing_matrix.tolist(),
                "dark_bias_linear_rgb":[0.,0.,0.]})
        elif self.background_method in DIRECT_BACKGROUND_METHODS:
            if self.direct_curve_convexity not in {
                    "none","increasing","decreasing"}:
                raise ValueError("direct_fit 的曲线凸性元数据无效")
            if self.direct_reconstruction_pipeline \
                    != SURFACE_RECONSTRUCTION_PIPELINE_VERSION:
                raise ValueError("direct_fit 的重建链元数据无效")
            frequencies=np.asarray(self.direct_coordinate_frequencies)
            base_texture=np.asarray(self.direct_base_texture)
            feature_mean=np.asarray(self.direct_geometry_feature_mean)
            feature_scale=np.asarray(self.direct_geometry_feature_scale)
            pca_components=np.asarray(self.direct_geometry_pca_components)
            pca_scale=np.asarray(self.direct_geometry_pca_scale)
            local_feature_mean=np.asarray(
                self.direct_local_geometry_feature_mean)
            local_feature_scale=np.asarray(
                self.direct_local_geometry_feature_scale)
            encoder_weights=tuple(np.asarray(value) for value in (
                self.direct_geometry_encoder_weights or ()))
            encoder_biases=tuple(np.asarray(value) for value in (
                self.direct_geometry_encoder_biases or ()))
            decoder_weights=tuple(np.asarray(value) for value in (
                self.direct_decoder_weights or ()))
            decoder_biases=tuple(np.asarray(value) for value in (
                self.direct_decoder_biases or ()))
            channel_decoder_weights=tuple(
                tuple(np.asarray(value) for value in decoder)
                for decoder in (self.direct_channel_decoder_weights or ()))
            channel_decoder_biases=tuple(
                tuple(np.asarray(value) for value in decoder)
                for decoder in (self.direct_channel_decoder_biases or ()))
            if base_texture.ndim!=3 or base_texture.shape[-1]!=3 \
                    or min(base_texture.shape[:2])<2 \
                    or not np.isfinite(base_texture).all() \
                    or np.any((base_texture<0)|(base_texture>1)):
                raise ValueError("direct_fit 的静态 B 纹理无效")
            if frequencies.ndim!=1 or frequencies.size<1 \
                    or not np.isfinite(frequencies).all() \
                    or np.any(frequencies<=0):
                raise ValueError("direct_fit 的坐标频率无效")
            if not isinstance(self.direct_geometry_descriptor_rows,int) \
                    or isinstance(self.direct_geometry_descriptor_rows,bool) \
                    or self.direct_geometry_descriptor_rows<4 \
                    or feature_mean.shape!=(
                        6+10*self.direct_geometry_descriptor_rows,) \
                    or feature_scale.shape!=feature_mean.shape \
                    or not np.isfinite(feature_mean).all() \
                    or not np.isfinite(feature_scale).all() \
                    or np.any(feature_scale<=0):
                raise ValueError("direct_fit 的几何特征归一化参数无效")
            if pca_components.ndim!=2 \
                    or pca_components.shape[0]!=feature_mean.size \
                    or pca_components.shape[1]<1 \
                    or pca_scale.shape!=(pca_components.shape[1],) \
                    or not np.isfinite(pca_components).all() \
                    or not np.isfinite(pca_scale).all() \
                    or np.any(pca_scale<=0):
                raise ValueError("direct_fit 的几何 PCA 参数无效")
            if local_feature_mean.shape!=(15,) \
                    or local_feature_scale.shape!=(15,) \
                    or not np.isfinite(local_feature_mean).all() \
                    or not np.isfinite(local_feature_scale).all() \
                    or np.any(local_feature_scale<=0):
                raise ValueError("direct_fit 的局部几何特征归一化参数无效")

            def validate_network(weights,biases,input_count,output_count,name):
                if not weights or len(weights)!=len(biases):
                    raise ValueError(f"direct_fit 的 {name} 网络无效")
                previous=input_count
                for weight,layer_bias in zip(weights,biases,strict=True):
                    if weight.ndim!=2 or weight.shape[0]!=previous \
                            or layer_bias.shape!=(weight.shape[1],) \
                            or not np.isfinite(weight).all() \
                            or not np.isfinite(layer_bias).all():
                        raise ValueError(f"direct_fit 的 {name} 网络层尺寸无效")
                    previous=weight.shape[1]
                if previous!=output_count:
                    raise ValueError(f"direct_fit 的 {name} 输出尺寸无效")

            latent_count=(encoder_weights[-1].shape[1]
                          if encoder_weights else -1)
            validate_network(
                encoder_weights,encoder_biases,feature_mean.size,latent_count,
                "geometry encoder")
            decoder_input_count=(2+4*frequencies.size+latent_count
                                 +pca_components.shape[1]+15)
            def validate_decoder(weights,biases,output_count,name):
                if not weights or len(weights)!=len(biases):
                    raise ValueError(f"{name} 网络无效")
                previous=decoder_input_count
                for index,(weight,layer_bias) in enumerate(zip(
                        weights,biases,strict=True)):
                    if weight.ndim!=2 or weight.shape[0]!=previous \
                            or layer_bias.shape!=(weight.shape[1],) \
                            or not np.isfinite(weight).all() \
                            or not np.isfinite(layer_bias).all() \
                            or (index==len(weights)-1
                                and weight.shape[1]!=output_count):
                        raise ValueError(f"{name} 网络层尺寸无效")
                    previous=weight.shape[1]+decoder_input_count
            if self.background_method=="direct_fit":
                validate_decoder(
                    decoder_weights,decoder_biases,3,"direct_fit decoder")
                if channel_decoder_weights or channel_decoder_biases:
                    raise ValueError("direct_fit 不应包含分通道 decoder")
            else:
                if decoder_weights or decoder_biases:
                    raise ValueError("direct_fit_3 不应包含共享 RGB decoder")
                if len(channel_decoder_weights)!=3 \
                        or len(channel_decoder_biases)!=3:
                    raise ValueError(
                        "direct_fit_3 必须包含 R/G/B 三个 decoder")
                for channel,weights,biases in zip(
                        CHANNELS,channel_decoder_weights,
                        channel_decoder_biases,strict=True):
                    validate_decoder(
                        weights,biases,1,f"direct_fit_3 {channel} decoder")
            data.update({
                "direct_session_correction_mode":"additive_bspline",
                "direct_base_mode":"robust_full_resolution_texture",
                "direct_delta_mode":"additive_logit",
                "direct_base_axis_order":["surface_row","surface_column","RGB"],
                "direct_base_texture":base_texture.tolist(),
                "direct_decoder_skip_mode":"input_every_layer",
                "direct_coordinate_frequencies":frequencies.tolist(),
                "direct_geometry_descriptor_rows":(
                    self.direct_geometry_descriptor_rows),
                "direct_curve_convexity":self.direct_curve_convexity,
                "direct_reconstruction_pipeline":(
                    self.direct_reconstruction_pipeline),
                "direct_geometry_feature_mean":feature_mean.tolist(),
                "direct_geometry_feature_scale":feature_scale.tolist(),
                "direct_geometry_pca_components":pca_components.tolist(),
                "direct_geometry_pca_scale":pca_scale.tolist(),
                "direct_local_geometry_feature_mean":local_feature_mean.tolist(),
                "direct_local_geometry_feature_scale":local_feature_scale.tolist(),
                "direct_geometry_encoder_weights":[
                    value.tolist() for value in encoder_weights],
                "direct_geometry_encoder_biases":[
                    value.tolist() for value in encoder_biases]})
            if self.background_method=="direct_fit":
                data.update({
                    "direct_decoder_weights":[
                        value.tolist() for value in decoder_weights],
                    "direct_decoder_biases":[
                        value.tolist() for value in decoder_biases]})
            else:
                data.update({
                    "direct_base_channel_mode":"independent_huber",
                    "direct_channel_decoder_order":list(CHANNELS),
                    "direct_channel_decoder_weights":[
                        [value.tolist() for value in decoder]
                        for decoder in channel_decoder_weights],
                    "direct_channel_decoder_biases":[
                        [value.tolist() for value in decoder]
                        for decoder in channel_decoder_biases]})
        else:
            # 用构造器复用全部尺寸/数值校验，再保存明确的缓存插值语义。
            validated=LightFieldModel.geometry_cache(
                residual_b,
                base_texture=np.asarray(self.geometry_cache_base_texture),
                anchor_coefficients=np.asarray(
                    self.geometry_cache_anchor_coefficients),
                descriptor_mean=np.asarray(
                    self.geometry_cache_descriptor_mean),
                descriptor_scale=np.asarray(
                    self.geometry_cache_descriptor_scale),
                pca_components=np.asarray(
                    self.geometry_cache_pca_components),
                pca_scale=np.asarray(self.geometry_cache_pca_scale),
                anchor_keys=np.asarray(self.geometry_cache_anchor_keys),
                curve_coefficients=self.geometry_cache_curve_coefficients,
                descriptor_huber_delta=(
                    self.geometry_cache_descriptor_huber_delta),
                interpolation_neighbors=(
                    self.geometry_cache_interpolation_neighbors),
                distance_power=self.geometry_cache_distance_power,
                distance_epsilon=self.geometry_cache_distance_epsilon,
                curve_convexity=self.direct_curve_convexity,
                reconstruction_pipeline=self.direct_reconstruction_pipeline,
                source_layout=source_layout)
            data.update({
                "geometry_cache_session_correction_mode":"additive_bspline",
                "geometry_cache_mode":"nearest_anchor_convex_interpolation",
                "geometry_cache_base_axis_order":[
                    "surface_row","surface_column","RGB"],
                "geometry_cache_base_texture":np.asarray(
                    validated.geometry_cache_base_texture).tolist(),
                "geometry_cache_anchor_axis_order":[
                    "anchor","RGB","surface_row","surface_column"],
                "geometry_cache_anchor_bspline_coefficients":np.asarray(
                    validated.geometry_cache_anchor_coefficients).tolist(),
                "geometry_cache_descriptor_basis":"robust_centerline_bspline",
                "geometry_cache_curve_coefficients":(
                    validated.geometry_cache_curve_coefficients),
                "geometry_cache_descriptor_huber_delta_mm":(
                    validated.geometry_cache_descriptor_huber_delta),
                "geometry_cache_descriptor_mean":np.asarray(
                    validated.geometry_cache_descriptor_mean).tolist(),
                "geometry_cache_descriptor_scale":np.asarray(
                    validated.geometry_cache_descriptor_scale).tolist(),
                "geometry_cache_pca_components":np.asarray(
                    validated.geometry_cache_pca_components).tolist(),
                "geometry_cache_pca_scale":np.asarray(
                    validated.geometry_cache_pca_scale).tolist(),
                "geometry_cache_anchor_keys":np.asarray(
                    validated.geometry_cache_anchor_keys).tolist(),
                "geometry_cache_interpolation_neighbors":(
                    validated.geometry_cache_interpolation_neighbors),
                "geometry_cache_distance_power":(
                    validated.geometry_cache_distance_power),
                "geometry_cache_distance_epsilon":(
                    validated.geometry_cache_distance_epsilon),
                "geometry_cache_curve_convexity":(
                    validated.direct_curve_convexity),
                "geometry_cache_reconstruction_pipeline":(
                    validated.direct_reconstruction_pipeline)})
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(output)

def choose_device(name: str = "gpu") -> jax.Device:
    if name not in {"gpu", "cuda"}:
        raise RuntimeError("光场程序只允许 JAX GPU；lightfield.device 必须为 gpu")
    try:
        devices = jax.devices("gpu")
    except RuntimeError as error:
        raise RuntimeError("JAX CUDA 已安装，但 NVIDIA GPU/驱动当前不可用") from error
    if not devices:
        raise RuntimeError("JAX CUDA 已安装，但没有可用的 NVIDIA GPU")
    return devices[0]

def point_set_to_grid(
    data: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(data["xyz"], np.float32); uv = np.asarray(data["uv"], np.float32)
    if "st" not in data:
        raise ValueError("XYZ-UV 映射缺少 st，请重新运行曲面重建")
    st = np.asarray(data["st"], np.float32)
    if "camera_depth" not in data:
        raise ValueError("UV-XYZ 映射缺少 camera_depth，请重新运行曲面重建")
    camera_depth = np.asarray(data["camera_depth"], np.float32)
    section = np.asarray(data["cross_section_index"]); alpha = np.asarray(data["cross_section_alpha"])
    rows, cols = np.unique(section), np.unique(alpha)
    if xyz.shape[0] != rows.size * cols.size: raise ValueError("UV-XYZ 点集不是完整的规则截面网格")
    order = np.lexsort((alpha, section))
    return (xyz[order].reshape(rows.size, cols.size, 3),
            uv[order].reshape(rows.size, cols.size, 2),
            st[order].reshape(rows.size, cols.size, 2),
            camera_depth[order].reshape(rows.size, cols.size))

def bgr_to_linear_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """把 OpenCV BGR 图像转换到光场计算使用的线性 RGB。"""
    srgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return np.where(srgb <= .04045,srgb/12.92,((srgb+.055)/1.055)**2.4)

def sample_rgb(image_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    image = bgr_to_linear_rgb(image_bgr)
    return cv2.remap(image, uv[...,0].astype(np.float32), uv[...,1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def sample_unsaturated_mask(image_bgr: np.ndarray,uv: np.ndarray,
                            saturation_threshold: int) -> np.ndarray:
    """仅当双线性采样覆盖的四个原始像素所有通道均低于阈值时有效。"""
    if not isinstance(saturation_threshold,int) or isinstance(saturation_threshold,bool) \
            or not 1<=saturation_threshold<=255:
        raise ValueError("saturation_threshold 必须是 1..255 的整数")
    height,width=image_bgr.shape[:2]
    x=uv[...,0]; y=uv[...,1]
    x0=np.floor(x).astype(np.int64); y0=np.floor(y).astype(np.int64)
    x1=x0+1; y1=y0+1
    inside=(x0>=0)&(y0>=0)&(x1<width)&(y1<height)
    x0c=np.clip(x0,0,width-1); x1c=np.clip(x1,0,width-1)
    y0c=np.clip(y0,0,height-1); y1c=np.clip(y1,0,height-1)
    neighbors=np.stack([image_bgr[y0c,x0c],image_bgr[y0c,x1c],
                        image_bgr[y1c,x0c],image_bgr[y1c,x1c]],axis=-2)
    return inside & np.all(neighbors<saturation_threshold,axis=(-2,-1))

def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, 1)
    return np.where(image <= .0031308, 12.92*image, 1.055*image**(1/2.4)-.055)


def bgr_to_linear_rgb_jax(image_bgr: Array) -> Array:
    """在当前 JAX 设备上把 uint8 BGR 图像转换为线性 RGB。"""
    srgb=image_bgr[...,::-1].astype(jnp.float32)/255
    return jnp.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)


def linear_rgb_to_bgr8_jax(image: Array) -> Array:
    """在当前 JAX 设备上把线性 RGB 转换为显示用 uint8 BGR。"""
    clipped=jnp.clip(image,0,1)
    srgb=jnp.where(clipped<=.0031308,12.92*clipped,
                   1.055*clipped**(1/2.4)-.055)
    return jnp.rint(srgb[...,::-1]*255).astype(jnp.uint8)


def _bilinear_sample_jax(image: Array,map_y: Array,map_x: Array,
                         *,constant_border: bool = False) -> Array:
    """对 HxWxC JAX 图像做双线性采样。"""
    height,width=image.shape[:2]
    x0=jnp.floor(map_x).astype(jnp.int32); y0=jnp.floor(map_y).astype(jnp.int32)
    x1=x0+1; y1=y0+1
    fx=(map_x-x0.astype(map_x.dtype))[...,None]
    fy=(map_y-y0.astype(map_y.dtype))[...,None]

    def gather(y: Array,x: Array) -> Array:
        value=image[jnp.clip(y,0,height-1),jnp.clip(x,0,width-1)]
        if constant_border:
            inside=(x>=0)&(x<width)&(y>=0)&(y<height)
            value=jnp.where(inside[...,None],value,0)
        return value

    top=gather(y0,x0)*(1-fx)+gather(y0,x1)*fx
    bottom=gather(y1,x0)*(1-fx)+gather(y1,x1)*fx
    return top*(1-fy)+bottom*fy


def sample_linear_rgb_jax(linear_rgb: Array,uv: Array) -> Array:
    """在 GPU 上按 UV 采样已经线性化的 RGB 图像。"""
    return _bilinear_sample_jax(linear_rgb,uv[...,1],uv[...,0],constant_border=True)

def signed_difference_bgr(original_bgr: np.ndarray,rendered_linear_rgb: np.ndarray,
                          valid_mask: np.ndarray,gain: float = 2.) -> np.ndarray:
    """显示 camera-rendered 有符号色差：中性灰为零，gain 放大微小差值。"""
    difference=bgr_to_linear_rgb(original_bgr)-rendered_linear_rgb
    return signed_residual_bgr(difference,valid_mask,gain)

def signed_residual_bgr(residual_linear_rgb: np.ndarray,valid_mask: np.ndarray,
                        gain: float = 2.) -> np.ndarray:
    """显示线性 RGB 有符号残差；中性灰表示零残差。"""
    if gain <= 0: raise ValueError("difference gain 必须为正数")
    residual=np.asarray(residual_linear_rgb,np.float32)
    if residual.ndim != 3 or residual.shape[-1] != 3:
        raise ValueError("residual_linear_rgb 必须是 HxWx3")
    if np.asarray(valid_mask).shape != residual.shape[:2]:
        raise ValueError("valid_mask 尺寸必须与残差图一致")
    visualization=np.clip(.5+.5*gain*residual,0,1)
    visualization=np.where((valid_mask>0)[...,None],visualization,0)
    return (cv2.cvtColor(visualization.astype(np.float32),cv2.COLOR_RGB2BGR)*255).round().astype(np.uint8)

def positive_residual_bgr(residual_linear_rgb: np.ndarray,valid_mask: np.ndarray,
                          gain: float = 2.) -> np.ndarray:
    """显示逐通道截断到非负的线性 RGB 色差；零色差严格显示为黑色。"""
    if gain <= 0: raise ValueError("difference gain 必须为正数")
    residual=np.asarray(residual_linear_rgb,np.float32)
    if residual.ndim != 3 or residual.shape[-1] != 3:
        raise ValueError("residual_linear_rgb 必须是 HxWx3")
    if np.asarray(valid_mask).shape != residual.shape[:2]:
        raise ValueError("valid_mask 尺寸必须与残差图一致")
    visualization=np.clip(gain*np.maximum(residual,0),0,1)
    visualization=np.where((valid_mask>0)[...,None],visualization,0)
    return (cv2.cvtColor(visualization.astype(np.float32),cv2.COLOR_RGB2BGR)*255).round().astype(np.uint8)

def _normalize(value: Array) -> Array:
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8)

def surface_normals(xyz: Array) -> Array:
    du = jnp.concatenate([(xyz[:,1]-xyz[:,0])[:,None], xyz[:,2:]-xyz[:,:-2],
                          (xyz[:,-1]-xyz[:,-2])[:,None]], axis=1)
    dv = jnp.concatenate([(xyz[1]-xyz[0])[None], xyz[2:]-xyz[:-2],
                          (xyz[-1]-xyz[-2])[None]], axis=0)
    normal = _normalize(jnp.cross(_normalize(du), _normalize(dv)))
    # 仓库整体重建定义相机朝向为世界坐标系 +Z；保持连续参数化方向，
    # 仅对整张曲面的朝向做一次统一翻转。
    return normal * jnp.where(jnp.mean(normal[...,2]) < 0, -1., 1.)

def _resample_curve(curve: Array, count: int) -> tuple[Array, Array, Array, Array, Array]:
    segment = jnp.linalg.norm(curve[1:] - curve[:-1], axis=-1)
    cumulative = jnp.concatenate([jnp.zeros(1, curve.dtype), jnp.cumsum(segment)])
    total = jnp.maximum(cumulative[-1], 1e-8); target = jnp.linspace(0., total, count)
    index = jnp.clip(jnp.searchsorted(cumulative, target, side="right"), 1, curve.shape[0]-1)
    fraction = ((target-cumulative[index-1])/jnp.maximum(segment[index-1],1e-8))[:,None]
    points = curve[index-1]*(1-fraction) + curve[index]*fraction
    weights = jnp.concatenate([(.5*(target[1]-target[0]))[None],
                               .5*(target[2:]-target[:-2]),
                               (.5*(target[-1]-target[-2]))[None]])
    return points, target/total, weights, index, fraction

def _interpolate_samples(values: Array, index: Array, fraction: Array) -> Array:
    """使用空间边缘弧长产生的同一插值计划采样附属量（例如法向）。"""
    return values[index-1]*(1-fraction) + values[index]*fraction

def _light_source_boundaries(xyz: Array, normals: Array,
                             layout: LightSourceLayout) -> tuple[tuple[Array,...],tuple[Array,...]]:
    """按展开后的灯带实例顺序返回边界曲线及其法向。"""
    edges_by_side={"left":xyz[:,0],"right":xyz[:,-1],"top":xyz[0],"bottom":xyz[-1]}
    normals_by_side={"left":normals[:,0],"right":normals[:,-1],
                     "top":normals[0],"bottom":normals[-1]}
    specs=light_source_specs(layout)
    return (tuple(edges_by_side[side] for _,side in specs),
            tuple(normals_by_side[side] for _,side in specs))

def _light_source_inward_directions(xyz: Array,
                                    layout: LightSourceLayout) -> tuple[Array,...]:
    """返回各边界指向接触面内部的局部 x 初始方向（不是沿灯带方向）。"""
    inward_by_side={"left":xyz[:,1]-xyz[:,0],
                    "right":xyz[:,-2]-xyz[:,-1],
                    "top":xyz[1]-xyz[0],
                    "bottom":xyz[-2]-xyz[-1]}
    return tuple(inward_by_side[side] for _,side in light_source_specs(layout))

def _curve_tangents(curve: Array) -> Array:
    return _normalize(jnp.concatenate([(curve[1]-curve[0])[None],
        curve[2:]-curve[:-2], (curve[-1]-curve[-2])[None]],axis=0))

def bspline_basis(x: Array, coefficient_count: int, degree: int = 3) -> Array:
    interior = coefficient_count-degree-1
    knots = jnp.concatenate([jnp.zeros(degree+1),
        jnp.arange(1, interior+1, dtype=x.dtype)/(interior+1), jnp.ones(degree+1)])
    basis = ((x[:,None]>=knots[:-1]) & (x[:,None]<knots[1:])).astype(x.dtype)
    endpoint = jax.nn.one_hot(jnp.full(x.shape, coefficient_count-1), basis.shape[1], dtype=x.dtype)
    basis = jnp.where((x == 1)[:,None], endpoint, basis)
    for p in range(1, degree+1):
        columns = []
        for i in range(knots.size-p-1):
            ld, rd = knots[i+p]-knots[i], knots[i+p+1]-knots[i+1]
            left = jnp.where(ld>0, (x-knots[i])/jnp.maximum(ld,1e-12)*basis[:,i], 0)
            right = jnp.where(rd>0, (knots[i+p+1]-x)/jnp.maximum(rd,1e-12)*basis[:,i+1], 0)
            columns.append(left+right)
        basis = jnp.stack(columns, axis=1)
    return basis[:,:coefficient_count]

def rgb_bspline_field(grid_shape: tuple[int,int],coefficients: Array) -> Array:
    """在规则归一化曲面坐标上计算 RGB 二维 B 样条场。"""
    rows,cols=grid_shape
    row_basis=bspline_basis(jnp.linspace(0.,1.,rows),coefficients.shape[1])
    column_basis=bspline_basis(jnp.linspace(0.,1.,cols),coefficients.shape[2])
    return jnp.einsum("hr,krc,wc->hwk",row_basis,coefficients,column_basis)

def _bspline_basis_numpy(x: np.ndarray,coefficient_count: int,degree: int = 3) -> np.ndarray:
    """与 ``bspline_basis`` 相同的 NumPy Cox-de Boor 基，用于离线线性拟合。"""
    values=np.asarray(x,np.float64).reshape(-1)
    interior=coefficient_count-degree-1
    knots=np.concatenate([np.zeros(degree+1),
        np.arange(1,interior+1,dtype=np.float64)/(interior+1),np.ones(degree+1)])
    basis=((values[:,None]>=knots[:-1])&(values[:,None]<knots[1:])).astype(np.float64)
    basis[values==1]=0
    basis[values==1,coefficient_count-1]=1
    for order in range(1,degree+1):
        columns=[]
        for index in range(knots.size-order-1):
            left_denominator=knots[index+order]-knots[index]
            right_denominator=knots[index+order+1]-knots[index+1]
            left=(values-knots[index])/left_denominator*basis[:,index] \
                if left_denominator>0 else np.zeros_like(values)
            right=(knots[index+order+1]-values)/right_denominator*basis[:,index+1] \
                if right_denominator>0 else np.zeros_like(values)
            columns.append(left+right)
        basis=np.stack(columns,axis=1)
    return basis[:,:coefficient_count]

def evaluate_rgb_bspline(coefficients: np.ndarray,
                         grid_shape: tuple[int,int]) -> np.ndarray:
    """在规则网格上以 NumPy 计算一个 3xRxC RGB B 样条场。"""
    values=np.asarray(coefficients,np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or min(values.shape[1:])<4:
        raise ValueError("RGB B 样条系数必须是 3xRxC，且 R/C 至少为 4")
    rows,columns=grid_shape
    row_basis=_bspline_basis_numpy(np.linspace(0,1,rows),values.shape[1])
    column_basis=_bspline_basis_numpy(np.linspace(0,1,columns),values.shape[2])
    return np.einsum("hr,krc,wc->hwk",row_basis,values,column_basis).astype(np.float32)


def direct_background_features_jax(
    coordinates: Array,frequencies: Array,
) -> Array:
    """把规范曲面坐标编码为 direct 解码器的多尺度 Fourier 特征。"""
    angles=2*jnp.pi*coordinates[...,None]*frequencies
    return jnp.concatenate([
        coordinates,jnp.sin(angles).reshape((*coordinates.shape[:-1],-1)),
        jnp.cos(angles).reshape((*coordinates.shape[:-1],-1))],axis=-1)


def _linear_resample_first_axis(values: Array,count: int) -> Array:
    positions=jnp.linspace(0.,values.shape[0]-1,count)
    lower=jnp.floor(positions).astype(jnp.int32)
    upper=jnp.minimum(lower+1,values.shape[0]-1)
    fraction=positions-lower
    return values[lower]*(1-fraction[...,None])+values[upper]*fraction[...,None]


def direct_geometry_descriptor_jax(xyz: Array,descriptor_rows: int = 16) -> Array:
    """从整体 XYZ 提取可在标定和在线完全复现的固定维几何描述。"""
    if descriptor_rows<4:
        raise ValueError("descriptor_rows 必须至少为 4")
    centerline=jnp.mean(xyz,axis=1)
    center=_linear_resample_first_axis(centerline,descriptor_rows)
    relative_center=center-jnp.mean(xyz,axis=(0,1))
    previous=center[1:-1]-center[:-2]
    following=center[2:]-center[1:-1]
    tangent=center[2:]-center[:-2]
    tangent/=jnp.maximum(jnp.linalg.norm(tangent,axis=-1,keepdims=True),1e-9)
    second=following-previous
    normal_second=second-jnp.sum(
        second*tangent,axis=-1,keepdims=True)*tangent
    spacing=.5*(jnp.linalg.norm(previous,axis=-1)
                +jnp.linalg.norm(following,axis=-1))
    curvature_inner=normal_second/jnp.maximum(spacing[...,None]**2,1e-12)
    curvature=jnp.concatenate([
        curvature_inner[:1],curvature_inner,curvature_inner[-1:]],axis=0)
    width=jnp.linalg.norm(xyz[:,-1]-xyz[:,0],axis=-1,keepdims=True)
    sampled_width=_linear_resample_first_axis(width,descriptor_rows)
    mean_normals=_linear_resample_first_axis(
        jnp.mean(surface_normals(xyz),axis=1),descriptor_rows)
    global_values=jnp.concatenate([
        jnp.mean(xyz,axis=(0,1)),jnp.std(xyz,axis=(0,1))])
    return jnp.concatenate([
        global_values,relative_center.reshape(-1),curvature.reshape(-1),
        sampled_width.reshape(-1),mean_normals.reshape(-1)])


def geometry_cache_descriptor_jax(
    xyz: Array,curve_coefficients: int = 12,huber_delta: float = .5,
) -> Array:
    """用鲁棒低频中心线/宽度样条描述整体弯曲，抑制局部接触突变。"""
    if curve_coefficients<4 or curve_coefficients>xyz.shape[0] \
            or huber_delta<=0:
        raise ValueError("geometry_cache descriptor 参数无效")
    centerline=jnp.mean(xyz,axis=1)
    width=jnp.linalg.norm(xyz[:,-1]-xyz[:,0],axis=-1,keepdims=True)
    targets=jnp.concatenate([centerline,width],axis=-1)
    basis=bspline_basis(
        jnp.linspace(0.,1.,xyz.shape[0],dtype=xyz.dtype),curve_coefficients)
    second=basis[:-2]-2*basis[1:-1]+basis[2:]
    regularization=(.01*(second.T@second)/max(second.shape[0],1)
                    +1e-7*jnp.eye(curve_coefficients,dtype=xyz.dtype))

    def solve(weight: Array) -> Array:
        normal=jnp.einsum("hr,h,hs->rs",basis,weight,basis) \
            /jnp.maximum(jnp.sum(weight),1.)+regularization
        rhs=jnp.einsum("hr,h,hd->rd",basis,weight,targets) \
            /jnp.maximum(jnp.sum(weight),1.)
        return jnp.linalg.solve(normal,rhs)

    coefficients=solve(jnp.ones((xyz.shape[0],),xyz.dtype))
    for _ in range(5):
        error=targets-jnp.einsum("hr,rd->hd",basis,coefficients)
        distance=jnp.sqrt(jnp.sum(error[...,:3]**2,axis=-1)
                          +error[...,3]**2+1e-12)
        coefficients=solve(jnp.minimum(1.,huber_delta/distance))
    return coefficients.reshape(-1)


def direct_local_geometry_feature_grid_jax(xyz: Array) -> Array:
    """为每个曲面顶点构造保留空间结构的 15 维局部几何特征。"""
    centerline=jnp.mean(xyz,axis=1)
    segment=centerline[1:]-centerline[:-1]
    tangent_inner=centerline[2:]-centerline[:-2]
    tangent=jnp.concatenate([
        segment[:1],tangent_inner,segment[-1:]],axis=0)
    tangent/=jnp.maximum(jnp.linalg.norm(tangent,axis=-1,keepdims=True),1e-9)
    previous=centerline[1:-1]-centerline[:-2]
    following=centerline[2:]-centerline[1:-1]
    center_tangent=tangent[1:-1]
    second=following-previous
    normal_second=second-jnp.sum(
        second*center_tangent,axis=-1,keepdims=True)*center_tangent
    spacing=.5*(jnp.linalg.norm(previous,axis=-1)
                +jnp.linalg.norm(following,axis=-1))
    curvature_inner=normal_second/jnp.maximum(spacing[...,None]**2,1e-12)
    curvature=jnp.concatenate([
        curvature_inner[:1],curvature_inner,curvature_inner[-1:]],axis=0)
    columns=xyz.shape[1]
    tangent_grid=jnp.broadcast_to(tangent[:,None,:],(xyz.shape[0],columns,3))
    curvature_grid=jnp.broadcast_to(
        curvature[:,None,:],(xyz.shape[0],columns,3))
    lateral=xyz-centerline[:,None,:]
    return jnp.concatenate([
        xyz,surface_normals(xyz),tangent_grid,curvature_grid,lateral],axis=-1)


def direct_local_geometry_features_jax(
    xyz: Array,coordinates: Array,
) -> Array:
    """在任意规范坐标采样局部 XYZ、法向、切向、曲率和横向位置。"""
    feature_grid=direct_local_geometry_feature_grid_jax(xyz)
    return sample_direct_local_geometry_feature_grid_jax(
        feature_grid,coordinates)


def sample_direct_local_geometry_feature_grid_jax(
    feature_grid: Array,coordinates: Array,
) -> Array:
    """从已计算的局部几何网格采样，供训练中复用以避免重复求导。"""
    map_y=jnp.clip(coordinates[...,0],0,1)*(feature_grid.shape[0]-1)
    map_x=jnp.clip(coordinates[...,1],0,1)*(feature_grid.shape[1]-1)
    return _bilinear_sample_jax(feature_grid,map_y,map_x)


def sample_direct_base_texture_jax(
    base_texture: Array,coordinates: Array,
) -> Array:
    """在规范坐标采样 direct 静态 B；不施加任何空间平滑。"""
    texture=jnp.asarray(base_texture,jnp.float32)
    map_y=jnp.clip(coordinates[...,0],0,1)*(texture.shape[0]-1)
    map_x=jnp.clip(coordinates[...,1],0,1)*(texture.shape[1]-1)
    return _bilinear_sample_jax(texture,map_y,map_x)


def direct_geometry_conditions_jax(
    xyz: Array,model: LightFieldModel,
) -> tuple[Array,Array]:
    """同时返回学习型 latent 和确定性 PCA 全局几何直连。"""
    required=(model.direct_geometry_feature_mean,
              model.direct_geometry_feature_scale,
              model.direct_geometry_pca_components,
              model.direct_geometry_pca_scale,
              model.direct_geometry_encoder_weights,
              model.direct_geometry_encoder_biases)
    if model.background_method not in DIRECT_BACKGROUND_METHODS or any(
            value is None for value in required):
        raise ValueError("当前模型不包含 direct geometry encoder")
    descriptor=direct_geometry_descriptor_jax(
        xyz,model.direct_geometry_descriptor_rows)
    normalized=(descriptor-model.direct_geometry_feature_mean) \
        /model.direct_geometry_feature_scale
    assert model.direct_geometry_pca_components is not None
    assert model.direct_geometry_pca_scale is not None
    pca=(normalized@model.direct_geometry_pca_components) \
        /model.direct_geometry_pca_scale
    values=normalized
    assert model.direct_geometry_encoder_weights is not None
    assert model.direct_geometry_encoder_biases is not None
    for index,(weight,bias) in enumerate(zip(
            model.direct_geometry_encoder_weights,
            model.direct_geometry_encoder_biases,strict=True)):
        values=values@weight+bias
        if index<len(model.direct_geometry_encoder_weights)-1:
            values=jax.nn.silu(values)
    return values,pca


def direct_geometry_latent_jax(xyz: Array,model: LightFieldModel) -> Array:
    """返回 direct 网络中维数可配置的学习型全局弯曲状态。"""
    return direct_geometry_conditions_jax(xyz,model)[0]


def direct_background_rgb_jax(
    coordinates: Array,xyz: Array,model: LightFieldModel,
) -> Array:
    """以 sigmoid(logit(B)+delta B) 求值几何条件绝对线性 RGB。"""
    if model.background_method not in DIRECT_BACKGROUND_METHODS \
            or model.direct_base_texture is None \
            or model.direct_coordinate_frequencies is None \
            or model.direct_local_geometry_feature_mean is None \
            or model.direct_local_geometry_feature_scale is None \
            or (model.background_method=="direct_fit" and (
                model.direct_decoder_weights is None
                or model.direct_decoder_biases is None)) \
            or (model.background_method=="direct_fit_3" and (
                model.direct_channel_decoder_weights is None
                or model.direct_channel_decoder_biases is None)):
        raise ValueError("当前模型不包含 direct_fit 几何条件神经场")
    latent,pca=direct_geometry_conditions_jax(xyz,model)
    coordinate_features=direct_background_features_jax(
        coordinates,model.direct_coordinate_frequencies)
    local_features=(direct_local_geometry_features_jax(xyz,coordinates)
                    -model.direct_local_geometry_feature_mean) \
        /model.direct_local_geometry_feature_scale
    latent_features=jnp.broadcast_to(
        latent,(*coordinate_features.shape[:-1],latent.shape[-1]))
    pca_features=jnp.broadcast_to(
        pca,(*coordinate_features.shape[:-1],pca.shape[-1]))
    network_input=jnp.concatenate([
        coordinate_features,latent_features,pca_features,local_features],axis=-1)
    def decode(
        weights: tuple[Array,...],biases: tuple[Array,...],
    ) -> Array:
        values=network_input
        for index,(weight,bias) in enumerate(zip(
                weights,biases,strict=True)):
            if index>0:
                values=jnp.concatenate([values,network_input],axis=-1)
            values=values@weight+bias
            if index<len(weights)-1:
                values=jax.nn.silu(values)
        return values

    if model.background_method=="direct_fit":
        assert model.direct_decoder_weights is not None
        assert model.direct_decoder_biases is not None
        values=decode(
            model.direct_decoder_weights,model.direct_decoder_biases)
    else:
        assert model.direct_channel_decoder_weights is not None
        assert model.direct_channel_decoder_biases is not None
        values=jnp.concatenate([
            decode(weights,biases)
            for weights,biases in zip(
                model.direct_channel_decoder_weights,
                model.direct_channel_decoder_biases,strict=True)
        ],axis=-1)
    base=sample_direct_base_texture_jax(
        model.direct_base_texture,coordinates)
    epsilon=jnp.asarray(1e-4,base.dtype)
    base=jnp.clip(base,epsilon,1-epsilon)
    base_logit=jnp.log(base)-jnp.log1p(-base)
    return jax.nn.sigmoid(base_logit+values)


def direct_background_field_jax(
    grid_shape: tuple[int,int],xyz: Array,model: LightFieldModel,
) -> Array:
    """把当前整体几何对应的 direct 神经场展开到规范曲面纹理。"""
    rows,columns=grid_shape
    s,t=jnp.meshgrid(
        jnp.linspace(0.,1.,rows),jnp.linspace(0.,1.,columns),indexing="ij")
    return direct_background_rgb_jax(jnp.stack([s,t],axis=-1),xyz,model)


def direct_background_field_chunked(
    grid_shape: tuple[int,int],xyz: Array | np.ndarray,
    model: LightFieldModel,*,chunk_size: int = 8192,
    device: jax.Device | None = None,
) -> np.ndarray:
    """按像素分块展开 direct 全场，避免训练后一次性占用数 GiB 激活显存。"""
    rows,columns=grid_shape
    if rows<1 or columns<1:
        raise ValueError("grid_shape 必须为正")
    if not isinstance(chunk_size,int) or isinstance(chunk_size,bool) \
            or chunk_size<1:
        raise ValueError("chunk_size 必须是正整数")
    s,t=np.meshgrid(
        np.linspace(0.,1.,rows,dtype=np.float32),
        np.linspace(0.,1.,columns,dtype=np.float32),indexing="ij")
    coordinates=np.stack([s,t],axis=-1).reshape(-1,2)
    model_gpu=model if device is None else jax.device_put(model,device)
    xyz_gpu=(np.asarray(xyz,np.float32) if device is None
             else jax.device_put(np.asarray(xyz,np.float32),device))
    evaluate=jax.jit(direct_background_rgb_jax)
    parts=[]
    for start in range(0,coordinates.shape[0],chunk_size):
        chunk=coordinates[start:start+chunk_size]
        if device is not None:
            chunk=jax.device_put(chunk,device)
        parts.append(np.asarray(evaluate(chunk,xyz_gpu,model_gpu)))
    return np.concatenate(parts,axis=0).reshape(rows,columns,3)


def geometry_cache_background_field_jax(
    grid_shape: tuple[int,int],xyz: Array,model: LightFieldModel,
) -> Array:
    """从当前整体几何在最近缓存锚点间凸插值得到完整 RGB 背景。"""
    required=(model.geometry_cache_base_texture,
              model.geometry_cache_anchor_coefficients,
              model.geometry_cache_descriptor_mean,
              model.geometry_cache_descriptor_scale,
              model.geometry_cache_pca_components,
              model.geometry_cache_pca_scale,
              model.geometry_cache_anchor_keys)
    if model.background_method!="geometry_cache" or any(
            value is None for value in required):
        raise ValueError("当前模型不包含 geometry_cache 背景")
    assert model.geometry_cache_descriptor_mean is not None
    assert model.geometry_cache_descriptor_scale is not None
    assert model.geometry_cache_pca_components is not None
    assert model.geometry_cache_pca_scale is not None
    assert model.geometry_cache_anchor_keys is not None
    assert model.geometry_cache_anchor_coefficients is not None
    assert model.geometry_cache_base_texture is not None
    descriptor=geometry_cache_descriptor_jax(
        xyz,model.geometry_cache_curve_coefficients,
        model.geometry_cache_descriptor_huber_delta)
    normalized=(descriptor-model.geometry_cache_descriptor_mean) \
        /model.geometry_cache_descriptor_scale
    key=(normalized@model.geometry_cache_pca_components) \
        /model.geometry_cache_pca_scale
    squared_distance=jnp.sum(
        (model.geometry_cache_anchor_keys-key[None])**2,axis=-1)
    negative_distance,indices=jax.lax.top_k(
        -squared_distance,model.geometry_cache_interpolation_neighbors)
    distance=jnp.sqrt(jnp.maximum(-negative_distance,0))
    weight=(distance+model.geometry_cache_distance_epsilon) \
        **(-model.geometry_cache_distance_power)
    weight/=jnp.sum(weight)
    coefficients=jnp.einsum(
        "a,akrc->krc",weight,
        model.geometry_cache_anchor_coefficients[indices])
    correction=rgb_bspline_field(grid_shape,coefficients)
    rows,columns=grid_shape
    s,t=jnp.meshgrid(
        jnp.linspace(0.,1.,rows),jnp.linspace(0.,1.,columns),indexing="ij")
    base=sample_direct_base_texture_jax(
        model.geometry_cache_base_texture,jnp.stack([s,t],axis=-1))
    return jnp.clip(base+correction,0,1)


def geometry_background_field_jax(
    grid_shape: tuple[int,int],xyz: Array,model: LightFieldModel,
) -> Array:
    """统一求值神经场或几何锚点缓存背景。"""
    if model.background_method in DIRECT_BACKGROUND_METHODS:
        return direct_background_field_jax(grid_shape,xyz,model)
    if model.background_method=="geometry_cache":
        return geometry_cache_background_field_jax(grid_shape,xyz,model)
    raise ValueError("当前模型不是几何条件背景模式")

@partial(jax.jit,static_argnames=(
    "row_coefficients","column_coefficients","max_iterations"))
def _fit_bspline_fields_pcg_jax(
    targets: Array,weights: Array,coefficient_priors: Array,
    initial_coefficients: Array,*,row_coefficients: int,
    column_coefficients: int,smooth_lambda: float,magnitude_lambda: float,
    prior_lambda: float,relative_tolerance: float,absolute_tolerance: float,
    max_iterations: int,
) -> tuple[Array,Array,Array]:
    """矩阵自由 GPU PCG；每个 target/weight 对应一张标量 B 样条场。"""
    rows,columns=targets.shape[1:]
    row_basis=bspline_basis(
        jnp.linspace(0.,1.,rows,dtype=targets.dtype),row_coefficients)
    column_basis=bspline_basis(
        jnp.linspace(0.,1.,columns,dtype=targets.dtype),column_coefficients)
    positive_weights=jnp.maximum(weights,0)
    denominators=jnp.maximum(
        jnp.sum(positive_weights,axis=(1,2)),1e-12)
    coefficient_count=row_coefficients*column_coefficients
    penalty_count=max(
        (row_coefficients-2)*column_coefficients
        +row_coefficients*(column_coefficients-2),1)
    smooth_scale=smooth_lambda/penalty_count
    diagonal_scale=(magnitude_lambda/coefficient_count+1e-9
                    +prior_lambda/coefficient_count)

    def second_difference_product(values: Array) -> Array:
        result=jnp.zeros_like(values)
        if row_coefficients>2:
            difference=(values[:,:-2,:]-2*values[:,1:-1,:]
                        +values[:,2:,:])
            result=result.at[:,:-2,:].add(difference)
            result=result.at[:,1:-1,:].add(-2*difference)
            result=result.at[:,2:,:].add(difference)
        if column_coefficients>2:
            difference=(values[:,:,:-2]-2*values[:,:,1:-1]
                        +values[:,:,2:])
            result=result.at[:,:,:-2].add(difference)
            result=result.at[:,:,1:-1].add(-2*difference)
            result=result.at[:,:,2:].add(difference)
        return result

    def data_adjoint(values: Array) -> Array:
        return jnp.einsum(
            "hr,mhw,wc->mrc",row_basis,positive_weights*values,
            column_basis)/denominators[:,None,None]

    def matvec(values: Array) -> Array:
        fitted=jnp.einsum(
            "hr,mrc,wc->mhw",row_basis,values,column_basis)
        return (data_adjoint(fitted)
                +smooth_scale*second_difference_product(values)
                +diagonal_scale*values)

    rhs=data_adjoint(targets) \
        +(prior_lambda/coefficient_count)*coefficient_priors

    # Jacobi 预条件器只存 MxRxC 对角线；不再生成 (R*C)^2 正规矩阵。
    data_diagonal=jnp.einsum(
        "hr,mhw,wc->mrc",row_basis**2,positive_weights,
        column_basis**2)/denominators[:,None,None]
    row_diagonal=jnp.zeros((row_coefficients,),targets.dtype)
    if row_coefficients>2:
        row_diagonal=row_diagonal.at[:-2].add(1)
        row_diagonal=row_diagonal.at[1:-1].add(4)
        row_diagonal=row_diagonal.at[2:].add(1)
    column_diagonal=jnp.zeros((column_coefficients,),targets.dtype)
    if column_coefficients>2:
        column_diagonal=column_diagonal.at[:-2].add(1)
        column_diagonal=column_diagonal.at[1:-1].add(4)
        column_diagonal=column_diagonal.at[2:].add(1)
    diagonal=jnp.maximum(
        data_diagonal
        +smooth_scale*(row_diagonal[:,None]+column_diagonal[None,:])
        +diagonal_scale,1e-12)

    value=initial_coefficients
    residual=rhs-matvec(value)
    preconditioned=residual/diagonal
    direction=preconditioned
    rz=jnp.sum(residual*preconditioned,axis=(1,2))
    rhs_norm=jnp.sqrt(jnp.sum(rhs**2,axis=(1,2)))
    tolerance=jnp.maximum(
        absolute_tolerance,relative_tolerance*rhs_norm)

    def condition(state: tuple[Array,Array,Array,Array,Array]) -> Array:
        iteration,_,current_residual,_,_=state
        residual_norm=jnp.sqrt(jnp.sum(
            current_residual**2,axis=(1,2)))
        return (iteration<max_iterations)&jnp.any(residual_norm>tolerance)

    def iteration(state: tuple[Array,Array,Array,Array,Array]):
        count,current_value,current_residual,current_direction,current_rz=state
        product=matvec(current_direction)
        product_inner=jnp.sum(
            current_direction*product,axis=(1,2))
        residual_norm=jnp.sqrt(jnp.sum(
            current_residual**2,axis=(1,2)))
        active=(residual_norm>tolerance) \
            &jnp.isfinite(product_inner)&(jnp.abs(product_inner)>1e-30)
        alpha=jnp.where(active,current_rz/product_inner,0)
        next_value=current_value+alpha[:,None,None]*current_direction
        next_residual=current_residual-alpha[:,None,None]*product
        next_preconditioned=next_residual/diagonal
        next_rz=jnp.sum(next_residual*next_preconditioned,axis=(1,2))
        beta=jnp.where(
            active&jnp.isfinite(next_rz)&(jnp.abs(current_rz)>1e-30),
            next_rz/current_rz,0)
        next_direction=(next_preconditioned
                        +beta[:,None,None]*current_direction)
        return (count+1,next_value,next_residual,next_direction,next_rz)

    initial_state=(jnp.asarray(0,jnp.int32),value,residual,direction,rz)
    iterations,solution,final_residual,_,_=jax.lax.while_loop(
        condition,iteration,initial_state)
    relative_residual=jnp.sqrt(jnp.sum(
        final_residual**2,axis=(1,2)))/jnp.maximum(rhs_norm,1e-12)
    return solution,iterations,relative_residual


def _fit_scalar_bspline_fields_gpu(
    targets: np.ndarray,weights: np.ndarray,row_coefficients: int,
    column_coefficients: int,smooth_lambda: float,magnitude_lambda: float,
    *,coefficient_priors: np.ndarray | None = None,prior_lambda: float = 0.,
    initial_coefficients: np.ndarray | None = None,
    relative_tolerance: float = 2e-6,max_iterations: int = 2000,
    device: jax.Device | None = None,
) -> np.ndarray:
    """把若干独立标量 B 样条正规方程以矩阵自由 PCG 放到 GPU。"""
    fields=np.asarray(targets,np.float32)
    spatial_weights=np.asarray(weights,np.float32)
    if fields.ndim!=3 or spatial_weights.shape!=fields.shape:
        raise ValueError("targets/weights 必须是相同尺寸的 MxHxW")
    if row_coefficients<4 or column_coefficients<4:
        raise ValueError("B 样条行列控制系数数必须至少为 4")
    if smooth_lambda<0 or magnitude_lambda<0 or prior_lambda<0:
        raise ValueError("B 样条正则强度不能为负")
    expected=(fields.shape[0],row_coefficients,column_coefficients)
    if coefficient_priors is None:
        priors=np.zeros(expected,np.float32)
    else:
        priors=np.asarray(coefficient_priors,np.float32)
        if priors.shape!=expected:
            raise ValueError(f"B 样条系数先验尺寸必须为 {expected}")
    if initial_coefficients is None:
        initial=priors if prior_lambda>0 else np.zeros(expected,np.float32)
    else:
        initial=np.asarray(initial_coefficients,np.float32)
        if initial.shape!=expected:
            raise ValueError(f"B 样条 PCG 初值尺寸必须为 {expected}")
    gpu=choose_device("gpu") if device is None else device
    if gpu.platform!="gpu":
        raise RuntimeError("B 样条拟合只允许 JAX GPU，不提供 CPU 求解路径")
    coefficient_count=row_coefficients*column_coefficients
    if coefficient_count>=2048:
        print(
            f"GPU B 样条 PCG 开始：fields={fields.shape[0]}，"
            f"samples={fields.shape[1]}x{fields.shape[2]}，"
            f"grid={row_coefficients}x{column_coefficients}；"
            "首次调用包含 XLA 编译")
    solution,iterations,relative_residual=_fit_bspline_fields_pcg_jax(
        jax.device_put(fields,gpu),jax.device_put(spatial_weights,gpu),
        jax.device_put(priors,gpu),jax.device_put(initial,gpu),
        row_coefficients=row_coefficients,
        column_coefficients=column_coefficients,
        smooth_lambda=float(smooth_lambda),
        magnitude_lambda=float(magnitude_lambda),
        prior_lambda=float(prior_lambda),
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=1e-9,max_iterations=int(max_iterations))
    host_solution,host_iterations,host_residual=jax.device_get(
        (solution,iterations,relative_residual))
    if coefficient_count>=2048:
        print(
            f"GPU B 样条 PCG 完成：iterations={int(host_iterations)}/"
            f"{max_iterations}，max_relative_residual="
            f"{float(np.max(host_residual)):.3e}")
    if not np.isfinite(host_solution).all():
        raise RuntimeError("GPU B 样条 PCG 产生了非有限结果")
    return np.asarray(host_solution,np.float32)

def _fit_rgb_bspline(target: np.ndarray,weight: np.ndarray,row_coefficients: int,
                     column_coefficients: int,smooth_lambda: float,
                     magnitude_lambda: float,*,
                     coefficient_prior: np.ndarray | None = None,
                     prior_lambda: float = 0.,
                     initial_coefficients: np.ndarray | None = None,
                     device: jax.Device | None = None) -> np.ndarray:
    """用共同空间权重在 GPU 上矩阵自由拟合一个或多个 RGB 场。"""
    fields=np.asarray(target,np.float32)
    single=fields.ndim==3
    if single: fields=fields[None]
    if fields.ndim!=4 or fields.shape[-1]!=3:
        raise ValueError("target 必须是 HxWx3 或 KxHxWx3")
    field_count,rows,columns,_=fields.shape
    spatial_weight=np.asarray(weight,np.float32)
    if spatial_weight.shape!=(rows,columns):
        raise ValueError("weight 尺寸必须与 target 空间尺寸一致")
    priors=None
    if coefficient_prior is not None:
        prior=np.asarray(coefficient_prior,np.float32)
        if single and prior.ndim==3:
            prior=prior[None]
        expected=(field_count,3,row_coefficients,column_coefficients)
        if prior.shape!=expected:
            raise ValueError(f"B 样条系数先验尺寸必须为 {expected}")
        priors=prior.reshape(field_count*3,row_coefficients,column_coefficients)
    initial=None
    if initial_coefficients is not None:
        initial=np.asarray(initial_coefficients,np.float32)
        if single and initial.ndim==3:
            initial=initial[None]
        expected=(field_count,3,row_coefficients,column_coefficients)
        if initial.shape!=expected:
            raise ValueError(f"B 样条 PCG 初值尺寸必须为 {expected}")
        initial=initial.reshape(
            field_count*3,row_coefficients,column_coefficients)
    scalar_fields=fields.transpose(0,3,1,2).reshape(
        field_count*3,rows,columns)
    scalar_weights=np.broadcast_to(
        spatial_weight,scalar_fields.shape).copy()
    coefficients=_fit_scalar_bspline_fields_gpu(
        scalar_fields,scalar_weights,row_coefficients,column_coefficients,
        smooth_lambda,magnitude_lambda,coefficient_priors=priors,
        prior_lambda=prior_lambda,initial_coefficients=initial,device=device
    ).reshape(field_count,3,row_coefficients,column_coefficients)
    return coefficients[0] if single else coefficients


def fit_rgb_bspline_field_gpu(
    target: np.ndarray,weight: np.ndarray,row_coefficients: int,
    column_coefficients: int,smooth_lambda: float,magnitude_lambda: float,
    *,device: jax.Device,
) -> np.ndarray:
    """拟合缓存背景使用的 RGB B-spline 空间场。"""
    return _fit_rgb_bspline(
        target,weight,row_coefficients,column_coefficients,
        smooth_lambda,magnitude_lambda,device=device)


def fit_startup_residual_bsession_model(
    residuals: np.ndarray,
    valid: np.ndarray,
    offline_b_coefficients: np.ndarray,
    offline_m_coefficients: np.ndarray,
    *,
    huber_delta: float = .04,
    smooth_lambda: float = .01,
    magnitude_lambda: float = 1e-4,
    outer_weight: float = .2,
    outer_fraction: float = .05,
    bsession_prior_lambda: float = .01,
    bsession_max_field_deviation: float = .35,
    bsession_coefficient_bounds: tuple[float,float] | None = None,
    channel_huber_ratio_min: float = .5,
    channel_huber_ratio_max: float = 2.,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    """拟合 Bsession，并把离线 raw M 等价变换为当前会话的正交基底。"""
    samples=np.asarray(residuals,np.float64)
    mask=np.asarray(valid,bool)
    b_coefficients=np.asarray(offline_b_coefficients,np.float64)
    m_coefficients=np.asarray(offline_m_coefficients,np.float64)
    if samples.ndim!=4 or samples.shape[-1]!=3 or mask.shape!=samples.shape[:3]:
        raise ValueError("residuals/valid 必须分别是 NxHxWx3 和 NxHxW")
    if samples.shape[0]<2:
        raise ValueError("启动残差模式至少需要两帧")
    if np.any(mask.reshape(mask.shape[0],-1).sum(axis=1)==0):
        raise ValueError("每个启动残差帧都必须包含有效曲面像素")
    if b_coefficients.ndim!=3 or b_coefficients.shape[0]!=3 \
            or min(b_coefficients.shape[1:])<4:
        raise ValueError("offline_b_coefficients 必须是有效的 3xRxC B 样条系数")
    if m_coefficients.ndim!=4 or m_coefficients.shape[0]<1 \
            or m_coefficients.shape[1:]!=b_coefficients.shape \
            or not np.isfinite(m_coefficients).all():
        raise ValueError("offline_m_coefficients 必须是有效的 Kx3xRxC M 样条系数")
    if huber_delta<=0 or smooth_lambda<0 or magnitude_lambda<0 \
            or bsession_prior_lambda<0:
        raise ValueError("启动残差拟合正则参数无效")
    if not 0<=outer_weight<=1 or not 0<=outer_fraction<.5:
        raise ValueError("outer_weight/outer_fraction 范围无效")
    if bsession_max_field_deviation<=0:
        raise ValueError("Bsession 空间场幅度上限必须为正")
    if bsession_coefficient_bounds is None:
        bsession_lower=-float(bsession_max_field_deviation)
        bsession_upper=float(bsession_max_field_deviation)
    else:
        if len(bsession_coefficient_bounds)!=2:
            raise ValueError("Bsession 控制系数范围必须包含两个数")
        bsession_lower,bsession_upper=map(float,bsession_coefficient_bounds)
        if not np.isfinite([bsession_lower,bsession_upper]).all() \
                or bsession_upper<=bsession_lower:
            raise ValueError("Bsession 控制系数范围无效")
    if channel_huber_ratio_min<=0 or channel_huber_ratio_max<channel_huber_ratio_min:
        raise ValueError("分通道 Huber 比例上下界无效")
    _,rows,columns,_=samples.shape
    row_coefficients,column_coefficients=b_coefficients.shape[1:]
    row_coordinate=np.linspace(0,1,rows)[:,None]
    column_coordinate=np.linspace(0,1,columns)[None,:]
    outer=((row_coordinate<outer_fraction)|(row_coordinate>1-outer_fraction)
           |(column_coordinate<outer_fraction)|(column_coordinate>1-outer_fraction))
    spatial=np.where(outer,outer_weight,1.).astype(np.float64)

    # 时间中位数去掉任何静态底色，因此通道噪声尺度不依赖离线 B 是否匹配。
    masked_bsession=np.ma.array(
        samples,mask=np.broadcast_to(~mask[...,None],samples.shape))
    temporal_median=np.ma.median(masked_bsession,axis=0).filled(0.)
    channel_scale=np.asarray([
        1.4826*np.median(np.abs(
            samples[...,channel]-temporal_median[...,channel])[mask])
        for channel in range(3)],np.float64)
    channel_scale=np.maximum(channel_scale,max(1e-4,.05*huber_delta))
    reference_scale=max(float(np.median(channel_scale)),1e-12)
    channel_huber=huber_delta*np.clip(
        channel_scale/reference_scale,
        channel_huber_ratio_min,channel_huber_ratio_max)

    # 直接拟合当前会话的固定底色 Bsession；B 只通过系数先验项提供软约束，
    # 不再作为独立在线场。三个通道分别计算 Huber 权重。
    fit_weight=mask.astype(np.float64)*spatial[None]
    weight_sum=fit_weight.sum(axis=0)
    target=np.sum(fit_weight[...,None]*samples,axis=0) \
        /np.maximum(weight_sum[...,None],1e-12)
    bsession_coefficients=_fit_rgb_bspline(
        target,weight_sum,row_coefficients,column_coefficients,
        smooth_lambda,magnitude_lambda,
        coefficient_prior=b_coefficients,prior_lambda=bsession_prior_lambda)
    for iteration in range(5):
        bsession_field=evaluate_rgb_bspline(bsession_coefficients,(rows,columns))
        error=samples-bsession_field[None]
        channel_targets=[]
        channel_weights=[]
        for channel in range(3):
            robust=np.minimum(
                1.,channel_huber[channel]/
                np.maximum(np.abs(error[...,channel]),1e-12))
            channel_weight=mask*spatial[None]*robust
            channel_weight_sum=channel_weight.sum(axis=0)
            channel_targets.append(np.sum(
                channel_weight*samples[...,channel],axis=0)
                /np.maximum(channel_weight_sum,1e-12))
            channel_weights.append(channel_weight_sum)
        # 三个通道的 Huber 权重不同，但线性系统可以作为三个独立 RHS
        # 在一次 GPU PCG 中并行求解；上一轮系数作为热启动。
        next_bsession=_fit_scalar_bspline_fields_gpu(
            np.stack(channel_targets),np.stack(channel_weights),
            row_coefficients,column_coefficients,smooth_lambda,
            magnitude_lambda,coefficient_priors=b_coefficients,
            prior_lambda=bsession_prior_lambda,
            initial_coefficients=bsession_coefficients)
        bsession_coefficients=np.clip(
            next_bsession,bsession_lower,bsession_upper)
        if row_coefficients*column_coefficients>=2048:
            print(f"GPU Bsession IRLS {iteration+1}/5")

    bsession_field=evaluate_rgb_bspline(bsession_coefficients,(rows,columns))
    centered=samples-bsession_field[None]
    coverage=mask.mean(axis=0)*spatial
    m_fields=np.stack([
        evaluate_rgb_bspline(coefficients,(rows,columns))
        for coefficients in m_coefficients])
    training_scores=np.zeros(
        (samples.shape[0],3,m_coefficients.shape[0]),np.float64)
    # 先在离线 raw M 上求逐通道启动分数，再以实际 Bsession 做保持总修正场
    # 不变的会话级正交化。分数拟合仍使用原有空间权重、Huber 和五轮 IRLS。
    for channel in range(3):
        fields=m_fields[...,channel]
        for index in range(samples.shape[0]):
            current_weight=mask[index]*spatial
            scores=np.zeros(m_coefficients.shape[0],np.float64)
            for _ in range(5):
                error=centered[index,...,channel]-np.einsum(
                    "k,khw->hw",scores,fields)
                robust_score=np.minimum(
                    1.,channel_huber[channel]/np.maximum(np.abs(error),1e-12))
                weight=current_weight*robust_score
                normal=np.einsum("hw,khw,lhw->kl",weight,fields,fields)
                normal+=1e-8*np.eye(m_coefficients.shape[0])
                rhs=np.einsum(
                    "hw,khw,hw->k",weight,fields,
                    centered[index,...,channel])
                scores=np.clip(np.linalg.solve(normal,rhs),-10.,10.)
            training_scores[index,channel]=scores

    bsession_coefficients=np.clip(
        bsession_coefficients,bsession_lower,bsession_upper)
    # 离线模型保留 raw M。启动分数先在 raw M 上拟合，再把每个通道的 M
    # 相对于本次实际 Bsession 正交化。同步变换 Bsession 系数后，修正场严格不变。
    session_alpha=np.zeros((3,m_coefficients.shape[0]),np.float64)
    session_m_coefficients=m_coefficients.copy()
    for channel in range(3):
        b_field=bsession_field[...,channel]
        b_energy=float(np.sum(coverage*b_field**2))
        for m_index in range(m_coefficients.shape[0]):
            m_field=m_fields[m_index,...,channel]
            cross=float(np.sum(coverage*b_field*m_field))
            session_alpha[channel,m_index]=cross/max(b_energy,1e-12)
            session_m_coefficients[m_index,channel]-=(
                session_alpha[channel,m_index]*bsession_coefficients[channel])
    field_coefficients=np.stack(
        [bsession_coefficients,*session_m_coefficients]).astype(np.float32)
    startup_scores=np.empty(
        (samples.shape[0],3,1+m_coefficients.shape[0]),np.float32)
    startup_scores[:,:,0]=(1+np.einsum(
        "nck,ck->nc",training_scores,session_alpha)).astype(np.float32)
    startup_scores[:,:,1:]=training_scores
    residual_fields=np.stack([
        evaluate_rgb_bspline(coefficients,(rows,columns))
        for coefficients in field_coefficients])
    fitted_bsession=np.broadcast_to(bsession_field[None],samples.shape)
    fitted_full=np.einsum(
        "nck,khwc->nhwc",startup_scores,residual_fields)
    valid_count=max(int(mask.sum()),1)

    def valid_rmse(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(
            np.where(mask[...,None],values**2,0),axis=(0,1,2))/valid_count)

    diagnostics={
        "raw_rmse_rgb":valid_rmse(samples),
        "bsession_rmse_rgb":valid_rmse(samples-fitted_bsession),
        "bsession_m_rmse_rgb":valid_rmse(samples-fitted_full),
        "cross_frame_floor_rmse_rgb":valid_rmse(
            samples-temporal_median[None]),
        "bspline_spatial_miss_rmse_rgb":valid_rmse(
            np.broadcast_to(
                (temporal_median-bsession_field)[None],samples.shape)),
    }
    diagnostics={key:value.astype(np.float32)
                 for key,value in diagnostics.items()}
    return (field_coefficients,startup_scores,channel_huber.astype(np.float32),
            diagnostics)


def fit_startup_direct_bsession_model(
    fields: np.ndarray,
    valid: np.ndarray,
    offline_session_correction_coefficients: np.ndarray,
    *,
    huber_delta: float = .04,
    smooth_lambda: float = .01,
    magnitude_lambda: float = 1e-4,
    outer_weight: float = .2,
    outer_fraction: float = .05,
    bsession_prior_lambda: float = .01,
    session_correction_bounds: tuple[float,float] = (-.15,.15),
    channel_huber_ratio_min: float = .5,
    channel_huber_ratio_max: float = 2.,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    """对“观测减神经场”的样本只拟合一个低频加性会话修正。"""
    samples=np.asarray(fields,np.float64)
    mask=np.asarray(valid,bool)
    prior=np.asarray(offline_session_correction_coefficients,np.float64)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or mask.shape!=samples.shape[:3] or samples.shape[0]<2:
        raise ValueError("direct 会话样本必须是至少两帧 NxHxWx3/NxHxW")
    if np.any(mask.reshape(mask.shape[0],-1).sum(axis=1)==0):
        raise ValueError("每个 direct 会话样本都必须包含有效曲面像素")
    if prior.ndim!=3 or prior.shape[0]!=3 or min(prior.shape[1:])<4 \
            or not np.isfinite(prior).all():
        raise ValueError("offline_session_correction_coefficients 尺寸无效")
    if huber_delta<=0 or smooth_lambda<0 or magnitude_lambda<0 \
            or bsession_prior_lambda<0:
        raise ValueError("direct 会话拟合正则参数无效")
    if not 0<=outer_weight<=1 or not 0<=outer_fraction<.5:
        raise ValueError("outer_weight/outer_fraction 范围无效")
    if len(session_correction_bounds)!=2:
        raise ValueError("session_correction_bounds 必须包含两个数")
    lower,upper=map(float,session_correction_bounds)
    if not np.isfinite([lower,upper]).all() or upper<=lower:
        raise ValueError("session_correction_bounds 无效")
    if channel_huber_ratio_min<=0 \
            or channel_huber_ratio_max<channel_huber_ratio_min:
        raise ValueError("分通道 Huber 比例上下界无效")
    _,rows,columns,_=samples.shape
    row_count,column_count=prior.shape[1:]
    row_coordinate=np.linspace(0,1,rows)[:,None]
    column_coordinate=np.linspace(0,1,columns)[None,:]
    outer=((row_coordinate<outer_fraction)|(row_coordinate>1-outer_fraction)
           |(column_coordinate<outer_fraction)|(column_coordinate>1-outer_fraction))
    spatial=np.where(outer,outer_weight,1.).astype(np.float64)

    masked=np.ma.array(
        samples,mask=np.broadcast_to(~mask[...,None],samples.shape))
    temporal_median=np.ma.median(masked,axis=0).filled(0.)
    channel_scale=np.asarray([
        1.4826*np.median(np.abs(
            samples[...,channel]-temporal_median[...,channel])[mask])
        for channel in range(3)],np.float64)
    channel_scale=np.maximum(channel_scale,max(1e-4,.05*huber_delta))
    reference_scale=max(float(np.median(channel_scale)),1e-12)
    channel_huber=huber_delta*np.clip(
        channel_scale/reference_scale,
        channel_huber_ratio_min,channel_huber_ratio_max)

    fit_weight=mask.astype(np.float64)*spatial[None]
    weight_sum=fit_weight.sum(axis=0)
    target=np.sum(fit_weight[...,None]*samples,axis=0) \
        /np.maximum(weight_sum[...,None],1e-12)
    coefficients=_fit_rgb_bspline(
        target,weight_sum,row_count,column_count,smooth_lambda,
        magnitude_lambda,coefficient_prior=prior,
        prior_lambda=bsession_prior_lambda)
    for iteration in range(5):
        field=evaluate_rgb_bspline(coefficients,(rows,columns))
        error=samples-field[None]
        channel_targets=[]
        channel_weights=[]
        for channel in range(3):
            robust=np.minimum(
                1.,channel_huber[channel]/
                np.maximum(np.abs(error[...,channel]),1e-12))
            channel_weight=mask*spatial[None]*robust
            channel_weight_sum=channel_weight.sum(axis=0)
            channel_targets.append(np.sum(
                channel_weight*samples[...,channel],axis=0)
                /np.maximum(channel_weight_sum,1e-12))
            channel_weights.append(channel_weight_sum)
        updated=_fit_scalar_bspline_fields_gpu(
            np.stack(channel_targets),np.stack(channel_weights),
            row_count,column_count,smooth_lambda,magnitude_lambda,
            coefficient_priors=prior,prior_lambda=bsession_prior_lambda,
            initial_coefficients=coefficients)
        coefficients=np.clip(updated,lower,upper)
        if row_count*column_count>=2048:
            print(f"GPU direct Bsession IRLS {iteration+1}/5")

    fitted=evaluate_rgb_bspline(coefficients,(rows,columns))
    valid_count=max(int(mask.sum()),1)

    def valid_rmse(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(
            np.where(mask[...,None],values**2,0),axis=(0,1,2))/valid_count)

    diagnostics={
        "raw_rmse_rgb":valid_rmse(samples),
        "bsession_rmse_rgb":valid_rmse(samples-fitted[None]),
        "bsession_m_rmse_rgb":valid_rmse(samples-fitted[None]),
        "cross_frame_floor_rmse_rgb":valid_rmse(
            samples-temporal_median[None]),
        "bspline_spatial_miss_rmse_rgb":valid_rmse(
            np.broadcast_to((temporal_median-fitted)[None],samples.shape)),
    }
    empty_scores=np.zeros((samples.shape[0],3,0),np.float32)
    return (coefficients.astype(np.float32),empty_scores,
            channel_huber.astype(np.float32),
            {key:value.astype(np.float32) for key,value in diagnostics.items()})


def _surface_diffusion_geometry(xyz: Array) -> tuple[Array,Array,Array,Array,Array]:
    """由规则 XYZ 网格构造集总顶点面积和非负余切拉普拉斯的边表示。"""
    rows,cols=xyz.shape[:2]
    indices=jnp.arange(rows*cols,dtype=jnp.int32).reshape(rows,cols)
    triangles=jnp.concatenate([
        jnp.stack([indices[:-1,:-1],indices[1:,:-1],indices[1:,1:]],axis=-1).reshape(-1,3),
        jnp.stack([indices[:-1,:-1],indices[1:,1:],indices[:-1,1:]],axis=-1).reshape(-1,3),
    ],axis=0)
    points=xyz.reshape(-1,3); pa,pb,pc=(points[triangles[:,i]] for i in range(3))
    twice_area=jnp.maximum(jnp.linalg.norm(jnp.cross(pb-pa,pc-pa),axis=-1),1e-8)
    vertex_area=jnp.zeros(rows*cols,xyz.dtype).at[triangles.reshape(-1)].add(
        jnp.repeat(twice_area/6,3))
    cot_a=jnp.sum((pb-pa)*(pc-pa),axis=-1)/twice_area
    cot_b=jnp.sum((pc-pb)*(pa-pb),axis=-1)/twice_area
    cot_c=jnp.sum((pa-pc)*(pb-pc),axis=-1)/twice_area
    edge_i=jnp.concatenate([triangles[:,1],triangles[:,2],triangles[:,0]])
    edge_j=jnp.concatenate([triangles[:,2],triangles[:,0],triangles[:,1]])
    # 钝角三角形会产生负余切权重；裁零后系统保持 M-matrix 和非负扩散。
    edge_weight=.5*jnp.maximum(jnp.concatenate([cot_a,cot_b,cot_c]),0)
    laplacian_diagonal=jnp.zeros(rows*cols,xyz.dtype)
    laplacian_diagonal=laplacian_diagonal.at[edge_i].add(edge_weight)
    laplacian_diagonal=laplacian_diagonal.at[edge_j].add(edge_weight)
    return vertex_area,edge_i,edge_j,edge_weight,laplacian_diagonal

def diffuse_surface_fields(xyz: Array, direct_fields: Array, scatter_length: Array,
                           cg_tolerance: float = 1e-4,
                           cg_max_iterations: int = 30) -> Array:
    """求解 (A + ell^2 L) S = A D，逐灯带在真实 XYZ 曲面拓扑内扩散。"""
    single=xyz.ndim==3
    xyz_batch=xyz[None] if single else xyz
    direct_batch=direct_fields[None] if single else direct_fields
    geometry=jax.vmap(_surface_diffusion_geometry)(xyz_batch)
    area,edge_i_batch,edge_j_batch,edge_weight,laplacian_diagonal=geometry
    # 网格拓扑只由静态 H/W 决定，各样本的边索引完全相同。
    edge_i,edge_j=edge_i_batch[0],edge_j_batch[0]
    batch_size,rows,cols=xyz_batch.shape[:3]
    source_count=direct_batch.shape[-1]
    flattened=direct_batch.reshape(batch_size,rows*cols,source_count)
    # 联合块对角 CG 使用一个全局残差；逐样本逐灯带归一化，避免弱通道被强通道的
    # 残差尺度掩盖。线性求解后再恢复原光强。
    field_scale=jnp.maximum(jnp.sqrt(jnp.mean(flattened**2,axis=1,keepdims=True)),1e-6)
    normalized=flattened/field_scale
    length2=scatter_length[None,None,:]**2
    diagonal=jnp.maximum(area[...,None]+length2*laplacian_diagonal[...,None],1e-8)
    def matvec(value):
        difference=value[:,edge_i,:]-value[:,edge_j,:]
        flux=length2*edge_weight[...,None]*difference
        result=area[...,None]*value
        result=result.at[:,edge_i,:].add(flux)
        return result.at[:,edge_j,:].add(-flux)
    # 不使用 jax.scipy.cg 的隐式 VJP：有限迭代时它按“精确线性解”反传，前向未
    # 收敛会导致梯度与实际 loss 严重不一致。这里显式展开固定步数 PCG，使反向
    # 梯度严格对应实际执行的近似解；每个样本/灯带拥有独立的步长和残差。
    rhs=area[...,None]*normalized
    initial=normalized
    residual=rhs-matvec(initial)
    preconditioned=residual/diagonal
    direction=preconditioned
    rz=jnp.sum(residual*preconditioned,axis=1,keepdims=True)
    rhs_norm=jnp.sqrt(jnp.sum(rhs*rhs,axis=1,keepdims=True))
    def pcg_iteration(_,state):
        value,current_residual,current_direction,current_rz=state
        product=matvec(current_direction)
        denominator=jnp.sum(current_direction*product,axis=1,keepdims=True)
        active=(jnp.sqrt(jnp.sum(current_residual**2,axis=1,keepdims=True))
                > cg_tolerance*jnp.maximum(rhs_norm,1e-8))
        alpha=jnp.where(active& (jnp.abs(denominator)>1e-20),
                        current_rz/denominator,0)
        value=value+alpha*current_direction
        next_residual=current_residual-alpha*product
        next_preconditioned=next_residual/diagonal
        next_rz=jnp.sum(next_residual*next_preconditioned,axis=1,keepdims=True)
        beta=jnp.where(active& (jnp.abs(current_rz)>1e-20),next_rz/current_rz,0)
        next_direction=next_preconditioned+beta*current_direction
        return value,next_residual,next_direction,next_rz
    solution,*_=jax.lax.fori_loop(0,cg_max_iterations,pcg_iteration,
                                  (initial,residual,direction,rz))
    output=(solution*field_scale).reshape(direct_batch.shape)
    return output[0] if single else output

def _direct_light_fields(xyz: Array,model: LightFieldModel,integration_nodes: int,
                         distance_epsilon: float) -> Array:
    """计算每条实体灯带尚未扩散、尚未按颜色聚合的直接光场。"""
    normals = surface_normals(xyz); points = xyz.reshape(-1,3); point_normals = normals.reshape(-1,3)
    edges, edge_normals = _light_source_boundaries(xyz,normals,model.source_layout)
    edge_inward = _light_source_inward_directions(xyz,model.source_layout)
    sources = []
    for source_index in range(len(edges)):
        edge, xi, weights, index, fraction = _resample_curve(
            edges[source_index], integration_nodes)
        edge_normal = _normalize(_interpolate_samples(
            edge_normals[source_index],index,fraction))
        inward = _interpolate_samples(
            edge_inward[source_index],index,fraction)
        tangent = _curve_tangents(edge)
        tangent = _normalize(tangent-jnp.sum(tangent*edge_normal,axis=-1,keepdims=True)*edge_normal)
        # 局部 x 位于曲面切平面内且垂直于灯带，正方向指向接触面内部。
        # 因此该自由度不会退化成灯带沿自身方向的整体滑动。
        local_x = inward-jnp.sum(inward*edge_normal,axis=-1,keepdims=True)*edge_normal
        local_x = local_x-jnp.sum(local_x*tangent,axis=-1,keepdims=True)*tangent
        local_x = _normalize(local_x)
        source = (edge + model.delta[source_index,0]*local_x
                  + model.delta[source_index,1]*edge_normal)
        intensity = (bspline_basis(xi, model.beta.shape[1])
                     @ model.beta[source_index])
        displacement = source[None] - points[:,None]
        distance2 = jnp.sum(displacement**2, axis=-1) + distance_epsilon**2
        kernel = jax.nn.relu(jnp.sum(point_normals[:,None]*displacement, axis=-1))/distance2**1.5
        sources.append(kernel @ (weights*intensity))
    return jnp.stack(sources,axis=-1).reshape((*xyz.shape[:2],len(sources)))

def physical_background_batch(xyz: Array,model: LightFieldModel,integration_nodes: int = 48,
                              distance_epsilon: float = .05,chunk_size: int = 65536,
                              diffusion_cg_tolerance: float = 1e-4,
                              diffusion_cg_max_iterations: int = 30) -> Array:
    """批量近场积分、隐式曲面扩散和可辨识光谱混合；不包含经验残差场。"""
    del chunk_size
    direct_fields=jax.vmap(lambda surface:_direct_light_fields(
        surface,model,integration_nodes,distance_epsilon))(xyz)
    scattered=jax.lax.cond(
        jnp.any(model.scatter_ratio>0),
        lambda fields:diffuse_surface_fields(xyz,fields,model.scatter_length,
                                             diffusion_cg_tolerance,
                                             diffusion_cg_max_iterations),
        lambda fields:fields,direct_fields)
    source_fields=(1-model.scatter_ratio)*direct_fields+model.scatter_ratio*scattered
    source_channels=jnp.asarray(
        [channel for channel,_ in light_source_specs(model.source_layout)],
        dtype=jnp.int32)
    channel_fields=source_fields @ jax.nn.one_hot(
        source_channels,3,dtype=source_fields.dtype)
    # 同色不同边先线性叠加；混合矩阵行表示光源颜色，列表示相机输出 RGB。
    return channel_fields @ model.mixing_matrix

def physical_background(xyz: Array, model: LightFieldModel, integration_nodes: int = 48,
                        distance_epsilon: float = .05, chunk_size: int = 65536,
                        diffusion_cg_tolerance: float = 1e-4,
                        diffusion_cg_max_iterations: int = 30) -> Array:
    """单曲面包装；实际计算与离线批处理共用同一个批量实现。"""
    return physical_background_batch(xyz[None],model,integration_nodes,distance_epsilon,
                                     chunk_size,diffusion_cg_tolerance,
                                     diffusion_cg_max_iterations)[0]

def irls_gain_bias(observed: Array, physical: Array, bias_prior: Array,
                   sigma_rgb: Array, iterations: int = 4,
                   lambda_gain: float = 100., lambda_bias: float = 100.,
                   max_gain_deviation: float = .25,
                   max_bias_deviation: float = .05,
                   gain_prior: Array | None = None,
                   valid_mask: Array | None = None) -> tuple[Array, Array, Array]:
    if lambda_gain < 0 or lambda_bias < 0:
        raise ValueError("lambda_gain 和 lambda_bias 必须大于或等于 0")
    if not 0 <= max_gain_deviation < 1:
        raise ValueError("max_gain_deviation 必须大于或等于 0 且小于 1")
    if max_bias_deviation < 0:
        raise ValueError("max_bias_deviation 必须大于或等于 0")
    gain_prior=(jnp.ones_like(bias_prior) if gain_prior is None else
                jnp.asarray(gain_prior,bias_prior.dtype))
    valid=(jnp.ones(observed.shape[:-1],jnp.bool_)
           if valid_mask is None else jnp.asarray(valid_mask,jnp.bool_))
    if valid.shape!=observed.shape[:-1]:
        raise ValueError("gain/bias valid_mask 尺寸必须与图像域一致")
    # 三维标准高斯向量范数（Chi, k=3）的中位数。
    chi3_median = 1.5381722544550522
    def weights_for_parameters(gain,bias):
        residual = observed-gain*physical-bias
        distance = jnp.sqrt(jnp.sum((residual/sigma_rgb)**2, axis=-1)+1e-12)
        # 径向残差是非负 Chi 分布，不能用中心化 MAD 作为“相对零点”的尺度。
        valid_median=jnp.nanmedian(jnp.where(valid,distance,jnp.nan))
        valid_median=jnp.nan_to_num(
            valid_median,nan=chi3_median,posinf=chi3_median,
            neginf=chi3_median)
        scale = jnp.maximum(valid_median/chi3_median,1e-6)
        z = distance/(4.685*scale+1e-6); weights = jnp.where(z<1,(1-z**2)**2,0)
        return weights*valid.astype(weights.dtype)
    def iteration(_, state):
        gain,bias=state; weights=weights_for_parameters(gain,bias)
        weighted=weights[...,None]
        sum_w=jnp.sum(weights)
        sum_p=jnp.sum(weighted*physical,axis=(0,1))
        sum_pp=jnp.sum(weighted*physical*physical,axis=(0,1))
        sum_o=jnp.sum(weighted*observed,axis=(0,1))
        sum_po=jnp.sum(weighted*physical*observed,axis=(0,1))
        # 对每个 RGB 通道联合求解带先验的 2x2 加权正规方程，避免增益和偏置顺序更新。
        a=sum_pp+lambda_gain; c=sum_w+lambda_bias; cross=sum_p
        rhs_gain=sum_po+lambda_gain*gain_prior
        rhs_bias=sum_o+lambda_bias*bias_prior
        determinant=jnp.maximum(a*c-cross*cross,1e-12)
        gain=(rhs_gain*c-cross*rhs_bias)/determinant
        bias=(a*rhs_bias-cross*rhs_gain)/determinant
        gain=jnp.clip(gain,gain_prior-max_gain_deviation,gain_prior+max_gain_deviation)
        bias=jnp.clip(bias,bias_prior-max_bias_deviation,bias_prior+max_bias_deviation)
        return gain,bias
    gain,bias=jax.lax.fori_loop(0,iterations,iteration,(gain_prior,bias_prior))
    # W_IRLS 必须和最终返回的 gain/bias 对应，而不是落后一轮。
    return gain,bias,weights_for_parameters(gain,bias)

def rasterize_attributes_jax(uv: Array,camera_depth: Array,attributes: Array,
                             image_shape: tuple[int,int],*,triangle_chunk: int = 64,
                             max_triangle_width: int = 128,
                             max_triangle_height: int = 64) -> tuple[Array,Array,Array]:
    """GPU z-buffer 光栅化，一次插值任意数量的顶点属性。

    三角形按固定小批次并行展开其包围盒，避免 Python 逐三角形循环。返回属性图、
    有效像素和包围盒是否超过配置容量；调用方必须拒绝 overflow，不能静默截断。
    """
    height,width=image_shape
    rows,columns=uv.shape[:2]
    vertex_count=rows*columns
    indices=jnp.arange(vertex_count,dtype=jnp.int32).reshape(rows,columns)
    triangles=jnp.concatenate([
        jnp.stack([indices[:-1,:-1],indices[1:,:-1],indices[1:,1:]],axis=-1).reshape(-1,3),
        jnp.stack([indices[:-1,:-1],indices[1:,1:],indices[:-1,1:]],axis=-1).reshape(-1,3),
    ],axis=0)
    triangle_count=triangles.shape[0]
    padded_count=((triangle_count+triangle_chunk-1)//triangle_chunk)*triangle_chunk
    padding=padded_count-triangle_count
    triangles=jnp.pad(triangles,((0,padding),(0,0)))
    triangle_enabled=jnp.arange(padded_count)<triangle_count

    flat_uv=uv.reshape(-1,2); flat_depth=camera_depth.reshape(-1)
    flat_attributes=attributes.reshape(-1,attributes.shape[-1])
    output_depth=jnp.full((height*width,),jnp.inf,jnp.float32)
    output_attributes=jnp.zeros((height*width,attributes.shape[-1]),attributes.dtype)
    offset_x=jnp.arange(max_triangle_width,dtype=jnp.int32)[None,None,:]
    offset_y=jnp.arange(max_triangle_height,dtype=jnp.int32)[None,:,None]
    candidate_count=triangle_chunk*max_triangle_height*max_triangle_width
    candidate_ids=jnp.arange(candidate_count,dtype=jnp.int32)
    missing_id=jnp.int32(candidate_count)

    def rasterize_chunk(chunk_index,state):
        current_depth,current_attributes,overflow=state
        start=chunk_index*triangle_chunk
        tri=jax.lax.dynamic_slice_in_dim(triangles,start,triangle_chunk,axis=0)
        enabled=jax.lax.dynamic_slice_in_dim(triangle_enabled,start,triangle_chunk,axis=0)
        points=flat_uv[tri]
        depths=flat_depth[tri]
        values=flat_attributes[tri]
        minimum=jnp.floor(jnp.min(points,axis=1)).astype(jnp.int32)
        maximum=jnp.ceil(jnp.max(points,axis=1)).astype(jnp.int32)
        required=maximum-minimum+1
        overflow=overflow|jnp.any(enabled&((required[:,0]>max_triangle_width)|
                                             (required[:,1]>max_triangle_height)))

        pixel_x=minimum[:,0,None,None]+offset_x
        pixel_y=minimum[:,1,None,None]+offset_y
        center_x=pixel_x.astype(jnp.float32)+.5
        center_y=pixel_y.astype(jnp.float32)+.5
        a=points[:,0]; b=points[:,1]; c=points[:,2]
        denominator=((b[:,1]-c[:,1])*(a[:,0]-c[:,0])+
                     (c[:,0]-b[:,0])*(a[:,1]-c[:,1]))
        safe_denominator=jnp.where(jnp.abs(denominator)>1e-6,denominator,1)
        wa=((b[:,1,None,None]-c[:,1,None,None])*(center_x-c[:,0,None,None])+
            (c[:,0,None,None]-b[:,0,None,None])*(center_y-c[:,1,None,None])) \
            /safe_denominator[:,None,None]
        wb=((c[:,1,None,None]-a[:,1,None,None])*(center_x-c[:,0,None,None])+
            (a[:,0,None,None]-c[:,0,None,None])*(center_y-c[:,1,None,None])) \
            /safe_denominator[:,None,None]
        wc=1-wa-wb
        inside=(wa>=-1e-4)&(wb>=-1e-4)&(wc>=-1e-4)
        in_image=(pixel_x>=0)&(pixel_x<width)&(pixel_y>=0)&(pixel_y<height)
        in_capacity=(offset_x<required[:,0,None,None])&(offset_y<required[:,1,None,None])
        candidate_depth=(wa*depths[:,0,None,None]+wb*depths[:,1,None,None]+
                         wc*depths[:,2,None,None])
        valid=(enabled[:,None,None]&inside&in_image&in_capacity&
               (jnp.abs(denominator)>1e-6)[:,None,None]&(candidate_depth>0))
        pixel_index=jnp.clip(pixel_y*width+pixel_x,0,height*width-1).reshape(-1)
        candidate_depth=jnp.where(valid,candidate_depth,jnp.inf).reshape(-1)

        chunk_depth=jnp.full_like(current_depth,jnp.inf)
        chunk_depth=chunk_depth.at[pixel_index].min(candidate_depth)
        is_winner=valid.reshape(-1)&jnp.isclose(
            candidate_depth,chunk_depth[pixel_index],rtol=1e-6,atol=1e-7)
        winner_ids=jnp.full((height*width,),missing_id,jnp.int32)
        winner_ids=winner_ids.at[pixel_index].min(jnp.where(
            is_winner,candidate_ids,missing_id))
        interpolation=(wa[...,None]*values[:,0,None,None,:]+
                       wb[...,None]*values[:,1,None,None,:]+
                       wc[...,None]*values[:,2,None,None,:]).reshape(
                           candidate_count,attributes.shape[-1])
        safe_winner=jnp.minimum(winner_ids,candidate_count-1)
        chunk_attributes=interpolation[safe_winner]
        take_chunk=chunk_depth<current_depth
        next_depth=jnp.minimum(current_depth,chunk_depth)
        next_attributes=jnp.where(take_chunk[:,None],chunk_attributes,current_attributes)
        return next_depth,next_attributes,overflow

    output_depth,output_attributes,overflow=jax.lax.fori_loop(
        0,padded_count//triangle_chunk,rasterize_chunk,
        (output_depth,output_attributes,jnp.asarray(False)))
    valid=jnp.isfinite(output_depth)
    return (output_attributes.reshape(height,width,-1),valid.reshape(height,width),overflow)

def surface_coordinate_grid_jax(grid_shape: tuple[int,int]) -> Array:
    """JAX 版规范曲面坐标网格。"""
    rows,columns=grid_shape
    row,column=jnp.meshgrid(jnp.linspace(0,1,rows,dtype=jnp.float32),
                            jnp.linspace(0,1,columns,dtype=jnp.float32),indexing="ij")
    return jnp.stack([row,column],axis=-1)

def sample_residual_correction_jax(coordinate_image: Array,b_texture: Array,
                                   m_textures: Array,valid_mask: Array) -> tuple[Array,Array]:
    """在 GPU 上用规范曲面坐标采样残差 B/M 纹理。"""
    map_y=jnp.clip(coordinate_image[...,0],0,1)*(b_texture.shape[0]-1)
    map_x=jnp.clip(coordinate_image[...,1],0,1)*(b_texture.shape[1]-1)
    b_image=_bilinear_sample_jax(b_texture,map_y,map_x)
    m_images=jax.vmap(lambda texture:_bilinear_sample_jax(texture,map_y,map_x))(m_textures)
    return (jnp.where(valid_mask[...,None],b_image,0),
            jnp.where(valid_mask[None,...,None],m_images,0))


def erode_mask_jax(mask: Array,radius: int) -> Array:
    """使用 JAX reduce-window 在 GPU 上腐蚀二值 mask。"""
    if radius<=0:
        return mask.astype(jnp.bool_)
    size=2*radius+1
    return jax.lax.reduce_window(mask.astype(jnp.bool_),jnp.asarray(True),
        jax.lax.bitwise_and,(size,size),(1,1),"SAME")


def _canonical_dense_uv_jax(
    uv: Array,sample_shape: tuple[int,int],
) -> Array:
    """把稀疏曲面 UV 双线性加密到固定规范曲面网格。"""
    sample_rows,sample_columns=sample_shape
    map_y,map_x=jnp.meshgrid(
        jnp.linspace(0,uv.shape[0]-1,sample_rows,dtype=uv.dtype),
        jnp.linspace(0,uv.shape[1]-1,sample_columns,dtype=uv.dtype),
        indexing="ij")
    return _bilinear_sample_jax(uv,map_y,map_x)


def sample_image_mask_to_canonical_jax(
    source_mask: Array,
    uv: Array,
    sample_shape: tuple[int,int],
) -> Array:
    """按规范残差的同一 UV 映射把图像域二值掩膜最近邻采样到曲面。"""
    dense_uv=_canonical_dense_uv_jax(uv,sample_shape)
    return _sample_image_mask_at_uv_jax(source_mask,dense_uv)


def _sample_image_mask_at_uv_jax(
    source_mask: Array,dense_uv: Array,
) -> Array:
    """按给定的稠密图像 UV 最近邻采样二值掩膜。"""
    nearest_x=jnp.rint(dense_uv[...,0]).astype(jnp.int32)
    nearest_y=jnp.rint(dense_uv[...,1]).astype(jnp.int32)
    inside=((nearest_x>=0)&(nearest_x<source_mask.shape[1])&
            (nearest_y>=0)&(nearest_y<source_mask.shape[0]))
    return (inside&source_mask[
        jnp.clip(nearest_y,0,source_mask.shape[0]-1),
        jnp.clip(nearest_x,0,source_mask.shape[1]-1)])


def build_canonical_residual_sample_jax(
    raw_residual: Array,
    frame_bgr: Array,
    valid_mask: Array,
    uv: Array,
    sample_shape: tuple[int,int],
    *,
    saturation_threshold: int,
    erode_pixels: int,
) -> tuple[Array,Array]:
    """在 GPU 上把图像域残差采样到固定分辨率规范曲面。

    ``frame_bgr`` 和 ``saturation_threshold`` 仅为兼容已有调用保留。规范背景
    必须学习相机实际输出（包括裁剪到 255 的平台），因此有效域只由几何投影
    决定；腐蚀也只作用于几何边界，不再扩大内部饱和区域。
    """
    del frame_bgr,saturation_threshold
    dense_uv=_canonical_dense_uv_jax(uv,sample_shape)
    source_valid=erode_mask_jax(valid_mask,erode_pixels)
    canonical_residual=_bilinear_sample_jax(
        raw_residual,dense_uv[...,1],dense_uv[...,0],constant_border=True)
    canonical_valid=_sample_image_mask_at_uv_jax(source_valid,dense_uv)
    return (jnp.where(canonical_valid[...,None],canonical_residual,0),
            canonical_valid)


def fit_residual_m_scores_jax(residual: Array,b_field: Array,
                              m_fields: Array,weight: Array,
                              huber_delta: float = .04,score_prior: float = 1e-4,
                              score_limit: float = 3.,iterations: int = 5) -> Array:
    """GPU/JAX 版逐帧鲁棒残差模式拟合。"""
    y=(residual-b_field).reshape(-1,3)
    design=m_fields.transpose(1,2,3,0).reshape(-1,3,m_fields.shape[0])
    base=jnp.maximum(weight.reshape(-1),0)

    def iteration(_,scores):
        error=y-jnp.einsum("pck,k->pc",design,scores)
        norm=jnp.linalg.norm(error,axis=1)
        robust=jnp.minimum(1.,huber_delta/jnp.maximum(norm,1e-12))
        current=base*robust
        denominator=jnp.maximum(jnp.sum(current),1e-12)
        normal=jnp.einsum("p,pck,pcl->kl",current,design,design)/denominator
        normal=normal+score_prior*jnp.eye(m_fields.shape[0],dtype=residual.dtype)
        rhs=jnp.einsum("p,pck,pc->k",current,design,y)/denominator
        return jnp.clip(jnp.linalg.solve(normal,rhs),-score_limit,score_limit)

    return jax.lax.fori_loop(0,iterations,iteration,
                             jnp.zeros(m_fields.shape[0],residual.dtype))


def fit_uniform_residual_correction_scores_jax(
    residual: Array,
    bsession_field: Array,
    m_fields: Array,
    valid_mask: Array,
) -> Array:
    """在有效域内等权、无 Huber/先验/边界地求固定 Bsession/M 系数。"""
    fields=jnp.concatenate([bsession_field[None],m_fields],axis=0)
    design=fields.transpose(1,2,3,0).reshape(-1,3,fields.shape[0])
    target=residual.reshape(-1,3)
    valid=jnp.maximum(valid_mask.reshape(-1),0).astype(residual.dtype)
    denominator=jnp.maximum(jnp.sum(valid),1e-12)
    normal=jnp.einsum(
        "p,pck,pcl->ckl",valid,design,design)/denominator
    rhs=jnp.einsum(
        "p,pck,pc->ck",valid,design,target)/denominator
    return _minimum_norm_residual_scores(normal,rhs)


def _minimum_norm_residual_scores(normal: Array,rhs: Array) -> Array:
    """用截断特征分解求每个 RGB 通道的无先验最小范数系数。"""
    # 固定基底可能线性相关；截断伪逆给出稳定的最小范数解。
    eigenvalues,eigenvectors=jnp.linalg.eigh(normal)
    largest=jnp.maximum(eigenvalues[:,-1],1e-20)
    threshold=jnp.maximum(largest[:,None]*1e-7,1e-12)
    inverse=jnp.where(
        eigenvalues>threshold,1/jnp.maximum(eigenvalues,threshold),0)
    projected=jnp.einsum("ckq,ck->cq",eigenvectors,rhs)
    return jnp.einsum("ckq,cq->ck",eigenvectors,projected*inverse)


def fit_uniform_huber_residual_correction_scores_jax(
    residual: Array,
    bsession_field: Array,
    m_fields: Array,
    valid_mask: Array,
    huber_delta: float = .04,
    iterations: int = 5,
) -> Array:
    """以有效像素为 uniform 基础权重，对数据项作逐通道 Huber IRLS。"""
    if huber_delta<=0:
        raise ValueError("uniform_huber 的 huber_delta 必须为正")
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1:
        raise ValueError("uniform_huber 的 iterations 必须是正整数")
    fields=jnp.concatenate([bsession_field[None],m_fields],axis=0)
    design=fields.transpose(1,2,3,0).reshape(-1,3,fields.shape[0])
    target=residual.reshape(-1,3)
    valid=jnp.maximum(valid_mask.reshape(-1),0).astype(residual.dtype)
    initial=fit_uniform_residual_correction_scores_jax(
        residual,bsession_field,m_fields,valid_mask)

    def iteration(_,scores):
        error=target-jnp.einsum("pck,ck->pc",design,scores)
        robust=jnp.minimum(
            1.,huber_delta/jnp.maximum(jnp.abs(error),1e-12))
        weight=valid[:,None]*robust
        denominator=jnp.maximum(jnp.sum(weight,axis=0),1e-12)
        normal=jnp.einsum(
            "pc,pck,pcl->ckl",weight,design,design) \
            /denominator[:,None,None]
        rhs=jnp.einsum("pc,pck,pc->ck",weight,design,target) \
            /denominator[:,None]
        return _minimum_norm_residual_scores(normal,rhs)

    return jax.lax.fori_loop(0,iterations,iteration,initial)


def signed_residual_bgr_jax(residual_linear_rgb: Array,valid_mask: Array,
                            gain: float = 2.) -> Array:
    """在 GPU 上生成中性灰为零的 uint8 BGR 有符号残差图。"""
    visualization=jnp.clip(.5+.5*gain*residual_linear_rgb,0,1)
    visualization=jnp.where(valid_mask[...,None],visualization,0)
    return jnp.rint(visualization[...,::-1]*255).astype(jnp.uint8)


def positive_residual_bgr_jax(residual_linear_rgb: Array,valid_mask: Array,
                              gain: float = 2.) -> Array:
    """在 GPU 上生成零点为黑色的 uint8 BGR 单边正色差图。"""
    visualization=jnp.clip(gain*jnp.maximum(residual_linear_rgb,0),0,1)
    visualization=jnp.where(valid_mask[...,None],visualization,0)
    return jnp.rint(visualization[...,::-1]*255).astype(jnp.uint8)
