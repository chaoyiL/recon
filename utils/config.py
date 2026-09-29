"""YAML 配置读取与基础校验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from utils.material_surface import (
    MaterialSurfaceTemplate,camera_calibration_sha256,
    material_reconstruction_sha256)

BACKGROUND_METHODS = ("physical_residual","direct_fit_3","direct_fit_s")
DIRECT_BACKGROUND_METHODS = ("direct_fit_3","direct_fit_s")


class ConfigError(ValueError):
    """配置文件内容无效。"""


@dataclass(frozen=True)
class CameraConfig:
    device: int
    exposure: float
    white_balance_temperature: float
    width: int | None
    height: int | None


@dataclass(frozen=True)
class DirectFit3Config:
    coordinate_frequencies: tuple[float,...]
    geometry_descriptor_rows: int
    geometry_encoder_width: int
    geometry_encoder_layers: int
    geometry_latent_dimensions: int
    geometry_pca_dimensions: int
    decoder_width: int
    decoder_layers: int
    steps: int
    batch_size: int
    frame_batch_size: int
    learning_rate: float
    base_huber_iterations: int
    adaptive_channel_weight_strength: float
    spatial_difference_weight: float
    spatial_difference_points_per_frame: int
    geometry_difference_weight: float
    geometry_difference_neighbor_count: int
    geometry_difference_points_per_pair: int
    validation_interval: int
    validation_frame_count: int
    validation_points_per_frame: int
    early_stopping_patience: int
    early_stopping_min_steps: int
    early_stopping_min_delta: float
    sample_erode_pixels: int
    session_correction_max_deviation: float


@dataclass(frozen=True)
class DirectFitSConfig:
    coordinate_frequencies: tuple[float,...]
    geometry_descriptor_rows: int
    geometry_encoder_width: int
    geometry_encoder_layers: int
    geometry_latent_dimensions: int
    geometry_pca_dimensions: int
    gru_hidden_dimensions: int
    appearance_descriptor_rows: int
    appearance_descriptor_columns: int
    appearance_pca_dimensions: int
    appearance_minimum_coverage: float
    appearance_memory_neighbors: int
    appearance_memory_epsilon: float
    color_trunk_width: int
    color_trunk_layers: int
    color_head_width: int
    color_head_layers: int
    synthetic_warp_steps: int
    warp_alignment_steps: int
    appearance_train_steps: int
    color_train_steps: int
    color_checkpoint_interval: int
    color_monitor_batch_count: int
    clip_length: int
    warmup_frames: int
    clip_batch_size: int
    points_per_frame: int
    synthetic_sequence_count: int
    synthetic_min_visible_fraction: float
    synthetic_warp_learning_rate: float
    warp_learning_rate: float
    appearance_learning_rate: float
    color_learning_rate: float
    gradient_clip_norm: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    base_huber_iterations: int
    base_frame_batch_size: int
    full_visibility_threshold: float
    minimum_visible_fraction: float
    measurement_prior_logit_limit: float
    synthetic_regularization_weight: float
    temporal_weight: float
    warp_distillation_weight: float
    appearance_supervision_weight: float
    appearance_score_clip: float
    frame_quality_full_confidence: float
    minimum_frame_quality_weight: float
    use_appearance_memory: bool
    use_recurrent_history: bool
    use_local_geometry: bool
    maximum_sequence_gap: int
    evaluation_points_per_frame: int
    sample_erode_pixels: int
    session_correction_max_deviation: float


@dataclass(frozen=True)
class MaterialSurfaceConfig:
    width_mm: float
    length_mm: float
    s_zero_endpoint: str
    initial_calibration_maximum_rms_px: float
    calibration_maximum_rms_px: float
    calibration_minimum_confidence: float
    match_confidence_scale_mm: float
    rms_confidence_scale_px: float
    confidence_floor: float
    visibility_depth_tolerance_mm: float
    bend_direction: str


@dataclass(frozen=True)
class ReconstructionConfig:
    calibration_file: Path
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    s1: float
    s2: float
    show_point_cloud: bool
    pair_fill_count: int
    sample_count: int
    side_edge_exclusion_ratio: float
    uv_boundary_smooth_lambda: float
    uv_boundary_huber_delta_px: float
    curve_convexity: str
    lightfield_rows: int
    lightfield_columns: int
    observation_rows: int
    observation_columns: int
    residual_coefficient_rows: int
    residual_coefficient_columns: int
    residual_texture_rows: int
    residual_texture_columns: int
    material_surface: MaterialSurfaceConfig

    @property
    def K(self) -> np.ndarray:
        return self.camera_matrix

    @property
    def geometry_rows(self) -> int:
        return self.sample_count

    @property
    def geometry_columns(self) -> int:
        return self.pair_fill_count+2

    @property
    def material_template(self) -> MaterialSurfaceTemplate:
        """按当前尺寸和重建语义确定性生成材料模板。"""
        material=self.material_surface
        return MaterialSurfaceTemplate(
            width_mm=material.width_mm,
            length_mm=material.length_mm,
            rows=self.geometry_rows,
            columns=self.geometry_columns,
            s_zero_endpoint=material.s_zero_endpoint,
            camera_sha256=camera_calibration_sha256(
                self.K,self.distortion_coefficients),
            reconstruction_sha256=material_reconstruction_sha256(
                s1=self.s1,s2=self.s2,
                geometry_rows=self.geometry_rows,
                geometry_columns=self.geometry_columns,
                bend_direction=material.bend_direction,
                uv_boundary_smooth_lambda=self.uv_boundary_smooth_lambda,
                uv_boundary_huber_delta_px=self.uv_boundary_huber_delta_px,
                side_edge_exclusion_ratio=self.side_edge_exclusion_ratio),
        )


def _merge_config_mappings(
    base: Mapping[str,Any], override: Mapping[str,Any],
) -> dict[str,Any]:
    """递归合并配置映射；子配置中的标量、列表和 null 直接覆盖父配置。"""
    result=deepcopy(dict(base))
    for key,value in override.items():
        if key=="extends":
            continue
        current=result.get(key)
        if isinstance(current,Mapping) and isinstance(value,Mapping):
            result[key]=_merge_config_mappings(current,value)
        else:
            result[key]=deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict[str,Any]:
    """读取 YAML；可用顶层 ``extends`` 继承同目录或绝对路径的基础配置。"""
    def load_one(path: Path,stack: tuple[Path,...]) -> dict[str,Any]:
        resolved=path.expanduser().resolve()
        if resolved in stack:
            chain=" -> ".join(str(item) for item in (*stack,resolved))
            raise ConfigError(f"配置 extends 存在循环: {chain}")
        try:
            with resolved.open("r",encoding="utf-8") as config_file:
                config=yaml.safe_load(config_file)
        except FileNotFoundError as error:
            raise ConfigError(f"配置文件不存在: {resolved}") from error
        except yaml.YAMLError as error:
            raise ConfigError(f"配置文件格式错误: {error}") from error
        if not isinstance(config,Mapping):
            raise ConfigError("配置文件根节点必须是字典")
        parent=config.get("extends")
        if parent is None:
            return deepcopy(dict(config))
        if not isinstance(parent,str) or not parent.strip():
            raise ConfigError("顶层 extends 必须是非空路径字符串")
        parent_path=Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path=resolved.parent/parent_path
        inherited=load_one(parent_path,(*stack,resolved))
        return _merge_config_mappings(inherited,config)

    return load_one(Path(config_path),())


def load_config_sections(
    config_path: str | Path,
    *section_names: str,
) -> tuple[dict[str, Any], ...]:
    """加载指定配置段，并拒绝缺失或非字典配置。"""
    config=load_config(config_path)

    sections: list[dict[str, Any]] = []
    for section_name in section_names:
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            raise ConfigError(f"缺少字典配置段: {section_name}")
        sections.append(dict(section))

    return tuple(sections)


def require_keys(section: Mapping[str, Any], section_name: str, *keys: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"配置段 {section_name} 缺少字段: {', '.join(missing)}")


def parse_background_method(lightfield: Mapping[str,Any]) -> str:
    """读取背景模型方法；旧配置默认保持物理光场加残差路径。"""
    background=lightfield.get("background")
    if background is None:
        return "physical_residual"
    if not isinstance(background,Mapping):
        raise ConfigError("lightfield.background 必须是字典")
    unknown=set(background)-{"method","model_files"}
    if unknown:
        raise ConfigError(f"lightfield.background 包含未知字段: {sorted(unknown)}")
    method=background.get("method","physical_residual")
    if method not in BACKGROUND_METHODS:
        raise ConfigError(
            "lightfield.background.method 必须是 physical_residual、"
            "direct_fit_3 或 direct_fit_s")
    return str(method)


def parse_direct_fit_3_config(lightfield: Mapping[str,Any]) -> DirectFit3Config:
    """读取 direct_fit_3 几何条件神经场和低频会话修正配置。"""
    section_name="direct_fit_3"
    direct=lightfield.get(section_name,{})
    if not isinstance(direct,Mapping):
        raise ConfigError(f"lightfield.{section_name} 必须是字典")
    unknown=set(direct)-{
        "neural_field","sample_filter","session_correction_max_deviation"}
    if unknown:
        raise ConfigError(
            f"lightfield.{section_name} 包含未知字段: {sorted(unknown)}")
    field=direct.get("neural_field",{})
    if not isinstance(field,Mapping):
        raise ConfigError(f"{section_name}.neural_field 必须是字典")
    field_unknown=set(field)-{
        "frequencies","geometry_descriptor_rows","geometry_encoder_width",
        "geometry_encoder_layers","geometry_latent_dimensions",
        "geometry_pca_dimensions",
        "decoder_width","decoder_layers","steps","batch_size",
        "frame_batch_size","learning_rate",
        "base_huber_iterations","adaptive_channel_weight_strength",
        "spatial_difference_weight",
        "spatial_difference_points_per_frame",
        "geometry_difference_weight","geometry_difference_neighbor_count",
        "geometry_difference_points_per_pair",
        "validation_interval","validation_frame_count",
        "validation_points_per_frame","early_stopping_patience",
        "early_stopping_min_steps","early_stopping_min_delta"}
    if field_unknown:
        raise ConfigError(
            f"{section_name}.neural_field 包含未知字段: {sorted(field_unknown)}")
    sample_filter=direct.get("sample_filter",{})
    if not isinstance(sample_filter,Mapping):
        raise ConfigError(f"{section_name}.sample_filter 必须是字典")
    sample_unknown=set(sample_filter)-{"erode_pixels"}
    if sample_unknown:
        raise ConfigError(
            f"{section_name}.sample_filter 包含未知字段: {sorted(sample_unknown)}")
    frequencies=field.get("frequencies",[1,2,4,8,16,32])
    if not isinstance(frequencies,(list,tuple)) or not frequencies \
            or any(not isinstance(value,Real) or isinstance(value,bool)
                   for value in frequencies) \
            or not np.isfinite(frequencies).all() \
            or any(float(value)<=0 for value in frequencies):
        raise ConfigError("direct background frequencies 必须是非空有限正数列表")

    def integer(section: Mapping[str,Any],name: str,default: int,
                minimum: int = 1) -> int:
        value=section.get(name,default)
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
            raise ConfigError(
                f"{section_name}.{name} 必须是大于等于 {minimum} 的整数")
        return value

    def number(section: Mapping[str,Any],name: str,default: float,
               *,allow_zero: bool = False) -> float:
        value=section.get(name,default)
        if not isinstance(value,Real) or isinstance(value,bool) \
                or not np.isfinite(float(value)) \
                or (float(value)<0 if allow_zero else float(value)<=0):
            qualifier="非负" if allow_zero else "正"
            raise ConfigError(f"{section_name}.{name} 必须是有限{qualifier}数")
        return float(value)

    session_max=number(
        direct,"session_correction_max_deviation",.15)
    result=DirectFit3Config(
        coordinate_frequencies=tuple(float(value) for value in frequencies),
        geometry_descriptor_rows=integer(
            field,"geometry_descriptor_rows",24,minimum=4),
        geometry_encoder_width=integer(
            field,"geometry_encoder_width",192),
        geometry_encoder_layers=integer(
            field,"geometry_encoder_layers",3),
        geometry_latent_dimensions=integer(
            field,"geometry_latent_dimensions",96),
        geometry_pca_dimensions=integer(
            field,"geometry_pca_dimensions",32),
        decoder_width=integer(field,"decoder_width",192),
        decoder_layers=integer(field,"decoder_layers",5),
        steps=integer(field,"steps",4000),
        batch_size=integer(field,"batch_size",16384),
        frame_batch_size=integer(field,"frame_batch_size",16),
        learning_rate=number(field,"learning_rate",1e-3),
        base_huber_iterations=integer(
            field,"base_huber_iterations",5),
        adaptive_channel_weight_strength=number(
            field,"adaptive_channel_weight_strength",0.,allow_zero=True),
        spatial_difference_weight=number(
            field,"spatial_difference_weight",1.,allow_zero=True),
        spatial_difference_points_per_frame=integer(
            field,"spatial_difference_points_per_frame",1024),
        geometry_difference_weight=number(
            field,"geometry_difference_weight",.25,allow_zero=True),
        geometry_difference_neighbor_count=integer(
            field,"geometry_difference_neighbor_count",16),
        geometry_difference_points_per_pair=integer(
            field,"geometry_difference_points_per_pair",512),
        validation_interval=integer(field,"validation_interval",100),
        validation_frame_count=integer(field,"validation_frame_count",64),
        validation_points_per_frame=integer(
            field,"validation_points_per_frame",512),
        early_stopping_patience=integer(
            field,"early_stopping_patience",10),
        early_stopping_min_steps=integer(
            field,"early_stopping_min_steps",1500,minimum=0),
        early_stopping_min_delta=number(
            field,"early_stopping_min_delta",5e-5,allow_zero=True),
        sample_erode_pixels=integer(
            sample_filter,"erode_pixels",2,minimum=0),
        session_correction_max_deviation=session_max)
    if result.adaptive_channel_weight_strength>1:
        raise ConfigError(
            f"{section_name}.adaptive_channel_weight_strength 必须不大于 1")
    if result.early_stopping_min_steps>result.steps:
        raise ConfigError(
            f"{section_name}.neural_field.early_stopping_min_steps 不能大于 steps")
    return result


def parse_direct_fit_s_config(lightfield: Mapping[str,Any]) -> DirectFitSConfig:
    """读取带 s 仿射 warp 和序列 GRU 的最小 direct_fit_s 配置。"""
    section_name="direct_fit_s"
    direct=lightfield.get(section_name,{})
    if not isinstance(direct,Mapping):
        raise ConfigError(f"lightfield.{section_name} 必须是字典")
    unknown=set(direct)-{
        "neural_field","training","measurement","loss","sequence","sample_filter",
        "ablation","session_correction_max_deviation"}
    if unknown:
        raise ConfigError(
            f"lightfield.{section_name} 包含未知字段: {sorted(unknown)}")

    def mapping(name: str) -> Mapping[str,Any]:
        value=direct.get(name,{})
        if not isinstance(value,Mapping):
            raise ConfigError(f"lightfield.{section_name}.{name} 必须是字典")
        return value

    field=mapping("neural_field")
    training=mapping("training")
    measurement=mapping("measurement")
    loss=mapping("loss")
    sequence=mapping("sequence")
    ablation=mapping("ablation")
    sample_filter=mapping("sample_filter")
    known={
        "neural_field":{
            "frequencies","geometry_descriptor_rows",
            "geometry_encoder_width","geometry_encoder_layers",
            "geometry_latent_dimensions","geometry_pca_dimensions",
            "gru_hidden_dimensions","appearance_descriptor_rows",
            "appearance_descriptor_columns","appearance_pca_dimensions",
            "appearance_minimum_coverage",
            "appearance_memory_neighbors","appearance_memory_epsilon",
            "color_trunk_width",
            "color_trunk_layers","color_head_width","color_head_layers"},
        "training":{
            "synthetic_warp_steps","warp_alignment_steps",
            "appearance_train_steps","color_train_steps",
            "color_checkpoint_interval",
            "color_monitor_batch_count","clip_length","warmup_frames",
            "clip_batch_size","points_per_frame",
            "synthetic_sequence_count","synthetic_min_visible_fraction",
            "synthetic_warp_learning_rate","warp_learning_rate",
            "appearance_learning_rate",
            "color_learning_rate","gradient_clip_norm",
            "base_huber_iterations","base_frame_batch_size",
            "adam_beta1","adam_beta2","adam_epsilon",
            "evaluation_points_per_frame"},
        "measurement":{
            "full_visibility_threshold","minimum_visible_fraction",
            "prior_logit_limit"},
        "loss":{
            "synthetic_regularization_weight","temporal_weight",
            "warp_distillation_weight","frame_quality_full_confidence",
            "minimum_frame_quality_weight","appearance_supervision_weight",
            "appearance_score_clip"},
        "sequence":{"maximum_sequence_gap"},
        "ablation":{"use_appearance_memory","use_recurrent_history",
                    "use_local_geometry"},
        "sample_filter":{"erode_pixels"},
    }
    for name,value in (("neural_field",field),("training",training),
                       ("measurement",measurement),("loss",loss),("sequence",sequence),
                       ("ablation",ablation),("sample_filter",sample_filter)):
        extra=set(value)-known[name]
        if extra:
            raise ConfigError(
                f"lightfield.{section_name}.{name} 包含未知字段: "
                f"{sorted(extra)}")

    def integer(section: Mapping[str,Any],name: str,default: int,
                minimum: int = 1) -> int:
        value=section.get(name,default)
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
            raise ConfigError(
                f"{section_name}.{name} 必须是大于等于 {minimum} 的整数")
        return int(value)

    def number(section: Mapping[str,Any],name: str,default: float,
               *,allow_zero: bool = False) -> float:
        value=section.get(name,default)
        if not isinstance(value,Real) or isinstance(value,bool) \
                or not np.isfinite(float(value)) \
                or (float(value)<0 if allow_zero else float(value)<=0):
            qualifier="非负" if allow_zero else "正"
            raise ConfigError(f"{section_name}.{name} 必须是有限{qualifier}数")
        return float(value)

    def boolean(section: Mapping[str,Any],name: str,default: bool) -> bool:
        value=section.get(name,default)
        if not isinstance(value,bool):
            raise ConfigError(f"{section_name}.{name} 必须是布尔值")
        return value

    frequencies=field.get("frequencies",[1,2,4,8,16,32,64])
    if not isinstance(frequencies,(list,tuple)) or not frequencies \
            or any(not isinstance(value,Real) or isinstance(value,bool)
                   for value in frequencies) \
            or not np.isfinite(frequencies).all() \
            or any(float(value)<=0 for value in frequencies):
        raise ConfigError("direct_fit_s frequencies 必须是非空有限正数列表")
    result=DirectFitSConfig(
        coordinate_frequencies=tuple(float(value) for value in frequencies),
        geometry_descriptor_rows=integer(
            field,"geometry_descriptor_rows",32,minimum=4),
        geometry_encoder_width=integer(field,"geometry_encoder_width",128),
        geometry_encoder_layers=integer(field,"geometry_encoder_layers",2),
        geometry_latent_dimensions=integer(
            field,"geometry_latent_dimensions",64),
        geometry_pca_dimensions=integer(field,"geometry_pca_dimensions",32),
        gru_hidden_dimensions=integer(field,"gru_hidden_dimensions",64),
        appearance_descriptor_rows=integer(
            field,"appearance_descriptor_rows",48,minimum=4),
        appearance_descriptor_columns=integer(
            field,"appearance_descriptor_columns",24,minimum=4),
        appearance_pca_dimensions=integer(
            field,"appearance_pca_dimensions",96),
        appearance_minimum_coverage=number(
            field,"appearance_minimum_coverage",.8),
        appearance_memory_neighbors=integer(
            field,"appearance_memory_neighbors",4),
        appearance_memory_epsilon=number(
            field,"appearance_memory_epsilon",1e-6),
        color_trunk_width=integer(field,"color_trunk_width",224),
        color_trunk_layers=integer(field,"color_trunk_layers",4),
        color_head_width=integer(field,"color_head_width",128),
        color_head_layers=integer(field,"color_head_layers",2),
        synthetic_warp_steps=integer(
            training,"synthetic_warp_steps",2500),
        warp_alignment_steps=integer(
            training,"warp_alignment_steps",2500),
        appearance_train_steps=integer(
            training,"appearance_train_steps",4000),
        color_train_steps=integer(training,"color_train_steps",2500),
        color_checkpoint_interval=integer(
            training,"color_checkpoint_interval",250),
        color_monitor_batch_count=integer(
            training,"color_monitor_batch_count",8),
        clip_length=integer(training,"clip_length",16,minimum=3),
        warmup_frames=integer(training,"warmup_frames",64,minimum=0),
        clip_batch_size=integer(training,"clip_batch_size",2,minimum=2),
        points_per_frame=integer(training,"points_per_frame",256),
        synthetic_sequence_count=integer(
            training,"synthetic_sequence_count",256),
        synthetic_min_visible_fraction=number(
            training,"synthetic_min_visible_fraction",.35),
        synthetic_warp_learning_rate=number(
            training,"synthetic_warp_learning_rate",8e-4),
        warp_learning_rate=number(training,"warp_learning_rate",4e-4),
        appearance_learning_rate=number(
            training,"appearance_learning_rate",4e-4),
        color_learning_rate=number(training,"color_learning_rate",4e-4),
        gradient_clip_norm=number(training,"gradient_clip_norm",1.),
        adam_beta1=number(training,"adam_beta1",.9),
        adam_beta2=number(training,"adam_beta2",.999),
        adam_epsilon=number(training,"adam_epsilon",1e-8),
        base_huber_iterations=integer(
            training,"base_huber_iterations",5),
        base_frame_batch_size=integer(
            training,"base_frame_batch_size",8),
        full_visibility_threshold=number(
            measurement,"full_visibility_threshold",.97),
        minimum_visible_fraction=number(
            measurement,"minimum_visible_fraction",.35),
        measurement_prior_logit_limit=number(
            measurement,"prior_logit_limit",4.),
        synthetic_regularization_weight=number(
            loss,"synthetic_regularization_weight",.02,allow_zero=True),
        temporal_weight=number(loss,"temporal_weight",.25,allow_zero=True),
        warp_distillation_weight=number(
            loss,"warp_distillation_weight",1.,allow_zero=True),
        appearance_supervision_weight=number(
            loss,"appearance_supervision_weight",.005,allow_zero=True),
        appearance_score_clip=number(loss,"appearance_score_clip",3.),
        frame_quality_full_confidence=number(
            loss,"frame_quality_full_confidence",.25),
        minimum_frame_quality_weight=number(
            loss,"minimum_frame_quality_weight",.2),
        use_appearance_memory=boolean(
            ablation,"use_appearance_memory",True),
        use_recurrent_history=boolean(
            ablation,"use_recurrent_history",True),
        use_local_geometry=boolean(
            ablation,"use_local_geometry",True),
        maximum_sequence_gap=integer(
            sequence,"maximum_sequence_gap",1,minimum=1),
        evaluation_points_per_frame=integer(
            training,"evaluation_points_per_frame",2048),
        sample_erode_pixels=integer(
            sample_filter,"erode_pixels",4,minimum=0),
        session_correction_max_deviation=number(
            direct,"session_correction_max_deviation",.30))
    if result.adam_beta1>=1 or result.adam_beta2>=1:
        raise ConfigError("direct_fit_s Adam beta 必须位于 (0,1)")
    if result.color_checkpoint_interval>result.color_train_steps:
        raise ConfigError(
            "direct_fit_s color_checkpoint_interval 不能大于 color_train_steps")
    if result.full_visibility_threshold>1:
        raise ConfigError(
            "direct_fit_s.full_visibility_threshold 必须不大于 1")
    if result.frame_quality_full_confidence>1 \
            or result.minimum_frame_quality_weight>1:
        raise ConfigError(
            "direct_fit_s 帧质量 confidence/weight 必须不大于 1")
    appearance_feature_count=(result.appearance_descriptor_rows
                              *result.appearance_descriptor_columns*3)
    if result.appearance_pca_dimensions>appearance_feature_count:
        raise ConfigError(
            "direct_fit_s appearance_pca_dimensions 不能大于外观描述维数")
    if result.appearance_minimum_coverage>1:
        raise ConfigError(
            "direct_fit_s appearance_minimum_coverage 必须不大于 1")
    if result.minimum_visible_fraction>=result.full_visibility_threshold:
        raise ConfigError(
            "direct_fit_s.minimum_visible_fraction 必须小于 "
            "full_visibility_threshold")
    if not result.minimum_visible_fraction \
            <=result.synthetic_min_visible_fraction<1:
        raise ConfigError(
            "direct_fit_s.synthetic_min_visible_fraction 必须位于 "
            "[minimum_visible_fraction,1) 区间")
    return result


def resolve_method_path(
    section: Mapping[str,Any],
    *,
    method: str,
    mapping_key: str,
    legacy_key: str,
    base: str | Path,
    section_name: str,
) -> Path:
    """按背景方法选择路径，并为旧的单路径配置保留兼容回退。"""
    if method not in BACKGROUND_METHODS:
        raise ConfigError(f"不支持的背景方法: {method}")
    mapping=section.get(mapping_key)
    value: object
    if mapping is None:
        value=section.get(legacy_key)
    elif not isinstance(mapping,Mapping):
        raise ConfigError(f"{section_name}.{mapping_key} 必须是字典")
    else:
        unknown=set(mapping)-set(BACKGROUND_METHODS)
        if unknown:
            raise ConfigError(
                f"{section_name}.{mapping_key} 包含未知字段: {sorted(unknown)}")
        value=mapping.get(method)
    if not isinstance(value,str) or not value.strip():
        raise ConfigError(
            f"{section_name}.{mapping_key}.{method} 必须是非空路径")
    path=Path(value).expanduser()
    return path if path.is_absolute() else Path(base).expanduser()/path


def resolve_background_model_path(
    lightfield: Mapping[str,Any],*,method: str,base: str | Path,
) -> Path:
    background=lightfield.get("background")
    if isinstance(background,Mapping) and "model_files" in background:
        return resolve_method_path(
            background,method=method,mapping_key="model_files",
            legacy_key="model_file",base=base,
            section_name="lightfield.background")
    return resolve_method_path(
        lightfield,method=method,mapping_key="model_files",
        legacy_key="model_file",base=base,section_name="lightfield")


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_camera_config(section: Mapping[str, Any]) -> CameraConfig:
    require_keys(
        section,"camera","device","exposure","white_balance_temperature",
        "width","height")

    device = section["device"]
    exposure = section["exposure"]
    white_balance_temperature = section["white_balance_temperature"]
    width = section["width"]
    height = section["height"]

    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ConfigError("camera.device 必须是非负整数")
    if not isinstance(exposure, Real) or isinstance(exposure, bool):
        raise ConfigError("camera.exposure 必须是数字")
    if (
        not isinstance(white_balance_temperature, Real)
        or isinstance(white_balance_temperature, bool)
        or not 1000<=float(white_balance_temperature)<=20000
    ):
        raise ConfigError("camera.white_balance_temperature 必须是 1000..20000 K 的数字")

    for name, value in (("width", width), ("height", height)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigError(f"camera.{name} 必须是正整数或 null")

    return CameraConfig(
        device=device,
        exposure=float(exposure),
        white_balance_temperature=float(white_balance_temperature),
        width=width,
        height=height,
    )


def load_camera_calibration(
    calibration_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """读取内参矩阵和畸变系数。"""
    path = Path(calibration_path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as calibration_file:
            data = yaml.safe_load(calibration_file)
    except FileNotFoundError as error:
        raise ConfigError(f"相机标定文件不存在: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"相机标定文件格式错误: {error}") from error

    if not isinstance(data, Mapping):
        raise ConfigError(f"相机标定文件根节点必须是字典: {path}")
    raw_matrix = data.get("camera_matrix")
    if not isinstance(raw_matrix, list) or len(raw_matrix) != 3:
        raise ConfigError(f"相机标定文件缺少有效 camera_matrix: {path}")

    try:
        camera_matrix = np.asarray(raw_matrix, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"camera_matrix 必须是 3x3 数值矩阵: {path}") from error

    if not np.isfinite(camera_matrix).all():
        raise ConfigError(f"camera_matrix 含有非有限值: {path}")
    if camera_matrix[0, 0] == 0 or camera_matrix[1, 1] == 0:
        raise ConfigError(f"camera_matrix 的 fx/fy 不能为 0: {path}")

    raw_distortion = data.get("distortion_coefficients")
    if not isinstance(raw_distortion, list) or not raw_distortion:
        raise ConfigError(f"相机标定文件缺少有效 distortion_coefficients: {path}")
    try:
        distortion = np.asarray(raw_distortion, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"distortion_coefficients 必须是数值列表: {path}") from error
    if not np.isfinite(distortion).all():
        raise ConfigError(f"distortion_coefficients 含有非有限值: {path}")
    return camera_matrix, distortion


def parse_reconstruction_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: str | Path,
    calibration_output: str | None = None,
) -> ReconstructionConfig:
    """解析 get_surface.reconstruction；内参从 camera_calibration 文件读取。"""
    if section is None:
        raw: dict[str, Any] = {}
    elif isinstance(section, Mapping):
        raw = dict(section)
    else:
        raise ConfigError("get_surface.reconstruction 必须是字典或 null")

    known = {
        "calibration_file",
        "show_point_cloud",
        "pair_fill_count",
        "sample_count",
        "geometry_grid",
        "lightfield_grid",
        "observation_grid",
        "residual_coefficient_grid",
        "residual_texture_grid",
        "side_edge_exclusion_ratio",
        "uv_boundary_smooth_lambda",
        "uv_boundary_huber_delta_px",
        "curve_convexity",
        "bend_direction",
        "material_surface",
    }
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"get_surface.reconstruction 包含未知字段: {sorted(unknown)}"
        )

    calibration_file = raw.get("calibration_file")
    if calibration_file is None:
        calibration_file = calibration_output or "camera_calibration.yaml"
    if not isinstance(calibration_file, str) or not calibration_file.strip():
        raise ConfigError(
            "get_surface.reconstruction.calibration_file 必须是非空字符串"
        )

    calibration_path = Path(calibration_file).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = Path(config_path).expanduser().parent / calibration_path

    show_point_cloud = raw.get("show_point_cloud", True)
    pair_fill_count = raw.get("pair_fill_count", 10)
    sample_count = raw.get("sample_count", 100)
    side_edge_exclusion_ratio=raw.get("side_edge_exclusion_ratio",.02)

    def parse_grid(name: str, fallback_rows: object,
                   fallback_columns: object, *,
                   minimum_rows: int = 4,
                   minimum_columns: int = 3) -> tuple[int,int]:
        value=raw.get(name)
        if value is None:
            rows=fallback_rows; columns=fallback_columns
        elif not isinstance(value,Mapping):
            raise ConfigError(f"get_surface.reconstruction.{name} 必须是字典")
        else:
            unknown_grid=set(value)-{"rows","columns"}
            if unknown_grid:
                raise ConfigError(
                    f"get_surface.reconstruction.{name} 包含未知字段: "
                    f"{sorted(unknown_grid)}")
            if "rows" not in value or "columns" not in value:
                raise ConfigError(
                    f"get_surface.reconstruction.{name} 必须同时配置 rows 和 columns")
            rows=value["rows"]; columns=value["columns"]
        for axis,axis_value,minimum in (
                ("rows",rows,minimum_rows),
                ("columns",columns,minimum_columns)):
            if (not isinstance(axis_value,int) or isinstance(axis_value,bool)
                    or axis_value<minimum):
                raise ConfigError(
                    f"get_surface.reconstruction.{name}.{axis} "
                    f"必须是大于等于 {minimum} 的整数")
        return int(rows),int(columns)

    legacy_columns=(pair_fill_count+2
                    if isinstance(pair_fill_count,int)
                    and not isinstance(pair_fill_count,bool)
                    else pair_fill_count)
    geometry_rows,geometry_columns=parse_grid(
        "geometry_grid",sample_count,legacy_columns)
    lightfield_rows,lightfield_columns=parse_grid(
        "lightfield_grid",geometry_rows,geometry_columns)
    observation_rows,observation_columns=parse_grid(
        "observation_grid",geometry_rows,geometry_columns,minimum_columns=4)
    residual_coefficient_rows,residual_coefficient_columns=parse_grid(
        "residual_coefficient_grid",min(24,observation_rows),
        min(12,observation_columns),minimum_columns=4)
    residual_texture_rows,residual_texture_columns=parse_grid(
        "residual_texture_grid",256,128,minimum_columns=4)
    if observation_rows<residual_coefficient_rows \
            or observation_columns<residual_coefficient_columns:
        raise ConfigError(
            "get_surface.reconstruction.observation_grid 不能小于 "
            "residual_coefficient_grid")
    # 新配置是公开接口；保留两个旧字段作为兼容别名。
    sample_count=geometry_rows
    pair_fill_count=geometry_columns-2
    uv_boundary_smooth_lambda = raw.get("uv_boundary_smooth_lambda", 10.0)
    uv_boundary_huber_delta_px = raw.get("uv_boundary_huber_delta_px", 2.0)
    legacy_curve_convexity = raw.get("curve_convexity")
    direction_field=(
        "curve_convexity"
        if "curve_convexity" in raw and "bend_direction" not in raw
        else "bend_direction")
    bend_direction = raw.get(
        "bend_direction",
        legacy_curve_convexity if legacy_curve_convexity is not None else "none")
    if legacy_curve_convexity is not None \
            and bend_direction != legacy_curve_convexity:
        raise ConfigError(
            "get_surface.reconstruction.bend_direction 与旧字段 "
            "curve_convexity 不能冲突")
    curve_convexity = bend_direction

    raw_material = raw.get("material_surface", {})
    if not isinstance(raw_material, Mapping):
        raise ConfigError(
            "get_surface.reconstruction.material_surface 必须是字典")
    known_material = {
        "width_mm","length_mm","s_zero_endpoint",
        "initial_calibration_maximum_rms_px",
        "calibration_maximum_rms_px","calibration_minimum_confidence",
        "match_confidence_scale_mm",
        "rms_confidence_scale_px","confidence_floor",
        "visibility_depth_tolerance_mm",
    }
    unknown_material = set(raw_material)-known_material
    if unknown_material:
        raise ConfigError(
            "get_surface.reconstruction.material_surface 包含未知字段: "
            f"{sorted(unknown_material)}")
    material_values = {
        "width_mm":raw_material.get("width_mm",22.),
        "length_mm":raw_material.get("length_mm",55.),
        "s_zero_endpoint":raw_material.get(
            "s_zero_endpoint","image_top"),
        "initial_calibration_maximum_rms_px":raw_material.get(
            "initial_calibration_maximum_rms_px",4.),
        "calibration_maximum_rms_px":raw_material.get(
            "calibration_maximum_rms_px",4.),
        "calibration_minimum_confidence":raw_material.get(
            "calibration_minimum_confidence",.05),
        "match_confidence_scale_mm":raw_material.get(
            "match_confidence_scale_mm",8.),
        "rms_confidence_scale_px":raw_material.get(
            "rms_confidence_scale_px",4.),
        "confidence_floor":raw_material.get("confidence_floor",1e-6),
        "visibility_depth_tolerance_mm":raw_material.get(
            "visibility_depth_tolerance_mm",1.),
    }

    for dimension_name in ("width_mm","length_mm"):
        dimension=material_values[dimension_name]
        if not isinstance(dimension,Real) or isinstance(dimension,bool) \
                or not np.isfinite(float(dimension)) or float(dimension)<=0:
            raise ConfigError(
                f"material_surface.{dimension_name} 必须是有限正数")
    width_mm=float(material_values["width_mm"])
    s1=width_mm/2
    s2=-width_mm/2

    for name, value in (("s1", s1), ("s2", s2)):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ConfigError(f"get_surface.reconstruction.{name} 必须是数字")
    if float(s1) <= float(s2):
        raise ConfigError("get_surface.reconstruction 要求 s1 > s2")
    if not isinstance(show_point_cloud, bool):
        raise ConfigError(
            "get_surface.reconstruction.show_point_cloud 必须是 true 或 false"
        )
    for name, value, minimum in (
        ("pair_fill_count", pair_fill_count, 0),
        ("sample_count", sample_count, 4),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigError(
                f"get_surface.reconstruction.{name} 必须是大于等于 {minimum} 的整数"
            )
    if (
        not isinstance(side_edge_exclusion_ratio,Real)
        or isinstance(side_edge_exclusion_ratio,bool)
        or not 0<=float(side_edge_exclusion_ratio)<.5
    ):
        raise ConfigError(
            "get_surface.reconstruction.side_edge_exclusion_ratio "
            "必须位于 [0,0.5)"
        )
    if (
        not isinstance(uv_boundary_smooth_lambda, Real)
        or isinstance(uv_boundary_smooth_lambda, bool)
        or float(uv_boundary_smooth_lambda) < 0
    ):
        raise ConfigError(
            "get_surface.reconstruction.uv_boundary_smooth_lambda 必须是非负数"
        )
    if (
        not isinstance(uv_boundary_huber_delta_px, Real)
        or isinstance(uv_boundary_huber_delta_px, bool)
        or float(uv_boundary_huber_delta_px) <= 0
    ):
        raise ConfigError(
            "get_surface.reconstruction.uv_boundary_huber_delta_px 必须是正数"
        )
    if curve_convexity not in ("none", "increasing", "decreasing"):
        raise ConfigError(
            f"get_surface.reconstruction.{direction_field} 必须是 "
            "none、increasing 或 decreasing"
        )
    for name in (
        "width_mm","length_mm",
        "initial_calibration_maximum_rms_px",
        "calibration_maximum_rms_px",
        "match_confidence_scale_mm",
        "rms_confidence_scale_px",
        "visibility_depth_tolerance_mm",
    ):
        value=material_values[name]
        if not isinstance(value,Real) or isinstance(value,bool) or float(value)<=0:
            raise ConfigError(f"material_surface.{name} 必须是正数")
    s_zero_endpoint=material_values["s_zero_endpoint"]
    if s_zero_endpoint not in ("image_top","image_bottom"):
        raise ConfigError(
            "material_surface.s_zero_endpoint 必须是 image_top 或 image_bottom")
    confidence_floor=material_values["confidence_floor"]
    calibration_minimum_confidence=(
        material_values["calibration_minimum_confidence"])
    if not isinstance(confidence_floor,Real) \
            or isinstance(confidence_floor,bool) \
            or not 0<float(confidence_floor)<1:
        raise ConfigError(
            "material_surface.confidence_floor 必须位于 (0,1)")
    if not isinstance(calibration_minimum_confidence,Real) \
            or isinstance(calibration_minimum_confidence,bool) \
            or not float(confidence_floor)<float(
                calibration_minimum_confidence)<1:
        raise ConfigError(
            "material_surface.calibration_minimum_confidence 必须大于 "
            "confidence_floor 且小于 1")
    camera_matrix, distortion = load_camera_calibration(calibration_path)
    return ReconstructionConfig(
        calibration_file=calibration_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        s1=float(s1),
        s2=float(s2),
        show_point_cloud=show_point_cloud,
        pair_fill_count=int(pair_fill_count),
        sample_count=int(sample_count),
        side_edge_exclusion_ratio=float(side_edge_exclusion_ratio),
        uv_boundary_smooth_lambda=float(uv_boundary_smooth_lambda),
        uv_boundary_huber_delta_px=float(uv_boundary_huber_delta_px),
        curve_convexity=str(curve_convexity),
        lightfield_rows=lightfield_rows,
        lightfield_columns=lightfield_columns,
        observation_rows=observation_rows,
        observation_columns=observation_columns,
        residual_coefficient_rows=residual_coefficient_rows,
        residual_coefficient_columns=residual_coefficient_columns,
        residual_texture_rows=residual_texture_rows,
        residual_texture_columns=residual_texture_columns,
        material_surface=MaterialSurfaceConfig(
            width_mm=float(material_values["width_mm"]),
            length_mm=float(material_values["length_mm"]),
            s_zero_endpoint=str(s_zero_endpoint),
            initial_calibration_maximum_rms_px=float(
                material_values["initial_calibration_maximum_rms_px"]),
            calibration_maximum_rms_px=float(
                material_values["calibration_maximum_rms_px"]),
            calibration_minimum_confidence=float(
                calibration_minimum_confidence),
            match_confidence_scale_mm=float(
                material_values["match_confidence_scale_mm"]),
            rms_confidence_scale_px=float(
                material_values["rms_confidence_scale_px"]),
            confidence_floor=float(confidence_floor),
            visibility_depth_tolerance_mm=float(
                material_values["visibility_depth_tolerance_mm"]),
            bend_direction=str(bend_direction),
        ),
    )
