"""YAML 配置读取与基础校验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

BACKGROUND_METHODS = (
    "physical_residual","direct_fit","direct_fit_3","geometry_cache")
DIRECT_BACKGROUND_METHODS = ("direct_fit","direct_fit_3")
GEOMETRY_BACKGROUND_METHODS = (
    "direct_fit","direct_fit_3","geometry_cache")


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
class DirectFitConfig:
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
    spatial_difference_validation_weight: float
    spatial_difference_points_per_frame: int
    geometry_difference_weight: float
    geometry_difference_validation_weight: float
    geometry_difference_neighbor_count: int
    geometry_difference_points_per_pair: int
    validation_interval: int
    validation_frame_count: int
    validation_points_per_frame: int
    early_stopping_patience: int
    early_stopping_min_steps: int
    early_stopping_min_delta: float
    sample_saturation_threshold: int
    sample_erode_pixels: int
    session_correction_max_deviation: float


@dataclass(frozen=True)
class GeometryCacheConfig:
    descriptor_curve_coefficients: int
    descriptor_pca_dimensions: int
    descriptor_huber_delta_mm: float
    anchor_count: int
    anchor_neighbor_count: int
    interpolation_neighbor_count: int
    interpolation_distance_power: float
    interpolation_distance_epsilon: float
    background_huber_delta: float
    background_huber_iterations: int
    fit_batch_size: int
    sample_saturation_threshold: int
    sample_erode_pixels: int
    session_correction_max_deviation: float


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

    @property
    def K(self) -> np.ndarray:
        return self.camera_matrix

    @property
    def geometry_rows(self) -> int:
        return self.sample_count

    @property
    def geometry_columns(self) -> int:
        return self.pair_fill_count+2


def load_config_sections(
    config_path: str | Path,
    *section_names: str,
) -> tuple[dict[str, Any], ...]:
    """加载指定配置段，并拒绝缺失或非字典配置。"""
    path = Path(config_path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"配置文件不存在: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"配置文件格式错误: {error}") from error

    if not isinstance(config, Mapping):
        raise ConfigError("配置文件根节点必须是字典")

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
            "lightfield.background.method 必须是 physical_residual、direct_fit、"
            "direct_fit_3 或 geometry_cache")
    return str(method)


def parse_direct_fit_config(lightfield: Mapping[str,Any]) -> DirectFitConfig:
    """读取 direct 几何条件神经场和低频会话修正配置。"""
    configured_method=parse_background_method(lightfield)
    section_name=("direct_fit_3"
                  if configured_method=="direct_fit_3"
                  and "direct_fit_3" in lightfield else "direct_fit")
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
        "spatial_difference_validation_weight",
        "spatial_difference_points_per_frame",
        "geometry_difference_weight","geometry_difference_neighbor_count",
        "geometry_difference_validation_weight",
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
    sample_unknown=set(sample_filter)-{"saturation_threshold","erode_pixels"}
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
    saturation_threshold=integer(
        sample_filter,"saturation_threshold",255)
    if saturation_threshold>255:
        raise ConfigError(
            f"{section_name}.sample_filter.saturation_threshold 必须不大于 255")
    result=DirectFitConfig(
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
        spatial_difference_validation_weight=number(
            field,"spatial_difference_validation_weight",1.,
            allow_zero=True),
        spatial_difference_points_per_frame=integer(
            field,"spatial_difference_points_per_frame",1024),
        geometry_difference_weight=number(
            field,"geometry_difference_weight",.25,allow_zero=True),
        geometry_difference_validation_weight=number(
            field,"geometry_difference_validation_weight",.25,
            allow_zero=True),
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
        sample_saturation_threshold=saturation_threshold,
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


def parse_geometry_cache_config(
    lightfield: Mapping[str,Any],
) -> GeometryCacheConfig:
    """读取几何背景锚点缓存、鲁棒聚合和最近邻插值配置。"""
    section=lightfield.get("geometry_cache",{})
    if not isinstance(section,Mapping):
        raise ConfigError("lightfield.geometry_cache 必须是字典")
    unknown=set(section)-{
        "descriptor","anchors","sample_filter",
        "session_correction_max_deviation"}
    if unknown:
        raise ConfigError(
            f"lightfield.geometry_cache 包含未知字段: {sorted(unknown)}")
    descriptor=section.get("descriptor",{})
    anchors=section.get("anchors",{})
    sample_filter=section.get("sample_filter",{})
    for name,value in (("descriptor",descriptor),("anchors",anchors),
                       ("sample_filter",sample_filter)):
        if not isinstance(value,Mapping):
            raise ConfigError(f"lightfield.geometry_cache.{name} 必须是字典")
    descriptor_unknown=set(descriptor)-{
        "curve_coefficients","pca_dimensions","huber_delta_mm"}
    anchor_unknown=set(anchors)-{
        "count","neighbor_count","interpolation_neighbor_count",
        "interpolation_distance_power","interpolation_distance_epsilon",
        "background_huber_delta","background_huber_iterations",
        "fit_batch_size"}
    sample_unknown=set(sample_filter)-{"saturation_threshold","erode_pixels"}
    if descriptor_unknown or anchor_unknown or sample_unknown:
        unknown_fields=sorted(
            descriptor_unknown|anchor_unknown|sample_unknown)
        raise ConfigError(
            "lightfield.geometry_cache 包含未知字段: "
            f"{unknown_fields}")

    def integer(mapping: Mapping[str,Any],name: str,default: int,
                minimum: int = 1) -> int:
        value=mapping.get(name,default)
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
            raise ConfigError(
                f"geometry_cache.{name} 必须是大于等于 {minimum} 的整数")
        return value

    def number(mapping: Mapping[str,Any],name: str,default: float) -> float:
        value=mapping.get(name,default)
        if not isinstance(value,Real) or isinstance(value,bool) \
                or not np.isfinite(float(value)) or float(value)<=0:
            raise ConfigError(f"geometry_cache.{name} 必须是有限正数")
        return float(value)

    saturation=integer(sample_filter,"saturation_threshold",255)
    if saturation>255:
        raise ConfigError(
            "geometry_cache.sample_filter.saturation_threshold 必须不大于 255")
    result=GeometryCacheConfig(
        descriptor_curve_coefficients=integer(
            descriptor,"curve_coefficients",12,minimum=4),
        descriptor_pca_dimensions=integer(
            descriptor,"pca_dimensions",16),
        descriptor_huber_delta_mm=number(
            descriptor,"huber_delta_mm",.5),
        anchor_count=integer(anchors,"count",48),
        anchor_neighbor_count=integer(anchors,"neighbor_count",8,minimum=2),
        interpolation_neighbor_count=integer(
            anchors,"interpolation_neighbor_count",4),
        interpolation_distance_power=number(
            anchors,"interpolation_distance_power",2.),
        interpolation_distance_epsilon=number(
            anchors,"interpolation_distance_epsilon",1e-3),
        background_huber_delta=number(
            anchors,"background_huber_delta",.04),
        background_huber_iterations=integer(
            anchors,"background_huber_iterations",5),
        fit_batch_size=integer(anchors,"fit_batch_size",4),
        sample_saturation_threshold=saturation,
        sample_erode_pixels=integer(
            sample_filter,"erode_pixels",4,minimum=0),
        session_correction_max_deviation=number(
            section,"session_correction_max_deviation",.3))
    if result.interpolation_neighbor_count>result.anchor_count:
        raise ConfigError(
            "geometry_cache.interpolation_neighbor_count 不能大于 anchor count")
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
        "s1",
        "s2",
        "show_point_cloud",
        "pair_fill_count",
        "sample_count",
        "geometry_grid",
        "lightfield_grid",
        "observation_grid",
        "residual_coefficient_grid",
        "residual_texture_grid",
        "uv_boundary_smooth_lambda",
        "uv_boundary_huber_delta_px",
        "curve_convexity",
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

    s1 = raw.get("s1", 11.0)
    s2 = raw.get("s2", -11.0)
    show_point_cloud = raw.get("show_point_cloud", True)
    pair_fill_count = raw.get("pair_fill_count", 10)
    sample_count = raw.get("sample_count", 100)

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
    curve_convexity = raw.get("curve_convexity", "none")

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
            "get_surface.reconstruction.curve_convexity 必须是 "
            "none、increasing 或 decreasing"
        )
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
    )
