"""使用 JAX 自动微分离线标定一条或多条独立线光源。"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import glob
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from get_surface import (parse_mask_refine, parse_prompts,
                         point_set_from_surface_grids,
                         save_generated_uv_xyz_map)
from utils.config import (load_config,parse_background_method,
                          parse_direct_fit_3_config,parse_direct_fit_s_config,
                          parse_reconstruction_config,
                          resolve_background_model_path)
from utils.direct_fit_s import fit_direct_fit_s_gpu
from utils.gpu_residual_fit import (
    fit_direct_geometry_conditioned_field_gpu,
    fit_residual_correction_model_gpu)
from utils.jax_reconstruction import (
    MATERIAL_UPDATE_FAILURE_NAMES,
    SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
    prepare_edge_curves_from_masks_jax,
    reconstruct_material_surface_with_diagnostics_from_masks_jax,
    reference_material_surface_state_jax)
from utils.lightfield import (DEFAULT_LIGHT_SOURCE_LAYOUT,LightFieldModel,
                              LightSourceLayout,
                              bounded_mixing_matrix,
                              bgr_to_linear_rgb_jax,
                              build_canonical_residual_sample_jax, choose_device,
                              direct_background_field_chunked,
                              evaluate_rgb_bspline,
                              fit_uniform_huber_residual_correction_scores_jax,
                              fit_uniform_residual_correction_scores_jax,
                              irls_gain_bias, physical_background_batch, point_set_to_grid,
                              light_source_specs, parse_light_source_layout,
                              rasterize_attributes_jax,
                              sample_rgb, sample_unsaturated_mask)
from utils.process import EdgeReconstructor, ReconstructionPointSet
from utils.material_surface import MATERIAL_COORDINATE_MODE
from utils.surface_segmentation import (
    SurfaceSegmentationBackend,refine_masks_numpy,
    parse_surface_segmentation_config)

# 保留模块级名称，便于既有调用方注入测试后端。
SurfaceSegmenter=SurfaceSegmentationBackend


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
VIDEO_FRAME_PATTERN=re.compile(
    r"^(video_\d+_.+)_frame_(\d+)$")


CALIBRATION_OBSERVATION_FORMAT_VERSION = 5
CALIBRATION_FAILURE_FORMAT_VERSION = 2
CALIBRATION_RECONSTRUCTION_PIPELINE = SURFACE_RECONSTRUCTION_PIPELINE_VERSION
_DEFAULT_MATERIAL_DIAGNOSTIC_DIR=object()
_RECONSTRUCTION_FAILURE_SUFFIX=".reconstruction_failure.npz"


def _material_sequence_identity(path: Path) -> tuple[str | None,int | None]:
    match=VIDEO_FRAME_PATTERN.match(path.stem)
    if match is None:
        return None,None
    return match.group(1),int(match.group(2))


def _surface_range_polygon(uv: np.ndarray) -> np.ndarray | None:
    """把 RxCx2 曲面的四周边界整理为 OpenCV 多边形。"""
    values=np.asarray(uv,np.float32)
    if values.ndim!=3 or values.shape[-1]!=2 \
            or min(values.shape[:2])<2:
        return None
    boundary=np.concatenate([
        values[:,0],values[-1,1:],values[-2::-1,-1],
        values[0,-2:0:-1]],axis=0)
    if not np.isfinite(boundary).all():
        return None
    return np.rint(boundary).astype(np.int32).reshape(-1,1,2)


def _draw_material_initialization_failure(
    frame: np.ndarray,
    refined_masks: np.ndarray,
    template_uv: np.ndarray,
    observed_uv: np.ndarray,
    *,
    template_length_mm: float,
    observed_length_mm: float,
    observed_rms_px: float,
    maximum_rms_px: float,
    reconstruction_valid: bool,
    title: str = "MATERIAL INITIALIZATION FAILED",
    matching_confidence: float | None = None,
    minimum_confidence: float | None = None,
    failure_reasons: tuple[str,...] = (),
) -> np.ndarray:
    """绘制材料候选失败的模板/实测范围诊断图。"""
    image=np.asarray(frame,np.uint8).copy()
    masks=np.asarray(refined_masks,np.bool_)
    if masks.ndim==3 and masks.shape[1:]==image.shape[:2]:
        combined=np.any(masks,axis=0)
        tint=image.copy()
        tint[combined]=(180,180,0)
        image=cv2.addWeighted(image,.78,tint,.22,0)
        contours,_=cv2.findContours(
            combined.astype(np.uint8),cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(255,255,0),1,cv2.LINE_AA)

    template_polygon=_surface_range_polygon(template_uv)
    observed_polygon=_surface_range_polygon(observed_uv)
    if template_polygon is not None:
        cv2.polylines(
            image,[template_polygon],True,(255,0,255),3,cv2.LINE_AA)
    if observed_polygon is not None:
        cv2.polylines(
            image,[observed_polygon],True,(0,255,0),3,cv2.LINE_AA)

    for uv,color in ((template_uv,(255,0,255)),(observed_uv,(0,255,0))):
        values=np.asarray(uv,np.float32)
        if values.ndim!=3 or values.shape[-1]!=2:
            continue
        center=np.mean(values,axis=1)
        if np.isfinite(center).all():
            line=np.rint(center).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(image,[line],False,color,1,cv2.LINE_AA)
            cv2.circle(image,tuple(line[0,0]),5,color,-1,cv2.LINE_AA)
            cv2.circle(image,tuple(line[-1,0]),5,color,-1,cv2.LINE_AA)

    length_ratio=(
        abs(observed_length_mm-template_length_mm)/template_length_mm
        if np.isfinite(observed_length_mm) and template_length_mm>0
        else np.inf)
    rms_ok=np.isfinite(observed_rms_px) and observed_rms_px<=maximum_rms_px
    lines=[
        title,
        f"MAGENTA template: {template_length_mm:.3f} mm; CYAN segmentation mask",
        (f"GREEN reconstruction: {observed_length_mm:.3f} mm, "
         f"normalized length difference={100*length_ratio:.2f}% (diagnostic only)"),
        (f"checks: reconstruction={int(reconstruction_valid)} "
         f"rms={int(rms_ok)}  "
         f"RMS={observed_rms_px:.3f} / {maximum_rms_px:.3f} px"),
    ]
    if matching_confidence is not None and minimum_confidence is not None:
        confidence_ok=(np.isfinite(matching_confidence)
                       and matching_confidence>=minimum_confidence)
        lines.append(
            f"confidence={matching_confidence:.6f} / "
            f"{minimum_confidence:.6f}, ok={int(confidence_ok)}")
    if failure_reasons:
        lines.append("failure: "+", ".join(failure_reasons))
    panel_height=18+24*len(lines)
    panel_height=min(panel_height,image.shape[0])
    panel_top=image.shape[0]-panel_height
    panel=image[panel_top:].copy()
    panel[:]=(16,16,16)
    image[panel_top:]=cv2.addWeighted(
        image[panel_top:],.25,panel,.75,0)
    for line_index,line in enumerate(lines):
        cv2.putText(
            image,line,(10,panel_top+24+24*line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            .55,(245,245,245),1,cv2.LINE_AA)
    return image


def _write_material_failure(
    output: Path,frame: np.ndarray,refined_masks: np.ndarray,
    template_uv: np.ndarray,observed_xyz: np.ndarray,observed_uv: np.ndarray,
    *,template_length_mm: float,
    observed_rms_px: float,maximum_rms_px: float,
    reconstruction_valid: bool,title: str,
    matching_confidence: float | None = None,
    minimum_confidence: float | None = None,
    failure_reasons: tuple[str,...] = (),
) -> tuple[Path,float]:
    curve=np.asarray(observed_xyz,np.float32)[:,0,1:3]
    observed_length=float(np.sum(np.linalg.norm(
        np.diff(curve,axis=0),axis=1)))
    diagnostic=_draw_material_initialization_failure(
        frame,refined_masks,template_uv,observed_uv,
        template_length_mm=template_length_mm,
        observed_length_mm=observed_length,
        observed_rms_px=observed_rms_px,maximum_rms_px=maximum_rms_px,
        reconstruction_valid=reconstruction_valid,title=title,
        matching_confidence=matching_confidence,
        minimum_confidence=minimum_confidence,
        failure_reasons=failure_reasons)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(
            str(output),diagnostic,[cv2.IMWRITE_JPEG_QUALITY,92]):
        raise RuntimeError(f"无法保存材料失败诊断图: {output}")
    return output,observed_length


def _write_material_initialization_failure(
    output: Path,frame: np.ndarray,refined_masks: np.ndarray,
    template_uv: np.ndarray,observed_xyz: np.ndarray,observed_uv: np.ndarray,
    *,template_length_mm: float,
    observed_rms_px: float,maximum_rms_px: float,
    reconstruction_valid: bool,failure_reasons: tuple[str,...] = (),
) -> tuple[Path,float]:
    return _write_material_failure(
        output,frame,refined_masks,template_uv,observed_xyz,observed_uv,
        template_length_mm=template_length_mm,
        observed_rms_px=observed_rms_px,maximum_rms_px=maximum_rms_px,
        reconstruction_valid=reconstruction_valid,
        title="MATERIAL CALIBRATION REJECTED (INITIAL)",
        failure_reasons=failure_reasons)


def _write_material_update_failure(
    output: Path,frame: np.ndarray,refined_masks: np.ndarray,
    previous_uv: np.ndarray,observed_xyz: np.ndarray,observed_uv: np.ndarray,
    *,template_length_mm: float,
    observed_rms_px: float,maximum_rms_px: float,
    reconstruction_valid: bool,matching_confidence: float,
    minimum_confidence: float,failure_reasons: tuple[str,...],
) -> tuple[Path,float]:
    return _write_material_failure(
        output,frame,refined_masks,previous_uv,observed_xyz,observed_uv,
        template_length_mm=template_length_mm,
        observed_rms_px=observed_rms_px,maximum_rms_px=maximum_rms_px,
        reconstruction_valid=reconstruction_valid,
        title="MATERIAL CALIBRATION REJECTED",
        matching_confidence=matching_confidence,
        minimum_confidence=minimum_confidence,
        failure_reasons=failure_reasons)


def _refine_calibration_masks(
    mask_tensor: object,
    mask_refine: object,
) -> np.ndarray:
    """使两种分割后端都使用完全相同的 mask 后处理。"""
    return refine_masks_numpy(mask_tensor,mask_refine)


def _bind_material_model(
    model: LightFieldModel,
    reconstruction: object,
) -> LightFieldModel:
    template=reconstruction.material_template
    return replace(
        model,reconstruction_pipeline=CALIBRATION_RECONSTRUCTION_PIPELINE,
        material_template_sha256=template.sha256)


def _expand_source_parameter(
    value: object,source_layout: LightSourceLayout,tail_shape: tuple[int,...],
    name: str,
) -> np.ndarray:
    """接受按 RGB 三色或按展开灯带 S 配置的参数，并统一展开到 S。"""
    array=np.asarray(value,np.float32)
    source_channels=np.asarray(
        [channel for channel,_ in light_source_specs(source_layout)],np.int32)
    source_count=source_channels.size
    if array.shape==(source_count,*tail_shape):
        return array
    if array.shape==(3,*tail_shape):
        return array[source_channels]
    raise ValueError(
        f"{name} 必须按 RGB 配置为 3x...，或按灯带顺序配置为 "
        f"{source_count}x...")


def _normalise_prompt_signature(prompts: dict[object,object]) -> list[dict[str,object]]:
    """保留 prompt 顺序及 label 类型，生成稳定的重建配置指纹。"""
    result=[]
    for label,group in prompts.items():
        result.append({
            "label_type":type(label).__name__,
            "label":label,
            "positive":[list(map(float,point)) for point in group["positive"]],
            "negative":[list(map(float,point)) for point in group.get("negative",[])],
        })
    return result


def _calibration_reconstruction_signature(
    *,segmentation: object,prompts: dict[object,object],mask_refine: object,
    reconstruction: object,
    raster_max_width: int,raster_max_height: int,
    independent_material_sequence: bool = False,
) -> str:
    """覆盖所有会改变离线 XYZ/UV/depth 的输入和算法语义。"""
    payload={
        "pipeline":CALIBRATION_RECONSTRUCTION_PIPELINE,
        "segmentation":{
            "mode":segmentation.mode,
            "description":segmentation.description,
            "frame_interval":int(segmentation.frame_interval),
            "liteseg_foreground_class":int(
                segmentation.liteseg_foreground_class),
            "liteseg_label":segmentation.liteseg_label,
        },
        "independent_material_sequence":bool(
            independent_material_sequence),
        "prompts":_normalise_prompt_signature(prompts),
        "mask_refine":{
            "pipeline":"largest_external_fill_v1",
            "enabled":bool(mask_refine.enabled),
        },
        "side_edge_extraction":"central_sides_without_endpoint_inference_v3",
        "side_edge_exclusion_ratio":float(
            reconstruction.side_edge_exclusion_ratio),
        "camera_matrix":np.asarray(reconstruction.K,np.float64).tolist(),
        "distortion":np.asarray(
            reconstruction.distortion_coefficients,np.float64).tolist(),
        "s1":float(reconstruction.s1),
        "s2":float(reconstruction.s2),
        "geometry_rows":int(reconstruction.geometry_rows),
        "geometry_columns":int(reconstruction.geometry_columns),
        "uv_boundary_smooth_lambda":float(
            reconstruction.uv_boundary_smooth_lambda),
        "uv_boundary_huber_delta_px":float(
            reconstruction.uv_boundary_huber_delta_px),
        "curve_convexity":str(reconstruction.curve_convexity),
        "material_template_sha256":reconstruction.material_template.sha256,
        "material_coordinate_mode":MATERIAL_COORDINATE_MODE,
        "material_match_confidence_scale_mm":float(
            reconstruction.material_surface.match_confidence_scale_mm),
        "material_rms_confidence_scale_px":float(
            reconstruction.material_surface.rms_confidence_scale_px),
        "material_confidence_floor":float(
            reconstruction.material_surface.confidence_floor),
        "material_calibration_maximum_rms_px":float(
            reconstruction.material_surface.calibration_maximum_rms_px),
        "material_calibration_minimum_confidence":float(
            reconstruction.material_surface.calibration_minimum_confidence),
        "material_raster_max_triangle_width":int(raster_max_width),
        "material_raster_max_triangle_height":int(raster_max_height),
    }
    encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),
                       ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(path: Path) -> tuple[str,int,int]:
    resolved=path.expanduser().resolve()
    stat=resolved.stat()
    return str(resolved),int(stat.st_size),int(stat.st_mtime_ns)


def _observation_metadata(
    image_path: Path,signature: str,curve_convexity: str,
    rotation_vector: np.ndarray,tx: float,rms_values: np.ndarray,
    *,material_template_sha256: str = "0"*64,
    sequence_id: str | None = None,sequence_frame_index: int | None = None,
    matching_confidence: float = 0.,
    visible_fraction: float = 0.,
    raw_observed_xyz: np.ndarray | None = None,
) -> dict[str,object]:
    source_path,source_size,source_mtime_ns=_source_identity(image_path)
    raw_xyz=np.asarray(raw_observed_xyz,np.float32) \
        if raw_observed_xyz is not None else np.empty((0,0,3),np.float32)
    if raw_xyz.size and (raw_xyz.ndim!=3 or raw_xyz.shape[-1]!=3
                         or not np.isfinite(raw_xyz).all()):
        raise ValueError("raw_observed_xyz 必须是有限的 RxCx3")
    raw_curve=raw_xyz[:,0] if raw_xyz.size else np.empty((0,3),np.float32)
    observed_length=(float(np.sum(np.linalg.norm(
        np.diff(raw_curve[:,1:3],axis=0),axis=1)))
        if raw_curve.shape[0]>1 else np.nan)
    return {
        "observation_format_version":np.asarray(
            CALIBRATION_OBSERVATION_FORMAT_VERSION,np.int32),
        "reconstruction_pipeline":np.asarray(
            CALIBRATION_RECONSTRUCTION_PIPELINE),
        "reconstruction_signature":np.asarray(signature),
        "curve_convexity":np.asarray(curve_convexity),
        "reconstruction_rotation_vector":np.asarray(
            rotation_vector,np.float64).reshape(3),
        "reconstruction_tx":np.asarray(tx,np.float64),
        "reconstruction_rms_px":np.asarray(rms_values,np.float32).reshape(-1),
        "material_coordinate_mode":np.asarray(MATERIAL_COORDINATE_MODE),
        "material_template_sha256":np.asarray(material_template_sha256),
        "material_sequence_id":np.asarray(sequence_id or ""),
        "material_sequence_frame_index":np.asarray(
            -1 if sequence_frame_index is None else sequence_frame_index,
            np.int64),
        "material_matching_confidence":np.asarray(
            matching_confidence,np.float32),
        "material_visible_fraction":np.asarray(visible_fraction,np.float32),
        "raw_observed_xyz":raw_xyz,
        "raw_observed_length_mm":np.asarray(observed_length,np.float32),
        "source_image_resolved":np.asarray(source_path),
        "source_image_size":np.asarray(source_size,np.int64),
        "source_image_mtime_ns":np.asarray(source_mtime_ns,np.int64),
    }


def _cached_observation_pose(
    observation_path: Path,image_path: Path,signature: str,
    curve_convexity: str,expected_shape: tuple[int,int],
    saturation_threshold: int,filter_original_saturation: bool,
) -> tuple[np.ndarray,float] | None:
    """仅接受由当前实时 JAX 路径、当前配置和当前源图生成的完整缓存。"""
    try:
        source_path,source_size,source_mtime_ns=_source_identity(image_path)
        with np.load(observation_path,allow_pickle=False) as data:
            required={
                "xyz","uv","st","camera_depth","rgb","valid_mask",
                "observation_format_version","reconstruction_pipeline",
                "reconstruction_signature","curve_convexity",
                "reconstruction_rotation_vector","reconstruction_tx",
                "material_coordinate_mode","material_template_sha256",
                "material_sequence_id","material_sequence_frame_index",
                "material_matching_confidence","material_visible_fraction",
                "raw_observed_xyz","raw_observed_length_mm",
                "source_image_resolved","source_image_size",
                "source_image_mtime_ns","saturation_threshold",
                "original_saturation_filter_enabled",
            }
            if not required.issubset(data.files):
                return None
            if int(data["observation_format_version"]) \
                    != CALIBRATION_OBSERVATION_FORMAT_VERSION:
                return None
            if str(data["reconstruction_pipeline"]) \
                    != CALIBRATION_RECONSTRUCTION_PIPELINE:
                return None
            if str(data["reconstruction_signature"])!=signature \
                    or str(data["curve_convexity"])!=curve_convexity:
                return None
            if str(data["source_image_resolved"])!=source_path \
                    or int(data["source_image_size"])!=source_size \
                    or int(data["source_image_mtime_ns"])!=source_mtime_ns:
                return None
            if int(data["saturation_threshold"])!=saturation_threshold \
                    or bool(data["original_saturation_filter_enabled"]) \
                    != filter_original_saturation:
                return None
            rows,columns=expected_shape
            if data["xyz"].shape!=(rows,columns,3) \
                    or data["uv"].shape!=(rows,columns,2) \
                    or data["st"].shape!=(rows,columns,2) \
                    or data["camera_depth"].shape!=(rows,columns) \
                    or data["rgb"].shape!=(rows,columns,3) \
                    or data["valid_mask"].shape!=(rows,columns):
                return None
            raw_xyz=np.asarray(data["raw_observed_xyz"],np.float32)
            if raw_xyz.shape!=(rows,columns,3) \
                    or not np.isfinite(raw_xyz).all() \
                    or not np.isfinite(float(data["raw_observed_length_mm"])):
                return None
            rotation=np.asarray(
                data["reconstruction_rotation_vector"],np.float64).reshape(3)
            tx=float(data["reconstruction_tx"])
            if not np.isfinite(rotation).all() or not np.isfinite(tx):
                return None
            return rotation,tx
    except (OSError,ValueError,KeyError):
        return None


def _reconstruction_failure_path(output_dir: Path,image_path: Path) -> Path:
    return output_dir/f"{image_path.stem}{_RECONSTRUCTION_FAILURE_SUFFIX}"


def _write_reconstruction_failure(
    output: Path,image_path: Path,signature: str,curve_convexity: str,
    *,stage: str,failure_reasons: tuple[str,...],
    saturation_threshold: int,filter_original_saturation: bool,
    sequence_id: str | None = None,sequence_frame_index: int | None = None,
    tracking_committed: bool = False,observed_length_mm: float = np.nan,
    reprojection_rms_px: float = np.nan,matching_confidence: float = np.nan,
) -> Path:
    """原子保存未进入标定集的帧，使失败结果也能参与完整缓存判定。"""
    if not stage or not failure_reasons:
        raise ValueError("重建失败记录必须包含阶段和至少一个失败原因")
    source_path,source_size,source_mtime_ns=_source_identity(image_path)
    fields={
        "failure_format_version":np.asarray(
            CALIBRATION_FAILURE_FORMAT_VERSION,np.int32),
        "reconstruction_pipeline":np.asarray(
            CALIBRATION_RECONSTRUCTION_PIPELINE),
        "reconstruction_signature":np.asarray(signature),
        "curve_convexity":np.asarray(curve_convexity),
        "result":np.asarray("failure"),
        "failure_stage":np.asarray(stage),
        "failure_reasons":np.asarray(failure_reasons),
        "tracking_committed":np.asarray(tracking_committed),
        "material_sequence_id":np.asarray(sequence_id or ""),
        "material_sequence_frame_index":np.asarray(
            -1 if sequence_frame_index is None else sequence_frame_index,
            np.int64),
        "observed_length_mm":np.asarray(observed_length_mm,np.float32),
        "reconstruction_rms_px":np.asarray(
            reprojection_rms_px,np.float32),
        "material_matching_confidence":np.asarray(
            matching_confidence,np.float32),
        "source_image_resolved":np.asarray(source_path),
        "source_image_size":np.asarray(source_size,np.int64),
        "source_image_mtime_ns":np.asarray(source_mtime_ns,np.int64),
        "saturation_threshold":np.asarray(saturation_threshold,np.int32),
        "original_saturation_filter_enabled":np.asarray(
            filter_original_saturation),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**fields)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _cached_reconstruction_failure(
    failure_path: Path,image_path: Path,signature: str,curve_convexity: str,
    saturation_threshold: int,filter_original_saturation: bool,
) -> tuple[str,...] | None:
    """校验失败缓存确实属于当前源帧、配置和重建实现。"""
    try:
        source_path,source_size,source_mtime_ns=_source_identity(image_path)
        with np.load(failure_path,allow_pickle=False) as data:
            required={
                "failure_format_version","reconstruction_pipeline",
                "reconstruction_signature","curve_convexity","result",
                "failure_stage","failure_reasons","source_image_resolved",
                "source_image_size","source_image_mtime_ns",
                "saturation_threshold","original_saturation_filter_enabled",
            }
            if not required.issubset(data.files):
                return None
            if int(data["failure_format_version"]) \
                    !=CALIBRATION_FAILURE_FORMAT_VERSION \
                    or str(data["reconstruction_pipeline"]) \
                    !=CALIBRATION_RECONSTRUCTION_PIPELINE \
                    or str(data["reconstruction_signature"])!=signature \
                    or str(data["curve_convexity"])!=curve_convexity \
                    or str(data["result"])!="failure":
                return None
            if str(data["source_image_resolved"])!=source_path \
                    or int(data["source_image_size"])!=source_size \
                    or int(data["source_image_mtime_ns"])!=source_mtime_ns:
                return None
            if int(data["saturation_threshold"])!=saturation_threshold \
                    or bool(data["original_saturation_filter_enabled"]) \
                    !=filter_original_saturation:
                return None
            if not str(data["failure_stage"]):
                return None
            reasons=tuple(
                str(value) for value in np.asarray(
                    data["failure_reasons"]).reshape(-1))
            return reasons if reasons and all(reasons) else None
    except (OSError,ValueError,KeyError):
        return None


def _resolve_paths(
    value: str | list[str],
    base: Path,
    field_name: str = "calibration.images",
) -> list[Path]:
    patterns = [value] if isinstance(value, str) else value
    if not isinstance(patterns, list) or not patterns or not all(isinstance(item,str) for item in patterns):
        raise ValueError(f"{field_name} 必须是路径、glob 或非空路径列表")
    paths: list[Path] = []
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        absolute_pattern = str(expanded if expanded.is_absolute() else base/expanded)
        matches = sorted(Path(item) for item in glob.glob(absolute_pattern))
        if not matches: raise FileNotFoundError(f"{field_name} 没有匹配文件: {absolute_pattern}")
        paths.extend(matches)
    return paths


def _resolve_optional_paths(
    value: str | list[str] | None,
    base: Path,
    field_name: str,
) -> list[Path]:
    if value is None:
        return []
    return _resolve_paths(value,base,field_name)


def _iter_physical_batch_indices(
    sample_count: int,
    batch_size: int,
    update_count: int,
    seed: int,
) -> Iterator[tuple[int,np.ndarray]]:
    """逐 epoch 洗牌并依次产生 Adam batch；最后一个 batch 不丢弃。"""
    for name,value in (("sample_count",sample_count),("batch_size",batch_size),
                       ("update_count",update_count)):
        if not isinstance(value,int) or isinstance(value,bool) or value<1:
            raise ValueError(f"{name} 必须是正整数")
    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("seed 必须是整数")
    batch_size=min(batch_size,sample_count)
    generator=np.random.default_rng(seed)
    yielded=0
    epoch=0
    while yielded<update_count:
        epoch+=1
        shuffled=generator.permutation(sample_count)
        for start in range(0,sample_count,batch_size):
            if yielded>=update_count:
                return
            yield epoch,shuffled[start:start+batch_size]
            yielded+=1


def _split_calibration_indices(
    paths: list[Path],
    independent_image_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray,np.ndarray]:
    """独立图片确定性划分；视频沿完整时间轴均匀留出验证帧。"""
    count=len(paths)
    if count<2:
        raise ValueError("背景标定至少需要两个观测样本")
    if not 0<=validation_fraction<1:
        raise ValueError("validation_fraction 必须位于 [0,1)")
    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("validation_seed 必须是整数")
    if not 0<=independent_image_count<=count:
        raise ValueError("独立图片数量无效")
    validation: set[int]=set()
    if validation_fraction>0 and independent_image_count>1:
        n_validation=min(
            independent_image_count-1,
            max(1,int(round(independent_image_count*validation_fraction))))
        generator=np.random.default_rng(seed)
        validation.update(int(index) for index in generator.choice(
            independent_image_count,n_validation,replace=False))
    video_groups: dict[str,list[int]]={}
    for index,path in enumerate(paths[independent_image_count:],
                                independent_image_count):
        stem=path.stem
        group=stem.rsplit("_frame_",1)[0] if "_frame_" in stem else stem
        video_groups.setdefault(group,[]).append(index)
    if validation_fraction>0:
        for indices in video_groups.values():
            indices.sort()
            if len(indices)>1:
                n_validation=min(
                    len(indices)-1,
                    max(1,int(round(len(indices)*validation_fraction))))
                # 连续视频通常从平直到大弯曲。若总把末尾留作验证，训练集会
                # 完全缺失极端弯曲，运行时一弯曲背景就外推失配。验证帧均匀
                # 分散到整条时间轴，剩余训练帧仍覆盖全部几何范围。
                positions=np.floor(
                    (np.arange(n_validation,dtype=np.float64)+.5)
                    *len(indices)/n_validation).astype(np.int64)
                validation.update(indices[int(position)]
                                  for position in positions)
    if validation_fraction>0 and not validation and count>1:
        validation.add(count-1)
    validation_indices=np.asarray(sorted(validation),np.int64)
    training_indices=np.asarray(
        [index for index in range(count) if index not in validation],np.int64)
    if training_indices.size<2:
        raise ValueError("标定/验证划分后训练样本不足两个")
    return training_indices,validation_indices


def _write_split_manifest(
    path: Path,source_paths: list[Path],training: np.ndarray,
    validation: np.ndarray,*,fraction: float,seed: int,
) -> None:
    data={
        "strategy":"independent_seeded_and_video_uniform",
        "validation_fraction":float(fraction),"validation_seed":int(seed),
        "training":[str(source_paths[index]) for index in training],
        "validation":[str(source_paths[index]) for index in validation],
    }
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        yaml.safe_dump(data,stream,allow_unicode=True,sort_keys=False)
    temporary.replace(path)


def extract_video_frames(
    video_paths: list[Path],
    output_dir: Path,
    *,
    frame_step: int = 1,
    max_frames_per_file: int | None = None,
    reuse_existing: bool = True,
    group_prefix: str = "video",
) -> list[Path]:
    """顺序解码视频，将参与标定的帧无损保存，供标定的两个阶段复用。"""
    if not isinstance(frame_step,int) or isinstance(frame_step,bool) or frame_step<1:
        raise ValueError("video_frame_step 必须是正整数")
    if max_frames_per_file is not None and (
        not isinstance(max_frames_per_file,int)
        or isinstance(max_frames_per_file,bool)
        or max_frames_per_file<1
    ):
        raise ValueError("video_max_frames_per_file 必须是正整数或 null")

    if not video_paths:
        return []
    if not isinstance(group_prefix,str) or not group_prefix.strip() \
            or any(character not in
                   "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                   for character in group_prefix):
        raise ValueError("视频抽帧 group_prefix 必须是非空字母数字下划线字符串")
    output_dir.mkdir(parents=True,exist_ok=True)
    extracted_paths: list[Path] = []
    for video_number,video_path in enumerate(video_paths,1):
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频扩展名: {video_path}")
        prefix=(f"{group_prefix}_{video_number:03d}_"
                f"{video_path.stem}_frame_")
        if reuse_existing:
            existing=sorted(output_dir.glob(f"{prefix}*.png"))
            if existing:
                if max_frames_per_file is not None:
                    existing=existing[:max_frames_per_file]
                if frame_step>1:
                    existing=[
                        path for path in existing
                        if int(path.stem.rsplit("_",1)[-1])%frame_step==0
                    ]
                if existing:
                    extracted_paths.extend(existing)
                    print(f"视频 {video_number}/{len(video_paths)}: {video_path.name}，"
                          f"复用已有 {len(existing)} 帧 -> {output_dir}")
                    continue

        capture=cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开标定视频: {video_path}")

        decoded_count=0
        selected_count=0
        reused_count=0
        try:
            while True:
                ok,frame=capture.read()
                if not ok or frame is None:
                    break
                frame_index=decoded_count
                decoded_count+=1
                if frame_index%frame_step!=0:
                    continue
                output_path=output_dir/f"{prefix}{frame_index:08d}.png"
                if reuse_existing and output_path.exists():
                    reused_count+=1
                elif not cv2.imwrite(str(output_path),frame):
                    raise RuntimeError(f"无法保存视频标定帧: {output_path}")
                extracted_paths.append(output_path)
                selected_count+=1
                if (max_frames_per_file is not None
                        and selected_count>=max_frames_per_file):
                    break
        finally:
            capture.release()
        if selected_count==0:
            raise RuntimeError(f"标定视频没有可读取帧: {video_path}")
        print(f"视频 {video_number}/{len(video_paths)}: {video_path.name}，"
              f"顺序读取 {decoded_count} 帧，选取 {selected_count} 帧"
              f"（复用 {reused_count}）-> {output_dir}")
    return extracted_paths

def save_calibration_observation(image_path: Path, frame: np.ndarray,
                                 point_set: ReconstructionPointSet,
                                 output_dir: Path, map_dir: Path,
                                 saturation_threshold: int = 250,
                                 *,filter_original_saturation: bool = False,
                                 vertex_valid_mask: np.ndarray | None = None,
                                 reconstruction_metadata: dict[str,object] | None = None,
                                 ) -> Path:
    """严格使用当前图像内部重建出的 point_set，保存映射并生成标定观测。"""
    map_path=save_generated_uv_xyz_map(
        map_dir/f"{image_path.stem}_uv_xyz.npz",point_set,
        metadata=reconstruction_metadata)
    with np.load(map_path) as data: xyz,uv,st,camera_depth=point_set_to_grid(data)
    rgb=sample_rgb(frame,uv)
    if filter_original_saturation:
        valid_mask=sample_unsaturated_mask(frame,uv,saturation_threshold)
    else:
        # 当前标定保留相机裁剪平台；这里只检查双线性采样范围。
        height,width=frame.shape[:2]
        valid_mask=((uv[...,0]>=0)&(uv[...,0]<width-1)&
                    (uv[...,1]>=0)&(uv[...,1]<height-1))
    if vertex_valid_mask is not None:
        material_valid=np.asarray(vertex_valid_mask,np.bool_)
        if material_valid.shape!=valid_mask.shape:
            raise ValueError("材料可见掩膜尺寸与标定观测网格不一致")
        valid_mask&=material_valid
    valid_count=int(valid_mask.sum()); total_count=valid_mask.size
    if valid_count==0: raise RuntimeError(f"{image_path.name} 没有有效空间点")
    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f"{image_path.stem}.npz"
    fields: dict[str,object]={
        "xyz":xyz,"uv":uv,"st":st,"camera_depth":camera_depth,
        "rgb":rgb,"valid_mask":valid_mask,
        "saturation_threshold":np.asarray(saturation_threshold),
        "image_shape":frame.shape[:2],
        "original_saturation_filter_enabled":np.asarray(
            filter_original_saturation),
        "source_image":np.asarray(str(image_path.expanduser().resolve())),
        "source_surface_map":np.asarray(str(map_path.expanduser().resolve())),
    }
    if reconstruction_metadata is not None:
        overlap=fields.keys()&reconstruction_metadata.keys()
        if overlap:
            raise ValueError(
                f"标定观测元数据不能覆盖数据字段: {sorted(overlap)}")
        fields.update(reconstruction_metadata)
    temporary=output.with_name(output.name+".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**fields)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    filter_description=(f"original threshold<{saturation_threshold}"
                        if filter_original_saturation else "image bounds only")
    print(f"观测 {image_path.name}: 有效 {valid_count}/{total_count} "
          f"({100*valid_count/total_count:.1f}%, {filter_description}) -> {output}")
    return output

def reconstruct_all_observations(image_paths: list[Path], all_config: dict,
                                 config_path: Path, output_dir: Path,
                                 map_dir: Path,saturation_threshold: int = 250,
                                 *,filter_original_saturation: bool = False,
                                 reuse_existing: bool = True,
                                 material_initialization_diagnostic_dir: object =
                                 _DEFAULT_MATERIAL_DIAGNOSTIC_DIR,
                                 material_update_failure_diagnostic_dir: object =
                                 _DEFAULT_MATERIAL_DIAGNOSTIC_DIR,
                                 independent_material_sequence_id: str | None =
                                 None,
                                 ) -> list[Path]:
    """对每个图片或视频帧执行配置的分割、全局重建、映射和 RGB 采样。"""
    if independent_material_sequence_id is not None and (
            not isinstance(independent_material_sequence_id,str)
            or not independent_material_sequence_id.strip()):
        raise ValueError(
            "independent_material_sequence_id 必须是非空字符串或 null")
    output_dir.mkdir(parents=True,exist_ok=True)
    map_dir.mkdir(parents=True,exist_ok=True)
    surface=all_config["get_surface"]
    prompts=parse_prompts(surface["prompts"])
    mask_refine=parse_mask_refine(surface.get("mask_refine"))
    segmentation=parse_surface_segmentation_config(
        surface,config_path=config_path)
    calibration_output=all_config.get("calibration",{}).get("output")
    reconstruction=parse_reconstruction_config(
        surface.get("reconstruction"),config_path=config_path,
        calibration_output=calibration_output)
    material_template=reconstruction.material_template
    runtime=all_config.get("lightfield",{}).get("runtime",{})
    if not isinstance(runtime,dict):
        raise ValueError("lightfield.runtime 必须是字典")
    raster_width_value=runtime.get("gpu_raster_max_triangle_width",64)
    raster_height_value=runtime.get("gpu_raster_max_triangle_height",32)
    if any(not isinstance(value,int) or isinstance(value,bool) or value<1
           for value in (raster_width_value,raster_height_value)):
        raise ValueError("runtime gpu_raster_max_triangle_* 参数必须为正整数")
    # 离线规范采样已经以 64x32 为显式安全下限，候选提交门控与它完全一致。
    raster_max_width=max(raster_width_value,64)
    raster_max_height=max(raster_height_value,32)
    signature=_calibration_reconstruction_signature(
        segmentation=segmentation,prompts=prompts,mask_refine=mask_refine,
        reconstruction=reconstruction,raster_max_width=raster_max_width,
        raster_max_height=raster_max_height,
        independent_material_sequence=
            independent_material_sequence_id is not None)
    expected_shape=(reconstruction.geometry_rows,
                    reconstruction.geometry_columns)
    pending: list[tuple[int,Path]]=[]
    outputs: list[Path | None]=[None]*len(image_paths)
    reused_poses: list[tuple[np.ndarray,float]]=[]
    reused=0; reused_failures=0
    for index,image_path in enumerate(image_paths):
        observation_path=output_dir/f"{image_path.stem}.npz"
        if reuse_existing and observation_path.exists():
            pose=_cached_observation_pose(
                observation_path,image_path,signature,
                reconstruction.curve_convexity,expected_shape,
                saturation_threshold,filter_original_saturation)
            if pose is not None:
                outputs[index]=observation_path
                reused_poses.append(pose)
                reused+=1
                continue
        failure_path=_reconstruction_failure_path(output_dir,image_path)
        if reuse_existing and failure_path.exists() \
                and _cached_reconstruction_failure(
                    failure_path,image_path,signature,
                    reconstruction.curve_convexity,saturation_threshold,
                    filter_original_saturation) is not None:
            reused_failures+=1
            continue
        pending.append((index,image_path))
    if reused_poses:
        first_rotation,first_tx=reused_poses[0]
        pose_consistent=all(
            np.allclose(rotation,first_rotation,rtol=0.,atol=1e-9)
            and np.isclose(tx,first_tx,rtol=0.,atol=1e-9)
            for rotation,tx in reused_poses[1:])
        if not pose_consistent:
            print("已有观测包含不同的固定外参；为保证同一重建坐标系，全部重新生成")
            outputs=[None]*len(image_paths)
            pending=list(enumerate(image_paths))
            reused_poses=[]
            reused=0
            reused_failures=0
    if not pending:
        completed=[path for path in outputs if path is not None]
        print(
            f"完整重建账本命中 {len(image_paths)}/{len(image_paths)}："
            f"成功 {len(completed)}，失败 {reused_failures}；跳过全部重建计算")
        if not completed:
            raise RuntimeError("缓存账本显示所有标定观测均未通过实时 JAX/凸性重建")
        return completed
    if reused or reused_failures:
        print(
            f"复用已有结果：成功 {reused}，失败 {reused_failures}，"
            f"未记录 {len(pending)}/{len(image_paths)}")
    elif reuse_existing and any(
            (output_dir/f"{path.stem}.npz").exists()
            or _reconstruction_failure_path(output_dir,path).exists()
            for path in image_paths):
        print("已有结果不含当前实时 JAX/凸性重建指纹；旧缓存将自动重建")
    # 材料状态具有时序依赖；只要任一观测缺失，就从输入起点重放全部序列，
    # 避免把缓存帧之后的状态接到错误前驱。
    if len(pending)!=len(image_paths):
        print("材料状态链存在待重建帧；为保证前驱一致性，全部观测重新生成")
        outputs=[None]*len(image_paths)
        pending=list(enumerate(image_paths))
        reused_poses=[]

    print(f"加载全局曲面分割模型: {segmentation.description}")
    segmenter=SurfaceSegmenter(
        segmentation,prompts=prompts,mask_refine=mask_refine)
    reconstructor=EdgeReconstructor(reconstruction.K,reconstruction.distortion_coefficients,
                                   reconstruction.s1,reconstruction.s2,
                                   sample_count=reconstruction.sample_count)
    if reused_poses:
        reconstructor.rotation_vector=reused_poses[0][0].copy()
        reconstructor.tx=float(reused_poses[0][1])
        reconstructor.calibrated=True
    device=choose_device(all_config.get("lightfield",{}).get("device","gpu"))
    camera_matrix_gpu=jax.device_put(
        np.asarray(reconstruction.K,np.float32),device)
    distortion_gpu=jax.device_put(np.asarray(
        reconstruction.distortion_coefficients,np.float32),device)
    inverse_camera_gpu=jax.device_put(np.asarray(
        np.linalg.inv(reconstruction.K),np.float32),device)
    template_st_gpu=jax.device_put(material_template.st,device)
    template_lengths_gpu=jax.device_put(
        material_template.segment_lengths_mm,device)
    template_x_gpu=jax.device_put(material_template.x_coordinates_mm,device)
    template_curve_gpu=jax.device_put(
        material_template.reference_curve_yz,device)
    template_angles_gpu=jax.device_put(
        material_template.reference_angles_rad,device)
    material_cfg=reconstruction.material_surface
    initialization_diagnostic_value=(
        all_config.get("lightfield",{}).get("calibration",{}).get(
            "material_initialization_diagnostic_dir",
            output_dir/"material_initialization_failures")
        if material_initialization_diagnostic_dir is
            _DEFAULT_MATERIAL_DIAGNOSTIC_DIR
        else material_initialization_diagnostic_dir)
    update_diagnostic_value=(
        all_config.get("lightfield",{}).get("calibration",{}).get(
            "material_update_failure_diagnostic_dir",
            output_dir/"material_update_failures")
        if material_update_failure_diagnostic_dir is
            _DEFAULT_MATERIAL_DIAGNOSTIC_DIR
        else material_update_failure_diagnostic_dir)

    def resolve_diagnostic_dir(value: object,name: str) -> Path | None:
        if value is None:
            return None
        if not isinstance(value,(str,Path)):
            raise ValueError(f"{name} 必须是路径字符串或 null")
        result=Path(value).expanduser()
        if not result.is_absolute():
            result=config_path.parent/result
        result.mkdir(parents=True,exist_ok=True)
        print(f"{name}: {result}")
        return result

    material_diagnostic_dir=resolve_diagnostic_dir(
        initialization_diagnostic_value,"材料初始化失败诊断目录")
    material_update_diagnostic_dir=resolve_diagnostic_dir(
        update_diagnostic_value,"材料更新失败诊断目录")

    # mask 已在 CPU 完成最大外轮廓保留和填洞，JAX 不再改变边界。
    prepare_curves_gpu=jax.jit(lambda masks:prepare_edge_curves_from_masks_jax(
        masks,camera_matrix_gpu,distortion_gpu,reconstructor.sample_count,
        side_edge_exclusion_ratio=
            reconstruction.side_edge_exclusion_ratio)[1:])
    reconstruct_geometry_gpu=jax.jit(
        lambda masks,rotation,tx,previous_state,initialized:
        reconstruct_material_surface_with_diagnostics_from_masks_jax(
            masks,camera_matrix_gpu,distortion_gpu,inverse_camera_gpu,
            rotation,reconstruction.s1,reconstruction.s2,tx,
            reconstructor.sample_count,
            reconstruction.pair_fill_count,
            reconstruction.uv_boundary_smooth_lambda,
            reconstruction.uv_boundary_huber_delta_px,
            previous_state,template_st_gpu,template_lengths_gpu,template_x_gpu,
            initialized=initialized,
            initial_calibration_maximum_rms_px=
                material_cfg.initial_calibration_maximum_rms_px,
            calibration_maximum_rms_px=
                material_cfg.calibration_maximum_rms_px,
            calibration_minimum_confidence=
                material_cfg.calibration_minimum_confidence,
            match_confidence_scale_mm=
                material_cfg.match_confidence_scale_mm,
            rms_confidence_scale_px=material_cfg.rms_confidence_scale_px,
            confidence_floor=material_cfg.confidence_floor,
            bend_direction=material_cfg.bend_direction,
            maximum_uv_triangle_width_px=raster_max_width,
            maximum_uv_triangle_height_px=raster_max_height,
            side_edge_exclusion_ratio=
                reconstruction.side_edge_exclusion_ratio))
    material_state_gpu=None
    material_sequence: str | None=None
    material_initialized=False
    try:
        for done,(index,image_path) in enumerate(pending,1):
            observation_path=output_dir/f"{image_path.stem}.npz"
            failure_record_path=_reconstruction_failure_path(
                output_dir,image_path)
            # 失败目录反映本次重建结果，避免已恢复帧继续遗留旧失败图。
            for diagnostic_dir,suffix in (
                (material_diagnostic_dir,"material_init_failure.jpg"),
                (material_update_diagnostic_dir,"material_update_failure.jpg"),
            ):
                if diagnostic_dir is None:
                    continue
                stale=diagnostic_dir/f"{image_path.stem}_{suffix}"
                try:
                    stale.unlink(missing_ok=True)
                except OSError as error:
                    print(f"警告：无法移除旧材料失败诊断图 {stale}: {error}")
            frame=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
            if frame is None: raise RuntimeError(f"无法读取标定观测帧: {image_path}")
            source_sequence_id,sequence_frame_index=_material_sequence_identity(
                image_path)
            sequence_id=source_sequence_id
            if sequence_id is None and independent_material_sequence_id \
                    is not None:
                sequence_id=independent_material_sequence_id
                sequence_frame_index=index
            reset_material_state=(
                sequence_id is None or sequence_id!=material_sequence)
            segmented=segmenter.segment_tensors(frame)
            labels,mask_tensor=segmented[:2]
            cleaned_masks=_refine_calibration_masks(mask_tensor,mask_refine)
            mask_gpu=jax.device_put(cleaned_masks,device)
            if not reconstructor.calibrated:
                left_curves,right_dense_curves,edge_valid=jax.device_get(
                    prepare_curves_gpu(mask_gpu))
                for label_index in np.flatnonzero(edge_valid):
                    try:
                        reconstructor.process_curves(
                            left_curves[label_index],
                            right_dense_curves[label_index])
                    except ValueError:
                        continue
                    break
                if not reconstructor.calibrated:
                    observation_path.unlink(missing_ok=True)
                    (map_dir/f"{image_path.stem}_uv_xyz.npz").unlink(
                        missing_ok=True)
                    _write_reconstruction_failure(
                        failure_record_path,image_path,signature,
                        reconstruction.curve_convexity,
                        stage="fixed_pose_initialization",
                        failure_reasons=("fixed_pose_initialization_failed",),
                        saturation_threshold=saturation_threshold,
                        filter_original_saturation=filter_original_saturation,
                        sequence_id=sequence_id,
                        sequence_frame_index=sequence_frame_index)
                    print(
                        f"跳过第 {index+1} 个观测帧：无法初始化固定外参 "
                        f"{image_path}")
                    continue
            rotation=cv2.Rodrigues(
                reconstructor.rotation_vector)[0].astype(np.float32)
            rotation_device=jax.device_put(rotation,device)
            tx_device=jax.device_put(
                np.asarray(reconstructor.tx,np.float32),device)
            if reset_material_state or material_state_gpu is None:
                material_state_gpu=reference_material_surface_state_jax(
                    template_curve_gpu,template_angles_gpu,template_x_gpu,
                    camera_matrix_gpu,distortion_gpu,rotation_device,tx_device)
                material_initialized=False
                material_sequence=sequence_id
            was_material_initialized=material_initialized
            material_result=reconstruct_geometry_gpu(
                mask_gpu,rotation_device,tx_device,material_state_gpu,
                jnp.asarray(was_material_initialized))
            material_state_gpu=material_result[1]
            (refined_masks,material_state_host,st_grid,reconstruction_valid,
             tracking_accepted,calibration_accepted,observed_xyz,observed_uv,
             observed_rms,failure_flags)=jax.device_get(material_result)
            tracking_committed=bool(tracking_accepted)
            if tracking_committed:
                material_initialized=True
            accepted=bool(calibration_accepted)
            if not accepted:
                failure_reasons=tuple(
                    name for name,failed in zip(
                        MATERIAL_UPDATE_FAILURE_NAMES,
                        np.asarray(failure_flags,np.bool_).tolist(),strict=True)
                    if failed)
                if not failure_reasons:
                    failure_reasons=("calibration_rejected",)
                failed_rms=float(material_state_host.reprojection_rms_px)
                failed_confidence=float(
                    material_state_host.matching_confidence)
                diagnostic_path=None
                observed_curve=np.asarray(observed_xyz,np.float32)[:,0,1:3]
                observed_length=float(np.sum(np.linalg.norm(
                    np.diff(observed_curve,axis=0),axis=1)))
                if not was_material_initialized \
                        and material_diagnostic_dir is not None:
                    diagnostic_path=(material_diagnostic_dir/
                                     f"{image_path.stem}_material_init_failure.jpg")
                    try:
                        diagnostic_path,observed_length=(
                            _write_material_initialization_failure(
                                diagnostic_path,frame,refined_masks,
                                np.asarray(material_state_host.uv),observed_xyz,
                                observed_uv,
                                template_length_mm=
                                    material_template.total_length_mm,
                                observed_rms_px=failed_rms,
                                maximum_rms_px=(material_cfg.
                                    initial_calibration_maximum_rms_px),
                                reconstruction_valid=bool(
                                    reconstruction_valid[0]),
                                failure_reasons=failure_reasons))
                    except (OSError,RuntimeError,ValueError) as error:
                        diagnostic_path=None
                        print(f"警告：材料初始化诊断图保存失败: {error}")
                elif was_material_initialized \
                        and material_update_diagnostic_dir is not None:
                    diagnostic_path=(material_update_diagnostic_dir/
                                     f"{image_path.stem}_material_update_failure.jpg")
                    try:
                        diagnostic_path,observed_length=(
                            _write_material_update_failure(
                                diagnostic_path,frame,refined_masks,
                                np.asarray(material_state_host.uv),observed_xyz,
                                observed_uv,
                                template_length_mm=
                                    material_template.total_length_mm,
                                observed_rms_px=failed_rms,
                                maximum_rms_px=
                                    material_cfg.calibration_maximum_rms_px,
                                reconstruction_valid=bool(
                                    reconstruction_valid[0]),
                                matching_confidence=failed_confidence,
                                minimum_confidence=
                                    material_cfg.calibration_minimum_confidence,
                                failure_reasons=failure_reasons))
                    except (OSError,RuntimeError,ValueError) as error:
                        diagnostic_path=None
                        print(f"警告：材料更新失败诊断图保存失败: {error}")
                if not was_material_initialized and not tracking_committed:
                    material_state_gpu=None
                    material_sequence=None
                observation_path.unlink(missing_ok=True)
                (map_dir/f"{image_path.stem}_uv_xyz.npz").unlink(
                    missing_ok=True)
                _write_reconstruction_failure(
                    failure_record_path,image_path,signature,
                    reconstruction.curve_convexity,
                    stage="material_calibration_admission",
                    failure_reasons=failure_reasons,
                    saturation_threshold=saturation_threshold,
                    filter_original_saturation=filter_original_saturation,
                    sequence_id=sequence_id,
                    sequence_frame_index=sequence_frame_index,
                    tracking_committed=tracking_committed,
                    observed_length_mm=observed_length,
                    reprojection_rms_px=failed_rms,
                    matching_confidence=failed_confidence)
                diagnostic_text=f"，diagnostic={diagnostic_path}" \
                    if diagnostic_path is not None else ""
                print(
                    f"跳过第 {index+1} 个观测帧：未通过标定准入，"
                    f"tracking={int(tracking_committed)}，"
                    f"sequence={sequence_id!r}，reasons={failure_reasons}，"
                    f"length={observed_length:.3f}/"
                    f"{material_template.total_length_mm:.3f}mm，"
                    f"RMS={failed_rms:.3f}px，"
                    f"confidence={failed_confidence:.6f}，"
                    f"{image_path}{diagnostic_text}")
                continue
            raw_observed_xyz=np.asarray(observed_xyz,np.float32)
            del refined_masks,observed_xyz,observed_uv,observed_rms
            xyz_grid=np.asarray(material_state_host.xyz)
            uv_grid=np.asarray(material_state_host.uv)
            depth_grid=np.asarray(material_state_host.camera_depth)
            material_visible=np.asarray(material_state_host.visible,np.bool_)
            rms_values=np.asarray(
                [material_state_host.reprojection_rms_px],np.float32)
            point_set=point_set_from_surface_grids(
                xyz_grid,uv_grid,st_grid,depth_grid,reconstruction.K,
                reconstruction.distortion_coefficients,
                surface_count=len(labels),
                surface_rows=reconstruction.geometry_rows)
            metadata=_observation_metadata(
                image_path,signature,reconstruction.curve_convexity,
                reconstructor.rotation_vector,reconstructor.tx,rms_values,
                material_template_sha256=material_template.sha256,
                sequence_id=sequence_id,
                sequence_frame_index=sequence_frame_index,
                matching_confidence=float(
                    material_state_host.matching_confidence),
                visible_fraction=float(
                    material_state_host.visible_fraction),
                raw_observed_xyz=raw_observed_xyz)
            # 先移除相反结果；若随后写盘中断，下次会把本帧识别为未完成并重放。
            failure_record_path.unlink(missing_ok=True)
            outputs[index]=save_calibration_observation(
                image_path,frame,point_set,output_dir,map_dir,saturation_threshold,
                filter_original_saturation=filter_original_saturation,
                vertex_valid_mask=material_visible,
                reconstruction_metadata=metadata)
            print(f"全局重建 {done}/{len(pending)} "
                  f"(总进度 {index+1}/{len(image_paths)}) 完成: {image_path.name}；"
                  f"convexity={reconstruction.curve_convexity}，"
                  f"RMS={float(np.max(rms_values)):.3f}px")
    finally:
        del segmenter
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    completed=[path for path in outputs if path is not None]
    if not completed:
        raise RuntimeError("所有标定观测均未通过实时 JAX/凸性重建")
    skipped=len(image_paths)-len(completed)
    if skipped:
        print(f"实时重建有效观测 {len(completed)}/{len(image_paths)}；"
              f"跳过 {skipped} 帧，不写入训练集")
    return completed


def _collect_direct_canonical_fields(
    source_images: list[Path],uv_values: list[np.ndarray],
    depth_values: list[np.ndarray],*,sample_shape: tuple[int,int],
    erode_pixels: int,raster_triangle_chunk: int,
    raster_max_width: int,raster_max_height: int,device: jax.Device,
) -> tuple[np.ndarray,np.ndarray]:
    """按现有 UV/有效域把绝对线性 RGB 采样到规范 observation_grid。"""
    first=cv2.imread(str(source_images[0]),cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"无法读取纯拟合标定图像: {source_images[0]}")
    image_shape=first.shape[:2]

    @jax.jit
    def sample_one(frame_bgr,uv,camera_depth):
        attributes=jnp.ones((*uv.shape[:2],1),jnp.float32)
        _,valid,overflow=rasterize_attributes_jax(
            uv,camera_depth,attributes,image_shape,
            triangle_chunk=raster_triangle_chunk,
            max_triangle_width=raster_max_width,
            max_triangle_height=raster_max_height)
        field=bgr_to_linear_rgb_jax(frame_bgr)
        canonical,canonical_valid=build_canonical_residual_sample_jax(
            field,frame_bgr,valid,uv,sample_shape,
            saturation_threshold=255,
            erode_pixels=erode_pixels)
        return canonical,canonical_valid,overflow

    fields=None; valid_fields=None
    for index,(image_path,uv,depth) in enumerate(zip(
            source_images,uv_values,depth_values,strict=True)):
        frame=first if index==0 else cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取纯拟合标定图像: {image_path}")
        if frame.shape[:2]!=image_shape:
            raise ValueError(f"纯拟合标定图像尺寸不一致: {image_path}")
        current,current_valid,overflow=sample_one(
            jax.device_put(jnp.asarray(frame,jnp.uint8),device),
            jax.device_put(jnp.asarray(uv,jnp.float32),device),
            jax.device_put(jnp.asarray(depth,jnp.float32),device))
        current,current_valid,overflow=jax.device_get(
            (current,current_valid,overflow))
        if bool(overflow):
            raise RuntimeError(
                "纯拟合规范采样超过 GPU 光栅化容量，请增大 runtime.gpu_raster_*" )
        if fields is None:
            fields=np.empty((len(source_images),*current.shape),np.float32)
            valid_fields=np.empty(
                (len(source_images),*current_valid.shape),np.bool_)
        fields[index]=current; valid_fields[index]=current_valid
        if index==0 or (index+1)%50==0 or index+1==len(source_images):
            print(f"绝对背景样本 {index+1}/{len(source_images)}: "
                  f"valid={int(np.count_nonzero(current_valid))}/"
                  f"{current_valid.size}")
    assert fields is not None and valid_fields is not None
    return fields,valid_fields


def _valid_rmse(values: np.ndarray,valid: np.ndarray) -> np.ndarray:
    count=max(int(np.count_nonzero(valid)),1)
    return np.sqrt(np.sum(
        np.where(valid[...,None],np.asarray(values,np.float64)**2,0),
        axis=tuple(range(values.ndim-1)))/count)


def _calibrate_direct_fit_3(
    *,raw: dict,cfg: dict,reconstruction: object,device: jax.Device,
    background_method: str,
    source_layout: LightSourceLayout,source_images: list[Path],
    uv_parts: list[np.ndarray],depth_parts: list[np.ndarray],xyz_all: np.ndarray,
    training_indices: np.ndarray,validation_indices: np.ndarray,
    model_output: Path,
) -> None:
    """训练 direct_fit_3 几何条件神经场，并报告同源留出集误差。"""
    if background_method!="direct_fit_3":
        raise ValueError("direct_fit_3 标定收到无效 background_method")
    runtime=raw.get("runtime",{})
    direct=parse_direct_fit_3_config(raw)
    sample_shape=(reconstruction.observation_rows,
                  reconstruction.observation_columns)
    raster_triangle_chunk=int(runtime.get("gpu_raster_triangle_chunk",256))
    raster_max_width=max(
        int(runtime.get("gpu_raster_max_triangle_width",24)),64)
    raster_max_height=max(
        int(runtime.get("gpu_raster_max_triangle_height",12)),32)
    canonical,canonical_valid=_collect_direct_canonical_fields(
        source_images,uv_parts,depth_parts,sample_shape=sample_shape,
        erode_pixels=direct.sample_erode_pixels,
        raster_triangle_chunk=raster_triangle_chunk,
        raster_max_width=raster_max_width,raster_max_height=raster_max_height,
        device=device)
    if training_indices.size<3:
        raise ValueError("direct_fit_3 统一神经场至少需要 3 个训练样本")
    checkpoint_path=model_output.with_suffix(".best_ckpt.npz")
    checkpoint_validation_indices=np.empty(0,np.int64)
    if validation_indices.size:
        checkpoint_validation_indices=validation_indices[np.linspace(
            0,validation_indices.size-1,
            min(direct.validation_frame_count,validation_indices.size),
            dtype=np.int64)]
    (base_texture,coordinate_frequencies,geometry_mean,geometry_scale,
     geometry_pca_components,geometry_pca_scale,
     local_geometry_mean,local_geometry_scale,
     encoder_weights,encoder_biases,decoder_weights,decoder_biases)=(
        fit_direct_geometry_conditioned_field_gpu(
        canonical[training_indices],canonical_valid[training_indices],
        surface_xyz=xyz_all[training_indices],device=device,
        validation_fields=(canonical[checkpoint_validation_indices]
                           if checkpoint_validation_indices.size else None),
        validation_valid=(canonical_valid[checkpoint_validation_indices]
                          if checkpoint_validation_indices.size else None),
        validation_surface_xyz=(xyz_all[checkpoint_validation_indices]
                                if checkpoint_validation_indices.size else None),
        checkpoint_path=(checkpoint_path
                         if checkpoint_validation_indices.size else None),
        frequencies=direct.coordinate_frequencies,
        geometry_descriptor_rows=direct.geometry_descriptor_rows,
        geometry_encoder_width=direct.geometry_encoder_width,
        geometry_encoder_layers=direct.geometry_encoder_layers,
        geometry_latent_dimensions=direct.geometry_latent_dimensions,
        geometry_pca_dimensions=direct.geometry_pca_dimensions,
        decoder_width=direct.decoder_width,
        decoder_layers=direct.decoder_layers,
        steps=direct.steps,batch_size=direct.batch_size,
        frame_batch_size=direct.frame_batch_size,
        learning_rate=direct.learning_rate,
        huber_delta=float(cfg.get("residual_huber_delta",.04)),
        base_huber_iterations=direct.base_huber_iterations,
        adaptive_channel_weight_strength=(
            direct.adaptive_channel_weight_strength),
        spatial_difference_weight=direct.spatial_difference_weight,
        spatial_difference_points_per_frame=(
            direct.spatial_difference_points_per_frame),
        geometry_difference_weight=direct.geometry_difference_weight,
        geometry_difference_neighbor_count=(
            direct.geometry_difference_neighbor_count),
        geometry_difference_points_per_pair=(
            direct.geometry_difference_points_per_pair),
        seed=int(cfg.get("training_seed",cfg.get("physical_seed",0))),
        validation_interval=direct.validation_interval,
        validation_points_per_frame=direct.validation_points_per_frame,
        early_stopping_patience=direct.early_stopping_patience,
        early_stopping_min_steps=direct.early_stopping_min_steps,
        early_stopping_min_delta=direct.early_stopping_min_delta,
        separate_channel_decoders=True))
    # 释放 Adam/训练图占用的编译缓存与碎片显存，再展开 observation_grid 全场。
    jax.clear_caches()
    gc.collect()
    residual_rows=reconstruction.residual_coefficient_rows
    residual_columns=reconstruction.residual_coefficient_columns
    session_correction=np.zeros((
        3,residual_rows,residual_columns),np.float32)
    common_model_arguments={
        "base_texture":base_texture,
        "coordinate_frequencies":coordinate_frequencies,
        "geometry_feature_mean":geometry_mean,
        "geometry_feature_scale":geometry_scale,
        "geometry_pca_components":geometry_pca_components,
        "geometry_pca_scale":geometry_pca_scale,
        "local_geometry_feature_mean":local_geometry_mean,
        "local_geometry_feature_scale":local_geometry_scale,
        "geometry_encoder_weights":encoder_weights,
        "geometry_encoder_biases":encoder_biases,
        "geometry_descriptor_rows":direct.geometry_descriptor_rows,
        "curve_convexity":reconstruction.curve_convexity,
        "reconstruction_pipeline":CALIBRATION_RECONSTRUCTION_PIPELINE,
        "source_layout":source_layout,
    }
    model=LightFieldModel.direct_fit_3(
        session_correction,
        channel_decoder_weights=decoder_weights,
        channel_decoder_biases=decoder_biases,
        **common_model_arguments)

    # 最佳 checkpoint 已包含生成最终模型所需的全部参数。先绑定模板并原子
    # 保存，再展开训练/验证集全场预测；后者耗时且显存占用高，不能让报告阶段
    # 的中断留下“checkpoint 完整但最终 YAML 不存在”的半成品状态。
    model=_bind_material_model(model,reconstruction)
    model.save(model_output)
    print(f"{background_method} 最佳参数模型已先行保存：{model_output}")

    # 与训练 batch_size 对齐：400x202 全场一次性 decode 会再申请 ~2GiB+ 激活。
    field_chunk_size=max(int(direct.batch_size),1)

    def predictions(
        current_model: LightFieldModel,indices: np.ndarray,
    ) -> np.ndarray:
        model_gpu=jax.device_put(current_model,device)
        return np.stack([
            direct_background_field_chunked(
                sample_shape,xyz_all[index],model_gpu,
                chunk_size=field_chunk_size,device=device)
            for index in indices])

    training_prediction=predictions(model,training_indices)
    training_base_rmse=_valid_rmse(
        canonical[training_indices]-base_texture[None],
        canonical_valid[training_indices])
    training_rmse=_valid_rmse(
        canonical[training_indices]-training_prediction,
        canonical_valid[training_indices])
    validation_rmse=np.full(3,np.nan,np.float64)
    validation_base_rmse=np.full(3,np.nan,np.float64)
    if validation_indices.size:
        validation_base_rmse=_valid_rmse(
            canonical[validation_indices]-base_texture[None],
            canonical_valid[validation_indices])
        validation_prediction=predictions(model,validation_indices)
        validation_rmse=_valid_rmse(
            canonical[validation_indices]-validation_prediction,
            canonical_valid[validation_indices])

    print(f"{background_method} 已保存 B + delta B + Bsession 模型："
          f"B-only train/validation RMSE RGB="
          f"{training_base_rmse.tolist()}/{validation_base_rmse.tolist()}，"
          f"B+deltaB train/validation RMSE RGB="
          f"{training_rmse.tolist()}/{validation_rmse.tolist()}")
    print(f"标定完成（background_method={background_method}，JAX device={device}）："
          f"{model_output}")


def _contiguous_role_sequences(
    indices: np.ndarray,sequence_ids: list[str],frame_indices: np.ndarray,
    *,maximum_gap: int,
) -> list[np.ndarray]:
    """按视频和原始帧号排序，并在失败/缺失帧处切断 GRU 序列。"""
    groups: dict[str,list[int]]={}
    for index in np.asarray(indices,np.int64):
        sequence_id=sequence_ids[int(index)]
        if not sequence_id:
            raise ValueError("direct_fit_s 只接受带视频序列元数据的观测")
        groups.setdefault(sequence_id,[]).append(int(index))
    result=[]
    for values in groups.values():
        values.sort(key=lambda value:int(frame_indices[value]))
        current=[values[0]]
        for value in values[1:]:
            if int(frame_indices[value])-int(frame_indices[current[-1]]) \
                    <=maximum_gap:
                current.append(value)
            else:
                result.append(np.asarray(current,np.int64))
                current=[value]
        result.append(np.asarray(current,np.int64))
    return result


def _calibrate_direct_fit_s(
    *,raw: dict,cfg: dict,reconstruction: object,device: jax.Device,
    source_layout: LightSourceLayout,source_images: list[Path],
    uv_parts: list[np.ndarray],depth_parts: list[np.ndarray],
    xyz_all: np.ndarray,raw_xyz_all: np.ndarray,
    trusted_indices: np.ndarray,trusted_sequences: list[np.ndarray],
    sequence_sequences: list[np.ndarray],model_output: Path,
) -> None:
    """训练只沿 s 做隐式区间对齐的 direct_fit_s 序列背景场。"""
    direct=parse_direct_fit_s_config(raw)
    runtime=raw.get("runtime",{})
    sample_shape=(reconstruction.observation_rows,
                  reconstruction.observation_columns)
    canonical,canonical_valid=_collect_direct_canonical_fields(
        source_images,uv_parts,depth_parts,sample_shape=sample_shape,
        erode_pixels=direct.sample_erode_pixels,
        raster_triangle_chunk=int(runtime.get("gpu_raster_triangle_chunk",256)),
        raster_max_width=max(
            int(runtime.get("gpu_raster_max_triangle_width",24)),64),
        raster_max_height=max(
            int(runtime.get("gpu_raster_max_triangle_height",12)),32),
        device=device)
    result=fit_direct_fit_s_gpu(
        canonical,canonical_valid,surface_xyz=xyz_all,
        raw_observed_xyz=raw_xyz_all,trusted_indices=trusted_indices,
        trusted_sequences=trusted_sequences,
        sequence_sequences=sequence_sequences,device=device,config=direct,
        huber_delta=float(cfg.get("residual_huber_delta",.04)),
        seed=int(cfg.get("training_seed",0)))
    session=np.zeros((
        3,reconstruction.residual_coefficient_rows,
        reconstruction.residual_coefficient_columns),np.float32)
    model=LightFieldModel.direct_fit_s(
        session,base_texture=result.base_texture,
        coordinate_frequencies=result.coordinate_frequencies,
        geometry_feature_mean=result.geometry_feature_mean,
        geometry_feature_scale=result.geometry_feature_scale,
        geometry_pca_components=result.geometry_pca_components,
        geometry_pca_scale=result.geometry_pca_scale,
        local_geometry_feature_mean=result.local_geometry_feature_mean,
        local_geometry_feature_scale=result.local_geometry_feature_scale,
        geometry_encoder_weights=result.geometry_encoder_weights,
        geometry_encoder_biases=result.geometry_encoder_biases,
        gru_input_weight=result.gru_input_weight,
        gru_recurrent_weight=result.gru_recurrent_weight,
        gru_bias=result.gru_bias,warp_weight=result.warp_weight,
        warp_bias=result.warp_bias,
        color_trunk_weights=result.color_trunk_weights,
        color_trunk_biases=result.color_trunk_biases,
        channel_head_weights=result.channel_head_weights,
        channel_head_biases=result.channel_head_biases,
        geometry_descriptor_rows=direct.geometry_descriptor_rows,
        curve_convexity=reconstruction.curve_convexity,
        reconstruction_pipeline=CALIBRATION_RECONSTRUCTION_PIPELINE,
        source_layout=source_layout)
    model=_bind_material_model(model,reconstruction)
    model.save(model_output)
    print(
        "direct_fit_s 已保存：可信平直帧="
        f"{trusted_indices.size}，循环序列段={len(sequence_sequences)}，"
        "warp=softmax[left,visible,right] 且 psi(s)=left+visible*s，"
        f"model={model_output}")

def main() -> None:
    parser=argparse.ArgumentParser(description="JAX 离线标定无局部形变背景光场")
    parser.add_argument("--config",default=Path(__file__).with_name("config.yaml")); args=parser.parse_args()
    config_path=Path(args.config).expanduser(); all_config=load_config(config_path); raw=all_config["lightfield"]
    cfg=raw["calibration"]
    background_method=parse_background_method(raw)
    model_output=resolve_background_model_path(
        raw,method=background_method,base=config_path.parent)
    surface=all_config["get_surface"]
    reconstruction=parse_reconstruction_config(
        surface.get("reconstruction"),config_path=config_path,
        calibration_output=all_config.get("calibration",{}).get("output"))
    source_layout=(
        parse_light_source_layout(raw.get("light_source_layout"))
        if background_method=="physical_residual"
        else DEFAULT_LIGHT_SOURCE_LAYOUT)
    output_dir=Path(cfg.get("sample_output_dir","assets/lightfield_calibration")).expanduser()
    if not output_dir.is_absolute(): output_dir=config_path.parent/output_dir
    video_frame_step=(
        cfg.get("direct_fit_s_video_frame_step",cfg.get("video_frame_step",1))
        if background_method=="direct_fit_s" else cfg.get("video_frame_step",1))
    video_max_frames=cfg.get("video_max_frames_per_file")
    reuse_existing=bool(cfg.get("reuse_existing",True))
    video_frame_dir=Path(
        cfg.get("video_frame_output_dir",output_dir/"video_frames")
    ).expanduser()
    if not video_frame_dir.is_absolute():
        video_frame_dir=config_path.parent/video_frame_dir
    trusted_flat_frame_paths: list[Path]=[]
    sequence_frame_paths: list[Path]=[]
    if background_method=="direct_fit_s":
        trusted_videos=_resolve_optional_paths(
            cfg.get("trusted_flat_videos"),config_path.parent,
            "lightfield.calibration.trusted_flat_videos")
        sequence_videos=_resolve_optional_paths(
            cfg.get("sequence_videos"),config_path.parent,
            "lightfield.calibration.sequence_videos")
        if not trusted_videos or not sequence_videos:
            raise ValueError(
                "direct_fit_s 必须同时配置 trusted_flat_videos 和 "
                "sequence_videos")
        trusted_video_set={path.expanduser().resolve()
                           for path in trusted_videos}
        sequence_video_set={path.expanduser().resolve()
                            for path in sequence_videos}
        if trusted_video_set&sequence_video_set:
            raise ValueError(
                "direct_fit_s 的可信平直视频和循环视频不能包含同一文件")
        trusted_flat_frame_paths=extract_video_frames(
            trusted_videos,video_frame_dir,frame_step=video_frame_step,
            max_frames_per_file=video_max_frames,reuse_existing=reuse_existing,
            group_prefix="trusted_flat_video")
        sequence_frame_paths=extract_video_frames(
            sequence_videos,video_frame_dir,frame_step=video_frame_step,
            max_frames_per_file=video_max_frames,reuse_existing=reuse_existing,
            group_prefix="bend_sequence_video")
        image_paths=[]
        video_frame_paths=[*trusted_flat_frame_paths,*sequence_frame_paths]
    else:
        image_paths=_resolve_optional_paths(
            cfg.get("images"),config_path.parent,
            "lightfield.calibration.images")
        video_paths=_resolve_optional_paths(
            cfg.get("videos"),config_path.parent,
            "lightfield.calibration.videos")
        if not image_paths and not video_paths:
            raise ValueError(
                "lightfield.calibration.images 和 videos 至少需要配置一项")
        video_frame_paths=extract_video_frames(
            video_paths,video_frame_dir,frame_step=video_frame_step,
            max_frames_per_file=video_max_frames,reuse_existing=reuse_existing)
    observation_image_paths=[*image_paths,*video_frame_paths]
    map_dir=Path(cfg.get("generated_map_dir","assets/lightfield_calibration/maps")).expanduser()
    if not map_dir.is_absolute(): map_dir=config_path.parent/map_dir
    saturation_threshold=cfg.get("saturation_threshold",250)
    if not isinstance(saturation_threshold,int) or isinstance(saturation_threshold,bool) \
            or not 1<=saturation_threshold<=255:
        raise ValueError("lightfield.calibration.saturation_threshold 必须是 1..255 的整数")
    if background_method=="direct_fit_s":
        print(
            f"direct_fit_s 标定输入: 可信平直帧 {len(trusted_flat_frame_paths)}，"
            f"非可信循环帧 {len(sequence_frame_paths)}"
            f"（reuse_existing={reuse_existing}）")
    else:
        print(f"标定输入: 图片 {len(image_paths)} 张，视频帧 {len(video_frame_paths)} 张"
              f"（reuse_existing={reuse_existing}）")
    paths=reconstruct_all_observations(
        observation_image_paths,all_config,config_path,output_dir,map_dir,
        saturation_threshold,filter_original_saturation=False,
        reuse_existing=reuse_existing)
    device=choose_device(raw.get("device","gpu"))
    xyz_parts=[]; raw_xyz_parts=[]; uv_parts=[]; st_parts=[]; depth_parts=[]
    rgb_parts=[]; valid_parts=[]; source_images=[]; sequence_ids=[]
    sequence_frame_indices=[]
    for path in paths:
        with np.load(path) as data:
            xyz_parts.append(data["xyz"]); uv_parts.append(data["uv"])
            if "raw_observed_xyz" not in data:
                raise ValueError(
                    f"标定样本缺少归一化前 raw_observed_xyz，请重新生成: {path}")
            raw_xyz_parts.append(data["raw_observed_xyz"])
            if "st" not in data:
                raise ValueError(f"标定样本缺少 st，请重新生成: {path}")
            st_parts.append(data["st"])
            if "camera_depth" not in data:
                raise ValueError(f"标定样本缺少 camera_depth，请重新生成: {path}")
            depth_parts.append(data["camera_depth"])
            rgb_parts.append(data["rgb"]); valid_parts.append(data["valid_mask"])
            source_images.append(Path(str(data["source_image"])))
            sequence_ids.append(str(data["material_sequence_id"]))
            sequence_frame_indices.append(int(
                data["material_sequence_frame_index"]))
    # 实时 JAX 重建无效的帧不会进入缓存；依据成功缓存恢复真实输入顺序，
    # 避免训练/验证划分继续引用已跳过的原始图像。
    observation_image_paths=source_images.copy()
    independent_sources={path.expanduser().resolve() for path in image_paths}
    independent_image_count=sum(
        source.expanduser().resolve() in independent_sources
        for source in source_images)
    try:
        # 全量数据只保留在 CPU；Adam 和后续物理预测仅把当前 batch 送入 GPU。
        xyz=np.asarray(np.stack(xyz_parts),dtype=np.float32)
        raw_xyz=np.asarray(np.stack(raw_xyz_parts),dtype=np.float32)
        observed=np.asarray(np.stack(rgb_parts),dtype=np.float32)
        valid=np.asarray(np.stack(valid_parts),dtype=np.bool_)
        # 当前物理模型只消费 XYZ；仍在这里堆叠 ST，以同步校验所有样本的网格结构。
        np.stack(st_parts)
    except ValueError as error:
        raise ValueError("所有标定样本必须使用相同的曲面网格尺寸") from error
    del xyz_parts,raw_xyz_parts,rgb_parts,valid_parts,st_parts
    print(
        "标定观测已由实时 JAX 重建链生成："
        f"convexity={reconstruction.curve_convexity}；"
        "XYZ/UV/depth 无事后补投影")
    validation_fraction=(0. if background_method=="direct_fit_s" else
                         float(cfg.get("validation_fraction",0.)))
    validation_seed=cfg.get("validation_seed",0)
    training_indices,validation_indices=_split_calibration_indices(
        observation_image_paths,independent_image_count,validation_fraction,
        validation_seed)
    _write_split_manifest(
        output_dir/"calibration_split.yaml",source_images,training_indices,
        validation_indices,fraction=validation_fraction,seed=validation_seed)
    print(f"标定/验证划分：train={training_indices.size}，"
          f"validation={validation_indices.size}，"
          f"manifest={output_dir/'calibration_split.yaml'}")
    local_cfg=all_config.get("local_reconstruction",{})
    configured_residual_method=(local_cfg.get("residual_method","uniform_huber")
                                if isinstance(local_cfg,dict)
                                else "uniform_huber")
    if configured_residual_method not in {"uniform","uniform_huber"}:
        raise ValueError("local_reconstruction.residual_method 无效")
    if background_method=="direct_fit_3":
        _calibrate_direct_fit_3(
            raw=raw,cfg=cfg,reconstruction=reconstruction,device=device,
            background_method=background_method,
            source_layout=source_layout,source_images=source_images,
            uv_parts=uv_parts,depth_parts=depth_parts,xyz_all=xyz,
            training_indices=training_indices,
            validation_indices=validation_indices,
            model_output=model_output)
        return
    if background_method=="direct_fit_s":
        trusted_sources={path.expanduser().resolve()
                         for path in trusted_flat_frame_paths}
        sequence_sources={path.expanduser().resolve()
                          for path in sequence_frame_paths}
        trusted_indices=np.asarray([
            index for index,path in enumerate(source_images)
            if path.expanduser().resolve() in trusted_sources],np.int64)
        bend_indices=np.asarray([
            index for index,path in enumerate(source_images)
            if path.expanduser().resolve() in sequence_sources],np.int64)
        if trusted_indices.size+bend_indices.size!=len(source_images):
            raise RuntimeError("direct_fit_s 观测角色映射不完整")
        direct_s=parse_direct_fit_s_config(raw)
        frame_index_values=np.asarray(sequence_frame_indices,np.int64)
        trusted_sequences=_contiguous_role_sequences(
            trusted_indices,sequence_ids,frame_index_values,
            maximum_gap=direct_s.maximum_sequence_gap)
        bend_sequences=_contiguous_role_sequences(
            bend_indices,sequence_ids,frame_index_values,
            maximum_gap=direct_s.maximum_sequence_gap)
        _calibrate_direct_fit_s(
            raw=raw,cfg=cfg,reconstruction=reconstruction,device=device,
            source_layout=source_layout,source_images=source_images,
            uv_parts=uv_parts,depth_parts=depth_parts,xyz_all=xyz,
            raw_xyz_all=raw_xyz,trusted_indices=trusted_indices,
            trusted_sequences=trusted_sequences,
            sequence_sequences=bend_sequences,model_output=model_output)
        return
    # 物理路径也只用训练划分拟合；保留完整数组供最后的留出集诊断。
    xyz_all=xyz; observed_all=observed; valid_all=valid
    source_images_all=source_images; uv_parts_all=uv_parts; depth_parts_all=depth_parts
    xyz=xyz_all[training_indices]
    observed=observed_all[training_indices]
    valid=valid_all[training_indices]
    source_images=[source_images_all[index] for index in training_indices]
    uv_parts=[uv_parts_all[index] for index in training_indices]
    depth_parts=[depth_parts_all[index] for index in training_indices]
    sample_count=int(xyz.shape[0])
    source_count=len(light_source_specs(source_layout))
    bounds=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["delta_bounds_mm"],source_layout,(2,2),"delta_bounds_mm")),device)
    initial=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["delta_initial_mm"],source_layout,(2,),"delta_initial_mm")),device)
    lower,upper=bounds[...,0],bounds[...,1]
    if bool(np.any(np.asarray(upper<=lower))):
        raise ValueError("delta_bounds_mm 的每个上界必须严格大于下界")
    ratio=jnp.clip((initial-lower)/(upper-lower),.001,.999)
    scatter_ratio_bounds=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["scatter_ratio_bounds"],source_layout,(2,),
        "scatter_ratio_bounds")),device)
    scatter_ratio_initial=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["scatter_ratio_initial"],source_layout,(),
        "scatter_ratio_initial")),device)
    scatter_length_bounds=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["scatter_length_bounds_mm"],source_layout,(2,),
        "scatter_length_bounds_mm")),device)
    scatter_length_initial=jax.device_put(jnp.asarray(_expand_source_parameter(
        cfg["scatter_length_initial_mm"],source_layout,(),
        "scatter_length_initial_mm")),device)
    ratio_lower,ratio_upper=scatter_ratio_bounds[:,0],scatter_ratio_bounds[:,1]
    length_lower,length_upper=scatter_length_bounds[:,0],scatter_length_bounds[:,1]
    if bool(np.any(np.asarray(ratio_lower<0))) or bool(np.any(np.asarray(ratio_upper>1))) \
            or bool(np.any(np.asarray(ratio_upper<=ratio_lower))):
        raise ValueError("scatter_ratio_bounds 必须是 [0,1] 内的有效上下界")
    if bool(np.any(np.asarray(length_lower<=0))) or bool(np.any(np.asarray(length_upper<=length_lower))):
        raise ValueError("scatter_length_bounds_mm 必须是正数范围")
    scatter_ratio_unit=jnp.clip((scatter_ratio_initial-ratio_lower)/(ratio_upper-ratio_lower),.001,.999)
    scatter_length_unit=jnp.clip((scatter_length_initial-length_lower)/(length_upper-length_lower),.001,.999)
    mixing_initial=jax.device_put(jnp.asarray(cfg["mixing_matrix_initial"],jnp.float32),device)
    if mixing_initial.shape != (3,3) or bool(np.any(np.asarray(mixing_initial<0))) \
            or bool(np.any(np.asarray(mixing_initial.sum(axis=1)<=0))):
        raise ValueError("mixing_matrix_initial 必须是各行和为正的非负 3x3 矩阵")
    mixing_initial=mixing_initial/mixing_initial.sum(axis=1,keepdims=True)
    mixing_max_offdiagonal=float(cfg.get("mixing_max_offdiagonal_sum",.2))
    if not 0<mixing_max_offdiagonal<1:
        raise ValueError("mixing_max_offdiagonal_sum 必须在 (0,1) 内")
    initial_leakage=1-jnp.diag(mixing_initial)
    if bool(np.any(np.asarray(initial_leakage>=mixing_max_offdiagonal))):
        raise ValueError("mixing_matrix_initial 每行的非对角和必须小于 mixing_max_offdiagonal_sum")
    leakage_unit=jnp.clip(initial_leakage/mixing_max_offdiagonal,.001,.999)
    raw_mixing=jnp.log(jnp.maximum(mixing_initial,1e-6))
    raw_mixing=raw_mixing.at[jnp.arange(3),jnp.arange(3)].set(
        jnp.log(leakage_unit/(1-leakage_unit)))
    residual_rows=reconstruction.residual_coefficient_rows
    residual_columns=reconstruction.residual_coefficient_columns
    residual_m_count=int(cfg.get("residual_m_count",1))
    residual_curvature_feature_count=int(
        cfg.get("residual_curvature_feature_count",residual_m_count))
    residual_curvature_curve_coefficients=int(
        cfg.get("residual_curvature_curve_coefficients",12))
    residual_curvature_smooth=float(
        cfg.get("lambda_residual_curvature_smooth",.01))
    residual_curvature_regression=float(
        cfg.get("lambda_residual_curvature_regression",.01))
    if residual_rows<4 or residual_columns<4 or residual_m_count<1:
        raise ValueError(
            "residual_row/column_coefficients 必须至少为 4，"
            "residual_m_count 必须至少为 1")
    if residual_curvature_feature_count<residual_m_count \
            or residual_curvature_feature_count>=sample_count:
        raise ValueError(
            "residual_curvature_feature_count 必须不小于 residual_m_count，"
            "且小于标定观测帧数")
    if residual_curvature_curve_coefficients<4 \
            or residual_curvature_curve_coefficients>xyz.shape[1]:
        raise ValueError(
            "residual_curvature_curve_coefficients 必须位于 [4, 曲面行数]")
    if residual_curvature_smooth<0 or residual_curvature_regression<=0:
        raise ValueError("曲率平滑强度必须非负，曲率残差回归强度必须为正")
    zero_residual_b=jax.device_put(
        jnp.zeros((3,residual_rows,residual_columns),jnp.float32),device)
    zero_residual_ms=jax.device_put(
        jnp.zeros((residual_m_count,3,residual_rows,residual_columns),jnp.float32),device)
    # 暗场基准 b0 是固定物理常量 [0,0,0]，不属于离线优化变量。
    params=(jnp.log(ratio/(1-ratio)),
            jnp.ones((source_count,int(cfg.get("spline_coefficients",6)))),
            jnp.log(scatter_ratio_unit/(1-scatter_ratio_unit)),
            jnp.log(scatter_length_unit/(1-scatter_length_unit)),
            raw_mixing)
    fixed_bias=jax.device_put(jnp.zeros((3,),jnp.float32),device)
    nodes=int(raw["integration_nodes"]); epsilon=float(raw["distance_epsilon_mm"])
    cg_tolerance=float(raw.get("diffusion_cg_tolerance",1e-4))
    cg_iterations=int(raw.get("diffusion_cg_max_iterations",30))
    if cg_tolerance<=0 or cg_iterations<1:
        raise ValueError("diffusion_cg_tolerance 必须为正，max_iterations 必须大于等于 1")
    huber=float(cfg.get("huber_delta",.03)); lbeta=float(cfg.get("lambda_beta",1e-3)); ldelta=float(cfg.get("lambda_delta",1e-3))
    lratio=float(cfg.get("lambda_scatter_ratio",1e-3))
    llength=float(cfg.get("lambda_scatter_length",1e-5))
    lmixing=float(cfg.get("lambda_mixing_matrix",1e-3))
    def decode(p):
        rd,rb,rr,rl,rm=p; delta=lower+jax.nn.sigmoid(rd)*(upper-lower)
        scatter_ratio=ratio_lower+jax.nn.sigmoid(rr)*(ratio_upper-ratio_lower)
        scatter_length=length_lower+jax.nn.sigmoid(rl)*(length_upper-length_lower)
        mixing_matrix=bounded_mixing_matrix(rm,mixing_max_offdiagonal)
        return LightFieldModel(delta,jax.nn.softplus(rb),fixed_bias,scatter_ratio,
                               scatter_length,mixing_matrix,zero_residual_b,
                               zero_residual_ms,source_layout)
    raw_physical_batch_size=cfg.get("physical_batch_size",8)
    raw_steps=cfg.get("steps",800)
    physical_seed=cfg.get("physical_seed",0)
    if not isinstance(raw_physical_batch_size,int) \
            or isinstance(raw_physical_batch_size,bool) \
            or raw_physical_batch_size<1:
        raise ValueError("physical_batch_size 必须是正整数")
    if not isinstance(raw_steps,int) or isinstance(raw_steps,bool) or raw_steps<1:
        raise ValueError("steps 必须是正整数")
    if not isinstance(physical_seed,int) or isinstance(physical_seed,bool):
        raise ValueError("physical_seed 必须是整数")
    physical_batch_size=min(raw_physical_batch_size,sample_count)
    steps=raw_steps

    def regularization_loss(p):
        model=decode(p)
        beta_mean=jnp.mean(jnp.diff(model.beta,n=2,axis=1)**2)
        delta_mean=jnp.mean((model.delta-initial)**2)
        ratio_mean=jnp.mean((model.scatter_ratio-scatter_ratio_initial)**2)
        length_mean=jnp.mean((model.scatter_length-scatter_length_initial)**2)
        mixing_mean=jnp.mean((model.mixing_matrix-mixing_initial)**2)
        return (lbeta*beta_mean+ldelta*delta_mean+lratio*ratio_mean
                +llength*length_mean+lmixing*mixing_mean)

    def batch_data_loss_sums(p,batch_xyz,batch_observed,batch_valid):
        model=decode(p)
        predictions=physical_background_batch(
            batch_xyz,model,nodes,epsilon,65536,cg_tolerance,cg_iterations)+model.bias
        error=predictions-batch_observed
        absolute=jnp.abs(error)
        huber_values=jnp.where(
            absolute<=huber,.5*error**2,huber*(absolute-.5*huber))
        return (jnp.sum(huber_values*batch_valid[...,None]),
                3*jnp.sum(batch_valid))

    def loss_fn(p,batch_xyz,batch_observed,batch_valid):
        numerator,denominator=batch_data_loss_sums(
            p,batch_xyz,batch_observed,batch_valid)
        # 数据项和正则项均为 mean，batch 大小变化时 lambda 含义保持稳定。
        return numerator/jnp.maximum(denominator,1)+regularization_loss(p)

    def put_physical_batch(indices):
        return (
            jax.device_put(np.ascontiguousarray(xyz[indices]),device),
            jax.device_put(np.ascontiguousarray(observed[indices]),device),
            jax.device_put(
                np.ascontiguousarray(valid[indices],dtype=np.float32),device),
        )

    lr=float(cfg.get("learning_rate",.02)); b1,b2=.9,.999; moments=jax.tree.map(jnp.zeros_like,params); variances=jax.tree.map(jnp.zeros_like,params)
    @jax.jit
    def step(p,m,v,index,batch_xyz,batch_observed,batch_valid):
        loss,grads=jax.value_and_grad(loss_fn)(
            p,batch_xyz,batch_observed,batch_valid)
        m=jax.tree.map(lambda a,g:b1*a+(1-b1)*g,m,grads); v=jax.tree.map(lambda a,g:b2*a+(1-b2)*g*g,v,grads)
        corrected_m=jax.tree.map(lambda a:a/(1-b1**index),m); corrected_v=jax.tree.map(lambda a:a/(1-b2**index),v)
        p=jax.tree.map(lambda a,ma,va:a-lr*ma/(jnp.sqrt(va)+1e-8),p,corrected_m,corrected_v)
        return p,m,v,loss

    evaluate_batch_data=jax.jit(batch_data_loss_sums)
    evaluate_regularization=jax.jit(regularization_loss)
    monitor_sample_count=min(sample_count,max(physical_batch_size,64))
    monitor_indices=np.linspace(
        0,sample_count-1,monitor_sample_count,dtype=np.int64)

    def evaluate_monitor_loss(p):
        numerator=0.; denominator=0.
        for start in range(0,monitor_sample_count,physical_batch_size):
            indices=monitor_indices[start:start+physical_batch_size]
            current_numerator,current_denominator=evaluate_batch_data(
                p,*put_physical_batch(indices))
            current_numerator.block_until_ready()
            numerator+=float(current_numerator)
            denominator+=float(current_denominator)
        return (numerator/max(denominator,1.)
                +float(evaluate_regularization(p)))

    initial_monitor_loss=evaluate_monitor_loss(params)
    best_params=params
    best_monitor_loss=initial_monitor_loss
    batches_per_epoch=(sample_count+physical_batch_size-1)//physical_batch_size
    print(f"物理模型 Adam：CPU samples={sample_count}，GPU batch_size="
          f"{physical_batch_size}，updates={steps}，"
          f"batches/epoch={batches_per_epoch}，"
          f"monitor_samples={monitor_sample_count}")
    for index,(epoch,batch_indices) in enumerate(_iter_physical_batch_indices(
            sample_count,physical_batch_size,steps,physical_seed),1):
        batch=put_physical_batch(batch_indices)
        params,moments,variances,loss=step(
            params,moments,variances,jnp.asarray(index,jnp.float32),*batch)
        # 防止 Python 比 GPU 快速排队很多 batch，确保峰值显存只含当前 batch。
        loss.block_until_ready()
        if index==1 or index%50==0 or index==steps:
            monitor_loss=evaluate_monitor_loss(params)
            if np.isfinite(monitor_loss) and monitor_loss<best_monitor_loss:
                best_params=params
                best_monitor_loss=monitor_loss
            model=decode(params); print(f"step={index:04d} epoch={epoch:03d} "
                                        f"batch_loss={float(loss):.7f} "
                                        f"monitor={monitor_loss:.7f} "
                                        f"best={best_monitor_loss:.7f} "
                                        f"delta[x,normal]={np.asarray(model.delta).tolist()} "
                                        f"scatter_ratio={np.asarray(model.scatter_ratio).tolist()} "
                                        f"scatter_length_mm={np.asarray(model.scatter_length).tolist()}")
    physical_model=decode(best_params)

    # 残差模型学习的是运行时 gain/bias 已经处理后的剩余误差，避免重复解释全局亮度。
    irls_cfg=raw["irls"]
    irls_sigma=jax.device_put(jnp.asarray(irls_cfg["sigma_rgb"],jnp.float32),device)
    def adjust_one(observation,prediction):
        gain,bias,_=irls_gain_bias(
            observation,prediction,physical_model.bias,irls_sigma,
            int(irls_cfg["iterations"]),float(irls_cfg["lambda_gain"]),
            float(irls_cfg["lambda_bias"]),float(irls_cfg["max_gain_deviation"]),
            float(irls_cfg["max_bias_deviation"]))
        return jnp.clip(gain*prediction+bias,0,1),gain,bias

    @jax.jit
    def predict_and_adjust_batch(batch_xyz,batch_observed,batch_valid):
        predictions=physical_background_batch(
            batch_xyz,physical_model,nodes,epsilon,65536,
            cg_tolerance,cg_iterations)
        adjusted,gains,biases=jax.vmap(adjust_one)(batch_observed,predictions)
        error=predictions+physical_model.bias-batch_observed
        absolute=jnp.abs(error)
        huber_values=jnp.where(
            absolute<=huber,.5*error**2,huber*(absolute-.5*huber))
        return (adjusted,gains,biases,
                jnp.sum(huber_values*batch_valid[...,None]),
                3*jnp.sum(batch_valid))

    adjusted_parts=[]; gain_parts=[]; bias_parts=[]
    physical_data_numerator=0.; physical_data_denominator=0.
    print("物理模型训练后预测：逐 batch 计算并立即回传 CPU")
    for start in range(0,sample_count,physical_batch_size):
        indices=np.arange(
            start,min(start+physical_batch_size,sample_count),dtype=np.int64)
        batch=put_physical_batch(indices)
        adjusted_batch,gain_batch,bias_batch,numerator,denominator=(
            predict_and_adjust_batch(*batch))
        adjusted_batch.block_until_ready()
        adjusted_parts.append(np.asarray(adjusted_batch))
        gain_parts.append(np.asarray(gain_batch))
        bias_parts.append(np.asarray(bias_batch))
        physical_data_numerator+=float(numerator)
        physical_data_denominator+=float(denominator)
        completed=min(start+physical_batch_size,sample_count)
        if completed==sample_count or completed%max(physical_batch_size*25,1)==0:
            print(f"物理预测 {completed}/{sample_count}")
    adjusted_predictions=np.concatenate(adjusted_parts,axis=0)
    calibration_gains=np.concatenate(gain_parts,axis=0)
    calibration_biases=np.concatenate(bias_parts,axis=0)
    physical_full_loss=(physical_data_numerator/max(physical_data_denominator,1.)
                        +float(regularization_loss(best_params)))
    del adjusted_parts,gain_parts,bias_parts,observed,valid

    residual_sample_rows=reconstruction.observation_rows
    residual_sample_columns=reconstruction.observation_columns
    residual_erode_pixels=int(cfg.get("residual_erode_pixels",6))
    residual_huber=float(cfg.get("residual_huber_delta",.04))
    residual_smooth=float(cfg.get("lambda_residual_smooth",.01))
    residual_magnitude=float(cfg.get("lambda_residual_magnitude",1e-4))
    residual_outer_weight=float(cfg.get("residual_outer_weight",.2))
    residual_outer_fraction=float(cfg.get("residual_outer_fraction",.05))
    residual_b_max_field=float(
        cfg["residual_b_max_field_deviation"])
    residual_m_max_field=float(
        cfg["residual_m_max_field_deviation"])
    residual_channel_huber_ratio_min=float(
        raw.get("runtime",{}).get("residual_channel_huber_ratio_min",.5))
    residual_channel_huber_ratio_max=float(
        raw.get("runtime",{}).get("residual_channel_huber_ratio_max",2.))
    if residual_sample_rows<residual_rows or residual_sample_columns<residual_columns:
        raise ValueError("observation_grid 不能小于 residual_coefficient_grid")
    if residual_erode_pixels<0:
        raise ValueError("residual_erode_pixels 必须大于等于 0")
    runtime_cfg=raw.get("runtime",{})
    raster_triangle_chunk=int(runtime_cfg.get("gpu_raster_triangle_chunk",256))
    # 离线标定网格可能比实时更密；这里给足包围盒容量，避免静默失败。
    raster_max_width=max(
        int(runtime_cfg.get("gpu_raster_max_triangle_width",24)),64)
    raster_max_height=max(
        int(runtime_cfg.get("gpu_raster_max_triangle_height",12)),32)
    if raster_triangle_chunk<1 or raster_max_width<1 or raster_max_height<1:
        raise ValueError("GPU 光栅化容量参数必须为正整数")
    prediction_values=np.asarray(adjusted_predictions)
    first_frame=cv2.imread(str(source_images[0]),cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"无法重新读取残差标定图像: {source_images[0]}")
    image_height,image_width=first_frame.shape[:2]

    @jax.jit
    def residual_sample_gpu(frame_bgr,uv_value,depth_value,prediction):
        rendered,valid_mask,overflow=rasterize_attributes_jax(
            uv_value,depth_value,prediction,(image_height,image_width),
            triangle_chunk=raster_triangle_chunk,
            max_triangle_width=raster_max_width,
            max_triangle_height=raster_max_height)
        raw_residual=bgr_to_linear_rgb_jax(frame_bgr)-rendered
        residual_sample,valid_sample=build_canonical_residual_sample_jax(
            raw_residual,frame_bgr,valid_mask,uv_value,
            (residual_sample_rows,residual_sample_columns),
            saturation_threshold=saturation_threshold,
            erode_pixels=residual_erode_pixels)
        return residual_sample,valid_sample,overflow

    print(f"GPU 残差采样：{len(source_images)} 帧，"
          f"canonical={residual_sample_rows}x{residual_sample_columns}")
    canonical_residuals=None; canonical_valid=None
    for index,(image_path,uv_value,depth_value,prediction) in enumerate(zip(
            source_images,uv_parts,depth_parts,prediction_values,strict=True),1):
        if index==1:
            frame=first_frame
        else:
            frame=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"无法重新读取残差标定图像: {image_path}")
        if frame.shape[:2]!=(image_height,image_width):
            raise ValueError(
                f"残差标定图像尺寸不一致: {image_path} "
                f"{frame.shape[:2]} != {(image_height,image_width)}")
        residual_sample,valid_sample,overflow=residual_sample_gpu(
            jax.device_put(jnp.asarray(frame,jnp.uint8),device),
            jax.device_put(jnp.asarray(uv_value,jnp.float32),device),
            jax.device_put(jnp.asarray(depth_value,jnp.float32),device),
            jax.device_put(jnp.asarray(prediction,jnp.float32),device))
        if bool(np.asarray(overflow)):
            raise RuntimeError(
                f"GPU 光栅化包围盒超出容量，请增大 runtime.gpu_raster_max_triangle_*："
                f" {image_path}")
        residual_np=np.asarray(residual_sample)
        valid_np=np.asarray(valid_sample)
        if canonical_residuals is None:
            canonical_residuals=np.empty(
                (len(source_images),*residual_np.shape),np.float32)
            canonical_valid=np.empty(
                (len(source_images),*valid_np.shape),np.bool_)
        canonical_residuals[index-1]=residual_np
        canonical_valid[index-1]=valid_np
        if index==1 or index%50==0 or index==len(source_images):
            print(f"残差样本 {index}/{len(source_images)}: "
                  f"valid={int(valid_np.sum())}/{valid_np.size}")
    if canonical_residuals is None or canonical_valid is None:
        raise RuntimeError("没有生成任何规范曲面残差样本")
    del adjusted_predictions,prediction_values,uv_parts,depth_parts,first_frame
    print(f"离线曲率引导 M：{sample_count} 个训练帧逐帧参与；"
          f"曲率特征数={residual_curvature_feature_count}")
    residual_b,residual_ms,training_scores=fit_residual_correction_model_gpu(
        canonical_residuals,canonical_valid,surface_xyz=xyz,
        device=device,
        row_coefficients=residual_rows,
        column_coefficients=residual_columns,m_count=residual_m_count,
        huber_delta=residual_huber,smooth_lambda=residual_smooth,
        magnitude_lambda=residual_magnitude,outer_weight=residual_outer_weight,
        outer_fraction=residual_outer_fraction,
        b_max_deviation=residual_b_max_field,
        m_max_deviation=residual_m_max_field,
        channel_huber_ratio_min=residual_channel_huber_ratio_min,
        channel_huber_ratio_max=residual_channel_huber_ratio_max,
        curvature_feature_count=residual_curvature_feature_count,
        curvature_curve_coefficients=residual_curvature_curve_coefficients,
        curvature_smooth_lambda=residual_curvature_smooth,
        curvature_regression_lambda=residual_curvature_regression,
        sample_batch_size=cfg.get("residual_gpu_sample_batch_size",8),
        pixel_chunk_size=cfg.get("residual_gpu_pixel_chunk_size",128),
        scale_sample_pixels=cfg.get("residual_gpu_scale_sample_pixels",512),
    )
    final_model=LightFieldModel(
        physical_model.delta,physical_model.beta,physical_model.bias,
        physical_model.scatter_ratio,physical_model.scatter_length,
        physical_model.mixing_matrix,jnp.asarray(residual_b),
        jnp.asarray(residual_ms),physical_model.source_layout)
    output=model_output
    final_model=_bind_material_model(final_model,reconstruction)
    final_model.save(output)
    b_field=evaluate_rgb_bspline(residual_b,(residual_sample_rows,residual_sample_columns))
    m_fields=np.stack([evaluate_rgb_bspline(item,(residual_sample_rows,residual_sample_columns))
                       for item in residual_ms])
    # RMSE 也按帧 batch 汇总，避免标定结束前额外生成两份 NxHxWx3 大数组。
    diagnostic_batch_size=min(
        int(cfg.get("residual_gpu_sample_batch_size",8)),sample_count)
    raw_squared_sum=np.zeros(3,np.float64)
    clean_squared_sum=np.zeros(3,np.float64)
    diagnostic_valid_count=0
    for start in range(0,sample_count,diagnostic_batch_size):
        stop=min(start+diagnostic_batch_size,sample_count)
        current_residual=canonical_residuals[start:stop]
        current_valid=canonical_valid[start:stop]
        fitted_correction=b_field[None]+np.einsum(
            "nck,khwc->nhwc",training_scores[start:stop],m_fields)
        cleaned=current_residual-fitted_correction
        raw_squared_sum+=np.sum(
            np.where(current_valid[...,None],current_residual**2,0),axis=(0,1,2))
        clean_squared_sum+=np.sum(
            np.where(current_valid[...,None],cleaned**2,0),axis=(0,1,2))
        diagnostic_valid_count+=int(current_valid.sum())
    denominator=max(diagnostic_valid_count,1)
    raw_rmse=np.sqrt(raw_squared_sum/denominator)
    clean_rmse=np.sqrt(clean_squared_sum/denominator)
    validation_raw_rmse=np.full(3,np.nan,np.float64)
    validation_clean_rmse=np.full(3,np.nan,np.float64)
    if validation_indices.size:
        validation_predictions=[]
        for start in range(0,validation_indices.size,physical_batch_size):
            indices=validation_indices[start:start+physical_batch_size]
            adjusted,_,_,_,_=predict_and_adjust_batch(
                jax.device_put(np.ascontiguousarray(xyz_all[indices]),device),
                jax.device_put(np.ascontiguousarray(observed_all[indices]),device),
                jax.device_put(np.ascontiguousarray(
                    valid_all[indices],dtype=np.float32),device))
            adjusted.block_until_ready()
            validation_predictions.append(np.asarray(adjusted))
        validation_predictions_np=np.concatenate(validation_predictions,axis=0)
        validation_residuals=[]; validation_masks=[]
        for local_index,global_index in enumerate(validation_indices):
            frame=cv2.imread(
                str(source_images_all[global_index]),cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(
                    f"无法读取验证图像: {source_images_all[global_index]}")
            residual_sample,valid_sample,overflow=residual_sample_gpu(
                jax.device_put(jnp.asarray(frame,jnp.uint8),device),
                jax.device_put(jnp.asarray(
                    uv_parts_all[global_index],jnp.float32),device),
                jax.device_put(jnp.asarray(
                    depth_parts_all[global_index],jnp.float32),device),
                jax.device_put(jnp.asarray(
                    validation_predictions_np[local_index],jnp.float32),device))
            residual_sample,valid_sample,overflow=jax.device_get(
                (residual_sample,valid_sample,overflow))
            if bool(overflow):
                raise RuntimeError("物理路径验证样本超过 GPU 光栅化容量")
            validation_residuals.append(np.asarray(residual_sample,np.float32))
            validation_masks.append(np.asarray(valid_sample,np.bool_))
        validation_residuals_np=np.stack(validation_residuals)
        validation_masks_np=np.stack(validation_masks)
        b_gpu=jax.device_put(jnp.asarray(b_field,jnp.float32),device)
        m_gpu=jax.device_put(jnp.asarray(m_fields,jnp.float32),device)
        all_fields_gpu=jnp.concatenate([b_gpu[None],m_gpu],axis=0)
        huber_delta=float(
            raw.get("runtime",{}).get("residual_score_huber_delta",.04))
        huber_iterations=int(
            raw.get("runtime",{}).get("residual_score_huber_iterations",5))

        @jax.jit
        def clean_validation_one(residual,mask):
            if configured_residual_method=="uniform":
                scores=fit_uniform_residual_correction_scores_jax(
                    residual,b_gpu,m_gpu,mask)
            else:
                scores=fit_uniform_huber_residual_correction_scores_jax(
                    residual,b_gpu,m_gpu,mask,huber_delta,huber_iterations)
            correction=jnp.einsum("ck,khwc->hwc",scores,all_fields_gpu)
            return residual-correction

        validation_clean=[]
        for residual,mask in zip(
                validation_residuals_np,validation_masks_np,strict=True):
            validation_clean.append(np.asarray(clean_validation_one(
                jax.device_put(residual,device),jax.device_put(mask,device))))
        validation_raw_rmse=_valid_rmse(
            validation_residuals_np,validation_masks_np)
        validation_clean_rmse=_valid_rmse(
            np.stack(validation_clean),validation_masks_np)
    print(f"mixing_matrix={np.asarray(final_model.mixing_matrix).tolist()}")
    print(f"calibration gain RGB mean={np.asarray(calibration_gains).mean(axis=0).tolist()} "
          f"bias mean={np.asarray(calibration_biases).mean(axis=0).tolist()}")
    print(f"离线残差标定已保存 B 和前 {residual_m_count} 个未正交化 raw M 模式；"
          "实时启动拟合 Bsession 后执行会话级等价正交化。")
    print(f"canonical residual RMSE raw={raw_rmse.tolist()} clean={clean_rmse.tolist()} "
          f"physical_full={physical_full_loss:.7f}")
    print("validation canonical residual RMSE "
          f"raw={validation_raw_rmse.tolist()} "
          f"clean={validation_clean_rmse.tolist()}")
    print(f"标定完成（JAX device={device}，samples={len(paths)}）: {output}")

if __name__=="__main__": main()
