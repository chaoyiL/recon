"""使用已知半径标定球建立线性 RGB 色差到局部坡度的查找表。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.spatial import cKDTree

from calibrate_lightfield import (
    _resolve_paths,
    reconstruct_all_observations,
)
from manual_norm_regions import (
    ManualEllipse,
    annotation_key,
    default_manual_ellipse,
    edit_manual_ellipse,
    ellipse_mask,
    eroded_ellipse_mask,
    load_manual_ellipses,
    save_manual_ellipses,
)
from recon import NormalCalibration
from utils.config import (GEOMETRY_BACKGROUND_METHODS,file_sha256,
                          parse_background_method,
                          parse_direct_fit_config,
                          parse_geometry_cache_config,
                          parse_reconstruction_config,
                          resolve_background_model_path,resolve_method_path)
from utils.lightfield import (
    LightFieldModel,
    bgr_to_linear_rgb_jax,
    build_canonical_residual_sample_jax,
    choose_device,
    geometry_background_field_jax,
    erode_mask_jax,
    fit_startup_residual_bsession_model,
    fit_startup_direct_bsession_model,
    fit_uniform_huber_residual_correction_scores_jax,
    fit_uniform_residual_correction_scores_jax,
    irls_gain_bias,
    physical_background,
    signed_residual_bgr,
    rasterize_attributes_jax,
    rgb_bspline_field,
    parse_light_source_layout,
    sample_image_mask_to_canonical_jax,
    sample_linear_rgb_jax,
    sample_residual_correction_jax,
)
from utils.jax_reconstruction import SURFACE_RECONSTRUCTION_PIPELINE_VERSION


@dataclass(frozen=True)
class BSessionContactExclusionConfig:
    dilation_pixels: int = 12
    minimum_background_fraction: float = .2

    @classmethod
    def from_mapping(
        cls,raw: dict | None,
    ) -> "BSessionContactExclusionConfig":
        values={} if raw is None else dict(raw)
        known={field for field in cls.__dataclass_fields__}
        unknown=set(values)-known
        if unknown:
            raise ValueError(
                "normal_calibration.bsession_contact_exclusion 包含未知字段: "+
                ", ".join(sorted(unknown)))
        result=cls(**values)
        if result.dilation_pixels<0 \
                or not 0<result.minimum_background_fraction<=1:
            raise ValueError(
                "normal_calibration.bsession_contact_exclusion 参数无效")
        return result


@dataclass(frozen=True)
class ManualContactSelection:
    accepted: bool
    reason: str
    ellipse: ManualEllipse
    surface_mask: np.ndarray
    contact_mask: np.ndarray
    sample_mask: np.ndarray
    plane_origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    center_plane: np.ndarray
    center_xyz: np.ndarray
    radius_mm: float
    indentation_mm: float
    plane_rms_mm: float


def _surface_plane_frame(xyz: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    points=np.asarray(xyz,np.float64)
    if points.ndim!=3 or points.shape[-1]!=3 or min(points.shape[:2])<2 \
            or not np.isfinite(points).all():
        raise ValueError("标定曲面 XYZ 必须是有限的规则点阵")
    origin=np.mean(points,axis=(0,1))
    horizontal=np.mean(points[:,-1]-points[:,0],axis=0)
    vertical=np.mean(points[-1]-points[0],axis=0)
    e1=horizontal/np.maximum(np.linalg.norm(horizontal),1e-12)
    vertical=vertical-np.dot(vertical,e1)*e1
    e2=vertical/np.maximum(np.linalg.norm(vertical),1e-12)
    normal=np.cross(e1,e2)
    normal/=np.maximum(np.linalg.norm(normal),1e-12)
    distance=np.einsum("...c,c->...",points-origin,normal)
    plane_rms=float(np.sqrt(np.mean(distance**2)))
    return origin,e1,e2,normal,plane_rms


def manual_contact_selection(
    ellipse: ManualEllipse,valid_mask: np.ndarray,surface_valid_mask: np.ndarray,
    xyz_image: np.ndarray,surface_xyz: np.ndarray,sphere_radius_mm: float,
    edge_margin_pixels: int,
) -> ManualContactSelection:
    """把人工图像椭圆确定性映射到接触平面；不执行阈值或质量筛选。"""
    valid=np.asarray(valid_mask,np.bool_)
    surface_valid=np.asarray(surface_valid_mask,np.bool_)
    xyz_pixels=np.asarray(xyz_image,np.float64)
    surface=valid&surface_valid&np.all(np.isfinite(xyz_pixels),axis=-1)
    contact=ellipse_mask(valid.shape,ellipse)&surface
    sample=eroded_ellipse_mask(valid.shape,ellipse,edge_margin_pixels)&surface
    origin,e1,e2,_,plane_rms=_surface_plane_frame(surface_xyz)
    rows,columns=np.nonzero(surface)
    if not ellipse.use_for_lut:
        center_plane=np.full(2,np.nan); center_xyz=np.full(3,np.nan)
        return ManualContactSelection(
            False,"manual_skip",ellipse,surface,contact,np.zeros_like(sample),
            origin,e1,e2,center_plane,center_xyz,np.nan,np.nan,plane_rms)
    if not rows.size or not np.any(contact) or not np.any(sample):
        raise ValueError("人工椭圆与当前有效曲面没有足够交集，请重新编辑该图片")
    nearest=int(np.argmin(
        (columns-ellipse.center_x)**2+(rows-ellipse.center_y)**2))
    center_xyz=xyz_pixels[rows[nearest],columns[nearest]]
    center_plane=np.asarray([
        (center_xyz-origin)@e1,(center_xyz-origin)@e2],np.float64)
    contours,_=cv2.findContours(
        contact.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    boundary=np.concatenate([item.reshape(-1,2) for item in contours],axis=0)
    boundary_xyz=xyz_pixels[boundary[:,1],boundary[:,0]]
    finite=np.all(np.isfinite(boundary_xyz),axis=-1)
    boundary_xyz=boundary_xyz[finite]
    boundary_plane=np.stack([
        (boundary_xyz-origin)@e1,(boundary_xyz-origin)@e2],axis=-1)
    radius_mm=float(np.median(np.linalg.norm(
        boundary_plane-center_plane,axis=-1)))
    indentation=(sphere_radius_mm-np.sqrt(max(
        sphere_radius_mm**2-min(radius_mm,sphere_radius_mm)**2,0)))
    return ManualContactSelection(
        True,"manual",ellipse,surface,contact,sample,origin,e1,e2,
        center_plane,center_xyz,radius_mm,float(indentation),plane_rms)


def dilate_manual_contact(mask: np.ndarray,dilation_pixels: int) -> np.ndarray:
    if dilation_pixels<0:
        raise ValueError("Bsession 人工接触区膨胀像素数不能为负数")
    result=np.asarray(mask,np.bool_)
    if dilation_pixels and np.any(result):
        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(2*dilation_pixels+1,2*dilation_pixels+1))
        result=cv2.dilate(result.astype(np.uint8),kernel)>0
    return result


def save_manual_verification(
    path: str | Path,frame_bgr: np.ndarray,residual_linear_rgb: np.ndarray,
    selection: ManualContactSelection,
) -> Path:
    frame=np.asarray(frame_bgr,np.uint8)
    residual=signed_residual_bgr(
        residual_linear_rgb,selection.surface_mask,gain=2.)
    masked=np.zeros_like(frame); masked[selection.surface_mask]=frame[selection.surface_mask]
    overlay=masked.copy(); overlay[selection.sample_mask]=(0,180,0)
    blended=cv2.addWeighted(masked,.58,overlay,.42,0)
    ellipse=selection.ellipse
    center=(int(np.rint(ellipse.center_x)),int(np.rint(ellipse.center_y)))
    axes=(max(1,int(np.rint(ellipse.semi_axis_x))),
          max(1,int(np.rint(ellipse.semi_axis_y))))
    color=(0,255,0) if selection.accepted else (0,0,255)
    cv2.ellipse(blended,center,axes,ellipse.angle_degrees,0,360,
                color,2,cv2.LINE_AA)
    cv2.drawMarker(
        blended,center,(255,0,255),cv2.MARKER_CROSS,18,2,cv2.LINE_AA)
    lines=[
        "MANUAL SELECTED" if selection.accepted else "MANUAL SKIPPED",
        f"center=({ellipse.center_x:.1f},{ellipse.center_y:.1f}) "
        f"semi-axes=({ellipse.semi_axis_x:.1f},{ellipse.semi_axis_y:.1f}) "
        f"angle={ellipse.angle_degrees:.1f} deg",
        f"physical median radius={selection.radius_mm:.3f} mm "
        f"plane RMS={selection.plane_rms_mm:.3f} mm",
        "green=manual ellipse/sample region magenta=manual center",
    ]
    for index,line in enumerate(lines):
        y=26+index*24
        cv2.putText(blended,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(blended,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.52,
                    color if index==0 else (255,255,255),1,cv2.LINE_AA)
    canvas=np.concatenate([blended,residual],axis=1)
    output=Path(path).expanduser(); output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),canvas):
        raise RuntimeError(f"无法写入人工标定检验图: {output}")
    return output


def sphere_slope_samples(
    residual_linear_rgb: np.ndarray,
    xyz_image: np.ndarray,
    selection: ManualContactSelection,
    sphere_radius_mm: float,
) -> tuple[np.ndarray,np.ndarray]:
    if not selection.accepted:
        raise ValueError("人工跳过的图片不能生成标定样本")
    xyz=np.asarray(xyz_image,np.float64)[selection.sample_mask]
    colors=np.asarray(
        residual_linear_rgb,np.float32)[selection.sample_mask]
    relative=xyz-selection.center_xyz
    xi=relative@selection.e1; eta=relative@selection.e2
    radius2=xi**2+eta**2
    inside=radius2<sphere_radius_mm**2
    zeta=np.sqrt(np.maximum(sphere_radius_mm**2-radius2[inside],1e-12))
    slopes=np.stack([-xi[inside]/zeta,-eta[inside]/zeta],axis=-1)
    finite=np.all(np.isfinite(colors[inside]),axis=-1)&np.all(
        np.isfinite(slopes),axis=-1)
    return colors[inside][finite].astype(np.float32),slopes[finite].astype(np.float32)


def save_pq_rgb_mapping(
    path: str | Path,
    colors: np.ndarray,
    slopes: np.ndarray,
    *,
    resolution: int = 640,
    display_percentile: float = 99.5,
) -> Path:
    """以 p、q 为横纵轴，按坡度网格叠加显示平均线性 RGB 和样本密度。"""
    color=np.asarray(colors,np.float64)
    slope=np.asarray(slopes,np.float64)
    if color.ndim!=2 or color.shape[1]!=3 or slope.shape!=(color.shape[0],2):
        raise ValueError("p-q RGB 可视化的颜色/坡度尺寸无效")
    if not isinstance(resolution,int) or isinstance(resolution,bool) or resolution<128:
        raise ValueError("pq_visualization_resolution 必须是至少 128 的整数")
    if not np.isfinite(display_percentile) or not 90<=display_percentile<=100:
        raise ValueError("pq_visualization_percentile 必须位于 [90, 100]")
    finite=(np.all(np.isfinite(color),axis=1)&
            np.all(np.isfinite(slope),axis=1))
    color=color[finite]; slope=slope[finite]
    if color.shape[0]==0:
        raise ValueError("没有有限的 p-q RGB 样本可供可视化")
    # p、q 使用相同的稳健对称尺度。离群点仍留在 LUT 训练样本中，只从这张
    # 诊断图隐藏，避免少量球面边缘发散值把主体压缩到中心。
    magnitude=np.max(np.abs(slope),axis=1)
    raw_max=float(np.max(magnitude))
    limit=max(float(np.percentile(magnitude,display_percentile))*1.05,1e-3)
    visible=magnitude<=limit
    hidden_count=int(np.count_nonzero(~visible))
    color=color[visible]; slope=slope[visible]
    x=np.rint((slope[:,0]+limit)/(2*limit)*(resolution-1)).astype(np.int32)
    y=np.rint((limit-slope[:,1])/(2*limit)*(resolution-1)).astype(np.int32)
    x=np.clip(x,0,resolution-1); y=np.clip(y,0,resolution-1)
    color_sum=np.zeros((resolution,resolution,3),np.float32)
    count=np.zeros((resolution,resolution),np.float32)
    np.add.at(color_sum,(y,x),color.astype(np.float32))
    np.add.at(count,(y,x),1)
    # 小范围高斯 splat 只用于显示连续性；颜色和密度始终分别归一，不改变均值。
    sigma=1.2
    blurred_sum=cv2.GaussianBlur(
        color_sum,(0,0),sigmaX=sigma,sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT)
    blurred_count=cv2.GaussianBlur(
        count,(0,0),sigmaX=sigma,sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT)
    occupied=blurred_count>1e-6
    mean_rgb=np.zeros_like(blurred_sum)
    mean_rgb[occupied]=blurred_sum[occupied]/blurred_count[occupied,None]
    # 有符号 RGB 不能直接套 sRGB 曲线；用统一的稳健对称尺度映射到中性灰，
    # 从而同时显示变亮和变暗，并保留三通道之间的相对幅值。
    display_scale=max(float(np.percentile(np.abs(color),99)),1e-6)
    display_rgb=np.clip(.5+.5*mean_rgb/display_scale,0,1)
    reference=float(np.percentile(blurred_count[occupied],95))
    alpha=np.clip(np.sqrt(blurred_count/max(reference,1e-6)),0,1)
    background=np.full_like(display_rgb,.025)
    display_rgb=np.where(
        occupied[...,None],
        background+(display_rgb-background)*alpha[...,None],background)
    plot=np.rint(display_rgb[...,::-1]*255).astype(np.uint8)

    top,left,right,bottom=70,82,30,66
    canvas=np.full(
        (top+resolution+bottom,left+resolution+right,3),18,np.uint8)
    canvas[top:top+resolution,left:left+resolution]=plot
    axis_color=(150,150,150); text_color=(225,225,225)
    zero=int(round((resolution-1)/2))
    cv2.line(canvas,(left,top+zero),(left+resolution-1,top+zero),
             axis_color,1,cv2.LINE_AA)
    cv2.line(canvas,(left+zero,top),(left+zero,top+resolution-1),
             axis_color,1,cv2.LINE_AA)
    ticks=np.linspace(-limit,limit,5)
    for value in ticks:
        px=left+int(round((value+limit)/(2*limit)*(resolution-1)))
        py=top+int(round((limit-value)/(2*limit)*(resolution-1)))
        cv2.line(canvas,(px,top+resolution),(px,top+resolution+6),
                 text_color,1,cv2.LINE_AA)
        cv2.line(canvas,(left-6,py),(left,py),text_color,1,cv2.LINE_AA)
        label=f"{value:.2f}"
        width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.42,1)[0][0]
        cv2.putText(canvas,label,(px-width//2,top+resolution+24),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,text_color,1,cv2.LINE_AA)
        cv2.putText(canvas,label,(8,py+5),cv2.FONT_HERSHEY_SIMPLEX,.42,
                    text_color,1,cv2.LINE_AA)
    title="p-q -> mean signed dRGB (brightness = sample density)"
    cv2.putText(canvas,title,(left,28),cv2.FONT_HERSHEY_SIMPLEX,.58,
                text_color,1,cv2.LINE_AA)
    summary=(f"shown={color.shape[0]} hidden={hidden_count} "
             f"range=+/-{limit:.3f} raw-max={raw_max:.3f} "
             f"dRGB=+/-{display_scale:.3f}")
    cv2.putText(canvas,summary,(left,51),cv2.FONT_HERSHEY_SIMPLEX,.42,
                text_color,1,cv2.LINE_AA)
    cv2.putText(canvas,"p",(left+resolution//2,top+resolution+54),
                cv2.FONT_HERSHEY_SIMPLEX,.65,text_color,1,cv2.LINE_AA)
    cv2.putText(canvas,"q (up)",(8,top-16),cv2.FONT_HERSHEY_SIMPLEX,.55,
                text_color,1,cv2.LINE_AA)
    output=Path(path).expanduser(); output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),canvas):
        raise RuntimeError(f"无法写入 p-q RGB 映射图: {output}")
    return output


def build_normal_lut(
    colors: np.ndarray,
    slopes: np.ndarray,
    *,
    size: int = 64,
    minimum_samples_per_node: int = 5,
    maximum_rms_angle_degrees: float = 5.,
) -> NormalCalibration:
    """按文档的角度 MAD、有效节点判据和 8 邻域 Shepard 插值构建 LUT。"""
    color=np.asarray(colors,np.float64); slope=np.asarray(slopes,np.float64)
    if color.ndim!=2 or color.shape[1]!=3 or slope.shape!=(color.shape[0],2) \
            or color.shape[0]<minimum_samples_per_node:
        raise ValueError("颜色/坡度标定样本数量或尺寸无效")
    if size<2 or minimum_samples_per_node<1 or maximum_rms_angle_degrees<=0:
        raise ValueError("LUT 构建参数无效")
    finite=np.all(np.isfinite(color),axis=1)&np.all(np.isfinite(slope),axis=1)
    color=color[finite]; slope=slope[finite]
    color_min=np.minimum(np.min(color,axis=0),0)
    color_max=np.maximum(np.max(color,axis=0),0)
    tiny=np.maximum(np.maximum(np.abs(color_min),np.abs(color_max))*1e-6,1e-6)
    same=color_max-color_min<tiny
    color_min=np.where(same,color_min-tiny,color_min)
    color_max=np.where(same,color_max+tiny,color_max)
    normalized=np.clip((color-color_min)/(color_max-color_min),0,1)
    indices=np.rint((size-1)*normalized).astype(np.int32)
    flat=np.ravel_multi_index(indices.T,(size,size,size))
    order=np.argsort(flat); flat=flat[order]; slope=slope[order]
    unique,starts,counts=np.unique(flat,return_index=True,return_counts=True)
    lut=np.zeros((size,size,size,2),np.float64)
    variance=np.zeros((size,size,size),np.float64)
    sample_counts=np.zeros((size,size,size),np.int32)
    original_valid=np.zeros((size,size,size),np.bool_)
    maximum_rms=np.deg2rad(maximum_rms_angle_degrees)
    for node,start,count in zip(unique,starts,counts,strict=True):
        values=slope[start:start+count]
        normals=np.concatenate([-values,np.ones((count,1))],axis=1)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        mean=np.sum(normals,axis=0)
        mean/=np.maximum(np.linalg.norm(mean),1e-12)
        angles=np.arccos(np.clip(normals@mean,-1,1))
        median=np.median(angles)
        mad=1.4826*np.median(np.abs(angles-median))
        kept=angles<=median+3*mad+1e-12
        kept_values=values[kept]
        if kept_values.shape[0]<minimum_samples_per_node:
            continue
        kept_normals=normals[kept]
        mean_normal=np.sum(kept_normals,axis=0)
        mean_normal/=np.maximum(np.linalg.norm(mean_normal),1e-12)
        rms=np.sqrt(np.mean(np.arccos(
            np.clip(kept_normals@mean_normal,-1,1))**2))
        if rms>maximum_rms:
            continue
        coordinate=np.unravel_index(int(node),(size,size,size))
        node_mean=np.mean(kept_values,axis=0)
        lut[coordinate]=node_mean
        variance[coordinate]=np.mean(np.sum((kept_values-node_mean)**2,axis=1))
        sample_counts[coordinate]=kept_values.shape[0]
        original_valid[coordinate]=True
    valid_coordinates=np.argwhere(original_valid)
    if valid_coordinates.shape[0]<1:
        raise ValueError("没有满足样本数和角度误差要求的 LUT 节点")
    sigma_ref2=max(float(np.median(variance[original_valid])),1e-8)
    missing_coordinates=np.argwhere(~original_valid)
    tree=cKDTree(valid_coordinates/(size-1))
    k=min(8,valid_coordinates.shape[0])
    for start in range(0,missing_coordinates.shape[0],65536):
        target=missing_coordinates[start:start+65536]
        distances,neighbors=tree.query(target/(size-1),k=k)
        if k==1:
            distances=distances[:,None]; neighbors=neighbors[:,None]
        weights=1/np.maximum(distances,1e-12)**2
        weights/=np.sum(weights,axis=1,keepdims=True)
        source=valid_coordinates[neighbors]
        source_slopes=lut[source[...,0],source[...,1],source[...,2]]
        filled=np.sum(weights[...,None]*source_slopes,axis=1)
        source_variances=variance[source[...,0],source[...,1],source[...,2]]
        filled_variance=np.sum(weights*(source_variances+
            np.sum((source_slopes-filled[:,None])**2,axis=-1)),axis=1)
        lut[target[:,0],target[:,1],target[:,2]]=filled
        variance[target[:,0],target[:,1],target[:,2]]=filled_variance
    # 无接触状态是解析已知约束。将零色差三线性查询涉及的八个节点固定为零坡度，
    # 避免完整 LUT 在传感器未接触区域产生虚假倾斜。
    zero_coordinate=(size-1)*np.clip(
        (0-color_min)/(color_max-color_min),0,1)
    lower=np.floor(zero_coordinate).astype(int)
    upper=np.minimum(lower+1,size-1)
    for ir in {lower[0],upper[0]}:
        for ig in {lower[1],upper[1]}:
            for ib in {lower[2],upper[2]}:
                lut[ir,ig,ib]=0
                variance[ir,ig,ib]=sigma_ref2
    return NormalCalibration(
        slopes=lut.astype(np.float32),variances=variance.astype(np.float32),
        color_min=color_min.astype(np.float32),color_max=color_max.astype(np.float32),
        sigma_ref2=sigma_ref2,original_valid=original_valid,
        sample_counts=sample_counts)


def _residual_textures(
    model: LightFieldModel,
    device: jax.Device,
    texture_shape: tuple[int,int],
    residual_field_coefficients: np.ndarray | None = None,
) -> tuple[jax.Array,jax.Array]:
    """将离线或会话残差系数展开为 GPU 规范曲面纹理。"""
    if residual_field_coefficients is None:
        b_coefficients=model.residual_b_coefficients
        m_coefficients=model.residual_m_coefficients
    else:
        fields=np.asarray(residual_field_coefficients,np.float32)
        expected=(1+model.residual_m_coefficients.shape[0],
                  *model.residual_b_coefficients.shape)
        if fields.shape!=expected or not np.isfinite(fields).all():
            raise ValueError(f"会话残差基底必须是有限的 {expected} 数组")
        fields=jax.device_put(jnp.asarray(fields,jnp.float32),device)
        b_coefficients=fields[0]
        m_coefficients=fields[1:]
    b_texture=rgb_bspline_field(texture_shape,b_coefficients)
    m_textures=jax.vmap(lambda coefficients:rgb_bspline_field(
        texture_shape,coefficients))(m_coefficients)
    return b_texture,m_textures


def _make_residual_renderer(
    lightfield_cfg: dict,model: LightFieldModel,device: jax.Device,
    residual_method: str,session_sample_shape: tuple[int,int],
) -> Callable:
    """建立与实时路径一致的可切换 GPU 净化与规范残差采样计算。"""
    if residual_method not in {"uniform","uniform_huber"}:
        raise ValueError("residual_method 必须是 uniform 或 uniform_huber")
    runtime=lightfield_cfg["runtime"]; irls=lightfield_cfg["irls"]
    calibration=lightfield_cfg["calibration"]
    sigma=jax.device_put(jnp.asarray(irls["sigma_rgb"],jnp.float32),device)
    nodes=int(lightfield_cfg["integration_nodes"])
    epsilon=float(lightfield_cfg["distance_epsilon_mm"])
    cg_tolerance=float(lightfield_cfg.get("diffusion_cg_tolerance",1e-4))
    cg_iterations=int(lightfield_cfg.get("diffusion_cg_max_iterations",30))
    raster_chunk=int(runtime.get("gpu_raster_triangle_chunk",64))
    raster_width=int(runtime.get("gpu_raster_max_triangle_width",128))
    raster_height=int(runtime.get("gpu_raster_max_triangle_height",64))
    erode_pixels=int(runtime.get("difference_erode_pixels",4))
    session_saturation=int(calibration.get("saturation_threshold",250))
    session_erode_pixels=int(calibration.get("residual_erode_pixels",6))
    if model.background_method in GEOMETRY_BACKGROUND_METHODS:
        sample_config=(parse_geometry_cache_config(lightfield_cfg)
                       if model.background_method=="geometry_cache" else
                       parse_direct_fit_config(lightfield_cfg))
        session_saturation=sample_config.sample_saturation_threshold
        session_erode_pixels=sample_config.sample_erode_pixels
    score_huber_delta=float(runtime.get("residual_score_huber_delta",.04))
    score_huber_iterations=int(runtime.get("residual_score_huber_iterations",5))
    if score_huber_delta<=0 or score_huber_iterations<1:
        raise ValueError("runtime residual_score_huber 参数无效")

    @jax.jit
    def render(frame_bgr,xyz,uv,st,camera_depth,b_texture,m_textures):
        frame_linear=bgr_to_linear_rgb_jax(frame_bgr)
        if model.background_method=="physical_residual":
            observed=sample_linear_rgb_jax(frame_linear,uv)
            physical=physical_background(
                xyz,model,nodes,epsilon,
                int(lightfield_cfg.get("gpu_chunk_size",65536)),
                cg_tolerance,cg_iterations)
            gain,bias,weights=irls_gain_bias(
                observed,physical,model.bias,sigma,int(irls["iterations"]),
                float(irls["lambda_gain"]),float(irls["lambda_bias"]),
                float(irls["max_gain_deviation"]),
                float(irls["max_bias_deviation"]),
                jnp.ones(3,jnp.float32))
            colors=jnp.clip(gain*physical+bias,0,1)
        else:
            # direct 的 gain/bias 在神经背景投影到图像域后拟合；这里的占位通道
            # 仅用于保持后续光栅属性布局一致。
            gain=jnp.ones(3,jnp.float32)
            bias=jnp.zeros(3,jnp.float32)
            weights=jnp.ones(uv.shape[:-1],jnp.float32)
            colors=jnp.zeros_like(xyz)
        attributes=jnp.concatenate([colors,st,weights[...,None],xyz],axis=-1)
        attribute_image,valid,overflow=rasterize_attributes_jax(
            uv,camera_depth,attributes,frame_bgr.shape[:2],
            triangle_chunk=raster_chunk,max_triangle_width=raster_width,
            max_triangle_height=raster_height)
        coordinate_image=attribute_image[...,3:6]
        xyz_image=attribute_image[...,6:9]
        if model.background_method=="physical_residual":
            background=attribute_image[...,:3]
            correction_target=frame_linear-background
        b_image,m_images=sample_residual_correction_jax(
            coordinate_image,b_texture,m_textures,valid)
        if model.background_method in GEOMETRY_BACKGROUND_METHODS:
            neural_texture=geometry_background_field_jax(
                b_texture.shape[:2],xyz,model)
            neural_image,_=sample_residual_correction_jax(
                coordinate_image,neural_texture,m_textures[:0],valid)
            gain,bias,weights=irls_gain_bias(
                frame_linear-b_image,neural_image,
                jnp.zeros(3,jnp.float32),sigma,
                int(irls["iterations"]),float(irls["lambda_gain"]),
                float(irls["lambda_bias"]),
                float(irls["max_gain_deviation"]),
                float(irls["max_bias_deviation"]),
                jnp.ones(3,jnp.float32),valid_mask=valid)
            adjusted_neural=jnp.clip(gain*neural_image+bias,0,1)
            background=jnp.clip(adjusted_neural+b_image,0,1)
            correction_target=frame_linear-adjusted_neural
        canonical_residual,canonical_valid=(
            build_canonical_residual_sample_jax(
                correction_target,frame_bgr,valid,uv,session_sample_shape,
                saturation_threshold=session_saturation,
                erode_pixels=session_erode_pixels))
        # 人工椭圆最终仍只与几何/投影有效域相交；不再进行自动色差筛选。
        difference_valid=erode_mask_jax(valid,erode_pixels)
        if model.background_method in GEOMETRY_BACKGROUND_METHODS:
            scores=jnp.ones((3,1),jnp.float32)
            fitted=background
        elif residual_method=="uniform":
            scores=fit_uniform_residual_correction_scores_jax(
                correction_target,b_image,m_images,difference_valid)
        else:
            scores=fit_uniform_huber_residual_correction_scores_jax(
                correction_target,b_image,m_images,difference_valid,
                score_huber_delta,score_huber_iterations)
        if model.background_method=="physical_residual":
            fields=jnp.concatenate([b_image[None],m_images],axis=0)
            fitted=jnp.einsum("ck,khwc->hwc",scores,fields)
        # 人工区域内的 LUT 采样和在线局部重建共享同一个有符号色差定义。
        cleaned_difference=(frame_linear-fitted
                            if model.background_method in
                            GEOMETRY_BACKGROUND_METHODS else
                            correction_target-fitted)
        cleaned=jnp.where(
            difference_valid[...,None],cleaned_difference,0)
        return (cleaned,difference_valid,valid,xyz_image,overflow,gain,bias,
                scores,canonical_residual,canonical_valid)

    return render


def fit_normal_calibration_residual_session(
    canonical_residuals: np.ndarray,
    canonical_valid: np.ndarray,
    model: LightFieldModel,
    lightfield_cfg: dict,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    """按实时启动参数拟合会话残差；direct 仅有低频加性 B。"""
    calibration=lightfield_cfg["calibration"]
    runtime=lightfield_cfg["runtime"]
    common={
        "huber_delta":float(runtime.get("residual_score_huber_delta",.04)),
        "smooth_lambda":float(calibration.get("lambda_residual_smooth",.01)),
        "magnitude_lambda":float(
            calibration.get("lambda_residual_magnitude",1e-4)),
        "outer_weight":float(calibration.get("residual_outer_weight",.2)),
        "outer_fraction":float(calibration.get("residual_outer_fraction",.05)),
        "bsession_prior_lambda":float(
            calibration["residual_bsession_prior_strength"]),
        "channel_huber_ratio_min":float(
            runtime.get("residual_channel_huber_ratio_min",.5)),
        "channel_huber_ratio_max":float(
            runtime.get("residual_channel_huber_ratio_max",2.)),
    }
    if getattr(model,"background_method","physical_residual") in \
            GEOMETRY_BACKGROUND_METHODS:
        session_config=(parse_geometry_cache_config(lightfield_cfg)
                        if model.background_method=="geometry_cache" else
                        parse_direct_fit_config(lightfield_cfg))
        b_coefficients,scores,channel_huber,diagnostics=(
            fit_startup_direct_bsession_model(
                canonical_residuals,canonical_valid,
                np.asarray(model.residual_b_coefficients),
                session_correction_bounds=(
                    -session_config.session_correction_max_deviation,
                    session_config.session_correction_max_deviation),
                **common))
        fields=np.stack([
            b_coefficients,*np.asarray(model.residual_m_coefficients)
        ]).astype(np.float32)
        return fields,scores,channel_huber,diagnostics
    return fit_startup_residual_bsession_model(
        canonical_residuals,canonical_valid,
        np.asarray(model.residual_b_coefficients),
        np.asarray(model.residual_m_coefficients),
        bsession_max_field_deviation=float(
            calibration["residual_bsession_max_field_deviation"]),
        **common,
    )


def _resolve_output(path_value: str,base: Path) -> Path:
    path=Path(path_value).expanduser()
    return path if path.is_absolute() else base/path


def main() -> None:
    parser=argparse.ArgumentParser(description="用人工椭圆标定球建立颜色差分到局部坡度标定")
    parser.add_argument("--config",default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--edit-manual-regions",action="store_true",
        help="重新逐张打开人工椭圆编辑器；默认仅编辑缺失标注")
    args=parser.parse_args()
    config_path=Path(args.config).expanduser()
    all_config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw=all_config.get("normal_calibration")
    if not isinstance(raw,dict):
        raise ValueError("config.yaml 缺少 normal_calibration 配置段")
    lightfield_cfg=all_config["lightfield"]
    background_method=parse_background_method(lightfield_cfg)
    sphere_radius=raw.get("sphere_radius_mm")
    if not isinstance(sphere_radius,(int,float)) or isinstance(sphere_radius,bool) \
            or sphere_radius<=0:
        raise ValueError(
            "normal_calibration.sphere_radius_mm 必须填写真实标定球半径（mm）")
    sphere_radius=float(sphere_radius)
    paths=_resolve_paths(raw.get("images"),config_path.parent)
    observation_dir=_resolve_output(
        raw.get("observation_dir","assets/normal_calibration/observations"),
        config_path.parent)
    map_dir=_resolve_output(
        raw.get("generated_map_dir","assets/normal_calibration/maps"),
        config_path.parent)
    verification_dir=_resolve_output(
        raw.get("verification_dir","assets/normal_calibration/verification"),
        config_path.parent)
    manual_regions_path=_resolve_output(
        raw.get("manual_regions_file",
                "assets/normal_calibration/manual_ellipses.yaml"),
        config_path.parent)
    output=resolve_method_path(
        raw,method=background_method,mapping_key="output_files",
        legacy_key="output",base=config_path.parent,
        section_name="normal_calibration")
    bsession_exclusion_cfg=BSessionContactExclusionConfig.from_mapping(
        raw.get("bsession_contact_exclusion"))
    manual_cfg=raw.get("manual_editor",{})
    if not isinstance(manual_cfg,dict):
        raise ValueError("normal_calibration.manual_editor 必须是映射")
    unknown_manual=set(manual_cfg)-{"edge_margin_pixels"}
    if unknown_manual:
        raise ValueError("normal_calibration.manual_editor 包含未知字段: "+
                         ", ".join(sorted(unknown_manual)))
    edge_margin_pixels=int(manual_cfg.get("edge_margin_pixels",2))
    if edge_margin_pixels<0:
        raise ValueError("manual_editor.edge_margin_pixels 不能为负数")
    surface=all_config["get_surface"]
    reconstruction=parse_reconstruction_config(
        surface.get("reconstruction"),config_path=config_path,
        calibration_output=all_config.get("calibration",{}).get("output"))
    session_sample_shape=(
        reconstruction.observation_rows,reconstruction.observation_columns)
    residual_texture_shape=(
        reconstruction.residual_texture_rows,
        reconstruction.residual_texture_columns)
    observations=reconstruct_all_observations(
        paths,all_config,config_path,observation_dir,map_dir,
        filter_original_saturation=False)
    # 无效帧按实时部署语义被拒绝；后续依据成功缓存中的源图重新配对。
    observation_sources=[]
    for observation_path in observations:
        with np.load(observation_path,allow_pickle=False) as data:
            observation_sources.append(Path(str(data["source_image"])))
    paths=observation_sources
    local_cfg=all_config.get("local_reconstruction",{})
    if not isinstance(local_cfg,dict):
        raise ValueError("local_reconstruction 必须是映射")
    configured_residual_method=local_cfg.get("residual_method","uniform")
    if configured_residual_method not in {"uniform","uniform_huber"}:
        raise ValueError(
            "local_reconstruction.residual_method 必须是 "
            "uniform 或 uniform_huber")
    model_path=resolve_background_model_path(
        lightfield_cfg,method=background_method,base=config_path.parent)
    device=choose_device(lightfield_cfg.get("device","gpu"))
    model=LightFieldModel.load(model_path,device)
    if model.background_method!=background_method:
        raise ValueError(
            "光场模型的 background_method 与当前配置不一致；请重新标定光场")
    model_sha256=file_sha256(model_path)
    configured_layout=parse_light_source_layout(
        lightfield_cfg.get("light_source_layout"))
    if model.source_layout!=configured_layout:
        raise ValueError("当前 RGB 灯带布局与光场模型不一致，请先重新标定光场")
    if background_method in GEOMETRY_BACKGROUND_METHODS \
            and model.direct_curve_convexity!=reconstruction.curve_convexity:
        raise ValueError(
            "direct 背景模型的曲线凸性语义与法向标定重建不一致；"
            "请先重新运行 calibrate-lightfield")
    configured_residual_coefficients=(
        reconstruction.residual_coefficient_rows,
        reconstruction.residual_coefficient_columns)
    if tuple(model.residual_b_coefficients.shape[1:]) \
            != configured_residual_coefficients:
        raise ValueError(
            "当前 residual_coefficient_grid 与光场模型不一致，请先重新标定光场")
    print(
        "法向标定观测已由实时 JAX 重建链生成："
        f"convexity={reconstruction.curve_convexity}；"
        "XYZ/UV/depth 无事后补投影")
    # 第一遍只用离线 B 生成供人工观察的色差。人工椭圆是接触区的唯一来源：
    # 它先从 Bsession 训练数据中剔除，第二遍再直接作为 LUT 采样区。
    renderer=_make_residual_renderer(
        lightfield_cfg,model,device,configured_residual_method,
        session_sample_shape)
    canonical_mask_sampler=jax.jit(
        lambda source_mask,uv:sample_image_mask_to_canonical_jax(
            source_mask,uv,session_sample_shape),device=device)
    offline_b_texture,offline_m_textures=_residual_textures(
        model,device,residual_texture_shape)
    canonical_residual_parts=[]; canonical_valid_parts=[]
    manual_records={}
    manual_regions=load_manual_ellipses(manual_regions_path)
    previous_ellipse: ManualEllipse | None=None
    target_name=("规范曲面残差" if background_method=="physical_residual"
                 else "几何背景外低频会话残差")
    print(f"法向标定背景方法：{background_method}；正在从 {len(paths)} 张图片"
          f"读取/编辑人工椭圆，并从椭圆外采集{target_name}……")
    for index,(image_path,observation_path) in enumerate(
            zip(paths,observations,strict=True),1):
        frame=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取标定球图像: {image_path}")
        with np.load(observation_path,allow_pickle=False) as data:
            xyz=np.asarray(data["xyz"],np.float32)
            uv=np.asarray(data["uv"],np.float32)
            st=np.asarray(data["st"],np.float32)
            depth=np.asarray(data["camera_depth"],np.float32)
        device_frame=jax.device_put(frame,device)
        device_xyz=jax.device_put(xyz,device)
        device_uv=jax.device_put(uv,device)
        device_results=renderer(
            device_frame,device_xyz,device_uv,jax.device_put(st,device),
            jax.device_put(depth,device),offline_b_texture,offline_m_textures)
        (cleaned,difference_valid,surface_valid,xyz_image,overflow,
         canonical_residual,canonical_valid)=jax.device_get((
            device_results[0],device_results[1],device_results[2],
            device_results[3],device_results[4],device_results[8],
            device_results[9]))
        if bool(overflow):
            raise RuntimeError(
                f"{image_path.name} 的投影三角形超过 GPU 光栅化容量")
        key=annotation_key(image_path)
        ellipse=manual_regions.get(key)
        if ellipse is not None and (
                ellipse.image_width!=frame.shape[1]
                or ellipse.image_height!=frame.shape[0]):
            if not args.edit_manual_regions:
                raise RuntimeError(
                    f"{image_path.name} 的人工椭圆图像尺寸已变化；请加 "
                    "--edit-manual-regions 重新标注")
            ellipse=None
        if ellipse is None or args.edit_manual_regions:
            initial=(ellipse if ellipse is not None else
                     default_manual_ellipse(difference_valid,previous_ellipse))
            edited=edit_manual_ellipse(
                frame,cleaned,difference_valid,initial,label=image_path.name,
                index=index,total=len(paths))
            if edited is None:
                save_manual_ellipses(manual_regions_path,manual_regions)
                raise RuntimeError(
                    f"人工椭圆编辑已中止；进度已保存到 {manual_regions_path}")
            ellipse=edited; manual_regions[key]=ellipse
            save_manual_ellipses(manual_regions_path,manual_regions)
        if ellipse.use_for_lut:
            previous_ellipse=ellipse
        selection=manual_contact_selection(
            ellipse,difference_valid,surface_valid,xyz_image,xyz,sphere_radius,
            edge_margin_pixels)
        image_exclusion=dilate_manual_contact(
            selection.contact_mask,bsession_exclusion_cfg.dilation_pixels)
        canonical_exclusion=np.asarray(jax.device_get(
            canonical_mask_sampler(
                jax.device_put(image_exclusion,device),device_uv)),np.bool_)
        canonical_valid=np.asarray(canonical_valid,np.bool_)
        background_valid=canonical_valid&~canonical_exclusion
        original_count=int(np.count_nonzero(canonical_valid))
        background_count=int(np.count_nonzero(background_valid))
        background_fraction=(background_count/original_count
                             if original_count else 0.)
        excluded_image_fraction=float(np.mean(image_exclusion[surface_valid])) \
            if np.any(surface_valid) else 0.
        used=(ellipse.use_for_lut and original_count>0 and
              background_fraction>=
              bsession_exclusion_cfg.minimum_background_fraction)
        manual_records[str(image_path)]={
            "manual_use_for_lut":ellipse.use_for_lut,
            "manual_center_x":ellipse.center_x,
            "manual_center_y":ellipse.center_y,
            "manual_semi_axis_x":ellipse.semi_axis_x,
            "manual_semi_axis_y":ellipse.semi_axis_y,
            "manual_angle_degrees":ellipse.angle_degrees,
            "bsession_excluded_image_fraction":excluded_image_fraction,
            "bsession_background_fraction":background_fraction,
            "bsession_used":used,
        }
        if used:
            canonical_residual_parts.append(
                np.asarray(canonical_residual,np.float32))
            canonical_valid_parts.append(background_valid)
        elif original_count:
            print(
                f"法向标定会话基底 {index}/{len(paths)} {image_path.name}: "
                f"人工接触区剔除后仅保留 {background_fraction:.1%} 背景，低于 "
                f"{bsession_exclusion_cfg.minimum_background_fraction:.1%}，跳过")
        else:
            print(f"法向标定会话基底 {index}/{len(paths)} {image_path.name}: "
                  "规范曲面没有有效像素，跳过")
        if index==1 or index%5==0 or index==len(paths):
            print(
                f"法向标定人工区域/会话残差采集 {index}/{len(paths)}："
                f"有效帧={len(canonical_residual_parts)}，"
                f"本帧={'使用' if ellipse.use_for_lut else '人工跳过'}，"
                f"排除={excluded_image_fraction:.1%}，"
                f"规范背景保留={background_fraction:.1%}")
    if len(canonical_residual_parts)<2:
        raise RuntimeError(
            "法向标定至少需要两张具有有效规范曲面残差的图片来拟合 Bsession")
    (session_fields,session_training_scores,session_channel_huber,
     session_diagnostics)=fit_normal_calibration_residual_session(
        np.stack(canonical_residual_parts),np.stack(canonical_valid_parts),
        model,lightfield_cfg)
    print(
        f"法向标定会话基底完成：frames={len(canonical_residual_parts)}，"
        f"Huber RGB={session_channel_huber.tolist()}，"
        "RMSE raw/Bsession/Bsession+M="
        f"{session_diagnostics['raw_rmse_rgb'].tolist()}/"
        f"{session_diagnostics['bsession_rmse_rgb'].tolist()}/"
        f"{session_diagnostics['bsession_m_rmse_rgb'].tolist()}，"
        "cross-frame-floor/B-spatial-miss="
        f"{session_diagnostics['cross_frame_floor_rmse_rgb'].tolist()}/"
        f"{session_diagnostics['bspline_spatial_miss_rmse_rgb'].tolist()}")
    session_training_image_count=len(session_training_scores)
    del (canonical_residual_parts,canonical_valid_parts,session_training_scores,
         offline_b_texture,offline_m_textures)
    session_b_texture,session_m_textures=_residual_textures(
        model,device,residual_texture_shape,
        residual_field_coefficients=session_fields)
    color_parts=[]; slope_parts=[]; records=[]
    for index,(image_path,observation_path) in enumerate(
            zip(paths,observations,strict=True),1):
        frame=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取标定球图像: {image_path}")
        with np.load(observation_path,allow_pickle=False) as data:
            xyz=np.asarray(data["xyz"],np.float32)
            uv=np.asarray(data["uv"],np.float32)
            st=np.asarray(data["st"],np.float32)
            depth=np.asarray(data["camera_depth"],np.float32)
        device_results=renderer(
            jax.device_put(frame,device),jax.device_put(xyz,device),
            jax.device_put(uv,device),jax.device_put(st,device),
            jax.device_put(depth,device),session_b_texture,session_m_textures)
        results=jax.device_get(device_results[:8])
        (cleaned,valid,surface_valid,xyz_image,overflow,gain,bias,
         scores)=results
        if bool(overflow):
            raise RuntimeError(
                f"{image_path.name} 的投影三角形超过 GPU 光栅化容量")
        ellipse=manual_regions[annotation_key(image_path)]
        selection=manual_contact_selection(
            ellipse,valid,surface_valid,xyz_image,xyz,sphere_radius,
            edge_margin_pixels)
        verification=save_manual_verification(
            verification_dir/f"{image_path.stem}_verification.png",frame,
            cleaned,selection)
        sample_count=0
        if selection.accepted:
            colors,slopes=sphere_slope_samples(
                cleaned,xyz_image,selection,sphere_radius)
            color_parts.append(colors); slope_parts.append(slopes)
            sample_count=colors.shape[0]
        records.append({
            "image":str(image_path),"accepted":selection.accepted,
            "reason":selection.reason,"sample_count":sample_count,
            "radius_mm":selection.radius_mm,
            "indentation_mm":selection.indentation_mm,
            "plane_rms_mm":selection.plane_rms_mm,
            "verification":str(verification),"gain_rgb":np.asarray(gain).tolist(),
            "bias_rgb":np.asarray(bias).tolist(),"residual_scores":np.asarray(scores).tolist(),
            **manual_records[str(image_path)],
        })
        print(f"法向标定 {index}/{len(paths)} {image_path.name}: "
              f"{'人工使用' if selection.accepted else '人工跳过'}，"
              f"samples={sample_count}，检验图={verification}")
    verification_dir.mkdir(parents=True,exist_ok=True)
    report=verification_dir/"manual_region_report.csv"
    with report.open("w",encoding="utf-8",newline="") as stream:
        fieldnames=[key for key in records[0] if key not in {"gain_rgb","bias_rgb","residual_scores"}]
        writer=csv.DictWriter(stream,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key:item[key] for key in fieldnames} for item in records])
    minimum_images=int(raw.get("minimum_accepted_images",5))
    mapping_visualization: Path | None=None
    if color_parts:
        colors=np.concatenate(color_parts); slopes=np.concatenate(slope_parts)
        mapping_visualization=save_pq_rgb_mapping(
            verification_dir/"pq_rgb_mapping.png",colors,slopes,
            resolution=int(raw.get("pq_visualization_resolution",640)),
            display_percentile=float(
                raw.get("pq_visualization_percentile",99.5)))
        print(f"p-q RGB 映射可视化: {mapping_visualization}")
    if len(color_parts)<minimum_images:
        raise RuntimeError(
            f"仅 {len(color_parts)} 张标定图被人工选用，少于要求的 {minimum_images} 张；"
            f"请查看 {report}"
            +(f" 和 {mapping_visualization}" if mapping_visualization else ""))
    calibration=build_normal_lut(
        colors,slopes,size=64,
        minimum_samples_per_node=int(raw.get("minimum_samples_per_node",5)),
        maximum_rms_angle_degrees=float(raw.get("maximum_rms_angle_degrees",5.)))
    residual_basis=(
        "bsession_orthogonal" if background_method=="physical_residual" else
        "geometry_anchor_cache" if background_method=="geometry_cache" else
        "direct_geometry_conditioned_neural_field")
    saved=calibration.save(
        output,sphere_radius_mm=np.asarray(sphere_radius,np.float32),
        residual_method=np.asarray(configured_residual_method),
        background_method=np.asarray(background_method),
        background_model_sha256=np.asarray(model_sha256),
        reconstruction_pipeline=np.asarray(
            SURFACE_RECONSTRUCTION_PIPELINE_VERSION),
        curve_convexity=np.asarray(reconstruction.curve_convexity),
        accepted_image_count=np.asarray(len(color_parts),np.int32),
        total_image_count=np.asarray(len(paths),np.int32),
        total_sample_count=np.asarray(colors.shape[0],np.int64),
        residual_basis=np.asarray(residual_basis),
        manual_regions_sha256=np.asarray(file_sha256(manual_regions_path)),
        bsession_training_image_count=np.asarray(
            session_training_image_count,np.int32),
        residual_bsession_coefficients=np.asarray(session_fields[0],np.float32),
        residual_session_m_coefficients=np.asarray(session_fields[1:],np.float32))
    print(f"法向标定完成: {saved}；接受 {len(color_parts)}/{len(paths)} 张，"
          f"samples={colors.shape[0]}，原始有效 LUT 节点="
          f"{int(np.count_nonzero(calibration.original_valid))}；报告={report}")


if __name__=="__main__":
    main()
