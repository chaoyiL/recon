"""单进程实时完成曲面重建与无局部形变背景光场渲染。"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# JAX 与 SAM2/PyTorch 共用 GPU，禁止 JAX 启动时预占大部分显存。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cv2
import jax
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
import numpy as np
import yaml

from get_surface import parse_mask_refine, parse_prompts
from recon import (LocalReconstructionResult,NormalCalibration,
                   fit_no_contact_residual_model,
                   parse_no_contact_constraints,
                   parse_zero_color_protection)
from utils.camera import open_camera
from utils.config import (GEOMETRY_BACKGROUND_METHODS,file_sha256,
                          parse_background_method,
                          parse_camera_config,
                          parse_direct_fit_config,
                          parse_geometry_cache_config,
                          parse_reconstruction_config,
                          resolve_background_model_path,resolve_method_path)
from utils.jax_local_reconstruction import (classify_trusted_no_contact_jax,
                                            reconstruct_local_surface_jax)
from utils.jax_reconstruction import (
    SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
    prepare_edge_curves_from_masks_jax,
    reconstruct_surface_from_masks_jax,resample_surface_batch_jax)
from utils.lightfield import (LightFieldModel, choose_device, irls_gain_bias,
    bgr_to_linear_rgb_jax, build_canonical_residual_sample_jax, erode_mask_jax,
    evaluate_rgb_bspline,geometry_background_field_jax,
    fit_uniform_huber_residual_correction_scores_jax,
    fit_uniform_residual_correction_scores_jax,
    fit_startup_direct_bsession_model,fit_startup_residual_bsession_model,
    linear_rgb_to_bgr8_jax,
    parse_light_source_layout, physical_background,
    point_set_to_grid, rasterize_attributes_jax, rgb_bspline_field,
    signed_residual_bgr_jax, sample_linear_rgb_jax,
    sample_residual_correction_jax)
from utils.process import (EdgeReconstructor, ReconstructionPointSet,
                           SurfaceMeshVisualizer)
from utils.sam2_surface import SurfaceSegmenter


def torch_tensor_to_jax(tensor: object) -> jax.Array:
    """通过 DLPack 共享 Torch Tensor，不经过 NumPy/CPU 拷贝。"""
    contiguous=getattr(tensor,"contiguous",None)
    if contiguous is None:
        raise TypeError("Torch→JAX 输入必须是 Tensor")
    return jax_dlpack.from_dlpack(contiguous())


def evaluate_startup_stability(
    residual_samples: np.ndarray,
    valid_masks: np.ndarray,
    gain_samples: np.ndarray,
    bias_samples: np.ndarray,
    *,
    residual_field_rmse_threshold: float,
    gain_range_threshold: float,
    bias_range_threshold: float,
    minimum_valid_overlap: float,
) -> dict[str,np.ndarray | float | bool]:
    """判断最近连续若干帧是否形成稳定的规范曲面残差窗口。"""
    samples=np.asarray(residual_samples,dtype=np.float64)
    masks=np.asarray(valid_masks,dtype=np.bool_)
    gains=np.asarray(gain_samples,dtype=np.float64)
    biases=np.asarray(bias_samples,dtype=np.float64)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3] \
            or gains.shape!=(samples.shape[0],3) \
            or biases.shape!=(samples.shape[0],3) \
            or samples.shape[0]<1:
        raise ValueError("启动稳定窗口的残差、掩膜、gain 或 bias 形状无效")
    if not np.isfinite(samples).all() or not np.isfinite(gains).all() \
            or not np.isfinite(biases).all():
        raise ValueError("启动稳定窗口必须全部为有限值")
    if residual_field_rmse_threshold<=0 or gain_range_threshold<0 \
            or bias_range_threshold<0 or not 0<minimum_valid_overlap<=1:
        raise ValueError("启动稳定阈值无效")

    common_valid=np.all(masks,axis=0)
    minimum_valid_count=max(int(np.min(np.sum(masks,axis=(1,2)))),1)
    valid_overlap=float(np.sum(common_valid)/minimum_valid_count)
    if np.any(common_valid):
        common_samples=samples[:,common_valid,:]
        window_mean=np.mean(common_samples,axis=0,keepdims=True)
        frame_field_rmse=np.sqrt(np.mean(
            (common_samples-window_mean)**2,axis=1))
        field_rmse=np.max(frame_field_rmse,axis=0)
    else:
        field_rmse=np.full(3,np.inf,np.float64)
    gain_range=np.ptp(gains,axis=0)
    bias_range=np.ptp(biases,axis=0)
    stable=(valid_overlap>=minimum_valid_overlap
            and np.all(field_rmse<=residual_field_rmse_threshold)
            and np.all(gain_range<=gain_range_threshold)
            and np.all(bias_range<=bias_range_threshold))
    return {
        "stable":stable,
        "residual_field_rmse_rgb":field_rmse,
        "gain_range_rgb":gain_range,
        "bias_range_rgb":bias_range,
        "valid_overlap":valid_overlap,
    }


def point_set_grid(
    point_set: ReconstructionPointSet,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """直接在内存中把整体重建点集恢复为规则曲面网格。"""
    return point_set_to_grid({
        "xyz": point_set.xyz,
        "uv": point_set.uv,
        "st": point_set.st,
        "camera_depth": point_set.camera_depth,
        "cross_section_index": point_set.cross_section_index,
        "cross_section_alpha": point_set.cross_section_alpha,
    })


def mask_original_frame(frame: np.ndarray,
                        masks: dict[str | int,np.ndarray]) -> np.ndarray:
    """仅保留所有 SAM2 曲面 mask 的并集区域，其他像素置零。"""
    union=np.zeros(frame.shape[:2],dtype=np.bool_)
    for label,mask in masks.items():
        if mask.shape[:2] != frame.shape[:2]:
            raise ValueError(f"label {label!r} 的 mask 尺寸与相机帧不一致")
        union |= np.asarray(mask,dtype=np.bool_)
    masked=np.zeros_like(frame)
    masked[union]=frame[union]
    return masked


def main() -> None:
    parser=argparse.ArgumentParser(description="实时整体曲面与局部形变重建")
    parser.add_argument(
        "--config",default=Path(__file__).with_name("config.yaml"),
        help="配置文件；默认使用程序目录下的 config.yaml")
    parser.add_argument("--image",help="单张图片诊断模式；省略时读取相机实时重建")
    parser.add_argument(
        "--output",default="assets/lightfield_output/background.png",
        help="仅单张图片模式使用的输出路径")
    parser.add_argument("--no-display",action="store_true",help="关闭所有实时窗口")
    args=parser.parse_args()

    config_path=Path(args.config).expanduser()
    all_config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg=all_config["lightfield"]; surface=all_config["get_surface"]
    local_cfg=all_config.get("local_reconstruction",{})
    if not isinstance(local_cfg,dict):
        raise ValueError("local_reconstruction 必须是映射")
    local_enabled=local_cfg.get("enabled",False)
    if not isinstance(local_enabled,bool):
        raise ValueError("local_reconstruction.enabled 必须是布尔值")
    configured_residual_method=local_cfg.get("residual_method","uniform")
    if configured_residual_method not in {"uniform","uniform_huber"}:
        raise ValueError(
            "local_reconstruction.residual_method 必须是 "
            "uniform 或 uniform_huber")
    local_depth_color_range=float(local_cfg.get("depth_color_range_mm",2.))
    if not np.isfinite(local_depth_color_range) or local_depth_color_range<=0:
        raise ValueError("local_reconstruction.depth_color_range_mm 必须是有限正数")
    local_deformation_geometry_gain=float(
        local_cfg.get("deformation_geometry_gain",3.))
    if not np.isfinite(local_deformation_geometry_gain) \
            or local_deformation_geometry_gain<1:
        raise ValueError(
            "local_reconstruction.deformation_geometry_gain 必须是大于等于 1 的有限数")
    local_show_coordinate_frame=local_cfg.get("show_coordinate_frame",False)
    if not isinstance(local_show_coordinate_frame,bool):
        raise ValueError(
            "local_reconstruction.show_coordinate_frame 必须是布尔值")
    local_show_surface_mesh=local_cfg.get("show_surface_mesh",True)
    if not isinstance(local_show_surface_mesh,bool):
        raise ValueError("local_reconstruction.show_surface_mesh 必须是布尔值")
    local_mesh_update_interval=int(local_cfg.get("mesh_update_interval",1))
    if local_mesh_update_interval<1:
        raise ValueError("local_reconstruction.mesh_update_interval 必须为正整数")
    display_cfg=cfg["runtime"].get("display",{})
    if not isinstance(display_cfg,dict):
        raise ValueError("lightfield.runtime.display 必须是映射")
    display_defaults={
        "camera_original":False,
        "physical_render":False,
        "physical_plus_residual":False,
        "signed_difference":False,
        "oracle_uniform_difference":True,
    }
    display_settings={}
    for name,default in display_defaults.items():
        value=display_cfg.get(name,default)
        if not isinstance(value,bool):
            raise ValueError(f"lightfield.runtime.display.{name} 必须是布尔值")
        display_settings[name]=value
    residual_method_label=(
        "uniform Huber" if configured_residual_method=="uniform_huber"
        else "uniform")
    oracle_method_specs=((
        configured_residual_method,"oracle_uniform_difference",
        residual_method_label,f"{residual_method_label} difference"),)
    enabled_oracle_specs=tuple(
        item for item in oracle_method_specs if display_settings[item[1]])
    displayed_oracle_methods=tuple(item[0] for item in enabled_oracle_specs)
    enabled_oracle_methods=displayed_oracle_methods
    displayed_oracle_indices=tuple(
        enabled_oracle_methods.index(method) for method in displayed_oracle_methods)
    display_fps=0.
    display_oracle_rmse: np.ndarray | None=None
    initialized_windows: set[str]=set()

    def show_runtime_views(
        camera_original: np.ndarray | None=None,
        physical_render: np.ndarray | None=None,
        physical_plus_residual: np.ndarray | None=None,
        signed_difference: np.ndarray | None=None,
        oracle_differences: np.ndarray | None=None,
    ) -> bool:
        """按配置更新可用窗口，返回用户是否请求退出。"""
        if args.no_display:
            return False
        oracle_images=(None,)*len(enabled_oracle_specs)
        if oracle_differences is not None:
            oracle_values=np.asarray(oracle_differences)
            if oracle_values.ndim!=4 \
                    or oracle_values.shape[0]!=len(enabled_oracle_methods) \
                    or oracle_values.shape[-1]!=3:
                raise ValueError("oracle 色差图数量与已计算方法不一致")
            oracle_images=tuple(
                oracle_values[index] for index in displayed_oracle_indices)
        views=(
            ("camera_original","camera original (mask only)",camera_original),
            ("physical_render","deformation-free lightfield",physical_render),
            ("physical_plus_residual",
             "physical + fitted residual lightfield",physical_plus_residual),
            ("signed_difference",
             "signed color difference (camera - rendered)",signed_difference),
        )+tuple(
            (spec[1],spec[3],image)
            for spec,image in zip(enabled_oracle_specs,oracle_images,strict=True))
        shown=False
        oracle_window_index=0
        for setting,window,image in views:
            if display_settings[setting] and image is not None:
                display_image=np.array(image,copy=True)
                fps_text=f"FPS: {display_fps:.1f}"
                (text_width,_),_=cv2.getTextSize(
                    fps_text,cv2.FONT_HERSHEY_SIMPLEX,.7,2)
                text_origin=(max(12,display_image.shape[1]-text_width-12),30)
                cv2.putText(
                    display_image,fps_text,text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(
                    display_image,fps_text,text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2,cv2.LINE_AA)
                if setting.startswith("oracle_") \
                        and display_oracle_rmse is not None:
                    rmse=display_oracle_rmse[
                        displayed_oracle_indices[oracle_window_index]]
                    rmse_text=(f"RMSE RGB: {rmse[0]:.4f}/{rmse[1]:.4f}/"
                               f"{rmse[2]:.4f}")
                    cv2.putText(
                        display_image,rmse_text,(12,30),
                        cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),4,cv2.LINE_AA)
                    cv2.putText(
                        display_image,rmse_text,(12,30),
                        cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),2,cv2.LINE_AA)
                if window not in initialized_windows:
                    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
                    preview_scale=min(
                        1.,640/max(display_image.shape[1],1),
                        480/max(display_image.shape[0],1))
                    preview_width=max(1,int(display_image.shape[1]*preview_scale))
                    preview_height=max(1,int(display_image.shape[0]*preview_scale))
                    cv2.resizeWindow(window,preview_width,preview_height)
                    if setting.startswith("oracle_"):
                        column=oracle_window_index%2
                        row=oracle_window_index//2
                        cv2.moveWindow(
                            window,column*(preview_width+12),
                            row*(preview_height+48))
                    initialized_windows.add(window)
                cv2.imshow(window,display_image)
                shown=True
            if setting.startswith("oracle_") and display_settings[setting]:
                oracle_window_index+=1
        return shown and cv2.waitKey(1)&0xff in (ord("q"),27)

    prompts=parse_prompts(surface["prompts"]); mask_refine=parse_mask_refine(surface.get("mask_refine"))
    center_band_d=float(surface.get("center_band_d",40))
    def gpu_mask_kernel(size: int) -> int:
        if not mask_refine.enabled or size<=0:
            return 0
        return size if size%2 else size+1
    gpu_mask_close_kernel=gpu_mask_kernel(mask_refine.close_kernel)
    gpu_mask_open_kernel=gpu_mask_kernel(mask_refine.open_kernel)
    gpu_mask_blur_kernel=gpu_mask_kernel(mask_refine.blur_kernel)
    calibration_output=all_config.get("calibration",{}).get("output")
    geometry=parse_reconstruction_config(surface.get("reconstruction"),config_path=config_path,
                                         calibration_output=calibration_output)
    print(
        "实时网格："
        f"整体几何={geometry.geometry_rows}x{geometry.geometry_columns}，"
        f"光场={geometry.lightfield_rows}x{geometry.lightfield_columns}，"
        f"高分辨率观测={geometry.observation_rows}x"
        f"{geometry.observation_columns}，"
        f"残差控制={geometry.residual_coefficient_rows}x"
        f"{geometry.residual_coefficient_columns}，"
        f"残差纹理={geometry.residual_texture_rows}x"
        f"{geometry.residual_texture_columns}")
    print("整体重建仅使用当前 SAM 边界：无时间先验、无历史状态拒绝门控")

    compile_sam=surface.get("torch_compile",True)
    if not isinstance(compile_sam,bool):
        raise ValueError("get_surface.torch_compile 必须是布尔值")
    sam_frame_interval=int(surface.get("sam_frame_interval",1))
    sam_memory_frames=int(surface.get("sam_memory_frames",7))
    if sam_frame_interval<1:
        raise ValueError("get_surface.sam_frame_interval 必须为正整数")
    if sam_memory_frames<1:
        raise ValueError("get_surface.sam_memory_frames 必须为正整数")
    print(f"加载实时 SAM2 曲面模型: {surface['model']}"
          f"（torch.compile={'on' if compile_sam else 'off'}，"
          f"memory={sam_memory_frames}，每 {sam_frame_interval} 帧更新一次）")
    segmenter=SurfaceSegmenter(
        model_id=surface["model"],mask_refine=mask_refine,
        compile_model=compile_sam,memory_frames=sam_memory_frames)
    reconstructor=EdgeReconstructor(geometry.K,geometry.distortion_coefficients,
                                   geometry.s1,geometry.s2,sample_count=geometry.sample_count)

    device=choose_device(cfg.get("device","gpu"))
    source_layout=parse_light_source_layout(cfg.get("light_source_layout"))
    background_method=parse_background_method(cfg)
    model_path=resolve_background_model_path(
        cfg,method=background_method,base=config_path.parent)
    model=LightFieldModel.load(model_path,device)
    if model.background_method!=background_method:
        raise ValueError(
            "配置的 background.method 与背景模型不一致，请重新标定或切换模型")
    if model.source_layout != source_layout:
        raise ValueError("当前 lightfield.light_source_layout 与标定模型不一致，请重新标定")
    if background_method in GEOMETRY_BACKGROUND_METHODS \
            and model.direct_curve_convexity!=geometry.curve_convexity:
        raise ValueError(
            "direct 背景模型的曲线凸性语义与实时重建不一致；"
            "请重新运行 calibrate-lightfield")
    configured_residual_coefficients=(
        geometry.residual_coefficient_rows,
        geometry.residual_coefficient_columns)
    if tuple(model.residual_b_coefficients.shape[1:]) \
            != configured_residual_coefficients:
        raise ValueError(
            "当前 residual_coefficient_grid 与光场模型不一致，请重新标定光场")
    local_calibration: NormalCalibration | None=None
    if local_enabled:
        local_model_path=resolve_method_path(
            local_cfg,method=background_method,
            mapping_key="calibration_files",legacy_key="calibration_file",
            base=config_path.parent,section_name="local_reconstruction")
        local_calibration=NormalCalibration.load(local_model_path)
        if local_calibration.residual_method is None:
            print("警告：法向 LUT 未记录 residual_method；当前配置为 "
                  f"{configured_residual_method}。请重新运行 calibrate-norm，"
                  "避免色差特征不一致。")
        elif local_calibration.residual_method!=configured_residual_method:
            raise ValueError(
                "法向 LUT 的 residual_method="
                f"{local_calibration.residual_method}，但实时配置为 "
                f"{configured_residual_method}；请重新运行 calibrate-norm")
        if local_calibration.background_method is None:
            if background_method in GEOMETRY_BACKGROUND_METHODS:
                raise ValueError(
                    "纯拟合模式要求法向 LUT 记录 background_method；"
                    "请重新运行 calibrate-norm")
            print("警告：旧法向 LUT 未记录 background_method/model hash；"
                  "建议重新运行 calibrate-norm。")
        elif local_calibration.background_method!=background_method:
            raise ValueError(
                "法向 LUT 的 background_method 与实时背景方法不一致；"
                "请重新运行 calibrate-norm")
        model_sha256=file_sha256(model_path)
        if background_method in GEOMETRY_BACKGROUND_METHODS \
                and local_calibration.background_model_sha256 is None:
            raise ValueError(
                "纯拟合模式要求法向 LUT 绑定背景模型哈希；"
                "请重新运行 calibrate-norm")
        if local_calibration.background_model_sha256 is not None \
                and local_calibration.background_model_sha256!=model_sha256:
            raise ValueError(
                "法向 LUT 对应的背景模型已变化；请重新运行 calibrate-norm")
        if local_calibration.reconstruction_pipeline \
                != SURFACE_RECONSTRUCTION_PIPELINE_VERSION \
                or local_calibration.curve_convexity!=geometry.curve_convexity:
            raise ValueError(
                "法向 LUT 的整体重建链/凸性语义与实时部署不一致；"
                "请重新运行 calibrate-norm")
    local_lsmr_atol=float(local_cfg.get("lsmr_atol",1e-7))
    local_lsmr_btol=float(local_cfg.get("lsmr_btol",1e-7))
    local_zero_color_inner,local_zero_color_outer=(
        parse_zero_color_protection(local_cfg))
    local_no_contact=parse_no_contact_constraints(local_cfg)
    local_linear_solver=local_cfg.get("linear_solver","lsmr")
    local_lsmr_max_iterations=local_cfg.get(
        "realtime_solver_max_iterations",
        local_cfg.get("realtime_lsmr_max_iterations",
                      local_cfg.get("lsmr_max_iterations")))
    local_spectral_iterations=local_cfg.get(
        "spectral_poisson_initialization_iterations",0)
    if local_lsmr_atol<=0 or local_lsmr_btol<=0 \
            or not isinstance(local_linear_solver,str) \
            or local_linear_solver not in {"lsmr","spectral_pcg"} \
            or (local_lsmr_max_iterations is not None and
                (not isinstance(local_lsmr_max_iterations,int)
                 or isinstance(local_lsmr_max_iterations,bool)
                 or local_lsmr_max_iterations<1)) \
            or not isinstance(local_spectral_iterations,int) \
            or isinstance(local_spectral_iterations,bool) \
            or local_spectral_iterations<0:
        raise ValueError("local_reconstruction 的局部线性求解参数无效")
    texture_rows=geometry.residual_texture_rows
    texture_columns=geometry.residual_texture_columns
    offline_session_correction_texture=rgb_bspline_field(
        (texture_rows,texture_columns),model.residual_b_coefficients)
    offline_residual_b_texture=offline_session_correction_texture
    offline_residual_m_textures=jax.vmap(
        lambda coefficients:rgb_bspline_field(
            (texture_rows,texture_columns),coefficients))(
                model.residual_m_coefficients)
    residual_m_count=int(model.residual_m_coefficients.shape[0])
    sigma=jax.device_put(jnp.asarray(cfg["irls"]["sigma_rgb"],jnp.float32),device)
    nodes=int(cfg["integration_nodes"]); epsilon=float(cfg["distance_epsilon_mm"])
    cg_tolerance=float(cfg.get("diffusion_cg_tolerance",1e-4))
    cg_iterations=int(cfg.get("diffusion_cg_max_iterations",30))
    if cg_tolerance<=0 or cg_iterations<1:
        raise ValueError("diffusion_cg_tolerance 必须为正，max_iterations 必须大于等于 1")
    iterations=int(cfg["irls"]["iterations"])
    lambda_gain=float(cfg["irls"]["lambda_gain"]); lambda_bias=float(cfg["irls"]["lambda_bias"])
    max_gain_deviation=float(cfg["irls"]["max_gain_deviation"])
    max_bias_deviation=float(cfg["irls"]["max_bias_deviation"])
    if not 0<=max_gain_deviation<1:
        raise ValueError("lightfield.irls.max_gain_deviation 必须大于或等于 0 且小于 1")
    if max_bias_deviation<0: raise ValueError("lightfield.irls.max_bias_deviation 必须大于或等于 0")
    difference_gain=float(cfg["runtime"].get("difference_gain",2.0))
    if difference_gain<=0: raise ValueError("lightfield.runtime.difference_gain 必须为正数")
    difference_erode_pixels=int(cfg["runtime"].get("difference_erode_pixels",4))
    if difference_erode_pixels<0:
        raise ValueError("lightfield.runtime.difference_erode_pixels 必须大于等于 0")
    residual_score_huber=float(cfg["runtime"].get("residual_score_huber_delta",.04))
    residual_score_huber_iterations=int(
        cfg["runtime"].get("residual_score_huber_iterations",5))
    residual_fit_pixel_stride=int(
        cfg["runtime"].get("residual_fit_pixel_stride",1))
    residual_channel_huber_ratio_min=float(
        cfg["runtime"].get("residual_channel_huber_ratio_min",.5))
    residual_channel_huber_ratio_max=float(
        cfg["runtime"].get("residual_channel_huber_ratio_max",2.))
    residual_diagnostic_interval=float(
        cfg["runtime"].get("residual_diagnostic_interval_seconds",2.))
    startup_stability_frames=int(
        cfg["runtime"].get("startup_stability_required_frames",5))
    startup_stability_residual_threshold=float(
        cfg["runtime"].get(
            "startup_stability_residual_field_rmse_threshold",.01))
    startup_stability_gain_threshold=float(
        cfg["runtime"].get("startup_stability_gain_range_threshold",.01))
    startup_stability_bias_threshold=float(
        cfg["runtime"].get("startup_stability_bias_range_threshold",.01))
    startup_stability_minimum_overlap=float(
        cfg["runtime"].get("startup_stability_minimum_valid_overlap",.9))
    if residual_score_huber<=0 or residual_score_huber_iterations<1 \
            or residual_fit_pixel_stride<1 \
            or residual_channel_huber_ratio_min<=0 \
            or residual_channel_huber_ratio_max<residual_channel_huber_ratio_min \
            or residual_diagnostic_interval<=0:
        raise ValueError("runtime 残差标定或诊断参数无效")
    if startup_stability_frames<1 \
            or startup_stability_residual_threshold<=0 \
            or startup_stability_gain_threshold<0 \
            or startup_stability_bias_threshold<0 \
            or not 0<startup_stability_minimum_overlap<=1:
        raise ValueError("runtime 启动稳定窗口参数无效")
    calibration_cfg=cfg["calibration"]
    startup_gain_bias_frame_count=int(
        cfg["runtime"].get("startup_gain_bias_frames",10))
    startup_frame_count=int(cfg["runtime"].get("startup_residual_frames",10))
    startup_sample_rows=geometry.observation_rows
    startup_sample_columns=geometry.observation_columns
    startup_erode_pixels=int(calibration_cfg.get("residual_erode_pixels",6))
    startup_saturation=int(calibration_cfg.get("saturation_threshold",250))
    startup_smooth=float(calibration_cfg.get("lambda_residual_smooth",.01))
    startup_magnitude=float(calibration_cfg.get("lambda_residual_magnitude",1e-4))
    startup_outer_weight=float(calibration_cfg.get("residual_outer_weight",.2))
    startup_outer_fraction=float(calibration_cfg.get("residual_outer_fraction",.05))
    startup_bsession_prior_strength=float(
        calibration_cfg["residual_bsession_prior_strength"])
    startup_bsession_max_field=float(
        calibration_cfg["residual_bsession_max_field_deviation"])
    if background_method in {"direct_fit","direct_fit_3"}:
        geometry_session_config=parse_direct_fit_config(cfg)
    elif background_method=="geometry_cache":
        geometry_session_config=parse_geometry_cache_config(cfg)
    else:
        geometry_session_config=None
    if background_method in GEOMETRY_BACKGROUND_METHODS:
        # direct 模型离线训练与实时 Bsession 必须使用同一套绝对背景筛选；
        # 否则 250/6 的物理路径规则会额外挖掉高亮和边缘规范样本。
        startup_saturation=geometry_session_config.sample_saturation_threshold
        startup_erode_pixels=geometry_session_config.sample_erode_pixels
    if startup_gain_bias_frame_count<2 or startup_frame_count<2:
        raise ValueError("启动 gain/bias 和 Bsession 标定帧数都必须至少为 2")
    if startup_sample_rows<model.residual_b_coefficients.shape[1] \
            or startup_sample_columns<model.residual_b_coefficients.shape[2]:
        raise ValueError("启动残差采样分辨率不能小于 B 样条系数网格")
    if startup_bsession_prior_strength<0 or startup_bsession_max_field<=0:
        raise ValueError("Bsession 先验强度或空间场幅度上限无效")
    raster_triangle_chunk=int(cfg["runtime"].get("gpu_raster_triangle_chunk",64))
    raster_max_width=int(cfg["runtime"].get("gpu_raster_max_triangle_width",128))
    raster_max_height=int(cfg["runtime"].get("gpu_raster_max_triangle_height",64))
    if raster_triangle_chunk<1 or raster_max_width<1 or raster_max_height<1:
        raise ValueError("runtime gpu_raster_* 参数必须为正整数")
    if geometry.distortion_coefficients.size != 5:
        raise ValueError("JAX GPU 实时重建当前要求 OpenCV 五参数畸变模型")
    camera_matrix_gpu=jax.device_put(
        np.asarray(geometry.K,np.float32),device)
    distortion_gpu=jax.device_put(
        np.asarray(geometry.distortion_coefficients,np.float32),device)
    inverse_camera_gpu=jax.device_put(
        np.asarray(np.linalg.inv(geometry.K),np.float32),device)
    if local_calibration is not None:
        local_calibration_gpu=jax.device_put((
            jnp.asarray(local_calibration.slopes,jnp.float32),
            jnp.asarray(local_calibration.variances,jnp.float32),
            jnp.asarray(local_calibration.color_min,jnp.float32),
            jnp.asarray(local_calibration.color_max,jnp.float32)),device)
    else:
        local_calibration_gpu=None

    prepare_curves_gpu=jax.jit(lambda masks:prepare_edge_curves_from_masks_jax(
        masks,camera_matrix_gpu,distortion_gpu,reconstructor.sample_count,
        center_band_d,close_kernel=gpu_mask_close_kernel,
        open_kernel=gpu_mask_open_kernel,blur_kernel=gpu_mask_blur_kernel)[1:])

    def reconstruct_lightfield(physical,observed,gain_prior,bias_prior):
        gain,bias,weights=irls_gain_bias(
            observed,physical,bias_prior,sigma,iterations,lambda_gain,lambda_bias,
            max_gain_deviation,max_bias_deviation,gain_prior)
        return jnp.clip(gain*physical+bias,0,1),gain,bias,weights

    def reconstruct_geometry_gpu_impl(raw_mask_stack,rotation,tx):
        """仅在 SAM 外轮廓更新时重建整体几何。"""
        (mask_stack,geometry_xyz,geometry_uv,_geometry_st,geometry_depth,
         reconstruction_rms,reconstruction_valid)=reconstruct_surface_from_masks_jax(
            raw_mask_stack,camera_matrix_gpu,distortion_gpu,inverse_camera_gpu,
            rotation,geometry.s1,geometry.s2,tx,reconstructor.sample_count,
            center_band_d,geometry.pair_fill_count,
            geometry.uv_boundary_smooth_lambda,
            geometry.uv_boundary_huber_delta_px,
            curve_convexity=geometry.curve_convexity,
            close_kernel=gpu_mask_close_kernel,
            open_kernel=gpu_mask_open_kernel,
            blur_kernel=gpu_mask_blur_kernel)
        surface_count=raw_mask_stack.shape[0]
        xyz,uv,st,camera_depth=resample_surface_batch_jax(
            geometry_xyz,geometry_uv,geometry_depth,
            surface_count=surface_count,source_rows=geometry.geometry_rows,
            target_rows=geometry.lightfield_rows,
            target_columns=geometry.lightfield_columns)
        observation_xyz,observation_uv,_,_=resample_surface_batch_jax(
            geometry_xyz,geometry_uv,geometry_depth,
            surface_count=surface_count,source_rows=geometry.geometry_rows,
            target_rows=geometry.observation_rows,
            target_columns=geometry.observation_columns)
        # 物理光场只依赖整体 XYZ 和冻结的标定模型，与当前
        # 相机颜色无关；跟随几何缓存，避免每帧重复积分和扩散 CG。
        if background_method=="physical_residual":
            physical=physical_background(
                xyz,model,nodes,epsilon,int(cfg.get("gpu_chunk_size",65536)),
                cg_tolerance,cg_iterations)
            direct_base_texture=jnp.zeros_like(
                offline_session_correction_texture)
        else:
            # 保持 geometry_state 的静态 pytree 结构；纯拟合路径不会消费该占位场。
            physical=jnp.zeros_like(xyz)
            # 几何条件背景只随整体几何更新；在 SAM 更新间直接复用这张纹理。
            direct_base_texture=geometry_background_field_jax(
                (texture_rows,texture_columns),xyz,model)
        coordinate_image,image_valid,coordinate_overflow=(
            rasterize_attributes_jax(
                uv,camera_depth,st,raw_mask_stack.shape[1:3],
                triangle_chunk=raster_triangle_chunk,
                max_triangle_width=raster_max_width,
                max_triangle_height=raster_max_height))
        geometry_state=(mask_stack,xyz,uv,st,camera_depth,physical,
                coordinate_image,image_valid,coordinate_overflow,
                observation_xyz,observation_uv,
                reconstruction_rms,reconstruction_valid,
                direct_base_texture)
        return geometry_state

    reconstruct_geometry_gpu=jax.jit(reconstruct_geometry_gpu_impl)

    @jax.jit
    def prepare_residual_images_gpu(
        geometry_state,residual_bsession_texture,residual_m_textures,
    ):
        correction_image,m_images=sample_residual_correction_jax(
            geometry_state[6],residual_bsession_texture,
            residual_m_textures,geometry_state[7])
        direct_base_image,_=sample_residual_correction_jax(
            geometry_state[6],geometry_state[13],
            residual_m_textures[:0],geometry_state[7])
        return correction_image,m_images,direct_base_image

    def reconstruct_and_render_gpu_impl(
        frame_bgr,geometry_state,
        residual_bsession_image,residual_m_images,direct_base_image,
        lightfield_gain_prior,lightfield_bias_prior,
    ):
        """在已缓存整体几何上每帧更新光场、残差和局部观测。"""
        (mask_stack,xyz,uv,st,camera_depth,physical,
         coordinate_image,coordinate_valid,coordinate_overflow,
         observation_xyz,observation_uv,
         reconstruction_rms,reconstruction_valid,
         _direct_base_texture)=geometry_state
        frame_linear=bgr_to_linear_rgb_jax(frame_bgr)
        if background_method=="physical_residual":
            observed=sample_linear_rgb_jax(frame_linear,uv)
            colors,gain,bias,weights=reconstruct_lightfield(
                physical,observed,lightfield_gain_prior,lightfield_bias_prior)
            attributes=jnp.concatenate([colors,weights[...,None]],axis=-1)
            attribute_image,valid,raster_overflow=rasterize_attributes_jax(
                uv,camera_depth,attributes,frame_bgr.shape[:2],
                triangle_chunk=raster_triangle_chunk,
                max_triangle_width=raster_max_width,
                max_triangle_height=raster_max_height)
            background=attribute_image[...,:3]
            valid=valid&coordinate_valid
            raster_overflow=raster_overflow|coordinate_overflow
            canonical_source=frame_linear-background
        else:
            valid=coordinate_valid
            raster_overflow=coordinate_overflow
            # 几何背景负责空间/弯曲结构；当前帧仍需鲁棒估计全局通道
            # gain/bias，吸收曝光和白平衡漂移。有效域显式进入 IRLS，背景外黑区
            # 不会把 bias 错误地拉回零。
            gain,bias,weights=irls_gain_bias(
                frame_linear-residual_bsession_image,direct_base_image,
                lightfield_bias_prior,sigma,
                iterations,lambda_gain,lambda_bias,max_gain_deviation,
                max_bias_deviation,lightfield_gain_prior,valid_mask=valid)
            adjusted_direct=jnp.clip(
                gain*direct_base_image+bias,0,1)
            background=jnp.clip(
                adjusted_direct+residual_bsession_image,0,1)
            # 启动会话拟合只看光度调整后的几何背景没有解释掉的空间量。
            canonical_source=frame_linear-adjusted_direct
        output=linear_rgb_to_bgr8_jax(background)
        mask_union=jnp.any(mask_stack,axis=0)
        masked_original=jnp.where(mask_union[...,None],frame_bgr,0)
        difference_valid=erode_mask_jax(valid,difference_erode_pixels)
        canonical_residual,canonical_residual_valid=(
            build_canonical_residual_sample_jax(
                canonical_source,frame_bgr,valid,uv,
                (startup_sample_rows,startup_sample_columns),
                saturation_threshold=startup_saturation,
                erode_pixels=startup_erode_pixels))
        # 物理路径拟合 [B,M] 全部系数；几何背景模式已解释整体弯曲，
        # 在线只使用逐帧 gain/bias 和冻结的低频会话修正。
        # 系数数量只有 3x(1+M)，不必在每个原图像素上建正规方程。
        # 规则子采样仍保持图像域 uniform 权重，同时显著减少 Huber IRLS
        # 的 O(H*W*M^2) 开销。
        fit_slice=(slice(None,None,residual_fit_pixel_stride),
                   slice(None,None,residual_fit_pixel_stride))
        fit_residual=canonical_source[fit_slice]
        fit_bsession=residual_bsession_image[fit_slice]
        fit_m_images=residual_m_images[(slice(None),*fit_slice)]
        fit_valid=difference_valid[fit_slice]
        if background_method in GEOMETRY_BACKGROUND_METHODS:
            residual_scores=jnp.ones((3,1),jnp.float32)
            fitted_bsession=background
            fitted_residual=background
            physical_plus_residual_output=linear_rgb_to_bgr8_jax(
                jnp.clip(fitted_residual,0,1))
            raw_residual=canonical_source
            bsession_residual=frame_linear-fitted_bsession
            cleaned_residual=bsession_residual
        else:
            if configured_residual_method=="uniform":
                residual_scores=fit_uniform_residual_correction_scores_jax(
                    fit_residual,fit_bsession,fit_m_images,fit_valid)
            else:
                residual_scores=fit_uniform_huber_residual_correction_scores_jax(
                    fit_residual,fit_bsession,fit_m_images,fit_valid,
                    residual_score_huber,
                    residual_score_huber_iterations)
            residual_fields=jnp.concatenate(
                [residual_bsession_image[None],residual_m_images],axis=0)
            fitted_residual=jnp.einsum(
                "ck,khwc->hwc",residual_scores,residual_fields)
            physical_plus_residual_output=linear_rgb_to_bgr8_jax(
                jnp.clip(background+fitted_residual,0,1))
            fitted_bsession=jnp.einsum(
                "ck,khwc->hwc",residual_scores[:,:1],residual_fields[:1])
            raw_residual=frame_linear-background
            bsession_residual=raw_residual-fitted_bsession
            cleaned_residual=raw_residual-fitted_residual
        uniform_signed_residual=jnp.where(
            difference_valid[...,None],cleaned_residual,0)
        # 显示和局部重建共享这一份 cleaned_residual，不再重复拟合 uniform。
        uniform_difference=signed_residual_bgr_jax(
            uniform_signed_residual,difference_valid,difference_gain)
        if display_settings["oracle_uniform_difference"]:
            oracle_differences=uniform_difference[None]
        else:
            oracle_differences=jnp.empty(
                (0,*raw_residual.shape),jnp.uint8)
        # 高分辨率观测网格独立于几何/光场网格，颜色直接从全分辨率色差图采样。
        local_color_residual=sample_linear_rgb_jax(
            uniform_signed_residual,observation_uv)
        local_valid_sample=sample_linear_rgb_jax(
            difference_valid[...,None].astype(jnp.float32),observation_uv)[...,0]
        local_color_valid=local_valid_sample>=1-1e-6
        difference=uniform_difference
        valid_count=jnp.maximum(jnp.sum(difference_valid),1)
        residual_stages=jnp.stack(
            [raw_residual,bsession_residual,cleaned_residual])
        residual_stage_rmse=jnp.sqrt(jnp.sum(jnp.where(
            difference_valid[None,...,None],residual_stages**2,0),
            axis=(1,2))/valid_count)
        uniform_diagnostics=(
            residual_stage_rmse[2][None],residual_scores[None])
        valid_u8=valid.astype(jnp.uint8)*255
        difference_valid_u8=difference_valid.astype(jnp.uint8)*255
        return (output,physical_plus_residual_output,masked_original,valid_u8,
                difference,difference_valid_u8,gain,bias,weights,
                residual_scores,raster_overflow,
                reconstruction_rms,reconstruction_valid,
                canonical_residual,canonical_residual_valid,
                residual_stage_rmse,uniform_diagnostics,oracle_differences,
                observation_xyz,local_color_residual,local_color_valid)

    reconstruct_and_render_gpu=jax.jit(reconstruct_and_render_gpu_impl)
    surface_mesh_enabled=local_show_surface_mesh and not args.no_display
    steady_view_indices=tuple(
        index for setting,index in (
            ("camera_original",2),("physical_render",0),
            ("physical_plus_residual",1),("signed_difference",4))
        if not args.no_display and display_settings[setting])
    if not args.no_display and enabled_oracle_specs:
        steady_view_indices=(*steady_view_indices,17)

    @jax.jit
    def reconstruct_and_render_steady_gpu(*arguments):
        results=reconstruct_and_render_gpu_impl(*arguments)
        views=tuple(results[index] for index in steady_view_indices)
        diagnostic_indices=(6,7,9,10,11,12,15,16)
        if local_enabled:
            local_inputs=results[18:21]
        elif surface_mesh_enabled:
            diagnostic_indices=(*diagnostic_indices,18)
            local_inputs=()
        else:
            local_inputs=()
        diagnostics=tuple(results[index] for index in diagnostic_indices)
        # 局部观测网格直接传给下一个 JAX executable，不把这三个
        # 大张量夹在 diagnostics 中每帧同步回 CPU。
        return views,diagnostics,local_inputs

    if local_enabled:
        assert local_calibration is not None
        assert local_calibration_gpu is not None
        local_slopes_gpu,local_variances_gpu,local_color_min_gpu,\
            local_color_max_gpu=local_calibration_gpu

        def reconstruct_local_gpu_impl(
            local_xyz,local_colors,local_valid,no_contact_center,
            no_contact_scale,no_contact_model_valid,previous,
        ):
            if local_no_contact.enabled:
                trusted_no_contact,_=classify_trusted_no_contact_jax(
                    local_colors,local_valid,no_contact_center,
                    no_contact_scale,no_contact_model_valid,
                    trusted_score_threshold=(
                        local_no_contact.trusted_score_threshold),
                    contact_guard_score_threshold=(
                        local_no_contact.contact_guard_score_threshold),
                    contact_guard_radius_pixels=(
                        local_no_contact.contact_guard_radius_pixels),
                    surface_edge_margin_pixels=(
                        local_no_contact.surface_edge_margin_pixels))
            else:
                trusted_no_contact=jnp.zeros(local_valid.shape,jnp.bool_)
            return reconstruct_local_surface_jax(
                local_xyz,local_colors,local_slopes_gpu,local_variances_gpu,
                local_color_min_gpu,local_color_max_gpu,
                local_calibration.sigma_ref2,local_valid,previous,
                zero_color_inner_radius=local_zero_color_inner,
                zero_color_outer_radius=local_zero_color_outer,
                trusted_no_contact_mask=trusted_no_contact,
                trusted_no_contact_confidence=(
                    local_no_contact.slope_confidence),
                displacement_zero_lambda_per_mm2=(
                    local_no_contact.displacement_zero_lambda_per_mm2
                    if local_no_contact.enabled else 0.),
                lsmr_atol=local_lsmr_atol,lsmr_btol=local_lsmr_btol,
                lsmr_max_iterations=local_lsmr_max_iterations,
                spectral_initialization_iterations=local_spectral_iterations,
                linear_solver=local_linear_solver)

        def local_summary(result,color_residual):
            displacement=result[1]; valid=result[8]
            trusted=result[14]&valid
            color_zero=(valid&(jnp.max(jnp.abs(color_residual),axis=-1)
                               <=local_zero_color_inner))
            forced_slope_zero=color_zero|trusted
            displacement_soft_zero=trusted&~result[9]
            any_valid=jnp.any(valid)
            minimum=jnp.where(
                any_valid,jnp.min(jnp.where(valid,displacement,jnp.inf)),0)
            maximum=jnp.where(
                any_valid,jnp.max(jnp.where(valid,displacement,-jnp.inf)),0)
            return (minimum,maximum,result[10],result[11],result[12],result[13],
                    jnp.sum(forced_slope_zero),
                    jnp.sum(displacement_soft_zero),
                    jnp.sum(result[9]&valid),jnp.sum(valid))

        @jax.jit
        def reconstruct_local_compact_gpu(
            local_xyz,local_colors,local_valid,no_contact_center,
            no_contact_scale,no_contact_model_valid,previous,
        ):
            result=reconstruct_local_gpu_impl(
                local_xyz,local_colors,local_valid,no_contact_center,
                no_contact_scale,no_contact_model_valid,previous)
            return result[1],local_summary(result,local_colors)

        @jax.jit
        def reconstruct_local_mesh_gpu(
            local_xyz,local_colors,local_valid,no_contact_center,
            no_contact_scale,no_contact_model_valid,previous,
        ):
            result=reconstruct_local_gpu_impl(
                local_xyz,local_colors,local_valid,no_contact_center,
                no_contact_scale,no_contact_model_valid,previous)
            display_xyz=(result[0]+(local_deformation_geometry_gain-1)
                         *result[2])
            return (result[1],local_summary(result,local_colors),
                    (display_xyz,result[1],result[8]))

        reconstruct_local_full_gpu=jax.jit(reconstruct_local_gpu_impl)
    else:
        reconstruct_local_compact_gpu=None
        reconstruct_local_mesh_gpu=None
        reconstruct_local_full_gpu=None

    cap=None
    if args.image:
        frame=cv2.imread(args.image,cv2.IMREAD_COLOR)
        if frame is None: raise RuntimeError(f"无法读取 {args.image}")
        frames=[frame]
    else:
        camera=parse_camera_config(all_config["camera"])
        cap=open_camera(
            camera.device,camera.exposure,camera.white_balance_temperature,
            camera.width,camera.height)
        frames=None
    surface_mesh_vis: SurfaceMeshVisualizer | None=None
    if surface_mesh_enabled:
        surface_mesh_vis=SurfaceMeshVisualizer(
            window_name=("Realtime surface: normal depth "
                         f"[-{local_depth_color_range:g}, "
                         f"+{local_depth_color_range:g}] mm; geometry "
                         f"x{local_deformation_geometry_gain:g}"),
            depth_range_mm=local_depth_color_range,
            show_coordinate_frame=local_show_coordinate_frame)
    # 外参在首个有效帧完成标定后保持固定；Rodrigues 和 host->device 传输只做一次。
    rotation_gpu=None
    tx_gpu=None
    startup_residuals: list[np.ndarray]=[]
    startup_valid_masks: list[np.ndarray]=[]
    startup_gain_values: list[np.ndarray]=[]
    startup_bias_values: list[np.ndarray]=[]
    startup_ready=frames is not None
    gain_bias_prior_ready=(
        frames is not None or background_method in GEOMETRY_BACKGROUND_METHODS)
    residual_bsession_texture=offline_residual_b_texture
    no_contact_shape=(geometry.observation_rows,geometry.observation_columns)
    no_contact_center_gpu=jax.device_put(
        jnp.zeros((*no_contact_shape,3),jnp.float32),device)
    no_contact_scale_gpu=jax.device_put(jnp.ones((3,),jnp.float32),device)
    no_contact_valid_gpu=jax.device_put(
        jnp.zeros(no_contact_shape,jnp.bool_),device)
    # 物理路径启动后把 raw M 相对 Bsession 等价重参数化；纯拟合路径只更新
    # 低频会话修正，弯曲背景完全由选定的几何条件背景模型给出。
    residual_m_textures=offline_residual_m_textures
    lightfield_gain_prior_values=np.ones(3,np.float32)
    lightfield_bias_prior_values=np.asarray(model.bias,np.float32)
    lightfield_gain_prior=jax.device_put(
        jnp.asarray(lightfield_gain_prior_values),device)
    lightfield_bias_prior=jax.device_put(
        jnp.asarray(lightfield_bias_prior_values),device)
    if startup_ready:
        print("单张图片模式无法拟合会话底色；使用离线背景模型。")
    else:
        if background_method=="physical_residual":
            print(
                f"启动标定将在最近连续 {startup_stability_frames} 帧达到稳定阈值后"
                "自动开始；随后采集 "
                f"{startup_gain_bias_frame_count} 帧确定 gain/bias 先验，并采集 "
                f"{startup_frame_count} 帧拟合 Bsession 并重参数化 raw M。")
        else:
            print(
                f"{background_method} 启动将在连续 "
                f"{startup_stability_frames} 帧稳定后，"
                f"采集 {startup_frame_count} 帧拟合低频会话背景修正；"
                "不执行物理光场，逐帧鲁棒拟合 RGB gain/bias。")
    last_residual_diagnostic=0.
    last_startup_stability_report=0.
    performance_window_started=time.perf_counter()
    performance_window_frames=0
    steady_warmup_remaining=3
    performance_window_fps=0.
    startup_stability_ready=frames is not None
    startup_stability_residuals: list[np.ndarray]=[]
    startup_stability_valid_masks: list[np.ndarray]=[]
    startup_stability_gains: list[np.ndarray]=[]
    startup_stability_biases: list[np.ndarray]=[]
    previous_local_displacement: jax.Array | None=None
    frame_number=0
    mask_gpu: jax.Array | None=None
    geometry_state_gpu: tuple[jax.Array,...] | None=None
    residual_images_gpu: tuple[jax.Array,jax.Array] | None=None

    try:
        while True:
            started=time.perf_counter()
            if frames is not None:
                frame=frames[0]
            else:
                ok,frame=cap.read()
                if not ok or frame is None: raise RuntimeError("读取相机帧失败")

            current_frame_number=frame_number
            frame_number+=1
            update_sam=(mask_gpu is None or frames is not None
                        or current_frame_number%sam_frame_interval==0)
            if update_sam:
                _,mask_tensor,frame_tensor=segmenter.segment_tensors(frame,prompts)
                mask_gpu=torch_tensor_to_jax(mask_tensor)
            else:
                # 外轮廓在 SAM 更新之间复用；当前相机帧仍上传，
                # 因此光场观测和局部色差每帧都是新的。
                frame_tensor=segmenter.upload_frame(frame)
            frame_gpu=torch_tensor_to_jax(frame_tensor)
            assert mask_gpu is not None
            if not reconstructor.calibrated:
                initial_left,initial_right,initial_valid=jax.device_get(
                    prepare_curves_gpu(mask_gpu))
                for label_index in np.flatnonzero(initial_valid):
                    try:
                        reconstructor.process_curves(
                            initial_left[label_index],initial_right[label_index])
                    except ValueError:
                        continue
                    break
                if not reconstructor.calibrated:
                    if frames is not None: raise RuntimeError("当前图像整体曲面重建失败")
                    failed=frame.copy(); cv2.putText(
                        failed,"surface reconstruction failed",(12,30),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                    if show_runtime_views(frame,failed): break
                    continue

            if rotation_gpu is None:
                rotation_np=cv2.Rodrigues(
                    reconstructor.rotation_vector)[0].astype(np.float32)
                rotation_gpu,tx_gpu=jax.device_put((
                    rotation_np,np.asarray(reconstructor.tx,np.float32)),device)

            if geometry_state_gpu is None or update_sam:
                geometry_state_gpu=reconstruct_geometry_gpu(
                    mask_gpu,rotation_gpu,tx_gpu)
                residual_images_gpu=None
            if residual_images_gpu is None:
                residual_images_gpu=prepare_residual_images_gpu(
                    geometry_state_gpu,residual_bsession_texture,
                    residual_m_textures)
            gpu_arguments=(
                frame_gpu,geometry_state_gpu,
                *residual_images_gpu,
                lightfield_gain_prior,lightfield_bias_prior)
            local_device_result=None
            local_host_result=None
            local_summary_values=None
            local_mesh_values=None
            if startup_ready and frames is None:
                device_views,device_diagnostics,local_inputs=(
                    reconstruct_and_render_steady_gpu(*gpu_arguments))
                if local_enabled:
                    assert reconstruct_local_compact_gpu is not None
                    assert reconstruct_local_mesh_gpu is not None
                    local_xyz_device,local_color_device,local_valid_device=(
                        local_inputs)
                    if previous_local_displacement is None:
                        previous_local_displacement=jnp.zeros(
                            local_xyz_device.shape[:2],jnp.float32)
                    update_mesh=(surface_mesh_enabled and
                                 current_frame_number%
                                 local_mesh_update_interval==0)
                    if update_mesh:
                        (previous_local_displacement,local_summary_device,
                         local_mesh_device)=reconstruct_local_mesh_gpu(
                            local_xyz_device,local_color_device,local_valid_device,
                            no_contact_center_gpu,no_contact_scale_gpu,
                            no_contact_valid_gpu,
                            previous_local_displacement)
                    else:
                        (previous_local_displacement,local_summary_device)=\
                            reconstruct_local_compact_gpu(
                                local_xyz_device,local_color_device,
                                local_valid_device,no_contact_center_gpu,
                                no_contact_scale_gpu,no_contact_valid_gpu,
                                previous_local_displacement)
                        local_mesh_device=None
                else:
                    local_summary_device=local_mesh_device=None
                (host_views,diagnostics,local_summary_values,
                 local_mesh_values)=jax.device_get(
                    (device_views,device_diagnostics,local_summary_device,
                     local_mesh_device))
                views_by_index=dict(zip(steady_view_indices,host_views))
                output=views_by_index.get(0)
                physical_plus_residual_output=views_by_index.get(1)
                masked_original=views_by_index.get(2)
                difference=views_by_index.get(4)
                oracle_differences=views_by_index.get(17)
                base_diagnostics=diagnostics[:8]
                (gain_values,bias_values,residual_scores,raster_overflow,
                 rms_values,reconstruction_valid,
                 residual_stage_rmse,uniform_diagnostics)=base_diagnostics
                if surface_mesh_enabled and not local_enabled:
                    local_xyz=diagnostics[8]
                    local_color_residual=local_color_valid=None
                else:
                    local_xyz=local_color_residual=local_color_valid=None
                valid=difference_valid=weights=None
                canonical_residual_value=canonical_valid_value=None
            else:
                gpu_results=reconstruct_and_render_gpu(*gpu_arguments)
                if startup_ready and local_enabled:
                    assert reconstruct_local_full_gpu is not None
                    local_xyz_device,local_color_device,local_valid_device=(
                        gpu_results[18:21])
                    if previous_local_displacement is None:
                        previous_local_displacement=jnp.zeros(
                            local_xyz_device.shape[:2],jnp.float32)
                    local_device_result=reconstruct_local_full_gpu(
                        local_xyz_device,local_color_device,local_valid_device,
                        no_contact_center_gpu,no_contact_scale_gpu,
                        no_contact_valid_gpu,
                        previous_local_displacement)
                    previous_local_displacement=local_device_result[1]
                gpu_results,local_host_result=jax.device_get(
                    (gpu_results,local_device_result))
                (output,physical_plus_residual_output,masked_original,valid,difference,
                 difference_valid,gain_values,bias_values,weights,residual_scores,
                 raster_overflow,rms_values,reconstruction_valid,
                 canonical_residual_value,canonical_valid_value,
                 residual_stage_rmse,uniform_diagnostics,
                 oracle_differences,local_xyz,local_color_residual,
                 local_color_valid)=gpu_results
            local_result: LocalReconstructionResult | None=None
            if local_host_result is not None:
                (local_output,local_displacement,local_vectors,local_normals,
                 local_slopes,local_observed,local_curvature,local_confidence,
                 local_result_valid,local_boundary,local_istop,local_iterations,
                 local_residual_norm,local_equation_count,
                 local_trusted_no_contact)=local_host_result
                if int(local_equation_count)>0:
                    local_result=LocalReconstructionResult(
                        xyz_out=np.asarray(local_output),
                        displacement=np.asarray(local_displacement),
                        displacement_vectors=np.asarray(local_vectors),
                        reference_normals=np.asarray(local_normals),
                        slopes=np.asarray(local_slopes),
                        observed_normals=np.asarray(local_observed),
                        curvature_correction=np.asarray(local_curvature),
                        confidence=np.asarray(local_confidence),
                        valid_mask=np.asarray(local_result_valid,np.bool_),
                        boundary_mask=np.asarray(local_boundary,np.bool_),
                        solver_istop=int(local_istop),
                        solver_iterations=int(local_iterations),
                        solver_residual_norm=float(local_residual_norm),
                        trusted_no_contact_mask=np.asarray(
                            local_trusted_no_contact,np.bool_))
                if np.any(local_result_valid):
                    local_valid_host=np.asarray(local_result_valid,np.bool_)
                    local_boundary_host=(
                        np.asarray(local_boundary,np.bool_)&local_valid_host)
                    local_trusted_host=np.asarray(
                        local_trusted_no_contact,np.bool_)&local_valid_host
                    local_color_zero_host=(local_valid_host&(
                        np.max(np.abs(np.asarray(local_color_residual)),axis=-1)
                        <=local_zero_color_inner))
                    local_pq_zero_host=local_color_zero_host|local_trusted_host
                    local_dprior_zero_host=(
                        local_trusted_host&~local_boundary_host)
                    local_summary_values=(
                        float(np.min(local_displacement[local_result_valid])),
                        float(np.max(local_displacement[local_result_valid])),
                        int(local_istop),int(local_iterations),
                        float(local_residual_norm),int(local_equation_count),
                        int(np.count_nonzero(local_pq_zero_host)),
                        int(np.count_nonzero(local_dprior_zero_host)),
                        int(np.count_nonzero(local_boundary_host)),
                        int(np.count_nonzero(local_result_valid)))
            display_oracle_rmse=np.asarray(uniform_diagnostics[0])
            if not bool(np.all(reconstruction_valid)):
                if frames is not None: raise RuntimeError("JAX GPU 曲面重建失败")
                continue
            if bool(raster_overflow):
                raise RuntimeError(
                    "投影三角形超过 GPU 光栅化容量；请增大 runtime."
                    "gpu_raster_max_triangle_width/height")
            # JAX device_get 返回的 NumPy 视图可能是只读的；OpenCV 绘字会原地修改。
            if output is not None:
                output=np.array(output,copy=True)
            rms=float(rms_values[-1])
            if not startup_ready:
                if not startup_stability_ready:
                    residual_sample=np.asarray(
                        canonical_residual_value,np.float32)
                    valid_sample=np.asarray(canonical_valid_value,np.bool_)
                    if not np.any(valid_sample):
                        print("当前帧没有有效的未饱和规范曲面残差像素，"
                              "不计入启动稳定窗口。")
                        continue
                    startup_stability_residuals.append(
                        np.asarray(residual_sample,np.float32))
                    startup_stability_valid_masks.append(
                        np.asarray(valid_sample,np.bool_))
                    startup_stability_gains.append(
                        np.asarray(gain_values,np.float32))
                    startup_stability_biases.append(
                        np.asarray(bias_values,np.float32))
                    for values in (
                            startup_stability_residuals,
                            startup_stability_valid_masks,
                            startup_stability_gains,
                            startup_stability_biases):
                        if len(values)>startup_stability_frames:
                            del values[0]
                    collected=len(startup_stability_residuals)
                    stability=None
                    if collected==startup_stability_frames:
                        stability=evaluate_startup_stability(
                            np.stack(startup_stability_residuals),
                            np.stack(startup_stability_valid_masks),
                            np.stack(startup_stability_gains),
                            np.stack(startup_stability_biases),
                            residual_field_rmse_threshold=(
                                startup_stability_residual_threshold),
                            gain_range_threshold=(
                                startup_stability_gain_threshold),
                            bias_range_threshold=(
                                startup_stability_bias_threshold),
                            minimum_valid_overlap=(
                                startup_stability_minimum_overlap))
                    status=(f"collecting stable window {collected}/"
                            f"{startup_stability_frames}" if stability is None else
                            "stable window accepted" if stability["stable"] else
                            "waiting for stable window")
                    cv2.putText(
                        output,"DO NOT TOUCH - "+status,
                        (12,32),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,255),2)
                    if stability is not None:
                        field_text="/".join(
                            f"{value:.4f}" for value in
                            stability["residual_field_rmse_rgb"])
                        cv2.putText(
                            output,
                            f"field RMS RGB={field_text} "
                            f"overlap={stability['valid_overlap']:.2f}",
                            (12,58),cv2.FONT_HERSHEY_SIMPLEX,.50,(0,0,255),2)
                        report_time=time.monotonic()
                        if stability["stable"] or report_time-\
                                last_startup_stability_report>=1.:
                            print(
                                "启动稳定窗口："
                                f"stable={stability['stable']}，"
                                "residual field RMS RGB="
                                f"{stability['residual_field_rmse_rgb'].tolist()}，"
                                f"gain range RGB={stability['gain_range_rgb'].tolist()}，"
                                f"bias range RGB={stability['bias_range_rgb'].tolist()}，"
                                f"valid overlap={stability['valid_overlap']:.3f}")
                            last_startup_stability_report=report_time
                    if stability is not None and stability["stable"]:
                        startup_stability_ready=True
                        startup_stability_residuals.clear()
                        startup_stability_valid_masks.clear()
                        startup_stability_gains.clear()
                        startup_stability_biases.clear()
                        print(
                            f"最近连续 {startup_stability_frames} 帧已达到稳定阈值；"
                            +("下一帧开始采集 gain/bias 标定数据。"
                               if background_method=="physical_residual" else
                               "下一帧开始采集绝对 Bsession 标定数据。"))
                    if show_runtime_views(
                            masked_original,output,physical_plus_residual_output,
                            difference,oracle_differences): break
                    continue
                if not gain_bias_prior_ready:
                    startup_gain_values.append(np.asarray(gain_values,np.float32))
                    startup_bias_values.append(np.asarray(bias_values,np.float32))
                    collected=len(startup_gain_values)
                    message=("DO NOT TOUCH SENSOR - calibrating gain/bias "
                             f"{collected}/{startup_gain_bias_frame_count}")
                    print("请勿触碰传感器：启动 gain/bias 标定 "
                          f"{collected}/{startup_gain_bias_frame_count}")
                    cv2.putText(output,message,(12,32),cv2.FONT_HERSHEY_SIMPLEX,
                                .68,(0,0,255),2)
                    if collected==startup_gain_bias_frame_count:
                        lightfield_gain_prior_values=np.mean(
                            np.stack(startup_gain_values),axis=0).astype(np.float32)
                        lightfield_bias_prior_values=np.mean(
                            np.stack(startup_bias_values),axis=0).astype(np.float32)
                        if not np.isfinite(lightfield_gain_prior_values).all() \
                                or not np.isfinite(lightfield_bias_prior_values).all():
                            raise RuntimeError(
                                "启动标定得到的 gain/bias 先验包含非有限值")
                        lightfield_gain_prior=jax.device_put(
                            jnp.asarray(lightfield_gain_prior_values),device)
                        lightfield_bias_prior=jax.device_put(
                            jnp.asarray(lightfield_bias_prior_values),device)
                        startup_gain_values.clear()
                        startup_bias_values.clear()
                        gain_bias_prior_ready=True
                        print("启动第一阶段完成："
                              f"gain_ref={lightfield_gain_prior_values.tolist()}，"
                              f"bias_ref={lightfield_bias_prior_values.tolist()}；"
                              "下面开始采集 Bsession 标定帧并准备 raw M 会话重参数化。")
                    if show_runtime_views(
                            masked_original,output,physical_plus_residual_output,
                            difference,oracle_differences): break
                    continue
                residual_sample=np.asarray(
                    canonical_residual_value,np.float32)
                valid_sample=np.asarray(canonical_valid_value,np.bool_)
                if not np.any(valid_sample):
                    print("当前帧没有有效的未饱和残差像素，不计入启动标定。")
                    continue
                startup_residuals.append(residual_sample)
                startup_valid_masks.append(valid_sample)
                collected=len(startup_residuals)
                message=(f"DO NOT TOUCH SENSOR - calibrating Bsession "
                         f"{collected}/{startup_frame_count}")
                print(f"请勿触碰传感器：启动残差标定 {collected}/{startup_frame_count}")
                cv2.putText(output,message,(12,32),cv2.FONT_HERSHEY_SIMPLEX,
                            .68,(0,0,255),2)
                if collected==startup_frame_count:
                    cv2.putText(output,"FITTING Bsession...",(12,60),
                                cv2.FONT_HERSHEY_SIMPLEX,.68,(0,0,255),2)
                    if show_runtime_views(
                            masked_original,output,physical_plus_residual_output,
                            difference,oracle_differences): break
                    print(f"{startup_frame_count} 帧采集完成，正在拟合并冻结"
                          "本次会话的 Bsession……")
                    startup_sample_values=np.stack(startup_residuals)
                    startup_valid_values=np.stack(startup_valid_masks)
                    if background_method in GEOMETRY_BACKGROUND_METHODS:
                        (startup_b_coefficients,_startup_scores,
                         startup_channel_huber,startup_diagnostics)=(
                            fit_startup_direct_bsession_model(
                                startup_sample_values,startup_valid_values,
                                np.asarray(model.residual_b_coefficients),
                                huber_delta=residual_score_huber,
                                smooth_lambda=startup_smooth,
                                magnitude_lambda=startup_magnitude,
                                outer_weight=startup_outer_weight,
                                outer_fraction=startup_outer_fraction,
                                bsession_prior_lambda=(
                                    startup_bsession_prior_strength),
                                session_correction_bounds=(
                                    -geometry_session_config.
                                    session_correction_max_deviation,
                                    geometry_session_config.
                                    session_correction_max_deviation),
                                channel_huber_ratio_min=(
                                    residual_channel_huber_ratio_min),
                                channel_huber_ratio_max=(
                                    residual_channel_huber_ratio_max)))
                        startup_m_coefficients=np.asarray(
                            model.residual_m_coefficients)
                    else:
                        (startup_fields,_startup_scores,startup_channel_huber,
                         startup_diagnostics)=(
                            fit_startup_residual_bsession_model(
                                startup_sample_values,startup_valid_values,
                                np.asarray(model.residual_b_coefficients),
                                np.asarray(model.residual_m_coefficients),
                                huber_delta=residual_score_huber,
                                smooth_lambda=startup_smooth,
                                magnitude_lambda=startup_magnitude,
                                outer_weight=startup_outer_weight,
                                outer_fraction=startup_outer_fraction,
                                bsession_prior_lambda=(
                                    startup_bsession_prior_strength),
                                bsession_max_field_deviation=(
                                    startup_bsession_max_field),
                                channel_huber_ratio_min=(
                                    residual_channel_huber_ratio_min),
                                channel_huber_ratio_max=(
                                    residual_channel_huber_ratio_max)))
                        startup_b_coefficients=startup_fields[0]
                        startup_m_coefficients=startup_fields[1:]
                    startup_bsession_texture_value=evaluate_rgb_bspline(
                        startup_b_coefficients,(texture_rows,texture_columns))
                    residual_bsession_texture=jax.device_put(
                        jnp.asarray(startup_bsession_texture_value,jnp.float32),device)
                    if background_method=="physical_residual":
                        startup_m_texture_values=np.stack([
                            evaluate_rgb_bspline(
                                coefficients,(texture_rows,texture_columns))
                            for coefficients in startup_m_coefficients])
                        residual_m_textures=jax.device_put(
                            jnp.asarray(startup_m_texture_values,jnp.float32),device)
                    if local_no_contact.enabled:
                        if background_method in GEOMETRY_BACKGROUND_METHODS:
                            startup_fitted_samples=evaluate_rgb_bspline(
                                startup_b_coefficients,no_contact_shape)[None]
                        else:
                            startup_sample_fields=np.stack([
                                evaluate_rgb_bspline(
                                    coefficients,no_contact_shape)
                                for coefficients in startup_fields])
                            startup_fitted_samples=np.einsum(
                                "nck,khwc->nhwc",_startup_scores,
                                startup_sample_fields)
                        startup_cleaned_samples=np.where(
                            startup_valid_values[...,None],
                            startup_sample_values-startup_fitted_samples,0)
                        no_contact_model=fit_no_contact_residual_model(
                            startup_cleaned_samples,startup_valid_values,
                            minimum_valid_fraction=(local_no_contact.
                                minimum_startup_valid_fraction),
                            minimum_channel_scale=(local_no_contact.
                                minimum_channel_scale))
                        no_contact_center_gpu=jax.device_put(
                            jnp.asarray(no_contact_model.center),device)
                        no_contact_scale_gpu=jax.device_put(
                            jnp.asarray(no_contact_model.channel_scale),device)
                        no_contact_valid_gpu=jax.device_put(
                            jnp.asarray(no_contact_model.valid_mask),device)
                        per_frame_coverage=np.mean(
                            startup_valid_values,axis=(1,2))
                        valid_counts=np.sum(startup_valid_values,axis=0)
                        all_frame_coverage=float(np.mean(
                            valid_counts==startup_valid_values.shape[0]))
                        required_count=int(np.ceil(
                            local_no_contact.minimum_startup_valid_fraction*
                            startup_valid_values.shape[0]))
                        print(
                            "可信无接触统计完成：scale RGB="
                            f"{no_contact_model.channel_scale.tolist()}，"
                            "coverage="
                            f"{float(np.mean(no_contact_model.valid_mask)):.3f}，"
                            f"required={required_count}/"
                            f"{startup_valid_values.shape[0]}，"
                            "per-frame min/median/max="
                            f"{float(np.min(per_frame_coverage)):.3f}/"
                            f"{float(np.median(per_frame_coverage)):.3f}/"
                            f"{float(np.max(per_frame_coverage)):.3f}，"
                            f"ever={float(np.mean(valid_counts>0)):.3f}，"
                            f"all={all_frame_coverage:.3f}")
                    residual_images_gpu=None
                    startup_ready=True
                    startup_residuals.clear()
                    startup_valid_masks.clear()
                    # 下一帧会编译稳态渲染和局部求解图。预热帧
                    # 不计入稳态吞吐率，避免显示数分钟才恢复。
                    last_residual_diagnostic=time.monotonic()
                    performance_window_started=time.perf_counter()
                    performance_window_frames=0
                    steady_warmup_remaining=3
                    print("启动背景标定完成：Bsession 已冻结；"
                          +("raw M 已完成会话级等价正交化；"
                             if background_method=="physical_residual" else
                             f"{background_method} 几何背景与会话修正已冻结；")+
                          f"Huber RGB={startup_channel_huber.tolist()}")
                    print(
                        "启动残差诊断：RMSE RGB "
                        f"raw={startup_diagnostics['raw_rmse_rgb'].tolist()}，"
                        f"Bsession={startup_diagnostics['bsession_rmse_rgb'].tolist()}，"
                        f"Bsession+M={startup_diagnostics['bsession_m_rmse_rgb'].tolist()}，"
                        "cross-frame-floor="
                        f"{startup_diagnostics['cross_frame_floor_rmse_rgb'].tolist()}，"
                        "B-spatial-miss="
                        f"{startup_diagnostics['bspline_spatial_miss_rmse_rgb'].tolist()}")
                if show_runtime_views(
                        masked_original,output,physical_plus_residual_output,
                        difference,oracle_differences): break
                continue
            if local_mesh_values is not None:
                mesh_grid,mesh_depth,mesh_valid=local_mesh_values
            elif local_result is not None:
                mesh_grid=(local_result.xyz_out
                           +(local_deformation_geometry_gain-1)
                           *local_result.displacement_vectors)
                mesh_depth=local_result.displacement
                mesh_valid=local_result.valid_mask
            elif not local_enabled and local_xyz is not None \
                    and current_frame_number%local_mesh_update_interval==0:
                mesh_grid=local_xyz
                mesh_depth=np.zeros(np.asarray(mesh_grid).shape[:2],np.float32)
                mesh_valid=None
            else:
                mesh_grid=mesh_depth=mesh_valid=None
            if surface_mesh_vis is not None and mesh_grid is not None:
                if not surface_mesh_vis.update(
                        mesh_grid,mesh_depth,valid_mask=mesh_valid):
                    break
            # 帧率统计在 GPU LSMR 同步回传和 Open3D mesh 更新之后结束，
            # 因而不再只反映 SAM/光场前半段耗时。
            instantaneous_fps=1/max(time.perf_counter()-started,1e-9)
            display_fps=(instantaneous_fps if display_fps==0 else
                         .85*display_fps+.15*instantaneous_fps)
            if steady_warmup_remaining>0:
                steady_warmup_remaining-=1
                performance_window_started=time.perf_counter()
                performance_window_frames=0
            else:
                performance_window_frames+=1
                performance_window_fps=(
                    performance_window_frames/max(
                        time.perf_counter()-performance_window_started,1e-9))
            bias_deviation_values=bias_values-lightfield_bias_prior_values
            oracle_rmse,_oracle_scores=uniform_diagnostics
            if output is not None:
                cv2.putText(
                    output,f"JAX/{device.platform} rms={rms:.2f}px",(12,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                cv2.putText(
                    output,
                    f"gain RGB={gain_values[0]:.3f}/{gain_values[1]:.3f}/"
                    f"{gain_values[2]:.3f}",(12,52),
                    cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),2)
                cv2.putText(
                    output,
                    f"bias dev RGB={bias_deviation_values[0]:+.3f}/"
                    f"{bias_deviation_values[1]:+.3f}/"
                    f"{bias_deviation_values[2]:+.3f}",
                    (12,76),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),2)
                for channel,(name,y) in enumerate(zip(
                        ("R","G","B"),(100,124,148))):
                    m_scores="/".join(
                        f"{value:+.2f}" for value in
                        residual_scores[channel,1:])
                    cv2.putText(
                        output,
                        f"uniform {name} B={residual_scores[channel,0]:+.2f} "
                        f"M={m_scores}",
                        (12,y),cv2.FONT_HERSHEY_SIMPLEX,.50,(0,255,0),2)
                if local_summary_values is not None:
                    (local_min,local_max,_,local_iterations,_,_,pq_zero_count,
                     dprior_zero_count,_,_local_valid_count)=\
                        local_summary_values
                    cv2.putText(
                        output,
                        f"local d={local_min:+.3f}..{local_max:+.3f} mm "
                        f"{('PCG' if local_linear_solver=='spectral_pcg' else 'LSMR')}="
                        f"{int(local_iterations)} PQ0={int(pq_zero_count)} "
                        f"Dprior0={int(dprior_zero_count)}",
                        (12,172),cv2.FONT_HERSHEY_SIMPLEX,.50,(0,255,0),2)
            residual_rmse=residual_stage_rmse[2]
            if difference is not None:
                difference=np.array(difference,copy=True)
                cv2.putText(
                    difference,
                    f"RMSE RGB={residual_rmse[0]:.4f}/{residual_rmse[1]:.4f}/"
                    f"{residual_rmse[2]:.4f}",
                    (12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),2)
            diagnostic_time=time.monotonic()
            if frames is None and diagnostic_time-last_residual_diagnostic \
                    >=residual_diagnostic_interval \
                    and performance_window_frames>=10:
                field_names=["Bsession",*(
                    f"M{index+1}" for index in range(residual_m_count))]
                def format_coefficients(values):
                    return ",".join(
                        f"{value:.3f}" if index==0 else f"{value:+.3f}"
                        for index,value in enumerate(values))
                matrix_text="; ".join(
                    f"{name}:A=[{format_coefficients(row)}]"
                    for name,row in zip(("R","G","B"),residual_scores))
                local_text=""
                if local_summary_values is not None:
                    (local_min,local_max,_,local_iterations,_,_,pq_zero_count,
                     dprior_zero_count,d_zero_count,
                     local_valid_count)=\
                        local_summary_values
                    denominator=max(local_valid_count,1)
                    local_text=(
                        f"；local d={local_min:+.3f}..{local_max:+.3f} mm，"
                        f"iterations={local_iterations}，"
                        f"PQ0={pq_zero_count}/{local_valid_count}"
                        f" ({pq_zero_count/denominator:.1%})，"
                        f"Dprior0={dprior_zero_count}/{local_valid_count}"
                        f" ({dprior_zero_count/denominator:.1%})，"
                        f"D0={d_zero_count}/{local_valid_count}"
                        f" ({d_zero_count/denominator:.1%})")
                print(
                    f"runtime {performance_window_fps:.1f} FPS；"
                    f"SAM 每 {sam_frame_interval} 帧更新；"
                    f"残差拟合 stride={residual_fit_pixel_stride}；"
                    "uniform 残差诊断：RMSE RGB "
                    f"raw={residual_stage_rmse[0].tolist()}，"
                    f"Bsession={residual_stage_rmse[1].tolist()}，"
                    f"Bsession+M={residual_stage_rmse[2].tolist()}；"
                    f"A[{','.join(field_names)}] {matrix_text}"
                    f"{local_text}")
                last_residual_diagnostic=diagnostic_time
                performance_window_started=time.perf_counter()
                performance_window_frames=0

            if frames is not None:
                path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),output)
                cv2.imwrite(
                    str(path.with_name(path.stem+"_physical_plus_residual.png")),
                    physical_plus_residual_output)
                cv2.imwrite(str(path.with_name(path.stem+"_mask.png")),valid)
                cv2.imwrite(str(path.with_name(path.stem+"_original.png")),masked_original)
                cv2.imwrite(str(path.with_name(path.stem+"_difference.png")),difference)
                cv2.imwrite(str(path.with_name(path.stem+"_difference_mask.png")),difference_valid)
                for spec,index in zip(
                        enabled_oracle_specs,displayed_oracle_indices,strict=True):
                    cv2.imwrite(str(path.with_name(
                        path.stem+f"_oracle_{spec[0]}_difference.png")),
                        oracle_differences[index])
                np.save(path.with_name(path.stem+"_irls_weights.npy"),np.asarray(weights))
                np.save(path.with_name(path.stem+"_residual_correction_scores.npy"),
                        residual_scores)
                np.save(path.with_name(path.stem+"_lightfield_gain_prior.npy"),
                        lightfield_gain_prior_values)
                np.save(path.with_name(path.stem+"_lightfield_bias_prior.npy"),
                        lightfield_bias_prior_values)
                np.save(path.with_name(path.stem+"_residual_rmse_rgb.npy"),
                        residual_rmse)
                np.save(path.with_name(path.stem+"_residual_stage_rmse_rgb.npy"),
                        residual_stage_rmse)
                if enabled_oracle_methods:
                    np.save(
                        path.with_name(path.stem+"_oracle_scores.npy"),
                        _oracle_scores)
                    np.save(
                        path.with_name(path.stem+"_oracle_rmse_rgb.npy"),
                        oracle_rmse)
                if local_result is not None:
                    local_path=path.with_name(
                        path.stem+"_local_reconstruction.npz")
                    local_result.save(local_path)
                print(f"已输出 {path}、物理+残差拟合场、有效掩膜和 IRLS 权重；"
                      f"gain={gain_values.tolist()} bias={bias_values.tolist()} "
                      f"gain_ref={lightfield_gain_prior_values.tolist()} "
                      f"bias_ref={lightfield_bias_prior_values.tolist()} "
                      f"residual_matrix_rgb_by_bsession_ms={residual_scores.tolist()} "
                      f"rmse_raw_bsession_bsession_m_rgb={residual_stage_rmse.tolist()}")
                break
            if show_runtime_views(
                masked_original,output,physical_plus_residual_output,
                difference,oracle_differences): break
    finally:
        if surface_mesh_vis is not None:
            surface_mesh_vis.close()
        if cap is not None: cap.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
