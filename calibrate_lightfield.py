"""使用 JAX 自动微分离线标定一条或多条独立线光源。"""
from __future__ import annotations
import argparse
import gc
import glob
import hashlib
import json
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
                         save_generated_uv_xyz_map,torch_tensor_to_jax)
from utils.config import (parse_background_method,
                          parse_direct_fit_config,
                          parse_geometry_cache_config,
                          parse_reconstruction_config,
                          resolve_background_model_path)
from utils.gpu_residual_fit import (
    fit_direct_geometry_conditioned_field_gpu,
    fit_robust_static_background_gpu,
    fit_residual_correction_model_gpu)
from utils.jax_reconstruction import (
    SURFACE_RECONSTRUCTION_PIPELINE_VERSION,
    prepare_edge_curves_from_masks_jax,reconstruct_surface_from_masks_jax)
from utils.lightfield import (LightFieldModel, LightSourceLayout,
                              bounded_mixing_matrix,
                              bgr_to_linear_rgb_jax,
                              build_canonical_residual_sample_jax, choose_device,
                              direct_background_field_chunked,
                              evaluate_rgb_bspline,
                              fit_rgb_bspline_field_gpu,
                              geometry_cache_descriptor_jax,
                              geometry_cache_background_field_jax,
                              fit_uniform_huber_residual_correction_scores_jax,
                              fit_uniform_residual_correction_scores_jax,
                              irls_gain_bias, physical_background_batch, point_set_to_grid,
                              light_source_specs, parse_light_source_layout,
                              rasterize_attributes_jax,
                              sample_rgb, sample_unsaturated_mask)
from utils.process import EdgeReconstructor, ReconstructionPointSet
from utils.sam2_surface import SurfaceSegmenter


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


CALIBRATION_OBSERVATION_FORMAT_VERSION = 2
CALIBRATION_RECONSTRUCTION_PIPELINE = SURFACE_RECONSTRUCTION_PIPELINE_VERSION


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
    *,model_id: str,prompts: dict[object,object],mask_refine: object,
    center_band_d: float,reconstruction: object,
) -> str:
    """覆盖所有会改变离线 XYZ/UV/depth 的输入和算法语义。"""
    payload={
        "pipeline":CALIBRATION_RECONSTRUCTION_PIPELINE,
        "model_id":model_id,
        "prompts":_normalise_prompt_signature(prompts),
        "mask_refine":{
            "enabled":bool(mask_refine.enabled),
            "close_kernel":int(mask_refine.close_kernel),
            "open_kernel":int(mask_refine.open_kernel),
            "blur_kernel":int(mask_refine.blur_kernel),
            "keep_largest":bool(mask_refine.keep_largest),
        },
        "center_band_d":float(center_band_d),
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
) -> dict[str,object]:
    source_path,source_size,source_mtime_ns=_source_identity(image_path)
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
            rotation=np.asarray(
                data["reconstruction_rotation_vector"],np.float64).reshape(3)
            tx=float(data["reconstruction_tx"])
            if not np.isfinite(rotation).all() or not np.isfinite(tx):
                return None
            return rotation,tx
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
    output_dir.mkdir(parents=True,exist_ok=True)
    extracted_paths: list[Path] = []
    for video_number,video_path in enumerate(video_paths,1):
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频扩展名: {video_path}")
        prefix=f"video_{video_number:03d}_{video_path.stem}_frame_"
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
    np.savez_compressed(output,**fields)
    filter_description=(f"original threshold<{saturation_threshold}"
                        if filter_original_saturation else "image bounds only")
    print(f"观测 {image_path.name}: 有效 {valid_count}/{total_count} "
          f"({100*valid_count/total_count:.1f}%, {filter_description}) -> {output}")
    return output

def reconstruct_all_observations(image_paths: list[Path], all_config: dict,
                                 config_path: Path, output_dir: Path,
                                 map_dir: Path,saturation_threshold: int = 250,
                                 *,filter_original_saturation: bool = False,
                                 reuse_existing: bool = True) -> list[Path]:
    """对每个图片或视频帧执行 SAM2 分割、全局重建、映射和 RGB 采样。"""
    output_dir.mkdir(parents=True,exist_ok=True)
    map_dir.mkdir(parents=True,exist_ok=True)
    surface=all_config["get_surface"]
    prompts=parse_prompts(surface["prompts"])
    mask_refine=parse_mask_refine(surface.get("mask_refine"))
    center_band_d=float(surface.get("center_band_d",40))
    calibration_output=all_config.get("calibration",{}).get("output")
    reconstruction=parse_reconstruction_config(
        surface.get("reconstruction"),config_path=config_path,
        calibration_output=calibration_output)
    signature=_calibration_reconstruction_signature(
        model_id=surface["model"],prompts=prompts,mask_refine=mask_refine,
        center_band_d=center_band_d,reconstruction=reconstruction)
    expected_shape=(reconstruction.geometry_rows,
                    reconstruction.geometry_columns)
    pending: list[tuple[int,Path]]=[]
    outputs: list[Path | None]=[None]*len(image_paths)
    reused_poses: list[tuple[np.ndarray,float]]=[]
    reused=0
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
    if reused:
        print(f"复用已有观测 {reused}/{len(image_paths)}，待重建 {len(pending)}")
    elif reuse_existing and any(
            (output_dir/f"{path.stem}.npz").exists() for path in image_paths):
        print("已有观测不含当前实时 JAX/凸性重建指纹；旧缓存将自动重建")
    if not pending:
        return [path for path in outputs if path is not None]

    print(f"加载 SAM2 全局曲面分割模型: {surface['model']}")
    segmenter=SurfaceSegmenter(model_id=surface["model"],mask_refine=mask_refine)
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

    def gpu_mask_kernel(size: int) -> int:
        if not mask_refine.enabled or size<=0:
            return 0
        return size if size%2 else size+1

    close_kernel=gpu_mask_kernel(mask_refine.close_kernel)
    open_kernel=gpu_mask_kernel(mask_refine.open_kernel)
    blur_kernel=gpu_mask_kernel(mask_refine.blur_kernel)
    prepare_curves_gpu=jax.jit(lambda masks:prepare_edge_curves_from_masks_jax(
        masks,camera_matrix_gpu,distortion_gpu,reconstructor.sample_count,
        center_band_d,close_kernel=close_kernel,open_kernel=open_kernel,
        blur_kernel=blur_kernel)[1:])
    reconstruct_geometry_gpu=jax.jit(
        lambda masks,rotation,tx:reconstruct_surface_from_masks_jax(
            masks,camera_matrix_gpu,distortion_gpu,inverse_camera_gpu,
            rotation,reconstruction.s1,reconstruction.s2,tx,
            reconstructor.sample_count,center_band_d,
            reconstruction.pair_fill_count,
            reconstruction.uv_boundary_smooth_lambda,
            reconstruction.uv_boundary_huber_delta_px,
            curve_convexity=reconstruction.curve_convexity,
            close_kernel=close_kernel,open_kernel=open_kernel,
            blur_kernel=blur_kernel))
    try:
        for done,(index,image_path) in enumerate(pending,1):
            frame=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
            if frame is None: raise RuntimeError(f"无法读取标定观测帧: {image_path}")
            # 每帧是一项独立标定观测，禁止由上一帧的 mask memory 改变样本定义。
            segmenter.reset()
            labels,mask_tensor,_=segmenter.segment_tensors(frame,prompts)
            mask_gpu=jax.device_put(torch_tensor_to_jax(mask_tensor),device)
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
                    print(
                        f"跳过第 {index+1} 个观测帧：无法初始化固定外参 "
                        f"{image_path}")
                    continue
            rotation=cv2.Rodrigues(
                reconstructor.rotation_vector)[0].astype(np.float32)
            (refined_masks,xyz_grid,uv_grid,st_grid,depth_grid,rms_values,
             reconstruction_valid)=jax.device_get(reconstruct_geometry_gpu(
                 mask_gpu,jax.device_put(rotation,device),
                 jax.device_put(np.asarray(reconstructor.tx,np.float32),device)))
            del refined_masks
            if not bool(np.all(reconstruction_valid)):
                invalid=np.flatnonzero(~np.asarray(reconstruction_valid,np.bool_))
                print(
                    f"跳过第 {index+1} 个观测帧：实时 JAX/凸性重建无效，"
                    f"surface={invalid.tolist()}，{image_path}")
                continue
            point_set=point_set_from_surface_grids(
                xyz_grid,uv_grid,st_grid,depth_grid,reconstruction.K,
                reconstruction.distortion_coefficients,
                surface_count=len(labels),
                surface_rows=reconstruction.geometry_rows)
            metadata=_observation_metadata(
                image_path,signature,reconstruction.curve_convexity,
                reconstructor.rotation_vector,reconstructor.tx,rms_values)
            outputs[index]=save_calibration_observation(
                image_path,frame,point_set,output_dir,map_dir,saturation_threshold,
                filter_original_saturation=filter_original_saturation,
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
    saturation_threshold: int,erode_pixels: int,raster_triangle_chunk: int,
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
            saturation_threshold=saturation_threshold,
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


def _farthest_geometry_anchor_indices(
    keys: np.ndarray,count: int,
) -> np.ndarray:
    """确定性 farthest-point 采样，覆盖几何键空间而不依赖采集顺序。"""
    values=np.asarray(keys,np.float64)
    if values.ndim!=2 or not 1<=count<=values.shape[0] \
            or not np.isfinite(values).all():
        raise ValueError("geometry cache anchor 输入无效")
    center=values.mean(axis=0)
    first=int(np.argmax(np.sum((values-center)**2,axis=1)))
    selected=[first]
    minimum_distance=np.sum((values-values[first])**2,axis=1)
    minimum_distance[first]=-np.inf
    while len(selected)<count:
        current=int(np.argmax(minimum_distance))
        selected.append(current)
        minimum_distance=np.minimum(
            minimum_distance,np.sum((values-values[current])**2,axis=1))
        minimum_distance[np.asarray(selected,np.int64)]=-np.inf
    return np.asarray(selected,np.int64)


def _calibrate_geometry_cache(
    *,raw: dict,cfg: dict,reconstruction: object,device: jax.Device,
    source_layout: LightSourceLayout,source_images: list[Path],
    uv_parts: list[np.ndarray],depth_parts: list[np.ndarray],xyz_all: np.ndarray,
    training_indices: np.ndarray,validation_indices: np.ndarray,
    model_output: Path,
) -> None:
    """标定独立的鲁棒几何锚点背景缓存与最近邻凸插值。"""
    cache=parse_geometry_cache_config(raw)
    runtime=raw.get("runtime",{})
    sample_shape=(reconstruction.observation_rows,
                  reconstruction.observation_columns)
    canonical,canonical_valid=_collect_direct_canonical_fields(
        source_images,uv_parts,depth_parts,sample_shape=sample_shape,
        saturation_threshold=cache.sample_saturation_threshold,
        erode_pixels=cache.sample_erode_pixels,
        raster_triangle_chunk=int(
            runtime.get("gpu_raster_triangle_chunk",256)),
        raster_max_width=max(int(
            runtime.get("gpu_raster_max_triangle_width",24)),64),
        raster_max_height=max(int(
            runtime.get("gpu_raster_max_triangle_height",12)),32),
        device=device)
    training_count=training_indices.size
    if training_count<3:
        raise ValueError("geometry_cache 至少需要 3 个训练样本")
    if cache.anchor_count>training_count:
        raise ValueError(
            "geometry_cache anchor count 不能大于训练样本数")
    if cache.anchor_neighbor_count>training_count:
        raise ValueError(
            "geometry_cache anchor neighbor_count 不能大于训练样本数")
    if cache.descriptor_curve_coefficients>xyz_all.shape[1]:
        raise ValueError(
            "geometry_cache curve_coefficients 不能大于几何网格行数")

    descriptor_one=jax.jit(lambda xyz:geometry_cache_descriptor_jax(
        xyz,cache.descriptor_curve_coefficients,
        cache.descriptor_huber_delta_mm))
    descriptor_parts=[]
    for start in range(0,training_count,cache.fit_batch_size):
        indices=training_indices[start:start+cache.fit_batch_size]
        descriptor_parts.append(np.stack([
            np.asarray(descriptor_one(jax.device_put(xyz_all[index],device)))
            for index in indices]))
    descriptors=np.concatenate(descriptor_parts,axis=0).astype(np.float64)
    descriptor_mean=descriptors.mean(axis=0)
    descriptor_scale=np.maximum(descriptors.std(axis=0),1e-6)
    normalized=(descriptors-descriptor_mean)/descriptor_scale
    _,singular,components_t=np.linalg.svd(normalized,full_matrices=False)
    tolerance=max(normalized.shape)*np.finfo(np.float64).eps*max(
        float(singular[0]) if singular.size else 0.,1.)
    rank=int(np.count_nonzero(singular>tolerance))
    pca_dimensions=min(
        cache.descriptor_pca_dimensions,rank,training_count-1)
    if pca_dimensions<1:
        raise ValueError("geometry_cache 几何描述没有有效变化维度")
    pca_components=components_t[:pca_dimensions].T
    unscaled_keys=normalized@pca_components
    pca_scale=np.maximum(unscaled_keys.std(axis=0),1e-6)
    keys=unscaled_keys/pca_scale
    anchor_indices=_farthest_geometry_anchor_indices(
        keys,cache.anchor_count)
    anchor_keys=keys[anchor_indices]

    train_fields=canonical[training_indices]
    train_valid=canonical_valid[training_indices]
    base_texture=fit_robust_static_background_gpu(
        train_fields,train_valid,device=device,
        huber_delta=cache.background_huber_delta,
        iterations=cache.background_huber_iterations,
        frame_batch_size=cache.fit_batch_size,shared_rgb_weights=True)
    residual_rows=reconstruction.residual_coefficient_rows
    residual_columns=reconstruction.residual_coefficient_columns
    anchor_coefficients=[]
    print("geometry_cache 锚点拟合："
          f"anchors={cache.anchor_count}，neighbors="
          f"{cache.anchor_neighbor_count}，PCA={pca_dimensions}，"
          f"B-spline={residual_rows}x{residual_columns}")
    for position,anchor_index in enumerate(anchor_indices,1):
        distance=np.sum((keys-keys[anchor_index])**2,axis=1)
        neighbors=np.argsort(distance)[:cache.anchor_neighbor_count]
        local_texture=fit_robust_static_background_gpu(
            train_fields[neighbors],train_valid[neighbors],device=device,
            huber_delta=cache.background_huber_delta,
            iterations=cache.background_huber_iterations,
            frame_batch_size=cache.fit_batch_size,shared_rgb_weights=True)
        coverage=np.any(train_valid[neighbors],axis=0).astype(np.float32)
        coefficients=fit_rgb_bspline_field_gpu(
            local_texture-base_texture,coverage,residual_rows,residual_columns,
            float(cfg.get("lambda_residual_smooth",.001)),
            float(cfg.get("lambda_residual_magnitude",1e-4)),device=device)
        anchor_coefficients.append(coefficients)
        if position==1 or position%8==0 or position==cache.anchor_count:
            print(f"geometry_cache anchor {position}/{cache.anchor_count}")
    anchor_coefficients=np.stack(anchor_coefficients).astype(np.float32)
    session_correction=np.zeros(
        (3,residual_rows,residual_columns),np.float32)
    model=LightFieldModel.geometry_cache(
        session_correction,base_texture=base_texture,
        anchor_coefficients=anchor_coefficients,
        descriptor_mean=descriptor_mean.astype(np.float32),
        descriptor_scale=descriptor_scale.astype(np.float32),
        pca_components=pca_components.astype(np.float32),
        pca_scale=pca_scale.astype(np.float32),
        anchor_keys=anchor_keys.astype(np.float32),
        curve_coefficients=cache.descriptor_curve_coefficients,
        descriptor_huber_delta=cache.descriptor_huber_delta_mm,
        interpolation_neighbors=cache.interpolation_neighbor_count,
        distance_power=cache.interpolation_distance_power,
        distance_epsilon=cache.interpolation_distance_epsilon,
        curve_convexity=reconstruction.curve_convexity,
        reconstruction_pipeline=CALIBRATION_RECONSTRUCTION_PIPELINE,
        source_layout=source_layout)
    model_gpu=jax.device_put(model,device)
    evaluate=jax.jit(lambda xyz:geometry_cache_background_field_jax(
        sample_shape,xyz,model_gpu))

    def predictions(indices: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(evaluate(
            jax.device_put(xyz_all[index],device))) for index in indices])

    training_rmse=_valid_rmse(
        canonical[training_indices]-predictions(training_indices),
        canonical_valid[training_indices])
    validation_rmse=np.full(3,np.nan,np.float64)
    if validation_indices.size:
        validation_rmse=_valid_rmse(
            canonical[validation_indices]-predictions(validation_indices),
            canonical_valid[validation_indices])
    model.save(model_output)
    print("geometry_cache 已保存：train/validation RMSE RGB="
          f"{training_rmse.tolist()}/{validation_rmse.tolist()}，"
          f"model={model_output}")


def _calibrate_direct_fit(
    *,raw: dict,cfg: dict,reconstruction: object,device: jax.Device,
    background_method: str,
    source_layout: LightSourceLayout,source_images: list[Path],
    uv_parts: list[np.ndarray],depth_parts: list[np.ndarray],xyz_all: np.ndarray,
    training_indices: np.ndarray,validation_indices: np.ndarray,
    model_output: Path,
) -> None:
    """训练 direct 几何条件神经场，并报告同源留出集误差。"""
    if background_method not in {"direct_fit","direct_fit_3"}:
        raise ValueError("direct 标定收到无效 background_method")
    runtime=raw.get("runtime",{})
    direct=parse_direct_fit_config(raw)
    sample_shape=(reconstruction.observation_rows,
                  reconstruction.observation_columns)
    raster_triangle_chunk=int(runtime.get("gpu_raster_triangle_chunk",256))
    raster_max_width=max(
        int(runtime.get("gpu_raster_max_triangle_width",24)),64)
    raster_max_height=max(
        int(runtime.get("gpu_raster_max_triangle_height",12)),32)
    canonical,canonical_valid=_collect_direct_canonical_fields(
        source_images,uv_parts,depth_parts,sample_shape=sample_shape,
        saturation_threshold=direct.sample_saturation_threshold,
        erode_pixels=direct.sample_erode_pixels,
        raster_triangle_chunk=raster_triangle_chunk,
        raster_max_width=raster_max_width,raster_max_height=raster_max_height,
        device=device)
    if training_indices.size<3:
        raise ValueError("direct_fit 统一神经场至少需要 3 个训练样本")
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
            direct.adaptive_channel_weight_strength
            if background_method=="direct_fit_3" else 0.),
        spatial_difference_weight=direct.spatial_difference_weight,
        spatial_difference_validation_weight=(
            direct.spatial_difference_validation_weight),
        spatial_difference_points_per_frame=(
            direct.spatial_difference_points_per_frame),
        geometry_difference_weight=direct.geometry_difference_weight,
        geometry_difference_validation_weight=(
            direct.geometry_difference_validation_weight),
        geometry_difference_neighbor_count=(
            direct.geometry_difference_neighbor_count),
        geometry_difference_points_per_pair=(
            direct.geometry_difference_points_per_pair),
        seed=int(cfg.get("physical_seed",0)),
        validation_interval=direct.validation_interval,
        validation_points_per_frame=direct.validation_points_per_frame,
        early_stopping_patience=direct.early_stopping_patience,
        early_stopping_min_steps=direct.early_stopping_min_steps,
        early_stopping_min_delta=direct.early_stopping_min_delta,
        separate_channel_decoders=(background_method=="direct_fit_3")))
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
    if background_method=="direct_fit_3":
        model=LightFieldModel.direct_fit_3(
            session_correction,
            channel_decoder_weights=decoder_weights,
            channel_decoder_biases=decoder_biases,
            **common_model_arguments)
    else:
        model=LightFieldModel.direct_fit(
            session_correction,decoder_weights=decoder_weights,
            decoder_biases=decoder_biases,**common_model_arguments)

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

    model.save(model_output)
    print(f"{background_method} 已保存 B + delta B + Bsession 模型："
          f"B-only train/validation RMSE RGB="
          f"{training_base_rmse.tolist()}/{validation_base_rmse.tolist()}，"
          f"B+deltaB train/validation RMSE RGB="
          f"{training_rmse.tolist()}/{validation_rmse.tolist()}")
    print(f"标定完成（background_method={background_method}，JAX device={device}）："
          f"{model_output}")

def main() -> None:
    parser=argparse.ArgumentParser(description="JAX 离线标定无局部形变背景光场")
    parser.add_argument("--config",default=Path(__file__).with_name("config.yaml")); args=parser.parse_args()
    config_path=Path(args.config).expanduser(); all_config=yaml.safe_load(config_path.read_text(encoding="utf-8")); raw=all_config["lightfield"]
    cfg=raw["calibration"]
    background_method=parse_background_method(raw)
    model_output=resolve_background_model_path(
        raw,method=background_method,base=config_path.parent)
    surface=all_config["get_surface"]
    reconstruction=parse_reconstruction_config(
        surface.get("reconstruction"),config_path=config_path,
        calibration_output=all_config.get("calibration",{}).get("output"))
    source_layout=parse_light_source_layout(raw.get("light_source_layout"))
    output_dir=Path(cfg.get("sample_output_dir","assets/lightfield_calibration")).expanduser()
    if not output_dir.is_absolute(): output_dir=config_path.parent/output_dir
    image_paths=_resolve_optional_paths(
        cfg.get("images"),config_path.parent,"lightfield.calibration.images")
    video_paths=_resolve_optional_paths(
        cfg.get("videos"),config_path.parent,"lightfield.calibration.videos")
    if not image_paths and not video_paths:
        raise ValueError(
            "lightfield.calibration.images 和 videos 至少需要配置一项")
    video_frame_step=cfg.get("video_frame_step",1)
    video_max_frames=cfg.get("video_max_frames_per_file")
    reuse_existing=bool(cfg.get("reuse_existing",True))
    video_frame_dir=Path(
        cfg.get("video_frame_output_dir",output_dir/"video_frames")
    ).expanduser()
    if not video_frame_dir.is_absolute():
        video_frame_dir=config_path.parent/video_frame_dir
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
    print(f"标定输入: 图片 {len(image_paths)} 张，视频帧 {len(video_frame_paths)} 张"
          f"（reuse_existing={reuse_existing}）")
    paths=reconstruct_all_observations(
        observation_image_paths,all_config,config_path,output_dir,map_dir,
        saturation_threshold,filter_original_saturation=False,
        reuse_existing=reuse_existing)
    device=choose_device(raw.get("device","gpu"))
    xyz_parts=[]; uv_parts=[]; st_parts=[]; depth_parts=[]; rgb_parts=[]; valid_parts=[]; source_images=[]
    for path in paths:
        with np.load(path) as data:
            xyz_parts.append(data["xyz"]); uv_parts.append(data["uv"])
            if "st" not in data:
                raise ValueError(f"标定样本缺少 st，请重新生成: {path}")
            st_parts.append(data["st"])
            if "camera_depth" not in data:
                raise ValueError(f"标定样本缺少 camera_depth，请重新生成: {path}")
            depth_parts.append(data["camera_depth"])
            rgb_parts.append(data["rgb"]); valid_parts.append(data["valid_mask"])
            source_images.append(Path(str(data["source_image"])))
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
        observed=np.asarray(np.stack(rgb_parts),dtype=np.float32)
        valid=np.asarray(np.stack(valid_parts),dtype=np.bool_)
        # 当前物理模型只消费 XYZ；仍在这里堆叠 ST，以同步校验所有样本的网格结构。
        np.stack(st_parts)
    except ValueError as error:
        raise ValueError("所有标定样本必须使用相同的曲面网格尺寸") from error
    del xyz_parts,rgb_parts,valid_parts,st_parts
    print(
        "标定观测已由实时 JAX 重建链生成："
        f"convexity={reconstruction.curve_convexity}；"
        "XYZ/UV/depth 无事后补投影")
    validation_fraction=float(cfg.get("validation_fraction",0.))
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
    if background_method=="geometry_cache":
        _calibrate_geometry_cache(
            raw=raw,cfg=cfg,reconstruction=reconstruction,device=device,
            source_layout=source_layout,source_images=source_images,
            uv_parts=uv_parts,depth_parts=depth_parts,xyz_all=xyz,
            training_indices=training_indices,
            validation_indices=validation_indices,
            model_output=model_output)
        return
    if background_method in {"direct_fit","direct_fit_3"}:
        _calibrate_direct_fit(
            raw=raw,cfg=cfg,reconstruction=reconstruction,device=device,
            background_method=background_method,
            source_layout=source_layout,source_images=source_images,
            uv_parts=uv_parts,depth_parts=depth_parts,xyz_all=xyz,
            training_indices=training_indices,
            validation_indices=validation_indices,
            model_output=model_output)
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
