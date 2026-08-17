"""GPU 分块拟合离线光场残差 B/M，避免构造全量逐像素正规方程。"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.lightfield import (
    _fit_bspline_fields_pcg_jax,
    bspline_basis,
    direct_background_features_jax,
    direct_geometry_descriptor_jax,
    direct_local_geometry_feature_grid_jax,
    rgb_bspline_field,
    sample_direct_base_texture_jax,
    sample_direct_local_geometry_feature_grid_jax,
)


Array = jax.Array


def _positive_integer(name: str,value: int) -> int:
    if not isinstance(value,int) or isinstance(value,bool) or value<1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _adaptive_channel_weights(
    channel_mean_squared_error: Array,strength: float,
) -> Array:
    """按通道 RMSE 分配均值为 1 的权重，并阻断权重本身的梯度。"""
    rmse=jnp.sqrt(jnp.maximum(channel_mean_squared_error,0)+1e-12)
    relative=rmse/jnp.maximum(jnp.mean(rmse),1e-12)
    weights=1+strength*(relative-1)
    return jax.lax.stop_gradient(weights)


def _save_direct_fit_checkpoint(
    path: str | Path,*,step: int,validation_rmse_rgb: np.ndarray,
    validation_difference_rmse_rgb: np.ndarray,
    validation_spatial_difference_rmse_rgb: np.ndarray,
    base_texture: np.ndarray,
    frequencies: np.ndarray,geometry_descriptor_rows: int,
    geometry_feature_mean: np.ndarray,geometry_feature_scale: np.ndarray,
    geometry_pca_components: np.ndarray,geometry_pca_scale: np.ndarray,
    local_feature_mean: np.ndarray,local_feature_scale: np.ndarray,
    parameters: object,moments: object,variances: object,
    separate_channel_decoders: bool = False,
) -> None:
    """原子覆盖 direct_fit 的最佳模型/优化器 checkpoint。"""
    output=Path(path).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True)
    host_parameters,host_moments,host_variances=jax.device_get(
        (parameters,moments,variances))
    encoder,decoder=host_parameters
    encoder_m,decoder_m=host_moments
    encoder_v,decoder_v=host_variances
    payload: dict[str,np.ndarray]={
        "checkpoint_format_version":np.asarray(
            4 if separate_channel_decoders else 3,np.int32),
        "step":np.asarray(step,np.int32),
        "validation_rmse_rgb":np.asarray(validation_rmse_rgb,np.float32),
        "validation_difference_rmse_rgb":np.asarray(
            validation_difference_rmse_rgb,np.float32),
        "validation_spatial_difference_rmse_rgb":np.asarray(
            validation_spatial_difference_rmse_rgb,np.float32),
        "direct_base_texture":np.asarray(base_texture,np.float32),
        "coordinate_frequencies":np.asarray(frequencies,np.float32),
        "geometry_descriptor_rows":np.asarray(
            geometry_descriptor_rows,np.int32),
        "geometry_feature_mean":np.asarray(geometry_feature_mean,np.float32),
        "geometry_feature_scale":np.asarray(geometry_feature_scale,np.float32),
        "geometry_pca_components":np.asarray(
            geometry_pca_components,np.float32),
        "geometry_pca_scale":np.asarray(geometry_pca_scale,np.float32),
        "local_geometry_feature_mean":np.asarray(local_feature_mean,np.float32),
        "local_geometry_feature_scale":np.asarray(local_feature_scale,np.float32),
        "encoder_layer_count":np.asarray(len(encoder),np.int32),
    }
    for prefix,layers in (
            ("encoder",encoder),("encoder_moment",encoder_m),
            ("encoder_variance",encoder_v)):
        for index,(weight,bias) in enumerate(layers):
            payload[f"{prefix}_weight_{index}"]=np.asarray(weight,np.float32)
            payload[f"{prefix}_bias_{index}"]=np.asarray(bias,np.float32)
    if separate_channel_decoders:
        payload["channel_decoder_count"]=np.asarray(3,np.int32)
        payload["channel_decoder_layer_count"]=np.asarray(
            len(decoder[0]),np.int32)
        for channel in range(3):
            for prefix,layers in (
                    (f"decoder_{channel}",decoder[channel]),
                    (f"decoder_{channel}_moment",decoder_m[channel]),
                    (f"decoder_{channel}_variance",decoder_v[channel])):
                for index,(weight,bias) in enumerate(layers):
                    payload[f"{prefix}_weight_{index}"]=np.asarray(
                        weight,np.float32)
                    payload[f"{prefix}_bias_{index}"]=np.asarray(
                        bias,np.float32)
    else:
        payload["decoder_layer_count"]=np.asarray(len(decoder),np.int32)
        for prefix,layers in (
                ("decoder",decoder),("decoder_moment",decoder_m),
                ("decoder_variance",decoder_v)):
            for index,(weight,bias) in enumerate(layers):
                payload[f"{prefix}_weight_{index}"]=np.asarray(
                    weight,np.float32)
                payload[f"{prefix}_bias_{index}"]=np.asarray(
                    bias,np.float32)
    temporary=output.with_name(output.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**payload)
    temporary.replace(output)


def _fit_direct_static_base_gpu(
    fields: np.ndarray,valid: np.ndarray,*,device: jax.Device,
    huber_delta: float,iterations: int,frame_batch_size: int,
    shared_rgb_weights: bool = False,
) -> np.ndarray:
    """在 GPU 上逐像素稳健拟合静态 B；只沿帧聚合，不做空间平滑。"""
    samples=np.asarray(fields,np.float32)
    mask=np.asarray(valid,np.bool_)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or mask.shape!=samples.shape[:3] or samples.shape[0]<2 \
            or huber_delta<=0 or iterations<1 or frame_batch_size<1:
        raise ValueError("direct 静态 B 输入或参数无效")
    shape=samples.shape[1:]

    @jax.jit
    def mean_statistics(batch,batch_valid):
        weight=batch_valid[...,None].astype(jnp.float32)
        return (jnp.sum(weight*batch,axis=0),jnp.sum(weight,axis=0))

    @jax.jit
    def robust_statistics(batch,batch_valid,center):
        error=batch-center[None]
        if shared_rgb_weights:
            magnitude=jnp.sqrt(jnp.mean(error**2,axis=-1,keepdims=True))
        else:
            magnitude=jnp.abs(error)
        weight=(batch_valid[...,None].astype(jnp.float32)
                *jnp.minimum(1.,huber_delta/jnp.maximum(magnitude,1e-12)))
        return (jnp.sum(weight*batch,axis=0),jnp.sum(weight,axis=0))

    numerator=jax.device_put(jnp.zeros(shape,jnp.float32),device)
    denominator=jax.device_put(jnp.zeros(shape,jnp.float32),device)
    for start in range(0,samples.shape[0],frame_batch_size):
        current_numerator,current_denominator=mean_statistics(
            jax.device_put(samples[start:start+frame_batch_size],device),
            jax.device_put(mask[start:start+frame_batch_size],device))
        numerator+=current_numerator; denominator+=current_denominator
    global_mean=jnp.sum(numerator,axis=(0,1)) \
        /jnp.maximum(jnp.sum(denominator,axis=(0,1)),1)
    base=jnp.where(
        denominator>0,numerator/jnp.maximum(denominator,1),
        jnp.broadcast_to(global_mean,shape))
    for iteration in range(iterations):
        numerator=jnp.zeros(shape,jnp.float32)
        denominator=jnp.zeros(shape,jnp.float32)
        for start in range(0,samples.shape[0],frame_batch_size):
            current_numerator,current_denominator=robust_statistics(
                jax.device_put(samples[start:start+frame_batch_size],device),
                jax.device_put(mask[start:start+frame_batch_size],device),base)
            numerator+=current_numerator; denominator+=current_denominator
        base=jnp.where(
            denominator>0,numerator/jnp.maximum(denominator,1),base)
        base.block_until_ready()
        print(f"direct 静态 B GPU Huber {iteration+1}/{iterations}")
    result=np.asarray(jax.device_get(jnp.clip(base,0,1)),np.float32)
    print("direct 静态 B：无空间平滑，coverage="
          f"{float(np.mean(np.any(mask,axis=0))):.1%}")
    return result


def fit_robust_static_background_gpu(
    fields: np.ndarray,valid: np.ndarray,*,device: jax.Device,
    huber_delta: float,iterations: int,frame_batch_size: int,
    shared_rgb_weights: bool = False,
) -> np.ndarray:
    """公开的逐像素分通道 Huber 背景聚合，供 direct/cache 模式复用。"""
    return _fit_direct_static_base_gpu(
        fields,valid,device=device,huber_delta=huber_delta,
        iterations=iterations,frame_batch_size=frame_batch_size,
        shared_rgb_weights=shared_rgb_weights)


def _surface_curvature_scores_gpu(
    surface_xyz: np.ndarray,
    feature_count: int,
    *,
    curve_coefficients: int,
    smooth_lambda: float,
    device: jax.Device,
) -> Array:
    """在 GPU 上从平滑 XYZ 中心线提取单位 RMS 曲率主分量。"""
    xyz=np.asarray(surface_xyz)
    if xyz.ndim!=4 or xyz.shape[-1]!=3 or min(xyz.shape[1:3])<2 \
            or not np.isfinite(xyz).all():
        raise ValueError("surface_xyz 必须是有限的 NxHxWx3 规则曲面")
    if feature_count<1 or feature_count>=xyz.shape[0]:
        raise ValueError("曲率特征数必须至少为 1 且小于样本数")
    if curve_coefficients<4 or curve_coefficients>xyz.shape[1]:
        raise ValueError("中心线 B 样条系数数必须位于 [4, 曲面行数]")
    if smooth_lambda<0:
        raise ValueError("中心线曲率平滑强度不能为负")

    # 先在 CPU 沿短的列方向取均值，只传 NxRowsx3 中心线而不是完整 XYZ。
    centerline=jax.device_put(
        np.mean(xyz,axis=2,dtype=np.float32),device)
    row_count=centerline.shape[1]
    basis=jax.device_put(bspline_basis(
        jnp.linspace(0.,1.,row_count),curve_coefficients),device)
    sample_second=jax.device_put(
        np.diff(np.eye(row_count,dtype=np.float32),n=2,axis=0),device)
    second_design=sample_second@basis
    normal=basis.T@basis/row_count
    normal+=smooth_lambda*(second_design.T@second_design) \
        /max(second_design.shape[0],1)
    normal+=1e-7*jnp.eye(curve_coefficients,dtype=jnp.float32)
    rhs=jnp.einsum("hr,nhd->rnd",basis,centerline)/row_count
    coefficients=jnp.linalg.solve(
        normal,rhs.reshape(curve_coefficients,-1)).reshape(
            curve_coefficients,xyz.shape[0],3).transpose(1,0,2)
    smoothed=jnp.einsum("hr,nrd->nhd",basis,coefficients)
    previous=smoothed[:,1:-1]-smoothed[:,:-2]
    following=smoothed[:,2:]-smoothed[:,1:-1]
    tangent=smoothed[:,2:]-smoothed[:,:-2]
    tangent/=jnp.maximum(jnp.linalg.norm(tangent,axis=-1,keepdims=True),1e-9)
    second=following-previous
    normal_second=second-jnp.sum(
        second*tangent,axis=-1,keepdims=True)*tangent
    local_spacing=.5*(jnp.linalg.norm(previous,axis=-1)
                      +jnp.linalg.norm(following,axis=-1))
    curvature=normal_second/jnp.maximum(local_spacing[...,None]**2,1e-12)
    features=curvature.reshape(xyz.shape[0],-1)
    features-=features.mean(axis=0,keepdims=True)
    scale=jnp.sqrt(jnp.mean(features**2,axis=0))
    scale_value=np.asarray(scale)
    active=np.flatnonzero(
        scale_value>max(float(np.max(scale_value))*1e-6,1e-12))
    if active.size<feature_count:
        raise ValueError("XYZ 曲率变化的有效维数小于请求的曲率特征数")
    features=features[:,jax.device_put(active.astype(np.int32),device)]
    u,singular,_=jnp.linalg.svd(features,full_matrices=False)
    scores=u[:,:feature_count]*singular[:feature_count]
    score_rms=jnp.maximum(jnp.sqrt(jnp.mean(scores**2,axis=0)),1e-12)
    return (scores/score_rms[None]).astype(jnp.float32)


def fit_direct_geometry_conditioned_field_gpu(
    fields: np.ndarray,valid: np.ndarray,*,surface_xyz: np.ndarray,
    validation_fields: np.ndarray | None = None,
    validation_valid: np.ndarray | None = None,
    validation_surface_xyz: np.ndarray | None = None,
    checkpoint_path: str | Path | None = None,
    device: jax.Device,
    frequencies: tuple[float,...] = (1.,2.,4.,8.,16.,32.),
    geometry_descriptor_rows: int = 24,
    geometry_encoder_width: int = 192,geometry_encoder_layers: int = 3,
    geometry_latent_dimensions: int = 96,
    geometry_pca_dimensions: int = 32,
    decoder_width: int = 192,decoder_layers: int = 5,
    steps: int = 4000,batch_size: int = 16384,frame_batch_size: int = 16,
    learning_rate: float = 1e-3,huber_delta: float = .04,
    seed: int = 0,
    base_huber_iterations: int = 5,
    adaptive_channel_weight_strength: float = 0.,
    spatial_difference_weight: float = 1.,
    spatial_difference_validation_weight: float = 1.,
    spatial_difference_points_per_frame: int = 1024,
    geometry_difference_weight: float = .25,
    geometry_difference_validation_weight: float = .25,
    geometry_difference_neighbor_count: int = 16,
    geometry_difference_points_per_pair: int = 512,
    validation_interval: int = 100,validation_points_per_frame: int = 512,
    early_stopping_patience: int = 10,early_stopping_min_steps: int = 1500,
    early_stopping_min_delta: float = 5e-5,
    separate_channel_decoders: bool = False,
) -> tuple[
    np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,
    np.ndarray,np.ndarray,
    tuple[np.ndarray,...],tuple[np.ndarray,...],
    tuple[np.ndarray,...],tuple[np.ndarray,...],
]:
    """联合训练“全局弯曲 + 逐点局部几何”条件的绝对 RGB 场。"""
    samples=np.asarray(fields,np.float32)
    mask=np.asarray(valid,np.bool_)
    surfaces=np.asarray(surface_xyz,np.float32)
    validation_inputs=(
        validation_fields,validation_valid,validation_surface_xyz)
    validation_enabled=all(value is not None for value in validation_inputs)
    if any(value is not None for value in validation_inputs) \
            and not validation_enabled:
        raise ValueError("direct validation fields/valid/surface_xyz 必须同时提供")
    validation_samples=(np.asarray(validation_fields,np.float32)
                        if validation_enabled else None)
    validation_mask=(np.asarray(validation_valid,np.bool_)
                     if validation_enabled else None)
    validation_surfaces=(np.asarray(validation_surface_xyz,np.float32)
                         if validation_enabled else None)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or mask.shape!=samples.shape[:3] or samples.shape[0]<2:
        raise ValueError("direct neural field 输入必须是 NxHxWx3/NxHxW")
    if surfaces.ndim!=4 or surfaces.shape[0]!=samples.shape[0] \
            or surfaces.shape[-1]!=3 or min(surfaces.shape[1:3])<3 \
            or not np.isfinite(surfaces).all():
        raise ValueError("surface_xyz 必须是与 direct 样本一一对应的 NxHxWx3")
    for name,value in (
            ("geometry_descriptor_rows",geometry_descriptor_rows),
            ("geometry_encoder_width",geometry_encoder_width),
            ("geometry_encoder_layers",geometry_encoder_layers),
            ("geometry_latent_dimensions",geometry_latent_dimensions),
            ("geometry_pca_dimensions",geometry_pca_dimensions),
            ("decoder_width",decoder_width),("decoder_layers",decoder_layers),
            ("steps",steps),("batch_size",batch_size),
            ("frame_batch_size",frame_batch_size),
            ("validation_interval",validation_interval),
            ("validation_points_per_frame",validation_points_per_frame),
            ("early_stopping_patience",early_stopping_patience),
            ("base_huber_iterations",base_huber_iterations),
            ("spatial_difference_points_per_frame",
             spatial_difference_points_per_frame),
            ("geometry_difference_neighbor_count",
             geometry_difference_neighbor_count),
            ("geometry_difference_points_per_pair",
             geometry_difference_points_per_pair)):
        _positive_integer(name,value)
    if geometry_descriptor_rows<4:
        raise ValueError("geometry_descriptor_rows 必须至少为 4")
    frequency_values=np.asarray(frequencies,np.float32)
    if frequency_values.ndim!=1 or frequency_values.size<1 \
            or not np.isfinite(frequency_values).all() \
            or np.any(frequency_values<=0):
        raise ValueError("direct neural field Fourier frequencies 必须是有限正数")
    if learning_rate<=0 or huber_delta<=0 \
            or not 0<=adaptive_channel_weight_strength<=1 \
            or spatial_difference_weight<0 \
            or spatial_difference_validation_weight<0 \
            or geometry_difference_weight<0 \
            or geometry_difference_validation_weight<0:
        raise ValueError("direct neural field 优化参数无效")
    if not isinstance(separate_channel_decoders,bool):
        raise ValueError("separate_channel_decoders 必须是布尔值")
    if adaptive_channel_weight_strength>0 and not separate_channel_decoders:
        raise ValueError("自适应通道 loss 权重仅支持独立通道 decoder")
    if validation_enabled and (not isinstance(early_stopping_min_steps,int) \
            or isinstance(early_stopping_min_steps,bool) \
            or not 0<=early_stopping_min_steps<=steps \
            or early_stopping_min_delta<0):
        raise ValueError("direct neural field 早停参数无效")
    if not np.any(mask):
        raise ValueError("direct neural field 没有有效样本")
    if validation_enabled:
        assert validation_samples is not None
        assert validation_mask is not None
        assert validation_surfaces is not None
        if validation_samples.ndim!=4 or validation_samples.shape[-1]!=3 \
                or validation_mask.shape!=validation_samples.shape[:3] \
                or validation_samples.shape[1:]!=samples.shape[1:] \
                or validation_surfaces.ndim!=4 \
                or validation_surfaces.shape[0]!=validation_samples.shape[0] \
                or validation_surfaces.shape[1:]!=surfaces.shape[1:] \
                or not np.isfinite(validation_surfaces).all():
            raise ValueError("direct validation 样本尺寸或数值无效")
        if np.any(validation_mask.reshape(
                validation_mask.shape[0],-1).sum(axis=1)==0):
            raise ValueError("direct validation 存在无有效像素的帧")

    base_texture=_fit_direct_static_base_gpu(
        samples,mask,device=device,huber_delta=huber_delta,
        iterations=base_huber_iterations,
        frame_batch_size=frame_batch_size)
    base_texture_gpu=jax.device_put(base_texture,device)

    # 分块提取全局描述和局部几何统计，避免按帧发起大量小 GPU 调用。
    analyze_geometry=jax.jit(jax.vmap(lambda value:(
        direct_geometry_descriptor_jax(value,geometry_descriptor_rows),
        direct_local_geometry_feature_grid_jax(value))))
    descriptor_batches=[]
    local_sum=np.zeros(15,np.float64)
    local_square_sum=np.zeros(15,np.float64)
    local_count=0
    for start in range(0,surfaces.shape[0],32):
        descriptor_batch,local_batch=jax.device_get(analyze_geometry(
            jax.device_put(surfaces[start:start+32],device)))
        descriptor_batches.append(np.asarray(descriptor_batch,np.float32))
        local_values=np.asarray(local_batch,np.float32).reshape(-1,15)
        local_sum+=local_values.sum(axis=0,dtype=np.float64)
        local_square_sum+=np.square(
            local_values,dtype=np.float64).sum(axis=0,dtype=np.float64)
        local_count+=local_values.shape[0]
    descriptors=np.concatenate(descriptor_batches,axis=0)
    feature_mean=descriptors.mean(axis=0,dtype=np.float64).astype(np.float32)
    feature_scale=descriptors.std(axis=0,dtype=np.float64).astype(np.float32)
    feature_scale=np.where(feature_scale>1e-6,feature_scale,1.).astype(np.float32)
    normalized_descriptors=(descriptors-feature_mean)/feature_scale
    if geometry_pca_dimensions>min(normalized_descriptors.shape):
        raise ValueError(
            "geometry_pca_dimensions 不能大于训练帧数或全局几何维数")
    _,_,pca_vh=np.linalg.svd(
        normalized_descriptors,full_matrices=False)
    geometry_pca_components=np.ascontiguousarray(
        pca_vh[:geometry_pca_dimensions].T,dtype=np.float32)
    pca_scores=normalized_descriptors@geometry_pca_components
    geometry_pca_scale=pca_scores.std(
        axis=0,dtype=np.float64).astype(np.float32)
    geometry_pca_scale=np.where(
        geometry_pca_scale>1e-6,geometry_pca_scale,1.).astype(np.float32)
    local_feature_mean=(local_sum/local_count).astype(np.float32)
    local_variance=np.maximum(
        local_square_sum/local_count-np.square(local_sum/local_count),0)
    local_feature_scale=np.sqrt(local_variance).astype(np.float32)
    local_feature_scale=np.where(
        local_feature_scale>1e-6,local_feature_scale,1.).astype(np.float32)

    generator=np.random.default_rng(seed)
    coordinate_feature_count=2+4*frequency_values.size

    def adjacent_pairs(frame_mask: np.ndarray) -> np.ndarray:
        """返回同一帧中右/下相邻且同时有效的扁平像素索引对。"""
        current=np.asarray(frame_mask,np.bool_)
        flat=np.arange(current.size,dtype=np.int64).reshape(current.shape)
        horizontal=np.stack([
            flat[:,:-1][current[:,:-1]&current[:,1:]],
            flat[:,1:][current[:,:-1]&current[:,1:]]],axis=-1)
        vertical=np.stack([
            flat[:-1][current[:-1]&current[1:]],
            flat[1:][current[:-1]&current[1:]]],axis=-1)
        return np.concatenate([horizontal,vertical],axis=0)

    validation_geometry_features=None
    validation_coordinates=None
    validation_targets=None
    validation_paired_geometry_features=None
    validation_paired_surfaces=None
    validation_difference_coordinates=None
    validation_difference_targets=None
    validation_difference_paired_targets=None
    validation_spatial_coordinates=None
    validation_spatial_neighbor_coordinates=None
    validation_spatial_targets=None
    validation_spatial_neighbor_targets=None
    if validation_enabled:
        assert validation_samples is not None
        assert validation_mask is not None
        assert validation_surfaces is not None
        descriptor_only=jax.jit(jax.vmap(lambda value:
            direct_geometry_descriptor_jax(value,geometry_descriptor_rows)))
        validation_descriptor_batches=[]
        for start in range(0,validation_surfaces.shape[0],32):
            validation_descriptor_batches.append(np.asarray(jax.device_get(
                descriptor_only(jax.device_put(
                    validation_surfaces[start:start+32],device))),np.float32))
        validation_descriptors=np.concatenate(
            validation_descriptor_batches,axis=0)
        validation_geometry_features=(
            validation_descriptors-feature_mean)/feature_scale
        validation_generator=np.random.default_rng(seed+104729)
        validation_flat=validation_samples.reshape(
            validation_samples.shape[0],-1,3)
        validation_flat_mask=validation_mask.reshape(
            validation_mask.shape[0],-1)
        validation_pixels=np.stack([
            validation_generator.choice(
                np.flatnonzero(frame_mask),validation_points_per_frame,
                replace=np.count_nonzero(frame_mask)<validation_points_per_frame)
            for frame_mask in validation_flat_mask])
        validation_rows=validation_pixels//samples.shape[2]
        validation_columns=validation_pixels%samples.shape[2]
        validation_coordinates=np.stack([
            validation_rows/max(samples.shape[1]-1,1),
            validation_columns/max(samples.shape[2]-1,1)],axis=-1).astype(
                np.float32)
        validation_targets=np.ascontiguousarray(validation_flat[
            np.arange(validation_samples.shape[0])[:,None],validation_pixels])
        validation_pca=(
            validation_geometry_features@geometry_pca_components
            /geometry_pca_scale)
        if validation_samples.shape[0]>1:
            validation_norm=np.sum(
                validation_pca**2,axis=1,keepdims=True)
            validation_distance=np.maximum(
                validation_norm+validation_norm.T
                -2*(validation_pca@validation_pca.T),0)
            np.fill_diagonal(validation_distance,np.inf)
            validation_pairs=np.argmin(validation_distance,axis=1)
        else:
            validation_pairs=np.zeros(1,np.int64)
        validation_difference_pixels=[]
        for frame_index,paired_index in enumerate(validation_pairs):
            common=np.flatnonzero(
                validation_flat_mask[frame_index]
                &validation_flat_mask[paired_index])
            if common.size==0:
                raise ValueError(
                    "direct validation 几何近邻帧没有共同有效 observation 点")
            validation_difference_pixels.append(validation_generator.choice(
                common,validation_points_per_frame,
                replace=common.size<validation_points_per_frame))
        validation_difference_pixels=np.stack(validation_difference_pixels)
        difference_rows=validation_difference_pixels//samples.shape[2]
        difference_columns=validation_difference_pixels%samples.shape[2]
        validation_difference_coordinates=np.stack([
            difference_rows/max(samples.shape[1]-1,1),
            difference_columns/max(samples.shape[2]-1,1)],axis=-1).astype(
                np.float32)
        validation_difference_targets=np.ascontiguousarray(validation_flat[
            np.arange(validation_samples.shape[0])[:,None],
            validation_difference_pixels])
        validation_difference_paired_targets=np.ascontiguousarray(
            validation_flat[
                validation_pairs[:,None],validation_difference_pixels])
        validation_paired_geometry_features=np.ascontiguousarray(
            validation_geometry_features[validation_pairs])
        validation_paired_surfaces=np.ascontiguousarray(
            validation_surfaces[validation_pairs])
        validation_spatial_pairs=[]
        for frame_index,frame_mask in enumerate(validation_mask):
            pairs=adjacent_pairs(frame_mask)
            if pairs.size==0:
                if spatial_difference_validation_weight>0:
                    raise ValueError("direct validation 帧没有相邻有效像素")
                first=np.flatnonzero(frame_mask)[0]
                pairs=np.asarray([[first,first]],np.int64)
            selection=validation_generator.choice(
                pairs.shape[0],validation_points_per_frame,
                replace=pairs.shape[0]<validation_points_per_frame)
            validation_spatial_pairs.append(pairs[selection])
        validation_spatial_pairs=np.stack(validation_spatial_pairs)
        spatial_source=validation_spatial_pairs[...,0]
        spatial_neighbor=validation_spatial_pairs[...,1]

        def flat_coordinates(indices):
            return np.stack([
                (indices//samples.shape[2])/max(samples.shape[1]-1,1),
                (indices%samples.shape[2])/max(samples.shape[2]-1,1),
            ],axis=-1).astype(np.float32)

        validation_spatial_coordinates=flat_coordinates(spatial_source)
        validation_spatial_neighbor_coordinates=flat_coordinates(
            spatial_neighbor)
        validation_spatial_targets=np.ascontiguousarray(validation_flat[
            np.arange(validation_samples.shape[0])[:,None],spatial_source])
        validation_spatial_neighbor_targets=np.ascontiguousarray(
            validation_flat[
                np.arange(validation_samples.shape[0])[:,None],
                spatial_neighbor])

    def initialize(layer_sizes: list[int],output_bias: np.ndarray | None = None):
        parameters=[]
        for index,(source_count,target_count) in enumerate(zip(
                layer_sizes[:-1],layer_sizes[1:],strict=True)):
            limit=np.sqrt(6/max(source_count+target_count,1))
            weight=generator.uniform(
                -limit,limit,(source_count,target_count)).astype(np.float32)
            bias=np.zeros(target_count,np.float32)
            if output_bias is not None and index==len(layer_sizes)-2:
                weight*=.05
                bias=output_bias.astype(np.float32)
            parameters.append((jnp.asarray(weight),jnp.asarray(bias)))
        return tuple(parameters)

    encoder=initialize([
        descriptors.shape[1],
        *([geometry_encoder_width]*geometry_encoder_layers),
        geometry_latent_dimensions])
    decoder_input_count=(coordinate_feature_count
                         +geometry_latent_dimensions
                         +geometry_pca_dimensions+15)
    # 完整条件输入跳连到每个隐层，避免局部几何在深层被洗掉。
    def initialize_decoder(output_count: int):
        decoder_sizes=[(decoder_input_count,decoder_width)]
        decoder_sizes.extend(
            (decoder_width+decoder_input_count,decoder_width)
            for _ in range(decoder_layers-1))
        decoder_sizes.append(
            (decoder_width+decoder_input_count,output_count))
        layers=[]
        for index,(source_count,target_count) in enumerate(decoder_sizes):
            limit=np.sqrt(6/max(source_count+target_count,1))
            weight=generator.uniform(
                -limit,limit,(source_count,target_count)).astype(np.float32)
            bias=np.zeros(target_count,np.float32)
            if index==len(decoder_sizes)-1:
                weight*=.05
            layers.append((jnp.asarray(weight),jnp.asarray(bias)))
        return tuple(layers)

    decoder=(tuple(initialize_decoder(1) for _ in range(3))
             if separate_channel_decoders else initialize_decoder(3))
    parameters=(encoder,decoder)
    parameters=jax.device_put(parameters,device)
    frequencies_gpu=jax.device_put(frequency_values,device)
    geometry_pca_components_gpu=jax.device_put(
        geometry_pca_components,device)
    geometry_pca_scale_gpu=jax.device_put(geometry_pca_scale,device)

    local_feature_mean_gpu=jax.device_put(local_feature_mean,device)
    local_feature_scale_gpu=jax.device_put(local_feature_scale,device)

    def forward(current,geometry_features,local_grids,coordinates):
        current_encoder,current_decoder=current
        latent=geometry_features
        for index,(weight,bias) in enumerate(current_encoder):
            latent=latent@weight+bias
            if index<len(current_encoder)-1:
                latent=jax.nn.silu(latent)
        coordinate_features=direct_background_features_jax(
            coordinates,frequencies_gpu)
        latent=jnp.broadcast_to(
            latent[:,None,:],(*coordinate_features.shape[:-1],latent.shape[-1]))
        pca=geometry_features@geometry_pca_components_gpu \
            /geometry_pca_scale_gpu
        pca=jnp.broadcast_to(
            pca[:,None,:],(*coordinate_features.shape[:-1],pca.shape[-1]))
        local_features=jax.vmap(
            sample_direct_local_geometry_feature_grid_jax)(
                local_grids,coordinates)
        local_features=(local_features-local_feature_mean_gpu) \
            /local_feature_scale_gpu
        network_input=jnp.concatenate([
            coordinate_features,latent,pca,local_features],axis=-1)
        def decode(layers):
            values=network_input
            for index,(weight,bias) in enumerate(layers):
                if index>0:
                    values=jnp.concatenate([values,network_input],axis=-1)
                values=values@weight+bias
                if index<len(layers)-1:
                    values=jax.nn.silu(values)
            return values
        values=(jnp.concatenate(
            [decode(layers) for layers in current_decoder],axis=-1)
                if separate_channel_decoders else decode(current_decoder))
        base=sample_direct_base_texture_jax(base_texture_gpu,coordinates)
        epsilon=jnp.asarray(1e-4,base.dtype)
        base=jnp.clip(base,epsilon,1-epsilon)
        return jax.nn.sigmoid(
            jnp.log(base)-jnp.log1p(-base)+values)

    def loss_fn(
        current,geometry_features,surface_batch,coordinates,targets,
        paired_geometry_features,paired_surface_batch,
        difference_coordinates,difference_targets,difference_paired_targets,
        spatial_coordinates,spatial_neighbor_coordinates,
        spatial_targets,spatial_neighbor_targets,
    ):
        local_grids=jax.vmap(direct_local_geometry_feature_grid_jax)(
            surface_batch)
        prediction=forward(
            current,geometry_features,local_grids,coordinates)
        error=prediction-targets
        absolute=jnp.abs(error)
        absolute_huber=jnp.where(
            absolute<=huber_delta,.5*error**2,
            huber_delta*(absolute-.5*huber_delta))
        channel_mse=jnp.mean(error**2,axis=(0,1))
        channel_mse_normalizer=1.
        spatial_huber=None
        if spatial_difference_weight>0:
            spatial_first=forward(
                current,geometry_features,local_grids,spatial_coordinates)
            spatial_second=forward(
                current,geometry_features,local_grids,
                spatial_neighbor_coordinates)
            spatial_error=(spatial_second-spatial_first) \
                -(spatial_neighbor_targets-spatial_targets)
            spatial_absolute=jnp.abs(spatial_error)
            spatial_huber=jnp.where(
                spatial_absolute<=huber_delta,.5*spatial_error**2,
                huber_delta*(spatial_absolute-.5*huber_delta))
            channel_mse+=spatial_difference_weight*jnp.mean(
                spatial_error**2,axis=(0,1))
            channel_mse_normalizer+=spatial_difference_weight
        difference_huber=None
        if geometry_difference_weight>0:
            paired_local_grids=jax.vmap(
                direct_local_geometry_feature_grid_jax)(paired_surface_batch)
            first=forward(current,geometry_features,local_grids,
                          difference_coordinates)
            second=forward(current,paired_geometry_features,
                           paired_local_grids,difference_coordinates)
            difference_error=(first-second) \
                -(difference_targets-difference_paired_targets)
            difference_absolute=jnp.abs(difference_error)
            difference_huber=jnp.where(
                difference_absolute<=huber_delta,
                .5*difference_error**2,
                huber_delta*(difference_absolute-.5*huber_delta))
            channel_mse+=geometry_difference_weight*jnp.mean(
                difference_error**2,axis=(0,1))
            channel_mse_normalizer+=geometry_difference_weight
        channel_weights=_adaptive_channel_weights(
            channel_mse/channel_mse_normalizer,
            adaptive_channel_weight_strength)
        total=jnp.mean(absolute_huber*channel_weights)
        if spatial_huber is not None:
            total+=spatial_difference_weight*jnp.mean(
                spatial_huber*channel_weights)
        if difference_huber is not None:
            total+=geometry_difference_weight*jnp.mean(
                difference_huber*channel_weights)
        return total,channel_weights

    beta1,beta2=.9,.999
    moments=jax.tree.map(jnp.zeros_like,parameters)
    variances=jax.tree.map(jnp.zeros_like,parameters)

    @jax.jit
    def step(current,moment,variance,index,geometry_features,
             surface_batch,coordinates,targets,paired_geometry_features,
             paired_surface_batch,difference_coordinates,
             difference_targets,difference_paired_targets,
             spatial_coordinates,spatial_neighbor_coordinates,
             spatial_targets,spatial_neighbor_targets):
        (loss,channel_weights),gradient=jax.value_and_grad(
            loss_fn,has_aux=True)(
            current,geometry_features,surface_batch,coordinates,targets,
            paired_geometry_features,paired_surface_batch,
            difference_coordinates,difference_targets,
            difference_paired_targets,spatial_coordinates,
            spatial_neighbor_coordinates,spatial_targets,
            spatial_neighbor_targets)
        moment=jax.tree.map(
            lambda old,value:beta1*old+(1-beta1)*value,moment,gradient)
        variance=jax.tree.map(
            lambda old,value:beta2*old+(1-beta2)*value*value,
            variance,gradient)
        corrected_m=jax.tree.map(
            lambda value:value/(1-beta1**index),moment)
        corrected_v=jax.tree.map(
            lambda value:value/(1-beta2**index),variance)
        progress=(index-1)/max(steps-1,1)
        rate=learning_rate*(.05+.95*.5*(1+jnp.cos(jnp.pi*progress)))
        current=jax.tree.map(
            lambda value,m,v:value-rate*m/(jnp.sqrt(v)+1e-8),
            current,corrected_m,corrected_v)
        return current,moment,variance,loss,channel_weights

    @jax.jit
    def validation_metrics(
        current,geometry_features,surface_batch,coordinates,targets,
        paired_geometry_features,paired_surface_batch,
        difference_coordinates,difference_targets,difference_paired_targets,
        spatial_coordinates,spatial_neighbor_coordinates,
        spatial_targets,spatial_neighbor_targets,
    ):
        local_grids=jax.vmap(direct_local_geometry_feature_grid_jax)(
            surface_batch)
        prediction=forward(
            current,geometry_features,local_grids,coordinates)
        absolute_rmse=jnp.sqrt(jnp.mean(
            (prediction-targets)**2,axis=(0,1)))
        paired_local_grids=jax.vmap(
            direct_local_geometry_feature_grid_jax)(paired_surface_batch)
        first=forward(
            current,geometry_features,local_grids,difference_coordinates)
        second=forward(
            current,paired_geometry_features,paired_local_grids,
            difference_coordinates)
        difference_error=(first-second) \
            -(difference_targets-difference_paired_targets)
        difference_rmse=jnp.sqrt(jnp.mean(
            difference_error**2,axis=(0,1)))
        spatial_first=forward(
            current,geometry_features,local_grids,spatial_coordinates)
        spatial_second=forward(
            current,geometry_features,local_grids,
            spatial_neighbor_coordinates)
        spatial_error=(spatial_second-spatial_first) \
            -(spatial_neighbor_targets-spatial_targets)
        spatial_difference_rmse=jnp.sqrt(jnp.mean(
            spatial_error**2,axis=(0,1)))
        return absolute_rmse,difference_rmse,spatial_difference_rmse

    validation_gpu=None
    if validation_enabled:
        assert validation_geometry_features is not None
        assert validation_surfaces is not None
        assert validation_coordinates is not None
        assert validation_targets is not None
        assert validation_paired_geometry_features is not None
        assert validation_paired_surfaces is not None
        assert validation_difference_coordinates is not None
        assert validation_difference_targets is not None
        assert validation_difference_paired_targets is not None
        assert validation_spatial_coordinates is not None
        assert validation_spatial_neighbor_coordinates is not None
        assert validation_spatial_targets is not None
        assert validation_spatial_neighbor_targets is not None
        validation_gpu=(
            jax.device_put(np.ascontiguousarray(
                validation_geometry_features),device),
            jax.device_put(np.ascontiguousarray(validation_surfaces),device),
            jax.device_put(validation_coordinates,device),
            jax.device_put(validation_targets,device),
            jax.device_put(validation_paired_geometry_features,device),
            jax.device_put(validation_paired_surfaces,device),
            jax.device_put(validation_difference_coordinates,device),
            jax.device_put(validation_difference_targets,device),
            jax.device_put(validation_difference_paired_targets,device),
            jax.device_put(validation_spatial_coordinates,device),
            jax.device_put(validation_spatial_neighbor_coordinates,device),
            jax.device_put(validation_spatial_targets,device),
            jax.device_put(validation_spatial_neighbor_targets,device))

    frame_count,rows,columns,_=samples.shape
    flattened=samples.reshape(frame_count,rows*columns,3)
    flattened_mask=mask.reshape(frame_count,rows*columns)
    valid_pixel_indices=[np.flatnonzero(value) for value in flattened_mask]
    empty_frames=[index for index,value in enumerate(valid_pixel_indices)
                  if value.size==0]
    if empty_frames:
        raise ValueError(
            f"direct neural field 存在 {len(empty_frames)} 个无有效像素的帧")
    spatial_pixel_pairs=[]
    for frame_index,frame_mask in enumerate(mask):
        pairs=adjacent_pairs(frame_mask)
        if pairs.size==0:
            if spatial_difference_weight>0:
                raise ValueError(
                    f"direct 训练帧 {frame_index} 没有相邻有效像素")
            first=valid_pixel_indices[frame_index][0]
            pairs=np.asarray([[first,first]],np.int64)
        spatial_pixel_pairs.append(pairs)
    effective_neighbor_count=min(
        geometry_difference_neighbor_count,frame_count-1)
    geometry_neighbors=None
    if geometry_difference_weight>0:
        # 在 PCA 白化后的主几何子空间找近邻，避免高维描述中的微小噪声
        # 主导差分帧选择；该状态也正是解码器获得的确定性直连。
        neighbor_features=pca_scores/geometry_pca_scale
        descriptor_norm=np.sum(
            neighbor_features**2,axis=1,keepdims=True)
        geometry_distance=np.maximum(
            descriptor_norm+descriptor_norm.T
            -2*(neighbor_features@neighbor_features.T),0)
        np.fill_diagonal(geometry_distance,np.inf)
        geometry_neighbors=np.argpartition(
            geometry_distance,effective_neighbor_count-1,axis=1
        )[:,:effective_neighbor_count]
        del geometry_distance,descriptor_norm,neighbor_features
    frames_per_step=min(frame_batch_size,frame_count)
    points_per_frame=max(1,int(np.ceil(batch_size/frames_per_step)))
    schedule_count=steps*frames_per_step
    frame_schedule=np.concatenate([
        generator.permutation(frame_count)
        for _ in range(int(np.ceil(schedule_count/frame_count)))
    ])[:schedule_count].reshape(steps,frames_per_step)
    decoder_description=("3xscalar" if separate_channel_decoders else "RGB")
    print(f"direct 统一几何条件神经场：samples={frame_count}，"
          f"grid={rows}x{columns}，encoder={geometry_encoder_layers}x"
          f"{geometry_encoder_width}->{geometry_latent_dimensions}，"
          f"decoder={decoder_description} {decoder_layers}x{decoder_width} "
          f"dense-skip，steps={steps}，"
          f"valid={mask.mean():.1%}，PCA={geometry_pca_dimensions}，"
          f"geometry_difference={geometry_difference_weight:g}x"
          f"{geometry_difference_points_per_pair}，"
          f"spatial_difference={spatial_difference_weight:g}x"
          f"{spatial_difference_points_per_frame}")
    if adaptive_channel_weight_strength>0:
        print("direct 自适应通道 loss："
              f"strength={adaptive_channel_weight_strength:g}，"
              "按综合 batch RMSE 动态调整 RGB 权重")
    if validation_enabled:
        assert validation_samples is not None
        print(f"direct validation：frames={validation_samples.shape[0]}，"
              f"points/frame={validation_points_per_frame}，"
              f"interval={validation_interval}，patience={early_stopping_patience}，"
              f"min_steps={early_stopping_min_steps}，"
              "geometry-difference weight="
              f"{geometry_difference_validation_weight:g}")
    best_parameters=None
    best_validation_score=np.inf
    best_validation_rmse=np.full(3,np.inf,np.float32)
    best_validation_difference_rmse=np.full(3,np.inf,np.float32)
    best_validation_spatial_difference_rmse=np.full(3,np.inf,np.float32)
    patience_reference_score=np.inf
    best_step=0
    stale_validation_count=0
    completed_steps=steps
    for index in range(1,steps+1):
        frame_indices=frame_schedule[index-1]
        pixel_indices=np.stack([
            generator.choice(valid_pixel_indices[frame_index],points_per_frame,
                             replace=valid_pixel_indices[frame_index].size
                             <points_per_frame)
            for frame_index in frame_indices])
        row_indices=pixel_indices//columns
        column_indices=pixel_indices%columns
        coordinates=np.stack([
            row_indices/max(rows-1,1),column_indices/max(columns-1,1)],
            axis=-1).astype(np.float32)
        targets=np.ascontiguousarray(flattened[frame_indices[:,None],pixel_indices])
        geometry_features=np.ascontiguousarray(
            normalized_descriptors[frame_indices])
        surface_batch=np.ascontiguousarray(surfaces[frame_indices])
        if geometry_difference_weight>0:
            assert geometry_neighbors is not None
            neighbor_rank=generator.integers(
                0,effective_neighbor_count,size=frames_per_step)
            paired_indices=geometry_neighbors[frame_indices,neighbor_rank]
            difference_pixels=[]
            for frame_index,paired_index in zip(
                    frame_indices,paired_indices,strict=True):
                common=np.flatnonzero(
                    flattened_mask[frame_index]&flattened_mask[paired_index])
                if common.size==0:
                    raise ValueError(
                        "几何近邻帧在 observation_grid 上没有共同有效点")
                difference_pixels.append(generator.choice(
                    common,geometry_difference_points_per_pair,
                    replace=common.size<geometry_difference_points_per_pair))
        else:
            # 保持 JIT 参数形状固定；loss 中权重为零时这些占位值不会参与计算。
            paired_indices=frame_indices
            difference_pixels=[generator.choice(
                valid_pixel_indices[frame_index],
                geometry_difference_points_per_pair,
                replace=valid_pixel_indices[frame_index].size
                <geometry_difference_points_per_pair)
                for frame_index in frame_indices]
        difference_pixel_indices=np.stack(difference_pixels)
        difference_rows=difference_pixel_indices//columns
        difference_columns=difference_pixel_indices%columns
        difference_coordinates=np.stack([
            difference_rows/max(rows-1,1),
            difference_columns/max(columns-1,1)],axis=-1).astype(np.float32)
        difference_targets=np.ascontiguousarray(flattened[
            frame_indices[:,None],difference_pixel_indices])
        difference_paired_targets=np.ascontiguousarray(flattened[
            paired_indices[:,None],difference_pixel_indices])
        selected_spatial_pairs=np.stack([
            pairs[generator.choice(
                pairs.shape[0],spatial_difference_points_per_frame,
                replace=pairs.shape[0]<spatial_difference_points_per_frame)]
            for pairs in (spatial_pixel_pairs[frame_index]
                          for frame_index in frame_indices)])
        spatial_source_indices=selected_spatial_pairs[...,0]
        spatial_neighbor_indices=selected_spatial_pairs[...,1]
        spatial_coordinates=np.stack([
            (spatial_source_indices//columns)/max(rows-1,1),
            (spatial_source_indices%columns)/max(columns-1,1)],axis=-1
        ).astype(np.float32)
        spatial_neighbor_coordinates=np.stack([
            (spatial_neighbor_indices//columns)/max(rows-1,1),
            (spatial_neighbor_indices%columns)/max(columns-1,1)],axis=-1
        ).astype(np.float32)
        spatial_targets=np.ascontiguousarray(flattened[
            frame_indices[:,None],spatial_source_indices])
        spatial_neighbor_targets=np.ascontiguousarray(flattened[
            frame_indices[:,None],spatial_neighbor_indices])
        paired_geometry_features=np.ascontiguousarray(
            normalized_descriptors[paired_indices])
        paired_surface_batch=np.ascontiguousarray(surfaces[paired_indices])
        parameters,moments,variances,loss,channel_weights=step(
            parameters,moments,variances,jnp.asarray(index,jnp.float32),
            jax.device_put(geometry_features,device),
            jax.device_put(surface_batch,device),
            jax.device_put(coordinates,device),jax.device_put(targets,device),
            jax.device_put(paired_geometry_features,device),
            jax.device_put(paired_surface_batch,device),
            jax.device_put(difference_coordinates,device),
            jax.device_put(difference_targets,device),
            jax.device_put(difference_paired_targets,device),
            jax.device_put(spatial_coordinates,device),
            jax.device_put(spatial_neighbor_coordinates,device),
            jax.device_put(spatial_targets,device),
            jax.device_put(spatial_neighbor_targets,device),
        )
        if index==1 or index%100==0 or index==steps:
            loss,logged_channel_weights=jax.device_get((loss,channel_weights))
            weight_message=(
                f" channel_weights_RGB={np.asarray(logged_channel_weights).tolist()}"
                if adaptive_channel_weight_strength>0 else "")
            print(f"direct neural field step={index:04d} loss={float(loss):.7f}"
                  f"{weight_message}")

        should_validate=(validation_enabled and (
            index%validation_interval==0 or index==steps))
        if should_validate:
            assert validation_gpu is not None
            (validation_rmse,validation_difference_rmse,
             validation_spatial_difference_rmse)=jax.device_get(
                validation_metrics(parameters,*validation_gpu))
            validation_rmse=np.asarray(validation_rmse,np.float32)
            validation_difference_rmse=np.asarray(
                validation_difference_rmse,np.float32)
            validation_spatial_difference_rmse=np.asarray(
                validation_spatial_difference_rmse,np.float32)
            absolute_score=float(np.mean(validation_rmse**2))
            difference_score=float(np.mean(validation_difference_rmse**2))
            spatial_difference_score=float(np.mean(
                validation_spatial_difference_rmse**2))
            validation_score=float(np.sqrt(
                (absolute_score+geometry_difference_validation_weight
                 *difference_score+spatial_difference_validation_weight
                 *spatial_difference_score)
                /(1+geometry_difference_validation_weight
                  +spatial_difference_validation_weight)))
            is_best=validation_score<best_validation_score
            significant_improvement=(validation_score
                <patience_reference_score-early_stopping_min_delta)
            if is_best:
                best_parameters=jax.device_get(parameters)
                best_validation_score=validation_score
                best_validation_rmse=validation_rmse
                best_validation_difference_rmse=validation_difference_rmse
                best_validation_spatial_difference_rmse=(
                    validation_spatial_difference_rmse)
                best_step=index
                if checkpoint_path is not None:
                    _save_direct_fit_checkpoint(
                        checkpoint_path,step=index,
                        validation_rmse_rgb=validation_rmse,
                        validation_difference_rmse_rgb=(
                            validation_difference_rmse),
                        validation_spatial_difference_rmse_rgb=(
                            validation_spatial_difference_rmse),
                        base_texture=base_texture,
                        frequencies=frequency_values,
                        geometry_descriptor_rows=geometry_descriptor_rows,
                        geometry_feature_mean=feature_mean,
                        geometry_feature_scale=feature_scale,
                        geometry_pca_components=geometry_pca_components,
                        geometry_pca_scale=geometry_pca_scale,
                        local_feature_mean=local_feature_mean,
                        local_feature_scale=local_feature_scale,
                        parameters=parameters,moments=moments,
                        variances=variances,
                        separate_channel_decoders=(
                            separate_channel_decoders))
            if significant_improvement:
                patience_reference_score=validation_score
                stale_validation_count=0
            else:
                stale_validation_count+=1
            marker=("best " if is_best else "") \
                +f"stale={stale_validation_count}/{early_stopping_patience}"
            print(f"direct validation step={index:04d} "
                  f"RMSE RGB={validation_rmse.tolist()}，"
                "geometry-difference RMSE RGB="
                f"{validation_difference_rmse.tolist()} "
                "spatial-difference RMSE RGB="
                f"{validation_spatial_difference_rmse.tolist()} "
                f"score={validation_score:.7f} {marker}")
            if index>=early_stopping_min_steps \
                    and stale_validation_count>=early_stopping_patience:
                completed_steps=index
                print(f"direct early stopping: step={index}，best_step={best_step}，"
                      f"best_RMSE_RGB={best_validation_rmse.tolist()}，"
                      "best_geometry_difference_RMSE_RGB="
                      f"{best_validation_difference_rmse.tolist()}，"
                      "best_spatial_difference_RMSE_RGB="
                      f"{best_validation_spatial_difference_rmse.tolist()}")
                break

    if best_parameters is not None:
        parameters=jax.device_put(best_parameters,device)
        checkpoint_message=(f"，checkpoint={Path(checkpoint_path).expanduser()}"
                            if checkpoint_path is not None else "")
        print(f"direct 恢复最佳参数：step={best_step}/{completed_steps}，"
              f"validation_RMSE_RGB={best_validation_rmse.tolist()}，"
              "validation_geometry_difference_RMSE_RGB="
              f"{best_validation_difference_rmse.tolist()}"
              "，validation_spatial_difference_RMSE_RGB="
              f"{best_validation_spatial_difference_rmse.tolist()}"
              f"{checkpoint_message}")

    host_encoder,host_decoder=jax.device_get(parameters)
    encoder_weights=tuple(np.asarray(value[0],np.float32)
                          for value in host_encoder)
    encoder_biases=tuple(np.asarray(value[1],np.float32)
                         for value in host_encoder)
    if separate_channel_decoders:
        decoder_weights=tuple(tuple(np.asarray(value[0],np.float32)
                              for value in decoder)
                              for decoder in host_decoder)
        decoder_biases=tuple(tuple(np.asarray(value[1],np.float32)
                             for value in decoder)
                             for decoder in host_decoder)
    else:
        decoder_weights=tuple(np.asarray(value[0],np.float32)
                              for value in host_decoder)
        decoder_biases=tuple(np.asarray(value[1],np.float32)
                             for value in host_decoder)
    # 训练图、Adam 矩和验证缓冲不再需要；清掉以免后续全场求值碎片化 OOM。
    del parameters,moments,variances,host_encoder,host_decoder
    if best_parameters is not None:
        del best_parameters
    jax.clear_caches()
    return (base_texture,frequency_values,feature_mean,feature_scale,
            geometry_pca_components,geometry_pca_scale,
            local_feature_mean,local_feature_scale,
            encoder_weights,encoder_biases,decoder_weights,decoder_biases)


def fit_residual_correction_model_gpu(
    residuals: np.ndarray,
    valid: np.ndarray,
    *,
    surface_xyz: np.ndarray,
    row_coefficients: int,
    column_coefficients: int,
    device: jax.Device,
    m_count: int = 2,
    huber_delta: float = .04,
    smooth_lambda: float = .01,
    magnitude_lambda: float = 1e-4,
    outer_weight: float = .2,
    outer_fraction: float = .05,
    b_max_deviation: float = .35,
    m_max_deviation: float = .15,
    channel_huber_ratio_min: float = .5,
    channel_huber_ratio_max: float = 2.,
    curvature_feature_count: int | None = None,
    curvature_curve_coefficients: int = 12,
    curvature_smooth_lambda: float = .01,
    curvature_regression_lambda: float = .01,
    sample_batch_size: int = 8,
    pixel_chunk_size: int = 128,
    scale_sample_pixels: int = 512,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """在 GPU 上分块学习 B 和曲率引导 M，输入全量样本始终留在 CPU。"""
    if device.platform!="gpu":
        raise RuntimeError("离线残差 B/M 拟合只允许 JAX GPU，不提供 CPU 兼容路径")
    samples=np.asarray(residuals,dtype=np.float32)
    mask=np.asarray(valid,dtype=np.bool_)
    if samples.ndim!=4 or samples.shape[-1]!=3 or mask.shape!=samples.shape[:3]:
        raise ValueError("residuals/valid 必须分别是 NxHxWx3 和 NxHxW")
    if samples.shape[0]<=m_count or m_count<1:
        raise ValueError("残差样本数必须大于 m_count，且 m_count 至少为 1")
    if row_coefficients<4 or column_coefficients<4:
        raise ValueError("残差 B 样条行列系数数必须至少为 4")
    if huber_delta<=0 or smooth_lambda<0 or magnitude_lambda<0:
        raise ValueError("残差拟合正则参数无效")
    if not 0<=outer_weight<=1 or not 0<=outer_fraction<.5:
        raise ValueError("outer_weight/outer_fraction 范围无效")
    if b_max_deviation<=0 or m_max_deviation<=0:
        raise ValueError("残差场幅度上限必须为正")
    b_lower,b_upper=-float(b_max_deviation),float(b_max_deviation)
    if channel_huber_ratio_min<=0 \
            or channel_huber_ratio_max<channel_huber_ratio_min:
        raise ValueError("分通道 Huber 比例上下界无效")
    if curvature_regression_lambda<=0:
        raise ValueError("曲率残差回归强度必须为正")
    sample_batch_size=min(
        _positive_integer("sample_batch_size",sample_batch_size),samples.shape[0])
    pixel_chunk_size=min(
        _positive_integer("pixel_chunk_size",pixel_chunk_size),
        samples.shape[1]*samples.shape[2])
    scale_sample_pixels=min(
        _positive_integer("scale_sample_pixels",scale_sample_pixels),
        samples.shape[1]*samples.shape[2])
    feature_count=(m_count if curvature_feature_count is None
                   else int(curvature_feature_count))
    if feature_count<m_count:
        raise ValueError("curvature_feature_count 不能小于 m_count")

    sample_count,rows,columns,_=samples.shape
    pixel_count=rows*columns
    spatial_y=np.linspace(0,1,rows,dtype=np.float32)[:,None]
    spatial_x=np.linspace(0,1,columns,dtype=np.float32)[None,:]
    outer=((spatial_y<outer_fraction)|(spatial_y>1-outer_fraction)
           |(spatial_x<outer_fraction)|(spatial_x>1-outer_fraction))
    spatial_np=np.where(outer,outer_weight,1.).astype(np.float32)
    coverage_np=mask.mean(axis=0,dtype=np.float32)*spatial_np
    spatial=jax.device_put(spatial_np,device)

    def fit_spline_fields(
        fields: Array,weight: Array,initial_coefficients: Array | None = None,
    ) -> Array:
        """在 GPU 上以矩阵自由 PCG 拟合共同权重的 RGB B 样条场。"""
        field_count=fields.shape[0]
        scalar_fields=fields.transpose(0,3,1,2).reshape(
            field_count*3,rows,columns)
        scalar_weights=jnp.broadcast_to(weight,scalar_fields.shape)
        coefficient_shape=(
            field_count*3,row_coefficients,column_coefficients)
        initial=(jnp.zeros(coefficient_shape,jnp.float32)
                 if initial_coefficients is None else
                 initial_coefficients.reshape(coefficient_shape))
        solution,iterations,relative_residual=_fit_bspline_fields_pcg_jax(
            scalar_fields,scalar_weights,
            jnp.zeros(coefficient_shape,jnp.float32),initial,
            row_coefficients=row_coefficients,
            column_coefficients=column_coefficients,
            smooth_lambda=float(smooth_lambda),
            magnitude_lambda=float(magnitude_lambda),prior_lambda=0.,
            relative_tolerance=2e-6,absolute_tolerance=1e-9,
            max_iterations=2000)
        solution.block_until_ready()
        print(
            f"GPU B 样条 PCG：fields={field_count*3}，"
            f"grid={row_coefficients}x{column_coefficients}，"
            f"iterations={int(np.asarray(iterations))}/2000，"
            f"max_relative_residual="
            f"{float(np.max(np.asarray(relative_residual))):.3e}")
        return solution.reshape(
            field_count,3,row_coefficients,column_coefficients)

    @jax.jit
    def initial_b_sums(batch_samples: Array,batch_mask: Array) -> tuple[Array,Array]:
        weight=batch_mask*spatial[None]
        return (jnp.sum(weight[...,None]*batch_samples,axis=0),
                jnp.sum(weight,axis=0))

    @jax.jit
    def robust_b_sums(
        batch_samples: Array,batch_mask: Array,b_field: Array,
    ) -> tuple[Array,Array]:
        error=batch_samples-b_field[None]
        robust=jnp.minimum(
            1.,huber_delta/jnp.maximum(jnp.linalg.norm(error,axis=-1),1e-12))
        weight=batch_mask*spatial[None]*robust
        return (jnp.sum(weight[...,None]*batch_samples,axis=0),
                jnp.sum(weight,axis=0))

    def accumulate_b(b_field: Array | None) -> tuple[Array,Array]:
        value_sum=jnp.zeros((rows,columns,3),jnp.float32)
        weight_sum=jnp.zeros((rows,columns),jnp.float32)
        for start in range(0,sample_count,sample_batch_size):
            stop=min(start+sample_batch_size,sample_count)
            batch_samples=jax.device_put(
                np.ascontiguousarray(samples[start:stop]),device)
            batch_mask=jax.device_put(
                np.ascontiguousarray(mask[start:stop],dtype=np.float32),device)
            if b_field is None:
                current_value,current_weight=initial_b_sums(
                    batch_samples,batch_mask)
            else:
                current_value,current_weight=robust_b_sums(
                    batch_samples,batch_mask,b_field)
            value_sum+=current_value
            weight_sum+=current_weight
            weight_sum.block_until_ready()
        target=value_sum/jnp.maximum(weight_sum[...,None],1e-12)
        return target,weight_sum

    print(f"GPU B 拟合：samples={sample_count}，batch={sample_batch_size}，"
          f"spline={row_coefficients}x{column_coefficients}")
    b_target,b_weight=accumulate_b(None)
    b_coefficients=fit_spline_fields(b_target[None],b_weight)[0]
    for iteration in range(5):
        b_field=rgb_bspline_field((rows,columns),b_coefficients)
        b_target,b_weight=accumulate_b(b_field)
        b_coefficients=jnp.clip(
            fit_spline_fields(
                b_target[None],b_weight,b_coefficients[None])[0],
            b_lower,b_upper)
        print(f"GPU B IRLS {iteration+1}/5")
    b_field=rgb_bspline_field((rows,columns),b_coefficients)
    b_field.block_until_ready()

    # 仅选取均匀分布的有限像素估计逐通道时间 MAD，避免全量 median 排序。
    scale_indices=np.linspace(
        0,pixel_count-1,scale_sample_pixels,dtype=np.int64)
    flat_samples=samples.reshape(sample_count,pixel_count,3)
    flat_mask=mask.reshape(sample_count,pixel_count)
    scale_samples=jax.device_put(
        np.ascontiguousarray(flat_samples[:,scale_indices]),device)
    scale_mask=jax.device_put(
        np.ascontiguousarray(flat_mask[:,scale_indices]),device)

    @jax.jit
    def estimate_channel_huber(values: Array,current_mask: Array) -> Array:
        masked=jnp.where(current_mask[...,None],values,jnp.nan)
        temporal_median=jnp.nanmedian(masked,axis=0)
        deviation=jnp.where(
            current_mask[...,None],jnp.abs(values-temporal_median[None]),jnp.nan)
        scale=1.4826*jnp.nanmedian(deviation,axis=(0,1))
        scale=jnp.nan_to_num(
            scale,nan=max(1e-4,.05*huber_delta),
            posinf=huber_delta,neginf=huber_delta)
        scale=jnp.maximum(scale,max(1e-4,.05*huber_delta))
        reference=jnp.maximum(jnp.median(scale),1e-12)
        return huber_delta*jnp.clip(
            scale/reference,channel_huber_ratio_min,channel_huber_ratio_max)

    channel_huber=estimate_channel_huber(scale_samples,scale_mask)
    channel_huber.block_until_ready()
    del scale_samples,scale_mask

    geometry_scores=_surface_curvature_scores_gpu(
        surface_xyz,feature_count,
        curve_coefficients=curvature_curve_coefficients,
        smooth_lambda=curvature_smooth_lambda,device=device)
    print(f"GPU M 曲率投影：features={feature_count}")
    eye=jnp.eye(feature_count,dtype=jnp.float32)

    @jax.jit
    def project_pixel_chunk(
        values: Array,current_mask: Array,b_values: Array,
        spatial_values: Array,current_huber: Array,
    ) -> Array:
        count=jnp.maximum(jnp.sum(current_mask,axis=0),1.)
        centered=values-b_values[None]
        robust=jnp.minimum(
            1.,(2*current_huber)/jnp.maximum(jnp.abs(centered),1e-12))
        weighted=centered*robust*jnp.sqrt(spatial_values)[None]
        weighted_mean=jnp.sum(current_mask*weighted,axis=0)/count
        centered_weighted=current_mask*(weighted-weighted_mean[None])
        normal=jnp.einsum(
            "np,nk,nl->pkl",current_mask,geometry_scores,
            geometry_scores)/count[:,None,None]
        normal+=curvature_regression_lambda*eye[None]
        rhs=jnp.einsum(
            "np,nk,np->pk",current_mask,geometry_scores,
            centered_weighted)/count[:,None]
        coefficients=jnp.linalg.solve(normal,rhs[...,None])[...,0]
        predicted=jnp.einsum("nk,pk->np",geometry_scores,coefficients)
        predicted*=current_mask
        predicted_mean=jnp.sum(predicted,axis=0)/count
        return current_mask*(predicted-predicted_mean[None])

    flat_b=b_field.reshape(pixel_count,3)
    flat_spatial=spatial.reshape(pixel_count)
    target_modes=jnp.zeros((m_count,pixel_count,3),jnp.float32)

    def projected_chunk(start: int,stop: int,channel: int) -> Array:
        return project_pixel_chunk(
            jax.device_put(np.ascontiguousarray(
                flat_samples[:,start:stop,channel]),device),
            jax.device_put(np.ascontiguousarray(
                flat_mask[:,start:stop],dtype=np.float32),device),
            flat_b[start:stop,channel],flat_spatial[start:stop],
            channel_huber[channel])

    for channel in range(3):
        print(f"GPU M 几何投影/Gram：channel={channel+1}/3，"
              f"pixel_chunk={pixel_chunk_size}")
        gram=jnp.zeros((sample_count,sample_count),jnp.float32)
        for start in range(0,pixel_count,pixel_chunk_size):
            stop=min(start+pixel_chunk_size,pixel_count)
            projected=projected_chunk(start,stop,channel)
            gram+=projected@projected.T
            if stop==pixel_count or (start//pixel_chunk_size+1)%16==0:
                gram.block_until_ready()
                print(f"  Gram pixels {stop}/{pixel_count}")
        gram=.5*(gram+gram.T)
        _,eigenvectors=jnp.linalg.eigh(gram)
        leading=jnp.flip(eigenvectors[:,-m_count:],axis=1)
        leading.block_until_ready()
        for start in range(0,pixel_count,pixel_chunk_size):
            stop=min(start+pixel_chunk_size,pixel_count)
            projected=projected_chunk(start,stop,channel)
            mode_chunk=(leading.T@projected)/np.sqrt(sample_count)
            target_modes=target_modes.at[:,start:stop,channel].set(mode_chunk)
            if stop==pixel_count or (start//pixel_chunk_size+1)%16==0:
                target_modes.block_until_ready()
                print(f"  modes pixels {stop}/{pixel_count}")

    coverage=jax.device_put(coverage_np,device)
    m_coefficients=fit_spline_fields(
        target_modes.reshape(m_count,rows,columns,3),coverage)
    # 先裁剪 raw M，再拟合所有训练帧的模式分数。
    raw_peak=jnp.maximum(jnp.max(jnp.abs(m_coefficients),axis=(2,3)),1e-12)
    raw_factor=jnp.minimum(1.,m_max_deviation/raw_peak)
    m_coefficients*=raw_factor[...,None,None]
    m_fields=jax.vmap(
        lambda coefficients:rgb_bspline_field((rows,columns),coefficients))(
            m_coefficients)

    @jax.jit
    def fit_score_batch(batch_samples: Array,batch_mask: Array) -> Array:
        centered=batch_samples-b_field[None]
        base=batch_mask[...,None]*spatial[None,...,None]

        def iteration(_,scores):
            prediction=jnp.einsum("bck,khwc->bhwc",scores,m_fields)
            error=centered-prediction
            robust=jnp.minimum(
                1.,channel_huber[None,None,None,:]
                /jnp.maximum(jnp.abs(error),1e-12))
            weight=base*robust
            normal=jnp.einsum(
                "bhwc,khwc,lhwc->bckl",weight,m_fields,m_fields)
            normal+=1e-8*jnp.eye(m_count,dtype=jnp.float32)[None,None]
            rhs=jnp.einsum(
                "bhwc,khwc,bhwc->bck",weight,m_fields,centered)
            return jnp.clip(
                jnp.linalg.solve(normal,rhs[...,None])[...,0],-10.,10.)

        return jax.lax.fori_loop(
            0,5,iteration,jnp.zeros(
                (batch_samples.shape[0],3,m_count),jnp.float32))

    print(f"GPU M 分数拟合：samples={sample_count}，batch={sample_batch_size}")
    score_parts=[]
    for start in range(0,sample_count,sample_batch_size):
        stop=min(start+sample_batch_size,sample_count)
        current_scores=fit_score_batch(
            jax.device_put(np.ascontiguousarray(samples[start:stop]),device),
            jax.device_put(np.ascontiguousarray(
                mask[start:stop],dtype=np.float32),device))
        current_scores.block_until_ready()
        score_parts.append(np.asarray(current_scores))
        if stop==sample_count or (start//sample_batch_size+1)%25==0:
            print(f"  M scores {stop}/{sample_count}")
    training_scores=jax.device_put(
        np.ascontiguousarray(np.concatenate(score_parts,axis=0)),device)

    @jax.jit
    def normalise_modes(
        coefficients: Array,scores: Array,
    ) -> tuple[Array,Array]:
        score_rms=jnp.maximum(jnp.sqrt(jnp.mean(scores**2,axis=0)),1e-6)
        coefficients*=score_rms.T[...,None,None]
        scores/=score_rms[None]
        peak=jnp.maximum(jnp.max(jnp.abs(coefficients),axis=(2,3)),1e-12)
        factor=jnp.minimum(1.,m_max_deviation/peak)
        coefficients*=factor[...,None,None]
        scores/=factor.T[None]
        flattened=coefficients.reshape(m_count,3,-1)
        pivot_index=jnp.argmax(jnp.abs(flattened),axis=-1)
        pivot=jnp.take_along_axis(
            flattened,pivot_index[...,None],axis=-1)[...,0]
        sign=jnp.where(pivot<0,-1.,1.)
        coefficients*=sign[...,None,None]
        scores*=sign.T[None]
        return coefficients,scores

    m_coefficients,training_scores=normalise_modes(
        m_coefficients,training_scores)
    m_coefficients.block_until_ready()
    return (np.asarray(b_coefficients,dtype=np.float32),
            np.asarray(m_coefficients,dtype=np.float32),
            np.asarray(training_scores,dtype=np.float32))
