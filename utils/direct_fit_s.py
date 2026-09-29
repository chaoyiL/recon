"""direct_fit_s：测长约束的 s-warp、GRU 与冻结 warp 的颜色拟合。"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from utils.config import DirectFitSConfig
from utils.gpu_residual_fit import _fit_direct_static_base_gpu
from utils.lightfield import (
    DIRECT_LOCAL_GEOMETRY_FEATURE_COUNT,
    LightFieldModel,
    direct_background_features_jax,
    direct_geometry_descriptor_jax,
    direct_local_geometry_feature_grid_jax,
    direct_s_background_rgb_jax,
    direct_s_background_rgb_with_scores_jax,
    direct_s_interval_from_logits_jax,
    direct_s_normalized_appearance_scores_jax,
    direct_s_recurrent_step_jax,
    sample_direct_base_texture_jax,
    sample_direct_local_geometry_feature_grid_jax,
)


@dataclass(frozen=True)
class DirectFitSTrainingResult:
    base_texture: np.ndarray
    coordinate_frequencies: np.ndarray
    geometry_feature_mean: np.ndarray
    geometry_feature_scale: np.ndarray
    geometry_pca_components: np.ndarray
    geometry_pca_scale: np.ndarray
    local_geometry_feature_mean: np.ndarray
    local_geometry_feature_scale: np.ndarray
    geometry_encoder_weights: tuple[np.ndarray,...]
    geometry_encoder_biases: tuple[np.ndarray,...]
    gru_input_weight: np.ndarray
    gru_recurrent_weight: np.ndarray
    gru_bias: np.ndarray
    warp_weight: np.ndarray
    warp_bias: np.ndarray
    appearance_mean: np.ndarray
    appearance_components: np.ndarray
    appearance_score_mean: np.ndarray
    appearance_score_scale: np.ndarray
    appearance_score_clip: float
    appearance_weight: np.ndarray
    appearance_bias: np.ndarray
    appearance_memory_hidden_mean: np.ndarray
    appearance_memory_hidden_scale: np.ndarray
    appearance_memory_anchors: np.ndarray
    appearance_memory_residuals: np.ndarray
    appearance_memory_neighbors: int
    appearance_memory_epsilon: float
    length_reference_mm: float
    color_trunk_weights: tuple[np.ndarray,...]
    color_trunk_biases: tuple[np.ndarray,...]
    channel_head_weights: tuple[tuple[np.ndarray,...],...]
    channel_head_biases: tuple[tuple[np.ndarray,...],...]
    evaluation_metrics: dict[str,dict[str,float]]


def _segments_with_minimum_length(
    sequences: list[np.ndarray],minimum: int,
) -> list[np.ndarray]:
    result=[]
    for sequence in sequences:
        values=np.asarray(sequence,np.int64).reshape(-1)
        if values.size>=minimum:
            result.append(values)
    return result


def _raw_surface_lengths(raw_surfaces: np.ndarray) -> np.ndarray:
    values=np.asarray(raw_surfaces,np.float32)
    curve=values[:,:,0,1:3]
    return np.sum(np.linalg.norm(np.diff(curve,axis=1),axis=-1),axis=1)


def _measured_visible_fractions(
    lengths: np.ndarray,reference: float,*,full_threshold: float,
    minimum: float,
) -> np.ndarray:
    ratio=np.asarray(lengths,np.float32)/np.float32(reference)
    clipped=np.clip(ratio,minimum,1.)
    return np.where(ratio>=full_threshold,1.,clipped).astype(np.float32)


def _crop_surface_rows(
    surface: np.ndarray,left: float,visible: float,
) -> np.ndarray:
    """从完整 raw XYZ 裁出已知 s 区间并重新采样回原行数。"""
    values=np.asarray(surface,np.float32)
    position=(left+visible*np.linspace(
        0.,1.,values.shape[0],dtype=np.float32))*(values.shape[0]-1)
    lower=np.floor(position).astype(np.int64)
    upper=np.minimum(lower+1,values.shape[0]-1)
    fraction=(position-lower).astype(np.float32)
    return np.ascontiguousarray(
        values[lower]*(1-fraction[:,None,None])
        +values[upper]*fraction[:,None,None],dtype=np.float32)


def _bilinear_sample_numpy(
    texture: np.ndarray,coordinates: np.ndarray,
) -> np.ndarray:
    """在最后两维为 (s,t) 的任意坐标阵列上采样 HxWxC 纹理。"""
    values=np.asarray(texture)
    points=np.asarray(coordinates,np.float32)
    y=np.clip(points[...,0],0,1)*(values.shape[0]-1)
    x=np.clip(points[...,1],0,1)*(values.shape[1]-1)
    y0=np.floor(y).astype(np.int64); x0=np.floor(x).astype(np.int64)
    y1=np.minimum(y0+1,values.shape[0]-1)
    x1=np.minimum(x0+1,values.shape[1]-1)
    fy=y-y0; fx=x-x0
    return ((1-fy)[...,None]*(1-fx)[...,None]*values[y0,x0]
            +(1-fy)[...,None]*fx[...,None]*values[y0,x1]
            +fy[...,None]*(1-fx)[...,None]*values[y1,x0]
            +fy[...,None]*fx[...,None]*values[y1,x1])


def _nearest_mask_numpy(mask: np.ndarray,coordinates: np.ndarray) -> np.ndarray:
    points=np.asarray(coordinates,np.float32)
    y=np.rint(np.clip(points[...,0],0,1)*(mask.shape[0]-1)).astype(np.int64)
    x=np.rint(np.clip(points[...,1],0,1)*(mask.shape[1]-1)).astype(np.int64)
    return np.asarray(mask,np.bool_)[y,x]


def _nearest_rms_distance(
    query: np.ndarray,reference: np.ndarray,*,exclude_self: bool=False,
) -> np.ndarray:
    """返回按特征维数归一化的最近欧氏距离。"""
    query_values=np.asarray(query,np.float64)
    reference_values=np.asarray(reference,np.float64)
    if query_values.ndim!=2 or reference_values.ndim!=2 \
            or query_values.shape[1]!=reference_values.shape[1]:
        raise ValueError("最近邻特征尺寸不一致")
    if exclude_self and query_values.shape[0]!=reference_values.shape[0]:
        raise ValueError("exclude_self 要求 query/reference 行数一致")
    result=[]
    reference_norm=np.sum(reference_values**2,axis=1)
    for start in range(0,query_values.shape[0],128):
        current=query_values[start:start+128]
        squared=(np.sum(current**2,axis=1,keepdims=True)
                 +reference_norm[None]
                 -2*current@reference_values.T)
        squared=np.maximum(squared,0)
        if exclude_self:
            rows=np.arange(current.shape[0])
            squared[rows,start+rows]=np.inf
        result.append(np.sqrt(np.min(squared,axis=1)
                              /max(query_values.shape[1],1)))
    return np.concatenate(result)


def _correlations(x: np.ndarray,y: np.ndarray) -> tuple[float,float]:
    first=np.asarray(x,np.float64).reshape(-1)
    second=np.asarray(y,np.float64).reshape(-1)
    if first.size<2 or np.std(first)<1e-12 or np.std(second)<1e-12:
        return float("nan"),float("nan")
    pearson=float(np.corrcoef(first,second)[0,1])
    first_rank=np.empty(first.size,np.float64)
    second_rank=np.empty(second.size,np.float64)
    first_rank[np.argsort(first,kind="stable")]=np.arange(first.size)
    second_rank[np.argsort(second,kind="stable")]=np.arange(second.size)
    spearman=float(np.corrcoef(first_rank,second_rank)[0,1])
    return pearson,spearman


def _fit_rgb_affine(
    prediction: np.ndarray,target: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    values=np.asarray(prediction,np.float64)
    truth=np.asarray(target,np.float64)
    gains=[]; biases=[]; corrected=[]
    for channel in range(3):
        x=values[...,channel].reshape(-1)
        y=truth[...,channel].reshape(-1)
        matrix=np.stack([x,np.ones_like(x)],axis=1)
        solution=np.linalg.lstsq(matrix,y,rcond=None)[0]
        gains.append(solution[0]); biases.append(solution[1])
        corrected.append(solution[0]*values[...,channel]+solution[1])
    return (np.asarray(gains),np.asarray(biases),
            np.clip(np.stack(corrected,axis=-1),0,1))


def _warp_candidate_errors(
    base: np.ndarray,coverage: np.ndarray,coordinates: np.ndarray,
    target: np.ndarray,visible: np.ndarray,q: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    visible_values=np.asarray(visible,np.float32).reshape(-1)
    q_values=np.asarray(q,np.float32).reshape(-1)
    left=(1-visible_values)*q_values
    warped=np.broadcast_to(
        np.asarray(coordinates,np.float32)[None],
        (visible_values.size,*coordinates.shape)).copy()
    warped[...,0]=(left[:,None]
                   +visible_values[:,None]*coordinates[None,:,0])
    prediction=_bilinear_sample_numpy(base,warped)
    common=(_nearest_mask_numpy(coverage,coordinates)[None]
            &_nearest_mask_numpy(coverage,warped))
    count=np.maximum(np.sum(common,axis=1)*3,1)
    sse=np.sum(np.where(
        common[...,None],(prediction-target[None])**2,0),axis=(1,2))
    return sse/count,sse,count


def _fit_warp_oracle(
    base: np.ndarray,coverage: np.ndarray,coordinates: np.ndarray,
    target: np.ndarray,predicted_interval: np.ndarray,
    minimum_visible: float,
) -> tuple[np.ndarray,float,float]:
    """以真实 RGB 搜索每帧最优仿射 s 区间，仅作为误差下界。"""
    coarse_count=25
    visible_axis=np.linspace(minimum_visible,1.,coarse_count,dtype=np.float32)
    q_axis=np.linspace(0.,1.,coarse_count,dtype=np.float32)
    visible_grid,q_grid=np.meshgrid(visible_axis,q_axis,indexing="ij")
    candidates_visible=visible_grid.reshape(-1)
    candidates_q=q_grid.reshape(-1)
    predicted_visible=float(predicted_interval[1])
    predicted_missing=max(1-predicted_visible,1e-8)
    predicted_q=float(np.clip(
        predicted_interval[0]/predicted_missing,0,1))
    candidates_visible=np.concatenate([
        candidates_visible,np.asarray([predicted_visible,1.],np.float32)])
    candidates_q=np.concatenate([
        candidates_q,np.asarray([predicted_q,.5],np.float32)])
    mse,_,_=_warp_candidate_errors(
        base,coverage,coordinates,target,candidates_visible,candidates_q)
    best=int(np.argmin(mse))
    visible0=float(candidates_visible[best]); q0=float(candidates_q[best])
    visible_step=(1-minimum_visible)/(coarse_count-1)
    q_step=1/(coarse_count-1)
    visible_refined=np.linspace(
        max(minimum_visible,visible0-visible_step),
        min(1.,visible0+visible_step),13,dtype=np.float32)
    q_refined=np.linspace(
        max(0.,q0-q_step),min(1.,q0+q_step),13,dtype=np.float32)
    visible_grid,q_grid=np.meshgrid(
        visible_refined,q_refined,indexing="ij")
    mse,sse,count=_warp_candidate_errors(
        base,coverage,coordinates,target,
        visible_grid.reshape(-1),q_grid.reshape(-1))
    best=int(np.argmin(mse))
    visible_best=float(visible_grid.reshape(-1)[best])
    q_best=float(q_grid.reshape(-1)[best])
    left=(1-visible_best)*q_best
    return (np.asarray([left,visible_best,1-visible_best-left],np.float32),
            float(sse[best]),float(count[best]))


def diagnose_direct_fit_s_gpu(
    model: LightFieldModel,*,training_fields: np.ndarray,
    training_valid: np.ndarray,training_raw_observed_xyz: np.ndarray,
    trusted_indices: np.ndarray,validation_fields: np.ndarray,
    validation_valid: np.ndarray,validation_surface_xyz: np.ndarray,
    validation_raw_observed_xyz: np.ndarray,
    validation_sequences: list[np.ndarray],device: jax.Device,
    evaluation_points_per_frame: int,
) -> dict[str,object]:
    """用冻结模型和留出 RGB 分解 direct_fit_s 的泛化误差。"""
    if model.background_method!="direct_fit_s":
        raise ValueError("误差诊断只接受 direct_fit_s 模型")
    training_samples=np.asarray(training_fields,np.float32)
    training_masks=np.asarray(training_valid,np.bool_)
    training_raw=np.asarray(training_raw_observed_xyz,np.float32)
    samples=np.asarray(validation_fields,np.float32)
    masks=np.asarray(validation_valid,np.bool_)
    surfaces=np.asarray(validation_surface_xyz,np.float32)
    raw_surfaces=np.asarray(validation_raw_observed_xyz,np.float32)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3] \
            or surfaces.ndim!=4 or raw_surfaces.ndim!=4 \
            or surfaces.shape[0]!=samples.shape[0] \
            or raw_surfaces.shape[0]!=samples.shape[0] \
            or surfaces.shape[-1]!=3 or raw_surfaces.shape[-1]!=3:
        raise ValueError("direct_fit_s 诊断验证数组尺寸不一致")
    if training_samples.ndim!=4 or training_masks.shape!=training_samples.shape[:3] \
            or training_raw.shape[0]!=training_samples.shape[0]:
        raise ValueError("direct_fit_s 诊断训练数组尺寸不一致")
    if evaluation_points_per_frame<1:
        raise ValueError("evaluation_points_per_frame 必须为正")
    required=(model.direct_base_texture,model.direct_s_appearance_mean,
              model.direct_s_appearance_components,
              model.direct_s_appearance_score_mean,
              model.direct_s_appearance_score_scale,
              model.direct_s_appearance_score_clip,
              model.direct_geometry_feature_mean,
              model.direct_geometry_feature_scale,
              model.direct_geometry_pca_components,
              model.direct_geometry_pca_scale,
              model.direct_s_appearance_memory_hidden_mean,
              model.direct_s_appearance_memory_hidden_scale,
              model.direct_s_appearance_memory_anchors,
              model.direct_s_minimum_visible_fraction)
    if any(value is None for value in required):
        raise ValueError("direct_fit_s 模型缺少诊断所需参数")

    frame_count=samples.shape[0]; rows,columns=samples.shape[1:3]
    coordinates=[]; targets=[]
    flattened=samples.reshape(frame_count,-1,3)
    for index in range(frame_count):
        available=np.flatnonzero(masks[index].reshape(-1))
        if not available.size:
            raise ValueError(f"验证帧 {index} 没有有效颜色点")
        positions=np.rint(np.linspace(
            0,available.size-1,evaluation_points_per_frame)).astype(np.int64)
        pixels=available[positions]
        coordinates.append(np.stack([
            (pixels//columns)/max(rows-1,1),
            (pixels%columns)/max(columns-1,1)],axis=-1))
        targets.append(flattened[index,pixels])
    coordinates=np.ascontiguousarray(coordinates,dtype=np.float32)
    targets=np.ascontiguousarray(targets,dtype=np.float32)

    hidden_count=int(model.direct_s_gru_recurrent_weight.shape[0])
    hidden_values=np.zeros((frame_count,hidden_count),np.float32)
    intervals=np.zeros((frame_count,3),np.float32)
    geometry_pca=np.zeros((
        frame_count,model.direct_geometry_pca_components.shape[1]),np.float32)
    normalized_scores=np.zeros((
        frame_count,model.direct_s_appearance_components.shape[0]),np.float32)
    predictions=np.zeros_like(targets)

    @jax.jit
    def forward_frame(hidden,surface,raw,point_coordinates):
        hidden,interval,_,pca=direct_s_recurrent_step_jax(raw,hidden,model)
        prediction=direct_s_background_rgb_jax(
            point_coordinates,surface,raw,hidden,interval,model)
        score=direct_s_normalized_appearance_scores_jax(hidden,model)
        return hidden,interval,pca,score,prediction

    visited=np.zeros((frame_count,),np.bool_)
    for sequence in validation_sequences:
        hidden=jnp.zeros((hidden_count,),jnp.float32)
        for source in np.asarray(sequence,np.int64):
            index=int(source)
            if index<0 or index>=frame_count or visited[index]:
                raise ValueError("验证序列索引越界或重复")
            output=jax.device_get(forward_frame(
                hidden,jax.device_put(surfaces[index],device),
                jax.device_put(raw_surfaces[index],device),
                jax.device_put(coordinates[index],device)))
            hidden=jax.device_put(output[0],device)
            hidden_values[index]=output[0]; intervals[index]=output[1]
            geometry_pca[index]=output[2]; normalized_scores[index]=output[3]
            predictions[index]=output[4]; visited[index]=True
    if not np.all(visited):
        raise ValueError("验证序列未覆盖全部验证帧")
    final_frame_rmse=np.sqrt(np.mean(
        (predictions-targets)**2,axis=(1,2),dtype=np.float64))
    final_rmse=float(np.sqrt(np.mean(
        (predictions-targets)**2,dtype=np.float64)))

    base=np.asarray(model.direct_base_texture,np.float32)
    base_coverage=np.any(
        training_masks[np.asarray(trusted_indices,np.int64)],axis=0)
    warped_coordinates=coordinates.copy()
    warped_coordinates[...,0]=(intervals[:,None,0]
                                +intervals[:,None,1]*coordinates[...,0])
    base_at_warp=_bilinear_sample_numpy(base,warped_coordinates)
    common=(_nearest_mask_numpy(base_coverage,coordinates)
            &_nearest_mask_numpy(base_coverage,warped_coordinates))
    predicted_warp_sse=float(np.sum(np.where(
        common[...,None],(base_at_warp-targets)**2,0),dtype=np.float64))
    predicted_warp_count=max(int(np.sum(common))*3,1)
    oracle_intervals=[]; oracle_sse=0.; oracle_count=0.
    minimum_visible=float(model.direct_s_minimum_visible_fraction)
    for index in range(frame_count):
        interval,sse,count=_fit_warp_oracle(
            base,base_coverage,coordinates[index],targets[index],
            intervals[index],minimum_visible)
        oracle_intervals.append(interval); oracle_sse+=sse; oracle_count+=count
    oracle_intervals=np.stack(oracle_intervals)

    appearance_mean=np.asarray(model.direct_s_appearance_mean,np.float32)
    components=np.asarray(model.direct_s_appearance_components,np.float32)
    appearance_rows,appearance_columns=appearance_mean.shape[:2]
    row_indices=np.rint(np.linspace(
        0,rows-1,appearance_rows)).astype(np.int64)
    column_indices=np.rint(np.linspace(
        0,columns-1,appearance_columns)).astype(np.int64)
    descriptor_targets=samples[:,row_indices[:,None],column_indices[None,:]]
    descriptor_valid=masks[:,row_indices[:,None],column_indices[None,:]]
    descriptor_s,descriptor_t=np.meshgrid(
        row_indices/max(rows-1,1),column_indices/max(columns-1,1),
        indexing="ij")
    descriptor_coordinates=np.stack(
        [descriptor_s,descriptor_t],axis=-1).astype(np.float32)
    descriptor_warped=np.broadcast_to(
        descriptor_coordinates[None],
        (frame_count,*descriptor_coordinates.shape)).copy()
    descriptor_warped[...,0]=(intervals[:,None,None,0]
                               +intervals[:,None,None,1]
                               *descriptor_coordinates[None,...,0])
    descriptor_base=_bilinear_sample_numpy(base,descriptor_warped)
    centered=np.where(
        descriptor_valid[...,None],
        descriptor_targets-descriptor_base-appearance_mean[None],0)
    component_flat=components.reshape(components.shape[0],-1)
    centered_flat=centered.reshape(frame_count,-1)
    valid_flat=np.repeat(
        descriptor_valid.reshape(frame_count,-1),3,axis=1).astype(np.float32)
    component_gpu=jax.device_put(component_flat,device)

    @jax.jit
    def masked_pca_projection(values,validity):
        def iteration(_,scores):
            residual=(values-scores@component_gpu)*validity
            return scores+residual@component_gpu.T
        initial=jnp.zeros((values.shape[0],component_gpu.shape[0]),values.dtype)
        return jax.lax.fori_loop(0,12,iteration,initial)

    oracle_scores=np.asarray(jax.device_get(masked_pca_projection(
        jax.device_put(centered_flat,device),
        jax.device_put(valid_flat,device))),np.float32)
    oracle_fields=(appearance_mean[None]+(
        oracle_scores@component_flat).reshape(
            frame_count,*appearance_mean.shape))
    sample_field_batch=jax.jit(jax.vmap(
        lambda field,point:sample_direct_base_texture_jax(field,point)))
    oracle_residual=np.asarray(jax.device_get(sample_field_batch(
        jax.device_put(oracle_fields,device),
        jax.device_put(coordinates,device))),np.float32)
    oracle_prediction=np.clip(base_at_warp+oracle_residual,0,1)
    oracle_frame_rmse=np.sqrt(np.mean(
        (oracle_prediction-targets)**2,axis=(1,2),dtype=np.float64))
    oracle_rmse=float(np.sqrt(np.mean(
        (oracle_prediction-targets)**2,dtype=np.float64)))
    score_mean=np.asarray(model.direct_s_appearance_score_mean,np.float32)
    score_scale=np.asarray(model.direct_s_appearance_score_scale,np.float32)
    score_clip=float(model.direct_s_appearance_score_clip)
    predicted_scores=(score_mean[None]
                      +normalized_scores*score_clip*score_scale[None])
    oracle_normalized=(oracle_scores-score_mean[None]) \
        /(score_clip*score_scale[None])
    oracle_decoder_predictions=np.zeros_like(targets)

    @jax.jit
    def oracle_decoder_frame(surface,raw,hidden,interval,score,point):
        return direct_s_background_rgb_with_scores_jax(
            point,surface,raw,hidden,interval,score,model)

    for index in range(frame_count):
        oracle_decoder_predictions[index]=jax.device_get(
            oracle_decoder_frame(
                jax.device_put(surfaces[index],device),
                jax.device_put(raw_surfaces[index],device),
                jax.device_put(hidden_values[index],device),
                jax.device_put(intervals[index],device),
                jax.device_put(np.clip(
                    oracle_normalized[index],-1,1),device),
                jax.device_put(coordinates[index],device)))
    oracle_decoder_rmse=float(np.sqrt(np.mean(
        (oracle_decoder_predictions-targets)**2,dtype=np.float64)))
    predicted_fields=(appearance_mean[None]+(
        predicted_scores@component_flat).reshape(
            frame_count,*appearance_mean.shape))
    predicted_residual=np.asarray(jax.device_get(sample_field_batch(
        jax.device_put(predicted_fields,device),
        jax.device_put(coordinates,device))),np.float32)
    predicted_prior=np.clip(base_at_warp+predicted_residual,0,1)
    predicted_prior_rmse=float(np.sqrt(np.mean(
        (predicted_prior-targets)**2,dtype=np.float64)))
    descriptor_reconstruction=(appearance_mean[None]+(
        oracle_scores@component_flat).reshape(
            frame_count,*appearance_mean.shape))
    descriptor_error=(descriptor_base+descriptor_reconstruction
                      -descriptor_targets)
    descriptor_count=max(int(np.sum(descriptor_valid))*3,1)
    descriptor_oracle_rmse=float(np.sqrt(np.sum(np.where(
        descriptor_valid[...,None],descriptor_error**2,0),dtype=np.float64)
        /descriptor_count))

    global_gain,global_bias,global_corrected=_fit_rgb_affine(
        predictions,targets)
    global_affine_rmse=float(np.sqrt(np.mean(
        (global_corrected-targets)**2,dtype=np.float64)))
    frame_gains=[]; frame_biases=[]; frame_corrected=[]
    for index in range(frame_count):
        gain,bias,corrected=_fit_rgb_affine(
            predictions[index],targets[index])
        frame_gains.append(gain); frame_biases.append(bias)
        frame_corrected.append(corrected)
    frame_gains=np.stack(frame_gains); frame_biases=np.stack(frame_biases)
    frame_affine_rmse=float(np.sqrt(np.mean(
        (np.stack(frame_corrected)-targets)**2,dtype=np.float64)))

    descriptor_batch=jax.jit(jax.vmap(
        lambda value:direct_geometry_descriptor_jax(
            value,model.direct_geometry_descriptor_rows)))
    training_descriptors=[]
    for start in range(0,training_raw.shape[0],32):
        training_descriptors.append(np.asarray(jax.device_get(
            descriptor_batch(jax.device_put(
                training_raw[start:start+32],device))),np.float32))
    validation_descriptors=[]
    for start in range(0,raw_surfaces.shape[0],32):
        validation_descriptors.append(np.asarray(jax.device_get(
            descriptor_batch(jax.device_put(
                raw_surfaces[start:start+32],device))),np.float32))
    training_descriptors=np.concatenate(training_descriptors)
    validation_descriptors=np.concatenate(validation_descriptors)
    feature_mean=np.asarray(model.direct_geometry_feature_mean,np.float32)
    feature_scale=np.asarray(model.direct_geometry_feature_scale,np.float32)
    training_normalized=(training_descriptors-feature_mean)/feature_scale
    validation_normalized=(validation_descriptors-feature_mean)/feature_scale
    geometry_components=np.asarray(
        model.direct_geometry_pca_components,np.float32)
    geometry_scale=np.asarray(model.direct_geometry_pca_scale,np.float32)
    training_pca=training_normalized@geometry_components/geometry_scale
    validation_pca=validation_normalized@geometry_components/geometry_scale
    descriptor_distance=_nearest_rms_distance(
        validation_normalized,training_normalized)
    descriptor_train_loo=_nearest_rms_distance(
        training_normalized,training_normalized,exclude_self=True)
    pca_distance=_nearest_rms_distance(validation_pca,training_pca)
    pca_train_loo=_nearest_rms_distance(
        training_pca,training_pca,exclude_self=True)
    hidden_mean=np.asarray(
        model.direct_s_appearance_memory_hidden_mean,np.float32)
    hidden_scale=np.asarray(
        model.direct_s_appearance_memory_hidden_scale,np.float32)
    anchors=np.asarray(model.direct_s_appearance_memory_anchors,np.float32)
    hidden_query=(hidden_values-hidden_mean)/hidden_scale
    hidden_distance=_nearest_rms_distance(hidden_query,anchors)
    hidden_train_loo=_nearest_rms_distance(
        anchors,anchors,exclude_self=True)

    def distance_metrics(
        values: np.ndarray,training_loo: np.ndarray,
    ) -> dict[str,object]:
        pearson,spearman=_correlations(values,final_frame_rmse)
        threshold=float(np.quantile(training_loo,.95))
        order=np.argsort(values)
        quartile=max(values.size//4,1)
        return {
            "nearest_distance_p10_median_p90":[float(value) for value in
                np.quantile(values,[.1,.5,.9])],
            "training_leave_one_out_p95":threshold,
            "validation_above_training_p95_fraction":float(np.mean(
                values>threshold)),
            "error_pearson":pearson,"error_spearman":spearman,
            "closest_quartile_frame_rmse":float(np.mean(
                final_frame_rmse[order[:quartile]])),
            "farthest_quartile_frame_rmse":float(np.mean(
                final_frame_rmse[order[-quartile:]])),
        }

    descriptor_rows=model.direct_geometry_descriptor_rows
    group_slices={
        "global_position_and_scale":slice(0,6),
        "relative_centerline":slice(6,6+3*descriptor_rows),
        "curvature":slice(6+3*descriptor_rows,6+6*descriptor_rows),
        "width":slice(6+6*descriptor_rows,6+7*descriptor_rows),
        "mean_normal":slice(6+7*descriptor_rows,6+10*descriptor_rows),
    }
    descriptor_groups={}
    for name,group in group_slices.items():
        group_validation=_nearest_rms_distance(
            validation_normalized[:,group],training_normalized[:,group])
        group_training=_nearest_rms_distance(
            training_normalized[:,group],training_normalized[:,group],
            exclude_self=True)
        descriptor_groups[name]=distance_metrics(
            group_validation,group_training)

    metrics: dict[str,object]={
        "evaluation":{
            "frame_count":int(frame_count),"points_per_frame":int(
                evaluation_points_per_frame),"final_rmse":final_rmse,
            "frame_rmse_median":float(np.median(final_frame_rmse)),
            "frame_rmse_p90":float(np.quantile(final_frame_rmse,.9)),
            "frame_rmse_p95":float(np.quantile(final_frame_rmse,.95)),
        },
        "warp_oracle":{
            "predicted_B_warp_rmse":float(np.sqrt(
                predicted_warp_sse/predicted_warp_count)),
            "oracle_B_warp_rmse":float(np.sqrt(
                oracle_sse/max(oracle_count,1))),
            "oracle_relative_mse_reduction":float(
                1-(oracle_sse/max(oracle_count,1))
                /(predicted_warp_sse/predicted_warp_count)),
            "interval_absolute_mae":float(np.mean(np.abs(
                oracle_intervals-intervals))),
            "predicted_visible_median":float(np.median(intervals[:,1])),
            "oracle_visible_median":float(np.median(
                oracle_intervals[:,1])),
        },
        "appearance_oracle":{
            "predicted_prior_rmse":predicted_prior_rmse,
            "fixed_training_basis_oracle_rmse":oracle_rmse,
            "fixed_training_basis_oracle_with_decoder_rmse":
                oracle_decoder_rmse,
            "descriptor_oracle_rmse":descriptor_oracle_rmse,
            "oracle_relative_mse_reduction":float(
                1-oracle_rmse**2/predicted_prior_rmse**2),
            "normalized_score_mae":float(np.mean(np.abs(
                normalized_scores-oracle_normalized))),
            "normalized_score_rmse":float(np.sqrt(np.mean(
                (normalized_scores-oracle_normalized)**2))),
            "oracle_score_outside_training_clip_fraction":float(np.mean(
                np.abs(oracle_normalized)>1)),
            "oracle_frame_rmse_median":float(np.median(oracle_frame_rmse)),
            "oracle_frame_rmse_p95":float(np.quantile(
                oracle_frame_rmse,.95)),
        },
        "photometric_oracle":{
            "original_final_rmse":final_rmse,
            "per_video_channel_gain_bias_rmse":global_affine_rmse,
            "per_video_relative_mse_reduction":float(
                1-global_affine_rmse**2/final_rmse**2),
            "per_video_rgb_gain":global_gain.tolist(),
            "per_video_rgb_bias":global_bias.tolist(),
            "per_frame_channel_gain_bias_rmse":frame_affine_rmse,
            "per_frame_relative_mse_reduction":float(
                1-frame_affine_rmse**2/final_rmse**2),
            "per_frame_gain_p10_median_p90":np.quantile(
                frame_gains,[.1,.5,.9],axis=0).tolist(),
            "per_frame_bias_p10_median_p90":np.quantile(
                frame_biases,[.1,.5,.9],axis=0).tolist(),
        },
        "geometry_distribution":{
            "standardized_descriptor":distance_metrics(
                descriptor_distance,descriptor_train_loo),
            "descriptor_groups":descriptor_groups,
            "geometry_pca":distance_metrics(pca_distance,pca_train_loo),
            "memory_hidden":distance_metrics(
                hidden_distance,hidden_train_loo),
        },
    }
    print(
        "direct_fit_s 冻结模型诊断："
        f"final={final_rmse:.7f}，"
        f"warp predicted/oracle={metrics['warp_oracle']['predicted_B_warp_rmse']:.7f}/"
        f"{metrics['warp_oracle']['oracle_B_warp_rmse']:.7f}，"
        f"appearance predicted/oracle={predicted_prior_rmse:.7f}/"
        f"{oracle_rmse:.7f}，oracle+decoder={oracle_decoder_rmse:.7f}，"
        f"gain-bias video/frame={global_affine_rmse:.7f}/{frame_affine_rmse:.7f}")
    return metrics


def fit_direct_fit_s_gpu(
    fields: np.ndarray,valid: np.ndarray,*,surface_xyz: np.ndarray,
    raw_observed_xyz: np.ndarray,observation_confidence: np.ndarray,
    trusted_indices: np.ndarray,
    trusted_sequences: list[np.ndarray],sequence_sequences: list[np.ndarray],
    device: jax.Device,config: DirectFitSConfig,huber_delta: float,seed: int,
    validation_fields: np.ndarray | None = None,
    validation_valid: np.ndarray | None = None,
    validation_surface_xyz: np.ndarray | None = None,
    validation_raw_observed_xyz: np.ndarray | None = None,
    validation_sequences: list[np.ndarray] | None = None,
) -> DirectFitSTrainingResult:
    """依次训练合成 warp、真实 B 对齐和冻结 warp 的颜色网络。"""
    samples=np.asarray(fields,np.float32)
    masks=np.asarray(valid,np.bool_)
    surfaces=np.asarray(surface_xyz,np.float32)
    raw_surfaces=np.asarray(raw_observed_xyz,np.float32)
    confidence=np.asarray(observation_confidence,np.float32).reshape(-1)
    trusted=np.asarray(trusted_indices,np.int64).reshape(-1)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3]:
        raise ValueError("direct_fit_s fields/valid 必须是 NxHxWx3/NxHxW")
    if surfaces.ndim!=4 or raw_surfaces.ndim!=4 \
            or surfaces.shape[0]!=samples.shape[0] \
            or raw_surfaces.shape[0]!=samples.shape[0] \
            or surfaces.shape[-1]!=3 or raw_surfaces.shape[-1]!=3 \
            or not np.isfinite(surfaces).all() \
            or not np.isfinite(raw_surfaces).all():
        raise ValueError("direct_fit_s 整体/原始观测 XYZ 尺寸或数值无效")
    if confidence.shape!=(samples.shape[0],) \
            or not np.isfinite(confidence).all() \
            or np.any(confidence<0) or np.any(confidence>1):
        raise ValueError("direct_fit_s 观测 confidence 必须是每帧 [0,1] 有限值")
    if trusted.size<2:
        raise ValueError("direct_fit_s 静态 B 至少需要两个可信平直帧")
    if not trusted_sequences or not sequence_sequences:
        raise ValueError("direct_fit_s 必须同时包含可信平直序列和弯曲序列")
    if huber_delta<=0:
        raise ValueError("direct_fit_s huber_delta 必须为正")

    base_texture=_fit_direct_static_base_gpu(
        samples[trusted],masks[trusted],device=device,huber_delta=huber_delta,
        iterations=config.base_huber_iterations,
        frame_batch_size=config.base_frame_batch_size)
    base_coverage=np.any(masks[trusted],axis=0)

    lengths=_raw_surface_lengths(raw_surfaces)
    length_reference=float(np.median(lengths[trusted]))
    if not np.isfinite(length_reference) or length_reference<=0:
        raise ValueError("direct_fit_s 可信平直帧无法建立有效弧长基准")
    visible_fractions=_measured_visible_fractions(
        lengths,length_reference,
        full_threshold=config.full_visibility_threshold,
        minimum=config.minimum_visible_fraction)
    visible_fractions[trusted]=1.
    sequence_mask=np.ones(samples.shape[0],np.bool_)
    sequence_mask[trusted]=False
    sequence_visible=visible_fractions[sequence_mask]
    frame_quality=np.maximum(
        config.minimum_frame_quality_weight,
        np.clip(confidence/config.frame_quality_full_confidence,0.,1.)
    ).astype(np.float32)
    # 平直 B 的可信帧由用户明确指定，不再被通用追踪分数降权。
    frame_quality[trusted]=1.
    print(
        "direct_fit_s 测长："
        f"trusted median={length_reference:.4f} mm，"
        f"trusted min/max={float(lengths[trusted].min()):.4f}/"
        f"{float(lengths[trusted].max()):.4f} mm，"
        f"sequence visible min/median/max="
        f"{float(sequence_visible.min()):.4f}/"
        f"{float(np.median(sequence_visible)):.4f}/"
        f"{float(sequence_visible.max()):.4f}，"
        f"sequence quality min/median="
        f"{float(frame_quality[sequence_mask].min()):.3f}/"
        f"{float(np.median(frame_quality[sequence_mask])):.3f}")

    appearance_rows=np.rint(np.linspace(
        0,samples.shape[1]-1,config.appearance_descriptor_rows
    )).astype(np.int64)
    appearance_columns=np.rint(np.linspace(
        0,samples.shape[2]-1,config.appearance_descriptor_columns
    )).astype(np.int64)
    appearance_fields=samples[:,appearance_rows[:,None],
                              appearance_columns[None,:]]
    appearance_valid=masks[:,appearance_rows[:,None],
                           appearance_columns[None,:]]
    appearance_base=base_texture[appearance_rows[:,None],
                                 appearance_columns[None,:]]
    appearance_residual=appearance_fields-appearance_base[None]
    appearance_coverage=np.mean(
        appearance_valid[sequence_mask],axis=0,dtype=np.float64)
    appearance_reliable=(
        appearance_coverage>=config.appearance_minimum_coverage)
    appearance_weight=(frame_quality[:,None,None,None]
                       *sequence_mask[:,None,None,None]
                       *appearance_valid[...,None]
                       *appearance_reliable[None,...,None])
    appearance_mean=np.sum(
        appearance_residual*appearance_weight,axis=0,dtype=np.float64) \
        /np.maximum(np.sum(appearance_weight,axis=0,dtype=np.float64),1e-8)
    appearance_mean=np.asarray(appearance_mean,np.float32)
    appearance_centered=np.where(
        (appearance_valid&appearance_reliable[None])[...,None],
        appearance_residual-appearance_mean[None],0
    ).reshape(samples.shape[0],-1).astype(np.float32)
    if config.appearance_pca_dimensions>min(appearance_centered.shape):
        raise ValueError(
            "direct_fit_s appearance_pca_dimensions 不能大于帧数或外观描述维数")
    weighted_appearance=(appearance_centered[sequence_mask]
                         *np.sqrt(frame_quality[sequence_mask,None]))
    _,appearance_singular,appearance_vh=np.linalg.svd(
        weighted_appearance,full_matrices=False)
    appearance_components=np.ascontiguousarray(
        appearance_vh[:config.appearance_pca_dimensions],dtype=np.float32)
    appearance_scores=appearance_centered@appearance_components.T
    quality_sum=max(float(np.sum(frame_quality[sequence_mask])),1e-8)
    appearance_score_mean=np.sum(
        appearance_scores[sequence_mask]
        *frame_quality[sequence_mask,None],axis=0,
        dtype=np.float64)/quality_sum
    appearance_score_scale=np.sqrt(np.sum(
        (appearance_scores[sequence_mask]-appearance_score_mean[None])**2
        *frame_quality[sequence_mask,None],axis=0,
        dtype=np.float64)/quality_sum)
    appearance_score_scale=np.where(
        appearance_score_scale>1e-6,appearance_score_scale,1.)
    appearance_score_mean=np.asarray(appearance_score_mean,np.float32)
    appearance_score_scale=np.asarray(appearance_score_scale,np.float32)
    appearance_targets=np.clip(
        (appearance_scores-appearance_score_mean[None])
        /appearance_score_scale[None],
        -config.appearance_score_clip,config.appearance_score_clip)
    appearance_targets=(appearance_targets
                        /config.appearance_score_clip).astype(np.float32)
    appearance_energy=float(np.sum(appearance_singular**2))
    appearance_explained=(float(np.sum(appearance_singular[
        :config.appearance_pca_dimensions]**2))/max(appearance_energy,1e-12))
    print(
        "direct_fit_s 动态外观监督："
        f"grid={config.appearance_descriptor_rows}x"
        f"{config.appearance_descriptor_columns}，dim="
        f"{config.appearance_pca_dimensions}，"
        f"reliable={100*float(np.mean(appearance_reliable)):.1f}%，"
        f"weighted explained={100*appearance_explained:.2f}%")

    analyze=jax.jit(jax.vmap(lambda raw,current:(
        direct_geometry_descriptor_jax(raw,config.geometry_descriptor_rows),
        direct_local_geometry_feature_grid_jax(current))))
    descriptor_parts=[]
    local_parts=[]
    for start in range(0,samples.shape[0],32):
        descriptor_batch,local_batch=jax.device_get(analyze(
            jax.device_put(raw_surfaces[start:start+32],device),
            jax.device_put(surfaces[start:start+32],device)))
        descriptor_parts.append(np.asarray(descriptor_batch,np.float32))
        local_parts.append(np.asarray(local_batch,np.float32))
    descriptors=np.ascontiguousarray(np.concatenate(descriptor_parts))
    local_grids=np.ascontiguousarray(np.concatenate(local_parts))
    feature_mean=descriptors.mean(axis=0,dtype=np.float64).astype(np.float32)
    feature_scale=descriptors.std(axis=0,dtype=np.float64).astype(np.float32)
    feature_scale=np.where(feature_scale>1e-6,feature_scale,1.).astype(np.float32)
    normalized=(descriptors-feature_mean)/feature_scale
    if config.geometry_pca_dimensions>min(normalized.shape):
        raise ValueError(
            "direct_fit_s geometry_pca_dimensions 不能大于帧数或描述维数")
    _,_,vh=np.linalg.svd(normalized,full_matrices=False)
    pca_components=np.ascontiguousarray(
        vh[:config.geometry_pca_dimensions].T,dtype=np.float32)
    pca_scores=normalized@pca_components
    pca_scale=pca_scores.std(axis=0,dtype=np.float64).astype(np.float32)
    pca_scale=np.where(pca_scale>1e-6,pca_scale,1.).astype(np.float32)
    local_flat=local_grids.reshape(-1,DIRECT_LOCAL_GEOMETRY_FEATURE_COUNT)
    local_mean=local_flat.mean(axis=0,dtype=np.float64).astype(np.float32)
    local_scale=local_flat.std(axis=0,dtype=np.float64).astype(np.float32)
    local_scale=np.where(local_scale>1e-6,local_scale,1.).astype(np.float32)

    trusted_clips=_segments_with_minimum_length(
        trusted_sequences,config.clip_length)
    sequence_clips=_segments_with_minimum_length(
        sequence_sequences,config.clip_length)
    if not trusted_clips or not sequence_clips:
        raise ValueError(
            "direct_fit_s 每类视频都至少要有一段不短于 clip_length 的连续帧")

    generator=np.random.default_rng(seed)
    frequencies=np.asarray(config.coordinate_frequencies,np.float32)
    coordinate_count=2+4*frequencies.size
    feature_count=descriptors.shape[1]
    latent_count=config.geometry_latent_dimensions
    hidden_count=config.gru_hidden_dimensions
    color_input_count=(coordinate_count+latent_count
                       +config.geometry_pca_dimensions
                       +DIRECT_LOCAL_GEOMETRY_FEATURE_COUNT+hidden_count
                       +config.appearance_pca_dimensions)

    def layer(source: int,target: int,*,output: bool = False):
        limit=np.sqrt(6/max(source+target,1))
        weight=generator.uniform(-limit,limit,(source,target)).astype(np.float32)
        if output:
            weight*=.05
        return jnp.asarray(weight),jnp.zeros((target,),jnp.float32)

    encoder=[]
    previous=feature_count
    for _ in range(config.geometry_encoder_layers):
        encoder.append(layer(previous,config.geometry_encoder_width))
        previous=config.geometry_encoder_width
    encoder.append(layer(previous,latent_count))
    gru_input,_=layer(latent_count+1,3*hidden_count)
    gru_recurrent,_=layer(hidden_count,3*hidden_count)
    gru_bias=jnp.zeros((3*hidden_count,),jnp.float32)
    # 两个输出分别为 q logit 和相对测长先验的 visible-logit 修正。
    warp_weight,_=layer(hidden_count,2,output=True)
    warp_bias=jnp.zeros((2,),jnp.float32)
    appearance_weight_head,_=layer(
        hidden_count,config.appearance_pca_dimensions,output=True)
    appearance_bias=jnp.zeros(
        (config.appearance_pca_dimensions,),jnp.float32)
    trunk=[]
    previous=color_input_count
    for _ in range(config.color_trunk_layers):
        trunk.append(layer(previous,config.color_trunk_width))
        previous=config.color_trunk_width+color_input_count
    heads=[]
    for _ in range(3):
        current=[]
        previous=config.color_trunk_width+color_input_count
        for index in range(config.color_head_layers):
            output=index==config.color_head_layers-1
            target=1 if output else config.color_head_width
            current.append(layer(previous,target,output=output))
            previous=target+color_input_count
        heads.append(tuple(current))
    parameters=jax.device_put({
        "encoder":tuple(encoder),
        "gru":(gru_input,gru_recurrent,gru_bias),
        "warp":(warp_weight,warp_bias),
        "appearance":(appearance_weight_head,appearance_bias),
        "trunk":tuple(trunk),"heads":tuple(heads)},device)
    frequencies_gpu=jax.device_put(frequencies,device)
    pca_components_gpu=jax.device_put(pca_components,device)
    pca_scale_gpu=jax.device_put(pca_scale,device)
    local_mean_gpu=jax.device_put(local_mean,device)
    local_scale_gpu=jax.device_put(local_scale,device)
    base_gpu=jax.device_put(base_texture,device)
    coverage_gpu=jax.device_put(base_coverage,device)
    appearance_mean_gpu=jax.device_put(appearance_mean,device)
    appearance_components_gpu=jax.device_put(appearance_components,device)
    appearance_score_mean_gpu=jax.device_put(appearance_score_mean,device)
    appearance_score_scale_gpu=jax.device_put(appearance_score_scale,device)

    def encode(current,descriptor):
        normalized_descriptor=(descriptor-jnp.asarray(feature_mean)) \
            /jnp.asarray(feature_scale)
        value=normalized_descriptor
        for index,(weight,bias) in enumerate(current["encoder"]):
            value=value@weight+bias
            if index<len(current["encoder"])-1:
                value=jax.nn.silu(value)
        pca=normalized_descriptor@pca_components_gpu/pca_scale_gpu
        return value,pca

    def recurrent(current,latent,visible,hidden):
        input_weight,recurrent_weight,bias=current["gru"]
        if not config.use_recurrent_history:
            hidden=jnp.zeros_like(hidden)
        recurrent_input=jnp.concatenate([latent,visible[None]],axis=-1)
        input_gates=recurrent_input@input_weight+bias
        recurrent_gates=hidden@recurrent_weight
        input_update,input_reset,input_candidate=jnp.split(input_gates,3,-1)
        recurrent_update,recurrent_reset,recurrent_candidate=jnp.split(
            recurrent_gates,3,-1)
        update=jax.nn.sigmoid(input_update+recurrent_update)
        reset=jax.nn.sigmoid(input_reset+recurrent_reset)
        candidate=jnp.tanh(input_candidate+reset*recurrent_candidate)
        return update*hidden+(1-update)*candidate

    def interval_from_hidden(current,hidden,measured_visible):
        weight,bias=current["warp"]
        return direct_s_interval_from_logits_jax(
            measured_visible,hidden@weight+bias,
            jnp.asarray(config.minimum_visible_fraction,jnp.float32),
            jnp.asarray(config.measurement_prior_logit_limit,jnp.float32))

    def sample_coverage(coordinates):
        row=jnp.rint(jnp.clip(coordinates[...,0],0,1)
                     *(coverage_gpu.shape[0]-1)).astype(jnp.int32)
        column=jnp.rint(jnp.clip(coordinates[...,1],0,1)
                        *(coverage_gpu.shape[1]-1)).astype(jnp.int32)
        return coverage_gpu[row,column]

    appearance_memory_anchors_gpu=None
    appearance_memory_residuals_gpu=None
    appearance_memory_hidden_mean_gpu=None
    appearance_memory_hidden_scale_gpu=None

    def parametric_appearance_from_hidden(current,hidden):
        weight,bias=current["appearance"]
        return jnp.tanh(hidden@weight+bias)

    def appearance_from_hidden(current,hidden):
        parametric=parametric_appearance_from_hidden(current,hidden)
        if not config.use_appearance_memory \
                or appearance_memory_anchors_gpu is None:
            return parametric
        assert appearance_memory_residuals_gpu is not None
        assert appearance_memory_hidden_mean_gpu is not None
        assert appearance_memory_hidden_scale_gpu is not None
        query=(hidden-appearance_memory_hidden_mean_gpu) \
            /appearance_memory_hidden_scale_gpu
        squared_distance=jnp.sum(
            (query[...,None,:]-appearance_memory_anchors_gpu)**2,axis=-1)
        _,indices=jax.lax.top_k(
            -squared_distance,config.appearance_memory_neighbors)
        selected_distance=jnp.take_along_axis(
            squared_distance,indices,axis=-1)
        weights=1/(selected_distance+config.appearance_memory_epsilon)
        weights/=jnp.sum(weights,axis=-1,keepdims=True)
        correction=jnp.sum(
            appearance_memory_residuals_gpu[indices]*weights[...,None],axis=-2)
        return jnp.clip(parametric+correction,-1.,1.)

    def decode(current,coordinates,local_grid,latent,pca,hidden,interval):
        warped=jnp.stack([
            interval[0]+interval[1]*coordinates[...,0],coordinates[...,1]],-1)
        coordinate_features=direct_background_features_jax(
            warped,frequencies_gpu)
        local=(sample_direct_local_geometry_feature_grid_jax(
            local_grid,coordinates)-local_mean_gpu)/local_scale_gpu
        if not config.use_local_geometry:
            local=jnp.zeros_like(local)
        broadcast=lambda value:jnp.broadcast_to(
            value,(*coordinates.shape[:-1],value.shape[-1]))
        # 记忆校正后的外观系数是当前帧最直接的光场状态。低秩分支用它
        # 重建大尺度颜色场，颜色 decoder 也显式读取它以补偿 PCA 网格插值
        # 和截断留下的高频残差；输入仍然只来自曲面/历史状态，不读取 RGB。
        normalized_scores=appearance_from_hidden(current,hidden)
        network_input=jnp.concatenate([
            coordinate_features,broadcast(latent),broadcast(pca),local,
            broadcast(hidden),broadcast(normalized_scores)],axis=-1)
        value=network_input
        for index,(weight,bias) in enumerate(current["trunk"]):
            if index>0:
                value=jnp.concatenate([value,network_input],axis=-1)
            value=jax.nn.silu(value@weight+bias)
        trunk_value=value

        def head(layers):
            output=jnp.concatenate([trunk_value,network_input],axis=-1)
            for index,(weight,bias) in enumerate(layers):
                if index>0:
                    output=jnp.concatenate([output,network_input],axis=-1)
                output=output@weight+bias
                if index<len(layers)-1:
                    output=jax.nn.silu(output)
            return output

        delta=jnp.concatenate([head(layers) for layers in current["heads"]],-1)
        # 低秩分支把 GRU 预测的逐帧外观状态显式还原为空间颜色场。
        # PCA 在观测坐标中由 target-B_warp 构造；高频材料纹理由完整
        # 分辨率 B 与 s-warp 保留，decoder 只修正低秩/插值残差。
        scores=(appearance_score_mean_gpu
                +normalized_scores*config.appearance_score_clip
                *appearance_score_scale_gpu)
        appearance_field=(appearance_mean_gpu+jnp.reshape(
            scores@appearance_components_gpu,appearance_mean_gpu.shape))
        base=(sample_direct_base_texture_jax(base_gpu,warped)
              +sample_direct_base_texture_jax(appearance_field,coordinates))
        base=jnp.clip(base,1e-4,1-1e-4)
        return jax.nn.sigmoid(jnp.log(base)-jnp.log1p(-base)+delta)

    def accumulate_warm_hidden(
        current,warm_descriptors,warm_visible,warm_mask,
    ):
        def frame(hidden,inputs):
            descriptor,visible,enabled=inputs
            latent,_=encode(current,descriptor)
            candidate=recurrent(current,latent,visible,hidden)
            return jnp.where(enabled,candidate,hidden),None
        hidden,_=jax.lax.scan(
            frame,jnp.zeros((hidden_count,),jnp.float32),
            (warm_descriptors,warm_visible,warm_mask))
        return hidden

    def frozen_warm_hidden(
        current,warm_descriptors,warm_visible,warm_mask,
    ):
        return jax.lax.stop_gradient(accumulate_warm_hidden(
            current,warm_descriptors,warm_visible,warm_mask))

    describe_synthetic=jax.jit(jax.vmap(
        lambda raw:direct_geometry_descriptor_jax(
            raw,config.geometry_descriptor_rows)))
    synthetic_descriptors=[]
    synthetic_visible=[]
    synthetic_targets=[]
    synthetic_source_sequences=trusted_clips+sequence_clips
    phase=np.linspace(0.,1.,config.clip_length,dtype=np.float32)
    envelope=np.sin(np.pi*phase)**2
    smooth_phase=phase*phase*(3-2*phase)
    for _ in range(config.synthetic_sequence_count):
        sequence=synthetic_source_sequences[int(generator.integers(
            0,len(synthetic_source_sequences)))]
        start=int(generator.integers(0,sequence.size-config.clip_length+1))
        indices=sequence[start:start+config.clip_length]
        deepest=float(generator.uniform(
            config.synthetic_min_visible_fraction,1.))
        missing=(1-deepest)*envelope
        rho_start,rho_end=generator.uniform(0.,1.,2)
        rho=rho_start+(rho_end-rho_start)*smooth_phase
        left=missing*rho
        requested_visible=1-missing
        cropped=np.stack([
            _crop_surface_rows(surfaces[index],float(current_left),
                               float(current_visible))
            for index,current_left,current_visible in zip(
                indices,left,requested_visible,strict=True)])
        current_descriptors=np.asarray(jax.device_get(describe_synthetic(
            jax.device_put(cropped,device))),np.float32)
        current_lengths=_raw_surface_lengths(cropped)
        current_visible=_measured_visible_fractions(
            current_lengths,length_reference,
            full_threshold=config.full_visibility_threshold,
            minimum=config.minimum_visible_fraction)
        # 真值来自对完整 55 mm 曲面的已知裁剪，不等同于带噪测长先验。
        current_missing=1-requested_visible
        targets=np.stack([
            current_missing*rho,requested_visible,
            current_missing*(1-rho)],axis=-1).astype(np.float32)
        synthetic_descriptors.append(current_descriptors)
        synthetic_visible.append(current_visible)
        synthetic_targets.append(targets)
    synthetic_descriptors=np.ascontiguousarray(
        np.stack(synthetic_descriptors),dtype=np.float32)
    synthetic_visible=np.ascontiguousarray(
        np.stack(synthetic_visible),dtype=np.float32)
    synthetic_targets=np.ascontiguousarray(
        np.stack(synthetic_targets),dtype=np.float32)
    print(
        "direct_fit_s 合成裁剪："
        f"sequences={config.synthetic_sequence_count}，"
        f"geometry_sources={len(synthetic_source_sequences)}，"
        f"measured-prior min={float(synthetic_visible.min()):.4f}，"
        f"target min={float(synthetic_targets[...,1].min()):.4f}")

    def synthetic_batch():
        selected=generator.integers(
            0,config.synthetic_sequence_count,size=config.clip_batch_size)
        return tuple(jax.device_put(value[selected],device) for value in (
            synthetic_descriptors,synthetic_visible,synthetic_targets))

    rows,columns=samples.shape[1:3]
    valid_pixels=[np.flatnonzero(value.reshape(-1)) for value in masks]
    if any(value.size==0 for value in valid_pixels):
        raise ValueError("direct_fit_s 存在没有有效采样点的帧")

    def sampled_clip(sequence: np.ndarray):
        start=int(generator.integers(0,sequence.size-config.clip_length+1))
        clip=sequence[start:start+config.clip_length]
        warm=np.zeros((config.warmup_frames,),np.int64)
        warm_mask=np.zeros((config.warmup_frames,),np.bool_)
        if config.warmup_frames:
            prefix=sequence[max(0,start-config.warmup_frames):start]
            if prefix.size:
                warm[-prefix.size:]=prefix
                warm_mask[-prefix.size:]=True
        return clip,warm,warm_mask

    flattened=samples.reshape(samples.shape[0],-1,3)

    def temporal_batch(
        sequences: list[np.ndarray],*,use_each_sequence_once: bool = False,
    ):
        clip_indices=[]
        warm_indices=[]
        warm_masks=[]
        if use_each_sequence_once:
            if len(sequences)!=config.clip_batch_size:
                raise ValueError("direct_fit_s 固定 batch 的序列数无效")
            chosen=sequences
        else:
            chosen=[sequences[int(generator.integers(0,len(sequences)))]
                    for _ in range(config.clip_batch_size)]
        for sequence in chosen:
            clip,warm,current_mask=sampled_clip(sequence)
            clip_indices.append(clip)
            warm_indices.append(warm)
            warm_masks.append(current_mask)
        clip_indices=np.stack(clip_indices)
        warm_indices=np.stack(warm_indices)
        warm_masks=np.stack(warm_masks)
        pixel_indices=np.empty((
            config.clip_batch_size,config.clip_length,
            config.points_per_frame),np.int64)
        for batch_index in range(config.clip_batch_size):
            for frame_index in range(config.clip_length):
                available=valid_pixels[int(clip_indices[
                    batch_index,frame_index])]
                pixel_indices[batch_index,frame_index]=generator.choice(
                    available,config.points_per_frame,
                    replace=available.size<config.points_per_frame)
        coordinates=np.stack([
            (pixel_indices//columns)/max(rows-1,1),
            (pixel_indices%columns)/max(columns-1,1)],axis=-1).astype(np.float32)
        target=np.empty((*pixel_indices.shape,3),np.float32)
        for batch_index in range(config.clip_batch_size):
            target[batch_index]=flattened[
                clip_indices[batch_index,:,None],pixel_indices[batch_index]]
        warm_descriptors=np.zeros((
            config.clip_batch_size,config.warmup_frames,feature_count),np.float32)
        warm_visible=np.ones((
            config.clip_batch_size,config.warmup_frames),np.float32)
        if config.warmup_frames:
            for batch_index in range(config.clip_batch_size):
                enabled=warm_masks[batch_index]
                warm_descriptors[batch_index,enabled]=descriptors[
                    warm_indices[batch_index,enabled]]
                warm_visible[batch_index,enabled]=visible_fractions[
                    warm_indices[batch_index,enabled]]
        values=(warm_descriptors,warm_visible,warm_masks,
                descriptors[clip_indices],visible_fractions[clip_indices],
                frame_quality[clip_indices],appearance_targets[clip_indices],
                local_grids[clip_indices],coordinates,target)
        return tuple(jax.device_put(value,device) for value in values)

    def mixed_color_batch():
        # 按真实帧数占比采样，避免 batch=2 时把仅占 24.5% 的平直帧放大到 50%。
        trusted_fraction=trusted.size/samples.shape[0]
        selected=[]
        for _ in range(config.clip_batch_size):
            choices=(trusted_clips if generator.random()<trusted_fraction
                     else sequence_clips)
            selected.append(choices[int(generator.integers(0,len(choices)))])
        return temporal_batch(selected,use_each_sequence_once=True)

    def synthetic_loss(current,batch):
        descriptor_batch,visible_batch,target_batch=batch

        def run_one(descriptor_sequence,visible_sequence):
            def frame(hidden,inputs):
                descriptor,visible=inputs
                latent,_=encode(current,descriptor)
                hidden=recurrent(current,latent,visible,hidden)
                return hidden,interval_from_hidden(current,hidden,visible)
            _,interval=jax.lax.scan(
                frame,jnp.zeros((hidden_count,),jnp.float32),
                (descriptor_sequence,visible_sequence))
            return interval

        predicted=jax.vmap(run_one)(descriptor_batch,visible_batch)
        error=predicted-target_batch
        loss=jnp.mean(error**2)
        metrics=jnp.concatenate([
            jnp.asarray([jnp.mean(jnp.abs(error))]),
            jnp.mean(predicted,axis=(0,1))])
        return loss,metrics

    def warp_alignment_loss(current,batch,synthetic):
        (warm_descriptor,warm_visible,warm_mask,descriptor_batch,
         visible_batch,quality_batch,_,_,coordinate_batch,target_batch)=batch

        def run_one(warm_d,warm_v,warm_m,descriptors_one,visible_one,
                    coordinates_one,targets_one):
            hidden=frozen_warm_hidden(current,warm_d,warm_v,warm_m)

            def frame(previous_hidden,inputs):
                descriptor,visible,coordinates,target=inputs
                latent,_=encode(current,descriptor)
                hidden=recurrent(current,latent,visible,previous_hidden)
                interval=interval_from_hidden(current,hidden,visible)
                warped=jnp.stack([
                    interval[0]+interval[1]*coordinates[...,0],
                    coordinates[...,1]],axis=-1)
                prediction=sample_direct_base_texture_jax(base_gpu,warped)
                covered=sample_coverage(warped)
                return hidden,(prediction,target,interval,covered)

            _,output=jax.lax.scan(frame,hidden,(
                descriptors_one,visible_one,coordinates_one,targets_one))
            return output

        prediction,target,interval,covered=jax.vmap(run_one)(
            warm_descriptor,warm_visible,warm_mask,descriptor_batch,
            visible_batch,coordinate_batch,target_batch)
        error=prediction-target
        absolute=jnp.abs(error)
        huber=jnp.where(
            absolute<=huber_delta,.5*error**2,
            huber_delta*(absolute-.5*huber_delta))
        pixel_weight=(covered.astype(jnp.float32)
                      *quality_batch[...,None])
        count=jnp.maximum(jnp.sum(pixel_weight)*3,1)
        photo=jnp.sum(huber*pixel_weight[...,None])/count
        rmse=jnp.sqrt(jnp.sum(error**2*pixel_weight[...,None])/count)
        endpoints=jnp.stack([interval[...,0],
                             interval[...,0]+interval[...,1]],axis=-1)
        temporal=jnp.mean((endpoints[:,1:]-endpoints[:,:-1])**2)
        synthetic_value,_=synthetic_loss(current,synthetic)
        loss=(photo+config.temporal_weight*temporal
              +config.synthetic_regularization_weight*synthetic_value)
        metrics=jnp.concatenate([jnp.asarray([
            photo,temporal,synthetic_value,rmse,
            jnp.mean(covered.astype(jnp.float32))]),
            jnp.mean(interval,axis=(0,1))])
        return loss,metrics

    def color_loss(current,batch,warp_teacher):
        (warm_descriptor,warm_visible,warm_mask,descriptor_batch,
         visible_batch,quality_batch,appearance_batch,local_batch,
         coordinate_batch,
         target_batch)=batch

        def run_one(warm_d,warm_v,warm_m,descriptors_one,visible_one,
                    locals_one,coordinates_one,targets_one):
            # 颜色损失需要穿过历史帧，否则 GRU 只能学到 loss clip
            # 内的短期状态。教师分支固定为 warp 阶段结束时的快照。
            hidden=accumulate_warm_hidden(
                current,warm_d,warm_v,warm_m)
            hidden=jax.lax.stop_gradient(hidden)
            teacher_hidden=frozen_warm_hidden(
                warp_teacher,warm_d,warm_v,warm_m)

            def frame(previous_state,inputs):
                previous_hidden,previous_teacher_hidden=previous_state
                descriptor,visible,local,coordinates,target=inputs
                latent,pca=encode(current,descriptor)
                latent=jax.lax.stop_gradient(latent)
                pca=jax.lax.stop_gradient(pca)
                hidden=recurrent(current,latent,visible,previous_hidden)
                hidden=jax.lax.stop_gradient(hidden)
                interval=interval_from_hidden(current,hidden,visible)
                interval=jax.lax.stop_gradient(interval)
                teacher_latent,_=encode(warp_teacher,descriptor)
                teacher_hidden=recurrent(
                    warp_teacher,teacher_latent,visible,
                    previous_teacher_hidden)
                teacher_interval=interval_from_hidden(
                    warp_teacher,teacher_hidden,visible)
                prediction=decode(
                    current,coordinates,local,latent,pca,hidden,interval)
                return (hidden,teacher_hidden),(
                    prediction,target,interval,teacher_interval,hidden)

            _,output=jax.lax.scan(frame,(hidden,teacher_hidden),(
                descriptors_one,visible_one,locals_one,coordinates_one,
                targets_one))
            return output

        prediction,target,interval,teacher_interval,hidden_sequence=jax.vmap(
            run_one)(
            warm_descriptor,warm_visible,warm_mask,descriptor_batch,
            visible_batch,local_batch,coordinate_batch,target_batch)
        error=prediction-target
        color_weight=quality_batch[...,None,None]
        color_count=jnp.maximum(
            jnp.sum(quality_batch)*error.shape[-2]*error.shape[-1],1)
        # 最终指标是 RMSE；颜色主损失直接使用质量加权 MSE，
        # 避免 Huber 把强自遮挡区域的大误差梯度截平。
        photo=jnp.sum(error**2*color_weight)/color_count
        interval_error=interval-jax.lax.stop_gradient(teacher_interval)
        interval_count=jnp.maximum(jnp.sum(quality_batch)*3,1)
        distillation=jnp.sum(
            interval_error**2*quality_batch[...,None])/interval_count
        appearance_prediction=appearance_from_hidden(current,hidden_sequence)
        appearance_error=appearance_prediction-appearance_batch
        appearance_count=jnp.maximum(
            jnp.sum(quality_batch)*config.appearance_pca_dimensions,1)
        appearance_loss=jnp.sum(
            appearance_error**2*quality_batch[...,None])/appearance_count
        loss=(photo+config.warp_distillation_weight*distillation
              +config.appearance_supervision_weight*appearance_loss)
        metrics=jnp.concatenate([
            jnp.asarray([
                jnp.sqrt(jnp.sum(
                    error**2*color_weight)/color_count),
                jnp.sum(jnp.abs(interval_error)*quality_batch[...,None])
                /interval_count,
                jnp.sum(jnp.abs(appearance_error)*quality_batch[...,None])
                /appearance_count]),
            jnp.mean(interval,axis=(0,1))])
        return loss,metrics

    beta1,beta2=config.adam_beta1,config.adam_beta2

    def make_step(loss_function,frozen: tuple[str,...]):
        @jax.jit
        def step(current,moment,variance,index,batch,rate):
            (loss,metrics),gradient=jax.value_and_grad(
                loss_function,has_aux=True)(current,batch)
            for name in frozen:
                gradient={**gradient,name:jax.tree.map(
                    jnp.zeros_like,gradient[name])}
            norm=jnp.sqrt(sum(
                jnp.sum(value**2) for value in jax.tree.leaves(gradient)))
            factor=jnp.minimum(
                1.,config.gradient_clip_norm/jnp.maximum(norm,1e-12))
            gradient=jax.tree.map(lambda value:value*factor,gradient)
            moment=jax.tree.map(
                lambda old,value:beta1*old+(1-beta1)*value,moment,gradient)
            variance=jax.tree.map(
                lambda old,value:beta2*old+(1-beta2)*value*value,
                variance,gradient)
            for name in frozen:
                moment={**moment,name:jax.tree.map(jnp.zeros_like,moment[name])}
                variance={**variance,name:jax.tree.map(
                    jnp.zeros_like,variance[name])}
            corrected_m=jax.tree.map(
                lambda value:value/(1-beta1**index),moment)
            corrected_v=jax.tree.map(
                lambda value:value/(1-beta2**index),variance)
            current=jax.tree.map(
                lambda value,m,v:value-rate*m/(jnp.sqrt(v)+config.adam_epsilon),
                current,corrected_m,corrected_v)
            return current,moment,variance,loss,metrics,norm
        return step

    synthetic_step=make_step(
        synthetic_loss,("appearance","trunk","heads"))
    warp_step=make_step(
        lambda current,batches:warp_alignment_loss(
            current,batches[0],batches[1]),
        ("appearance","trunk","heads"))
    print(
        "direct_fit_s："
        f"trusted={trusted.size}，all={samples.shape[0]}，"
        f"clips={len(trusted_clips)}+{len(sequence_clips)}，"
        f"GRU={hidden_count}，warmup={config.warmup_frames}，"
        "warp=measured-prior+learned-visible/side，"
        "appearance=full-sequence encoder+GRU，color-trainable=decoder，"
        "ablation(memory/history/local_geometry)="
        f"{config.use_appearance_memory}/{config.use_recurrent_history}/"
        f"{config.use_local_geometry}，"
        f"trunk={config.color_trunk_layers}x{config.color_trunk_width}，"
        f"heads=3x{config.color_head_layers}")
    stages=(
        ("synthetic_warp",config.synthetic_warp_steps,
         config.synthetic_warp_learning_rate,synthetic_step,
         synthetic_batch,"interval_mae/mean_interval"),
        ("warp_alignment",config.warp_alignment_steps,
         config.warp_learning_rate,warp_step,
         lambda:(temporal_batch(sequence_clips),synthetic_batch()),
         "photo/temp/synthetic/rmse/coverage/mean_interval"),
    )
    for name,count,rate,step,batch_factory,metric_names in stages:
        moments=jax.tree.map(jnp.zeros_like,parameters)
        variances=jax.tree.map(jnp.zeros_like,parameters)
        print(f"direct_fit_s 阶段 {name}: steps={count}, lr={rate:g}")
        for index in range(1,count+1):
            parameters,moments,variances,loss,metrics,norm=step(
                parameters,moments,variances,
                jnp.asarray(index,jnp.float32),batch_factory(),
                jnp.asarray(rate,jnp.float32))
            if index==1 or index%100==0 or index==count:
                loss_value,metric_values,norm_value=jax.device_get(
                    (loss,metrics,norm))
                print(
                    f"direct_fit_s {name} {index}/{count}: "
                    f"loss={float(loss_value):.7f}, "
                    f"{metric_names}={np.asarray(metric_values).tolist()}, "
                    f"grad={float(norm_value):.5f}")

    # 动态外观必须相对于最终 s-warp 后的高分辨率静态 B 建模。若相对
    # identity B 做 PCA，高频材料纹理的微小坐标位移会被错误塞进低分辨率
    # 外观基，导致即使 score 完全正确也无法在全分辨率重建。
    @jax.jit
    def predict_interval_sequence(current,descriptor_sequence,visible_sequence):
        def frame(hidden,inputs):
            descriptor,visible=inputs
            latent,_=encode(current,descriptor)
            hidden=recurrent(current,latent,visible,hidden)
            return hidden,interval_from_hidden(current,hidden,visible)
        _,interval_sequence=jax.lax.scan(
            frame,jnp.zeros((hidden_count,),jnp.float32),
            (descriptor_sequence,visible_sequence))
        return interval_sequence

    learned_intervals=np.zeros((samples.shape[0],3),np.float32)
    learned_intervals[:,1]=1.
    for sequence in trusted_clips+sequence_clips:
        learned_intervals[sequence]=np.asarray(jax.device_get(
            predict_interval_sequence(
                parameters,jax.device_put(descriptors[sequence],device),
                jax.device_put(visible_fractions[sequence],device))),np.float32)
    appearance_s=(appearance_rows/max(rows-1,1)).astype(np.float32)
    appearance_t=(appearance_columns/max(columns-1,1)).astype(np.float32)
    grid_s,grid_t=np.meshgrid(appearance_s,appearance_t,indexing="ij")
    appearance_coordinates=np.stack([grid_s,grid_t],axis=-1)
    warped_appearance_coordinates=np.empty((
        samples.shape[0],*appearance_coordinates.shape),np.float32)
    warped_appearance_coordinates[...,0]=(learned_intervals[:,None,None,0]
        +learned_intervals[:,None,None,1]*appearance_coordinates[None,...,0])
    warped_appearance_coordinates[...,1]=appearance_coordinates[None,...,1]
    sample_base_batch=jax.jit(jax.vmap(
        lambda value:sample_direct_base_texture_jax(base_gpu,value)))
    warped_appearance_base=[]
    for start in range(0,samples.shape[0],32):
        warped_appearance_base.append(np.asarray(jax.device_get(
            sample_base_batch(jax.device_put(
                warped_appearance_coordinates[start:start+32],device))),
            np.float32))
    warped_appearance_base=np.concatenate(warped_appearance_base)
    appearance_residual=appearance_fields-warped_appearance_base
    appearance_mean=np.sum(
        appearance_residual*appearance_weight,axis=0,dtype=np.float64) \
        /np.maximum(np.sum(appearance_weight,axis=0,dtype=np.float64),1e-8)
    appearance_mean=np.asarray(appearance_mean,np.float32)
    appearance_centered=np.where(
        (appearance_valid&appearance_reliable[None])[...,None],
        appearance_residual-appearance_mean[None],0
    ).reshape(samples.shape[0],-1).astype(np.float32)
    weighted_appearance=(appearance_centered[sequence_mask]
                         *np.sqrt(frame_quality[sequence_mask,None]))
    _,appearance_singular,appearance_vh=np.linalg.svd(
        weighted_appearance,full_matrices=False)
    appearance_components=np.ascontiguousarray(
        appearance_vh[:config.appearance_pca_dimensions],dtype=np.float32)
    appearance_scores=appearance_centered@appearance_components.T
    quality_sum=max(float(np.sum(frame_quality[sequence_mask])),1e-8)
    appearance_score_mean=np.sum(
        appearance_scores[sequence_mask]
        *frame_quality[sequence_mask,None],axis=0,
        dtype=np.float64)/quality_sum
    appearance_score_scale=np.sqrt(np.sum(
        (appearance_scores[sequence_mask]-appearance_score_mean[None])**2
        *frame_quality[sequence_mask,None],axis=0,
        dtype=np.float64)/quality_sum)
    appearance_score_scale=np.where(
        appearance_score_scale>1e-6,appearance_score_scale,1.)
    appearance_score_mean=np.asarray(appearance_score_mean,np.float32)
    appearance_score_scale=np.asarray(appearance_score_scale,np.float32)
    appearance_targets=np.clip(
        (appearance_scores-appearance_score_mean[None])
        /appearance_score_scale[None],
        -config.appearance_score_clip,config.appearance_score_clip)
    appearance_targets=(appearance_targets
                        /config.appearance_score_clip).astype(np.float32)
    appearance_energy=float(np.sum(appearance_singular**2))
    appearance_explained=(float(np.sum(appearance_singular[
        :config.appearance_pca_dimensions]**2))/max(appearance_energy,1e-12))
    appearance_mean_gpu=jax.device_put(appearance_mean,device)
    appearance_components_gpu=jax.device_put(appearance_components,device)
    appearance_score_mean_gpu=jax.device_put(appearance_score_mean,device)
    appearance_score_scale_gpu=jax.device_put(appearance_score_scale,device)
    oracle_scores=(appearance_score_mean[None]
                   +appearance_targets*config.appearance_score_clip
                   *appearance_score_scale[None])
    oracle_residual=(appearance_mean[None]+(
        oracle_scores@appearance_components).reshape(
            samples.shape[0],*appearance_mean.shape))
    oracle_error=(warped_appearance_base+oracle_residual-appearance_fields)
    oracle_mask=(appearance_valid&appearance_reliable[None]
                 &sequence_mask[:,None,None])
    oracle_count=max(int(np.sum(oracle_mask))*3,1)
    oracle_rmse=float(np.sqrt(np.sum(
        np.where(oracle_mask[...,None],oracle_error**2,0),dtype=np.float64)
        /oracle_count))
    print(
        "direct_fit_s warp 后动态外观基："
        f"weighted explained={100*appearance_explained:.2f}%，"
        f"descriptor_oracle_rmse={oracle_rmse:.7f}")

    # 外观状态先在完整序列上从零 hidden 连续展开。随机短 clip 无法稳定
    # 学到一次平直-弯曲-平直循环中的长程相位，而运行时恰好也是从序列
    # 起点连续推进，因此这里直接令训练和运行的状态轨迹一致。
    warp_teacher={name:jax.tree.map(jax.lax.stop_gradient,parameters[name])
                  for name in ("encoder","gru","warp")}
    state_sequences=trusted_clips+sequence_clips
    state_length=max(sequence.size for sequence in state_sequences)
    state_count=len(state_sequences)
    state_descriptors=np.zeros((state_count,state_length,feature_count),np.float32)
    state_visible=np.ones((state_count,state_length),np.float32)
    state_quality=np.zeros((state_count,state_length),np.float32)
    state_targets=np.zeros((
        state_count,state_length,config.appearance_pca_dimensions),np.float32)
    state_mask=np.zeros((state_count,state_length),np.bool_)
    for index,sequence in enumerate(state_sequences):
        count=sequence.size
        state_descriptors[index,:count]=descriptors[sequence]
        state_visible[index,:count]=visible_fractions[sequence]
        state_quality[index,:count]=frame_quality[sequence]
        state_targets[index,:count]=appearance_targets[sequence]
        state_mask[index,:count]=True
    state_batch=tuple(jax.device_put(value,device) for value in (
        state_descriptors,state_visible,state_quality,state_targets,state_mask))

    def appearance_sequence_loss(current,batch):
        descriptor_batch,visible_batch,quality_batch,target_batch,mask_batch=batch

        def run_one(descriptor_sequence,visible_sequence,enabled_sequence):
            def frame(states,inputs):
                hidden,teacher_hidden=states
                descriptor,visible,enabled=inputs
                latent,_=encode(current,descriptor)
                candidate=recurrent(current,latent,visible,hidden)
                hidden=jnp.where(enabled,candidate,hidden)
                prediction=appearance_from_hidden(current,hidden)
                interval=interval_from_hidden(current,hidden,visible)
                teacher_latent,_=encode(warp_teacher,descriptor)
                teacher_candidate=recurrent(
                    warp_teacher,teacher_latent,visible,teacher_hidden)
                teacher_hidden=jnp.where(
                    enabled,teacher_candidate,teacher_hidden)
                teacher_interval=interval_from_hidden(
                    warp_teacher,teacher_hidden,visible)
                return (hidden,teacher_hidden),(
                    prediction,interval,teacher_interval)
            _,outputs=jax.lax.scan(
                frame,(jnp.zeros((hidden_count,),jnp.float32),
                       jnp.zeros((hidden_count,),jnp.float32)),
                (descriptor_sequence,visible_sequence,enabled_sequence))
            return outputs

        prediction,interval,teacher_interval=jax.vmap(run_one)(
            descriptor_batch,visible_batch,mask_batch)
        weight=quality_batch*mask_batch.astype(jnp.float32)
        appearance_error=prediction-target_batch
        appearance_count=jnp.maximum(
            jnp.sum(weight)*config.appearance_pca_dimensions,1)
        appearance_loss=jnp.sum(
            appearance_error**2*weight[...,None])/appearance_count
        interval_error=interval-jax.lax.stop_gradient(teacher_interval)
        interval_count=jnp.maximum(jnp.sum(weight)*3,1)
        distillation=jnp.sum(
            interval_error**2*weight[...,None])/interval_count
        loss=(appearance_loss
              +config.warp_distillation_weight*distillation)
        metrics=jnp.asarray([
            jnp.sum(jnp.abs(appearance_error)*weight[...,None])
            /appearance_count,
            jnp.sum(jnp.abs(interval_error)*weight[...,None])/interval_count])
        return loss,metrics

    appearance_step=make_step(
        appearance_sequence_loss,
        ("encoder","gru","warp","trunk","heads"))
    moments=jax.tree.map(jnp.zeros_like,parameters)
    variances=jax.tree.map(jnp.zeros_like,parameters)
    print(
        "direct_fit_s 阶段 appearance_sequence: "
        f"steps={config.appearance_train_steps}, "
        f"lr={config.appearance_learning_rate:g}, "
        f"sequences={state_count}, max_frames={state_length}")
    for index in range(1,config.appearance_train_steps+1):
        parameters,moments,variances,loss,metrics,norm=appearance_step(
            parameters,moments,variances,jnp.asarray(index,jnp.float32),
            state_batch,jnp.asarray(config.appearance_learning_rate,jnp.float32))
        if index==1 or index%100==0 \
                or index==config.appearance_train_steps:
            loss_value,metric_values,norm_value=jax.device_get(
                (loss,metrics,norm))
            print(
                f"direct_fit_s appearance_sequence {index}/"
                f"{config.appearance_train_steps}: loss={float(loss_value):.7f}, "
                "appearance_mae/warp_teacher_mae="
                f"{np.asarray(metric_values).tolist()}, "
                f"grad={float(norm_value):.5f}")

    @jax.jit
    def state_hidden_trajectories(current,descriptor_batch,visible_batch,
                                  mask_batch):
        def run_one(descriptor_sequence,visible_sequence,enabled_sequence):
            def frame(hidden,inputs):
                descriptor,visible,enabled=inputs
                latent,_=encode(current,descriptor)
                candidate=recurrent(current,latent,visible,hidden)
                hidden=jnp.where(enabled,candidate,hidden)
                return hidden,hidden
            _,hidden_sequence=jax.lax.scan(
                frame,jnp.zeros((hidden_count,),jnp.float32),
                (descriptor_sequence,visible_sequence,enabled_sequence))
            return hidden_sequence
        return jax.vmap(run_one)(
            descriptor_batch,visible_batch,mask_batch)

    hidden_trajectories=np.asarray(jax.device_get(state_hidden_trajectories(
        parameters,state_batch[0],state_batch[1],state_batch[4])),np.float32)
    memory_hidden=np.ascontiguousarray(hidden_trajectories[state_mask])
    memory_targets=np.ascontiguousarray(state_targets[state_mask])
    if config.appearance_memory_neighbors>memory_hidden.shape[0]:
        raise ValueError(
            "direct_fit_s appearance_memory_neighbors 超过有效状态数")
    memory_prediction=np.asarray(jax.device_get(
        parametric_appearance_from_hidden(
            parameters,jax.device_put(memory_hidden,device))),np.float32)
    memory_hidden_mean=memory_hidden.mean(axis=0,dtype=np.float64).astype(
        np.float32)
    memory_hidden_scale=memory_hidden.std(axis=0,dtype=np.float64).astype(
        np.float32)
    memory_hidden_scale=np.where(
        memory_hidden_scale>1e-5,memory_hidden_scale,1.).astype(np.float32)
    memory_anchors=np.ascontiguousarray(
        (memory_hidden-memory_hidden_mean)/memory_hidden_scale,dtype=np.float32)
    memory_residuals=np.ascontiguousarray(
        memory_targets-memory_prediction,dtype=np.float32)
    if config.use_appearance_memory:
        appearance_memory_anchors_gpu=jax.device_put(memory_anchors,device)
        appearance_memory_residuals_gpu=jax.device_put(
            memory_residuals,device)
        appearance_memory_hidden_mean_gpu=jax.device_put(
            memory_hidden_mean,device)
        appearance_memory_hidden_scale_gpu=jax.device_put(
            memory_hidden_scale,device)
    memory_predict_batch=jax.jit(
        lambda value:appearance_from_hidden(parameters,value))
    memory_reconstructed_parts=[]
    memory_check_batch=32
    for start in range(0,memory_hidden.shape[0],memory_check_batch):
        current=memory_hidden[start:start+memory_check_batch]
        count=current.shape[0]
        if count<memory_check_batch:
            current=np.concatenate([
                current,np.repeat(current[-1:],memory_check_batch-count,axis=0)])
        prediction=np.asarray(jax.device_get(memory_predict_batch(
            jax.device_put(current,device))),np.float32)
        memory_reconstructed_parts.append(prediction[:count])
    memory_reconstructed=np.concatenate(memory_reconstructed_parts)
    print(
        "direct_fit_s 外观状态记忆："
        f"anchors={memory_hidden.shape[0]}，"
        f"neighbors={config.appearance_memory_neighbors}，"
        f"enabled={config.use_appearance_memory}，"
        f"training_mae={float(np.mean(np.abs(
            memory_reconstructed-memory_targets))):.7f}")

    # 外观状态训练完成后全部冻结，只让颜色 decoder 学低秩截断与局部
    # 几何残差，避免稀疏像素批次再次扰动完整序列上学到的状态轨迹。
    color_step=make_step(
        lambda current,batches:color_loss(
            current,batches[0],batches[1]),
        ("encoder","gru","warp","appearance"))
    monitor_batches=tuple(
        mixed_color_batch() for _ in range(config.color_monitor_batch_count))
    monitor_color=jax.jit(
        lambda current,batch:color_loss(
            current,batch,warp_teacher)[1])

    def monitor_rmse(current) -> float:
        values=np.asarray([
            float(np.asarray(jax.device_get(
                monitor_color(current,batch)))[0])
            for batch in monitor_batches],np.float64)
        return float(np.sqrt(np.mean(values**2)))

    moments=jax.tree.map(jnp.zeros_like,parameters)
    variances=jax.tree.map(jnp.zeros_like,parameters)
    best_parameters=parameters
    best_color_rmse=monitor_rmse(parameters)
    best_color_step=0
    print(
        "direct_fit_s 阶段 color: "
        f"steps={config.color_train_steps}, lr={config.color_learning_rate:g}, "
        "trainable=trunk+heads, frozen=encoder+GRU+warp+appearance, "
        f"warp-distillation={config.warp_distillation_weight:g}, "
        f"monitor-rmse={best_color_rmse:.7f}")
    for index in range(1,config.color_train_steps+1):
        parameters,moments,variances,loss,metrics,norm=color_step(
            parameters,moments,variances,
            jnp.asarray(index,jnp.float32),
            (mixed_color_batch(),warp_teacher),
            jnp.asarray(config.color_learning_rate,jnp.float32))
        if index==1 or index%100==0 or index==config.color_train_steps:
            loss_value,metric_values,norm_value=jax.device_get(
                (loss,metrics,norm))
            print(
                f"direct_fit_s color {index}/{config.color_train_steps}: "
                f"loss={float(loss_value):.7f}, "
                "rmse/warp_teacher_mae/appearance_mae/mean_interval="
                f"{np.asarray(metric_values).tolist()}, "
                f"grad={float(norm_value):.5f}")
        if index%config.color_checkpoint_interval==0 \
                or index==config.color_train_steps:
            current_monitor=monitor_rmse(parameters)
            if current_monitor<best_color_rmse:
                best_parameters=parameters
                best_color_rmse=current_monitor
                best_color_step=index
            print(
                f"direct_fit_s color checkpoint {index}: "
                f"monitor_rmse={current_monitor:.7f}, "
                f"best={best_color_rmse:.7f}@{best_color_step}")
    parameters=best_parameters
    print(
        "direct_fit_s color 恢复最佳 checkpoint："
        f"step={best_color_step}，monitor_rmse={best_color_rmse:.7f}")

    @jax.jit
    def evaluate_frame(current,hidden,descriptor,visible,local,coordinates,
                       target):
        latent,pca=encode(current,descriptor)
        hidden=recurrent(current,latent,visible,hidden)
        interval=interval_from_hidden(current,hidden,visible)
        warped=jnp.stack([
            interval[0]+interval[1]*coordinates[...,0],coordinates[...,1]],-1)
        identity_base=sample_direct_base_texture_jax(base_gpu,coordinates)
        warped_base=sample_direct_base_texture_jax(base_gpu,warped)
        normalized_scores=appearance_from_hidden(current,hidden)
        scores=(appearance_score_mean_gpu
                +normalized_scores*config.appearance_score_clip
                *appearance_score_scale_gpu)
        appearance_field=(appearance_mean_gpu+jnp.reshape(
            scores@appearance_components_gpu,appearance_mean_gpu.shape))
        appearance_prior=jnp.clip(
            warped_base+sample_direct_base_texture_jax(
                appearance_field,coordinates),0,1)
        prediction=decode(
            current,coordinates,local,latent,pca,hidden,interval)
        common=sample_coverage(coordinates)&sample_coverage(warped)
        identity_error=identity_base-target
        warp_error=warped_base-target
        appearance_error=appearance_prior-target
        color_error=prediction-target
        base_count=jnp.sum(common)*3
        return (hidden,interval,
                jnp.sum(jnp.where(common[...,None],identity_error**2,0)),
                jnp.sum(jnp.where(common[...,None],warp_error**2,0)),
                base_count,jnp.sum(appearance_error**2),
                jnp.sum(color_error**2),color_error.size)

    def evaluation_geometry(
        evaluation_surfaces: np.ndarray,evaluation_raw: np.ndarray,
    ) -> tuple[np.ndarray,np.ndarray]:
        descriptor_parts=[]
        local_parts=[]
        for start in range(0,evaluation_surfaces.shape[0],32):
            descriptor_batch,local_batch=jax.device_get(analyze(
                jax.device_put(evaluation_raw[start:start+32],device),
                jax.device_put(evaluation_surfaces[start:start+32],device)))
            descriptor_parts.append(np.asarray(descriptor_batch,np.float32))
            local_parts.append(np.asarray(local_batch,np.float32))
        return (np.ascontiguousarray(np.concatenate(descriptor_parts)),
                np.ascontiguousarray(np.concatenate(local_parts)))

    def evaluate(
        name: str,sequences: list[np.ndarray],*,
        evaluation_samples: np.ndarray,evaluation_valid: np.ndarray,
        evaluation_descriptors: np.ndarray,
        evaluation_local_grids: np.ndarray,
        evaluation_visible: np.ndarray,
    ) -> dict[str,float]:
        identity_sse=0.; warp_sse=0.; base_count=0
        appearance_sse=0.; color_sse=0.; color_count=0
        intervals=[]; frame_rmse=[]
        measured_values=[]
        evaluation_flattened=evaluation_samples.reshape(
            evaluation_samples.shape[0],-1,3)
        evaluation_valid_pixels=[
            np.flatnonzero(value.reshape(-1)) for value in evaluation_valid]
        point_count=config.evaluation_points_per_frame
        for sequence in sequences:
            hidden=jnp.zeros((hidden_count,),jnp.float32)
            for source in sequence:
                available=evaluation_valid_pixels[int(source)]
                positions=np.rint(np.linspace(
                    0,available.size-1,point_count)).astype(np.int64)
                pixels=available[positions]
                coordinates=np.stack([
                    (pixels//columns)/max(rows-1,1),
                    (pixels%columns)/max(columns-1,1)],axis=-1).astype(np.float32)
                target=evaluation_flattened[int(source),pixels]
                output=jax.device_get(evaluate_frame(
                    parameters,hidden,
                    jax.device_put(evaluation_descriptors[int(source)],device),
                    jax.device_put(evaluation_visible[int(source)],device),
                    jax.device_put(evaluation_local_grids[int(source)],device),
                    jax.device_put(coordinates,device),
                    jax.device_put(target,device)))
                hidden=jax.device_put(output[0],device)
                intervals.append(np.asarray(output[1],np.float64))
                measured_values.append(float(evaluation_visible[int(source)]))
                identity_sse+=float(output[2]); warp_sse+=float(output[3])
                base_count+=int(output[4]); appearance_sse+=float(output[5])
                color_sse+=float(output[6]); color_count+=int(output[7])
                frame_rmse.append(np.sqrt(
                    float(output[6])/max(int(output[7]),1)))
        values=np.stack(intervals)
        identity_rmse=np.sqrt(identity_sse/max(base_count,1))
        warp_rmse=np.sqrt(warp_sse/max(base_count,1))
        appearance_rmse=np.sqrt(appearance_sse/max(color_count,1))
        color_rmse=np.sqrt(color_sse/max(color_count,1))
        measured_values=np.asarray(measured_values,np.float64)
        missing=1-values[:,1]
        active=missing>.01
        q=values[active,0]/missing[active]
        q_text=(np.quantile(q,[.05,.5,.95]).tolist()
                if q.size else [float("nan")]*3)
        frame_values=np.asarray(frame_rmse,np.float64)
        metrics={
            "frame_count":float(frame_values.size),
            "B_identity_rmse":float(identity_rmse),
            "B_warp_rmse":float(warp_rmse),
            "appearance_prior_rmse":float(appearance_rmse),
            "final_rmse":float(color_rmse),
            "frame_rmse_median":float(np.median(frame_values)),
            "frame_rmse_p90":float(np.quantile(frame_values,.9)),
            "frame_rmse_p95":float(np.quantile(frame_values,.95)),
            "frame_rmse_max":float(np.max(frame_values)),
        }
        print(
            f"direct_fit_s 确定性评估 {name}: "
            f"B_identity_rmse={identity_rmse:.7f}，"
            f"B_warp_rmse={warp_rmse:.7f}，"
            f"appearance_prior_rmse={appearance_rmse:.7f}，"
            f"final_rmse={color_rmse:.7f}，"
            f"interval min={values.min(axis=0).tolist()}，"
            f"median={np.median(values,axis=0).tolist()}，"
            f"max={values.max(axis=0).tolist()}，"
            f"measured/learned visible median="
            f"{float(np.median(measured_values)):.5f}/"
            f"{float(np.median(values[:,1])):.5f}，"
            f"visible<0.99={100*float(np.mean(active)):.1f}%，"
            f"q p5/median/p95={q_text}，"
            "frame_rmse median/p90/p95/max="
            f"{metrics['frame_rmse_median']:.7f}/"
            f"{metrics['frame_rmse_p90']:.7f}/"
            f"{metrics['frame_rmse_p95']:.7f}/"
            f"{metrics['frame_rmse_max']:.7f}")
        return metrics

    evaluation_metrics={
        "trusted":evaluate(
            "trusted",trusted_sequences,evaluation_samples=samples,
            evaluation_valid=masks,evaluation_descriptors=descriptors,
            evaluation_local_grids=local_grids,
            evaluation_visible=visible_fractions),
        "sequence":evaluate(
            "sequence",sequence_sequences,evaluation_samples=samples,
            evaluation_valid=masks,evaluation_descriptors=descriptors,
            evaluation_local_grids=local_grids,
            evaluation_visible=visible_fractions),
    }
    validation_values=(validation_fields,validation_valid,
                       validation_surface_xyz,validation_raw_observed_xyz)
    if any(value is not None for value in validation_values):
        if any(value is None for value in validation_values):
            raise ValueError(
                "direct_fit_s 验证 fields/valid/surface/raw 必须同时提供")
        validation_samples=np.asarray(validation_fields,np.float32)
        validation_masks=np.asarray(validation_valid,np.bool_)
        validation_surfaces=np.asarray(validation_surface_xyz,np.float32)
        validation_raw=np.asarray(validation_raw_observed_xyz,np.float32)
        if validation_samples.ndim!=4 \
                or validation_samples.shape[-1]!=3 \
                or validation_masks.shape!=validation_samples.shape[:3] \
                or validation_surfaces.shape[0]!=validation_samples.shape[0] \
                or validation_raw.shape[0]!=validation_samples.shape[0]:
            raise ValueError("direct_fit_s 验证数组尺寸不一致")
        validation_descriptors,validation_local=evaluation_geometry(
            validation_surfaces,validation_raw)
        validation_visible=_measured_visible_fractions(
            _raw_surface_lengths(validation_raw),length_reference,
            full_threshold=config.full_visibility_threshold,
            minimum=config.minimum_visible_fraction)
        current_validation_sequences=(validation_sequences or [
            np.arange(validation_samples.shape[0],dtype=np.int64)])
        evaluation_metrics["validation"]=evaluate(
            "validation",current_validation_sequences,
            evaluation_samples=validation_samples,
            evaluation_valid=validation_masks,
            evaluation_descriptors=validation_descriptors,
            evaluation_local_grids=validation_local,
            evaluation_visible=validation_visible)

    host=jax.device_get(parameters)
    return DirectFitSTrainingResult(
        base_texture=base_texture,coordinate_frequencies=frequencies,
        geometry_feature_mean=feature_mean,geometry_feature_scale=feature_scale,
        geometry_pca_components=pca_components,geometry_pca_scale=pca_scale,
        local_geometry_feature_mean=local_mean,
        local_geometry_feature_scale=local_scale,
        geometry_encoder_weights=tuple(
            np.asarray(value[0],np.float32) for value in host["encoder"]),
        geometry_encoder_biases=tuple(
            np.asarray(value[1],np.float32) for value in host["encoder"]),
        gru_input_weight=np.asarray(host["gru"][0],np.float32),
        gru_recurrent_weight=np.asarray(host["gru"][1],np.float32),
        gru_bias=np.asarray(host["gru"][2],np.float32),
        warp_weight=np.asarray(host["warp"][0],np.float32),
        warp_bias=np.asarray(host["warp"][1],np.float32),
        appearance_mean=appearance_mean,
        appearance_components=appearance_components.reshape(
            config.appearance_pca_dimensions,
            config.appearance_descriptor_rows,
            config.appearance_descriptor_columns,3),
        appearance_score_mean=appearance_score_mean,
        appearance_score_scale=appearance_score_scale,
        appearance_score_clip=config.appearance_score_clip,
        appearance_weight=np.asarray(host["appearance"][0],np.float32),
        appearance_bias=np.asarray(host["appearance"][1],np.float32),
        appearance_memory_hidden_mean=memory_hidden_mean,
        appearance_memory_hidden_scale=memory_hidden_scale,
        appearance_memory_anchors=memory_anchors,
        appearance_memory_residuals=(
            memory_residuals if config.use_appearance_memory
            else np.zeros_like(memory_residuals)),
        appearance_memory_neighbors=config.appearance_memory_neighbors,
        appearance_memory_epsilon=config.appearance_memory_epsilon,
        length_reference_mm=length_reference,
        color_trunk_weights=tuple(
            np.asarray(value[0],np.float32) for value in host["trunk"]),
        color_trunk_biases=tuple(
            np.asarray(value[1],np.float32) for value in host["trunk"]),
        channel_head_weights=tuple(tuple(
            np.asarray(value[0],np.float32) for value in head)
            for head in host["heads"]),
        channel_head_biases=tuple(tuple(
            np.asarray(value[1],np.float32) for value in head)
            for head in host["heads"]),
        evaluation_metrics=evaluation_metrics)
