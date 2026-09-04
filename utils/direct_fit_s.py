"""direct_fit_s 的连续序列训练；推理算子位于 utils.lightfield。"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from utils.config import DirectFitSConfig
from utils.gpu_residual_fit import _fit_direct_static_base_gpu
from utils.lightfield import (
    DIRECT_LOCAL_GEOMETRY_FEATURE_COUNT,
    direct_background_features_jax,
    direct_geometry_descriptor_jax,
    direct_local_geometry_feature_grid_jax,
    sample_direct_base_texture_jax,
    sample_direct_local_geometry_feature_grid_jax,
)


Array = jax.Array


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
    color_trunk_weights: tuple[np.ndarray,...]
    color_trunk_biases: tuple[np.ndarray,...]
    channel_head_weights: tuple[tuple[np.ndarray,...],...]
    channel_head_biases: tuple[tuple[np.ndarray,...],...]


def _segments_with_minimum_length(
    sequences: list[np.ndarray],minimum: int,
) -> list[np.ndarray]:
    result=[]
    for sequence in sequences:
        values=np.asarray(sequence,np.int64).reshape(-1)
        if values.size>=minimum:
            result.append(values)
    return result


def _cycle_samples(
    sequences: list[np.ndarray],*,cycles_per_video: int,sample_count: int,
) -> list[np.ndarray]:
    result=[]
    for sequence in sequences:
        values=np.asarray(sequence,np.int64).reshape(-1)
        boundaries=np.rint(np.linspace(
            0,values.size,cycles_per_video+1)).astype(np.int64)
        for start,stop in zip(boundaries[:-1],boundaries[1:],strict=True):
            current=values[start:stop]
            if current.size<2:
                raise ValueError(
                    "direct_fit_s 的 cycles_per_video 使某个循环少于两帧")
            positions=np.rint(np.linspace(
                0,current.size-1,sample_count)).astype(np.int64)
            result.append(current[positions])
    return result


def fit_direct_fit_s_gpu(
    fields: np.ndarray,valid: np.ndarray,*,surface_xyz: np.ndarray,
    raw_observed_xyz: np.ndarray, trusted_indices: np.ndarray,
    trusted_sequences: list[np.ndarray],sequence_sequences: list[np.ndarray],
    device: jax.Device,config: DirectFitSConfig,huber_delta: float,seed: int,
) -> DirectFitSTrainingResult:
    """按 B、颜色预训练、warp、联合微调四阶段训练最小序列网络。"""
    samples=np.asarray(fields,np.float32)
    masks=np.asarray(valid,np.bool_)
    surfaces=np.asarray(surface_xyz,np.float32)
    raw_surfaces=np.asarray(raw_observed_xyz,np.float32)
    trusted=np.asarray(trusted_indices,np.int64).reshape(-1)
    if samples.ndim!=4 or samples.shape[-1]!=3 \
            or masks.shape!=samples.shape[:3]:
        raise ValueError("direct_fit_s fields/valid 必须是 NxHxWx3/NxHxW")
    expected_surface=(samples.shape[0],)
    if surfaces.ndim!=4 or raw_surfaces.ndim!=4 \
            or surfaces.shape[0:1]!=expected_surface \
            or raw_surfaces.shape[0:1]!=expected_surface \
            or surfaces.shape[-1]!=3 or raw_surfaces.shape[-1]!=3 \
            or not np.isfinite(surfaces).all() \
            or not np.isfinite(raw_surfaces).all():
        raise ValueError("direct_fit_s 整体/原始观测 XYZ 尺寸或数值无效")
    if trusted.size<2:
        raise ValueError("direct_fit_s 静态 B 至少需要两个可信平直帧")
    if not trusted_sequences or not sequence_sequences:
        raise ValueError("direct_fit_s 必须同时包含可信平直序列和弯曲循环序列")
    if huber_delta<=0:
        raise ValueError("direct_fit_s huber_delta 必须为正")

    base_texture=_fit_direct_static_base_gpu(
        samples[trusted],masks[trusted],device=device,huber_delta=huber_delta,
        iterations=config.base_huber_iterations,
        frame_batch_size=config.base_frame_batch_size)

    analyze=jax.jit(jax.vmap(lambda raw,current:(
        direct_geometry_descriptor_jax(
            raw,config.geometry_descriptor_rows),
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
        vh[:config.geometry_pca_dimensions].T,np.float32)
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
            "direct_fit_s 每类视频都至少要有一段不短于 clip_length 的连续成功帧")
    cycles=_cycle_samples(
        sequence_sequences,cycles_per_video=config.cycles_per_video,
        sample_count=config.cycle_sample_frames)
    generator=np.random.default_rng(seed)
    frequency_values=np.asarray(config.coordinate_frequencies,np.float32)
    coordinate_count=2+4*frequency_values.size
    feature_count=descriptors.shape[1]
    latent_count=config.geometry_latent_dimensions
    hidden_count=config.gru_hidden_dimensions
    color_input_count=(coordinate_count+latent_count
                       +config.geometry_pca_dimensions
                       +DIRECT_LOCAL_GEOMETRY_FEATURE_COUNT+hidden_count)

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
    gru_input,_=layer(latent_count,3*hidden_count)
    gru_recurrent,_=layer(hidden_count,3*hidden_count)
    gru_bias=jnp.zeros((3*hidden_count,),jnp.float32)
    warp_weight,_=layer(hidden_count,3,output=True)
    # softmax 不能表达严格 0；用配置的有限 epsilon 初始化，identity loss
    # 会继续把两端压低。
    identity_epsilon=config.warp_identity_epsilon
    warp_bias=jnp.log(jnp.asarray([
        identity_epsilon,1-2*identity_epsilon,identity_epsilon],jnp.float32))
    trunk=[]
    previous=color_input_count
    for index in range(config.color_trunk_layers):
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
    parameters={
        "encoder":tuple(encoder),
        "gru":(gru_input,gru_recurrent,gru_bias),
        "warp":(warp_weight,warp_bias),
        "trunk":tuple(trunk),"heads":tuple(heads),
    }
    parameters=jax.device_put(parameters,device)
    moments=jax.tree.map(jnp.zeros_like,parameters)
    variances=jax.tree.map(jnp.zeros_like,parameters)
    frequencies_gpu=jax.device_put(frequency_values,device)
    pca_components_gpu=jax.device_put(pca_components,device)
    pca_scale_gpu=jax.device_put(pca_scale,device)
    local_mean_gpu=jax.device_put(local_mean,device)
    local_scale_gpu=jax.device_put(local_scale,device)
    base_gpu=jax.device_put(base_texture,device)

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

    def recurrent(current,latent,hidden):
        input_weight,recurrent_weight,bias=current["gru"]
        input_gates=latent@input_weight+bias
        recurrent_gates=hidden@recurrent_weight
        input_update,input_reset,input_candidate=jnp.split(input_gates,3,-1)
        recurrent_update,recurrent_reset,recurrent_candidate=jnp.split(
            recurrent_gates,3,-1)
        update=jax.nn.sigmoid(input_update+recurrent_update)
        reset=jax.nn.sigmoid(input_reset+recurrent_reset)
        candidate=jnp.tanh(input_candidate+reset*recurrent_candidate)
        return update*hidden+(1-update)*candidate

    def decode(current,coordinates,local_grid,latent,pca,hidden,interval,
               force_identity,detach_conditions):
        effective=jnp.where(
            force_identity,jnp.asarray([0.,1.,0.],jnp.float32),interval)
        warped=jnp.stack([
            effective[0]+effective[1]*coordinates[...,0],coordinates[...,1]],
            axis=-1)
        coordinate_features=direct_background_features_jax(
            warped,frequencies_gpu)
        local=(sample_direct_local_geometry_feature_grid_jax(
            local_grid,coordinates)-local_mean_gpu)/local_scale_gpu
        broadcast=lambda value:jnp.broadcast_to(
            value,(*coordinates.shape[:-1],value.shape[-1]))
        color_latent=(jax.lax.stop_gradient(latent)
                      if detach_conditions else latent)
        color_hidden=(jax.lax.stop_gradient(hidden)
                      if detach_conditions else hidden)
        network_input=jnp.concatenate([
            coordinate_features,broadcast(color_latent),broadcast(pca),local,
            broadcast(color_hidden)],axis=-1)
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
        base=sample_direct_base_texture_jax(base_gpu,warped)
        base=jnp.clip(base,1e-4,1-1e-4)
        return jax.nn.sigmoid(jnp.log(base)-jnp.log1p(-base)+delta)

    def run_clip(current,descriptors_batch,local_batch,coordinates,targets,
                 trusted_role,force_identity,detach_conditions):
        def run_one(descriptor_sequence,local_sequence,coordinate_sequence,
                    target_sequence,is_trusted):
            def frame(hidden,inputs):
                descriptor,local_grid,current_coordinates,current_target=inputs
                latent,pca=encode(current,descriptor)
                hidden=recurrent(current,latent,hidden)
                warp_weight_value,warp_bias_value=current["warp"]
                interval=jax.nn.softmax(
                    hidden@warp_weight_value+warp_bias_value)
                prediction=decode(
                    current,current_coordinates,local_grid,latent,pca,hidden,
                    interval,force_identity,detach_conditions)
                return hidden,(prediction,current_target,interval)
            initial=jnp.zeros((hidden_count,),jnp.float32)
            _,outputs=jax.lax.scan(frame,initial,(
                descriptor_sequence,local_sequence,coordinate_sequence,
                target_sequence))
            prediction,target,interval=outputs
            error=prediction-target
            absolute=jnp.abs(error)
            photometric=jnp.mean(jnp.where(
                absolute<=huber_delta,.5*error**2,
                huber_delta*(absolute-.5*huber_delta)))
            identity=jnp.mean((interval-jnp.asarray([0.,1.,0.]))**2)
            temporal=jnp.mean((interval[1:]-interval[:-1])**2)
            crop=jnp.mean((1-interval[:,1])**2)
            return (photometric,
                    jnp.where(is_trusted,identity,0.),
                    jnp.where(is_trusted,0.,temporal),
                    jnp.where(is_trusted,0.,crop),
                    jnp.mean(error**2),jnp.mean(interval,axis=0))
        return jax.vmap(run_one)(
            descriptors_batch,local_batch,coordinates,targets,trusted_role)

    def cycle_loss(current,cycle_descriptors):
        def run_one(values):
            def frame(hidden,descriptor):
                latent,_=encode(current,descriptor)
                hidden=recurrent(current,latent,hidden)
                weight,bias=current["warp"]
                return hidden,jax.nn.softmax(hidden@weight+bias)
            _,interval=jax.lax.scan(
                frame,jnp.zeros((hidden_count,),jnp.float32),values)
            endpoint=config.cycle_endpoint_frames
            return jnp.mean((jnp.mean(interval[:endpoint],axis=0)
                             -jnp.mean(interval[-endpoint:],axis=0))**2)
        return jnp.mean(jax.vmap(run_one)(cycle_descriptors))

    def loss_fn(current,batch,cycle_batch,force_identity,use_regularizers,
                detach_conditions):
        metrics=run_clip(
            current,*batch,force_identity,detach_conditions)
        photometric=jnp.mean(metrics[0])
        rmse=jnp.sqrt(jnp.mean(metrics[4]))
        untrusted=(~batch[4]).astype(jnp.float32)
        mean_interval=jnp.sum(
            metrics[5]*untrusted[:,None],axis=0) \
            /jnp.maximum(jnp.sum(untrusted),1.)
        if not use_regularizers:
            auxiliary=jnp.concatenate([jnp.stack([
                photometric,jnp.asarray(0.),jnp.asarray(0.),jnp.asarray(0.),
                jnp.asarray(0.),rmse]),mean_interval])
            return photometric,auxiliary
        identity=jnp.mean(metrics[1])
        temporal=jnp.mean(metrics[2])
        crop=jnp.mean(metrics[3])
        closure=cycle_loss(current,cycle_batch)
        total=(photometric+config.identity_weight*identity
               +config.temporal_weight*temporal
               +config.minimum_crop_weight*crop
               +config.cycle_weight*closure)
        auxiliary=jnp.concatenate([jnp.stack([
            photometric,identity,temporal,closure,crop,rmse]),mean_interval])
        return total,auxiliary

    beta1,beta2=config.adam_beta1,config.adam_beta2
    total_steps=(config.color_pretrain_steps+config.warp_train_steps
                 +config.joint_finetune_steps)

    def make_step(*,force_identity: bool,use_regularizers: bool,
                  freeze_color: bool):
        @jax.jit
        def step(current,moment,variance,index,batch,cycle_batch,rate):
            (loss,metrics),gradient=jax.value_and_grad(
                loss_fn,has_aux=True)(
                    current,batch,cycle_batch,force_identity,use_regularizers,
                    freeze_color)
            if freeze_color:
                gradient={**gradient,
                          "trunk":jax.tree.map(jnp.zeros_like,
                                                gradient["trunk"]),
                          "heads":jax.tree.map(jnp.zeros_like,
                                                gradient["heads"])}
            norm=jnp.sqrt(sum(jnp.sum(value**2) for value in
                              jax.tree.leaves(gradient)))
            factor=jnp.minimum(1.,config.gradient_clip_norm/jnp.maximum(norm,1e-12))
            gradient=jax.tree.map(lambda value:value*factor,gradient)
            moment=jax.tree.map(
                lambda old,value:beta1*old+(1-beta1)*value,moment,gradient)
            variance=jax.tree.map(
                lambda old,value:beta2*old+(1-beta2)*value*value,
                variance,gradient)
            if freeze_color:
                moment={**moment,
                        "trunk":jax.tree.map(jnp.zeros_like,moment["trunk"]),
                        "heads":jax.tree.map(jnp.zeros_like,moment["heads"])}
                variance={
                    **variance,
                    "trunk":jax.tree.map(jnp.zeros_like,variance["trunk"]),
                    "heads":jax.tree.map(jnp.zeros_like,variance["heads"])}
            corrected_m=jax.tree.map(
                lambda value:value/(1-beta1**index),moment)
            corrected_v=jax.tree.map(
                lambda value:value/(1-beta2**index),variance)
            current=jax.tree.map(
                lambda value,m,v:value-rate*m/(
                    jnp.sqrt(v)+config.adam_epsilon),
                current,corrected_m,corrected_v)
            return current,moment,variance,loss,metrics,norm
        return step

    steps={
        "color":make_step(force_identity=True,use_regularizers=False,
                          freeze_color=False),
        "warp":make_step(force_identity=False,use_regularizers=True,
                         freeze_color=True),
        "joint":make_step(force_identity=False,use_regularizers=True,
                          freeze_color=False),
    }

    rows,columns=samples.shape[1:3]
    valid_pixels=[np.flatnonzero(value.reshape(-1)) for value in masks]
    if any(value.size==0 for value in valid_pixels):
        raise ValueError("direct_fit_s 存在没有有效采样点的帧")

    def sample_clip(sequence: np.ndarray) -> np.ndarray:
        maximum=sequence.size-config.clip_length
        start=int(generator.integers(0,maximum+1))
        return sequence[start:start+config.clip_length]

    def training_batch():
        trusted_count=int(round(
            config.clip_batch_size*config.trusted_clip_fraction))
        trusted_count=min(max(trusted_count,0),config.clip_batch_size)
        if trusted_count==0 and config.identity_weight>0:
            trusted_count=1
        if trusted_count==config.clip_batch_size \
                and (config.temporal_weight>0 or config.minimum_crop_weight>0):
            trusted_count-=1
        roles=np.asarray(
            [True]*trusted_count
            +[False]*(config.clip_batch_size-trusted_count),np.bool_)
        generator.shuffle(roles)
        clip_indices=[]
        for role in roles:
            choices=trusted_clips if role else sequence_clips
            clip_indices.append(sample_clip(choices[int(
                generator.integers(0,len(choices)))]))
        clip_indices=np.stack(clip_indices)
        pixel_indices=np.empty((
            config.clip_batch_size,config.clip_length,
            config.points_per_frame),np.int64)
        for batch_index in range(config.clip_batch_size):
            for frame_index in range(config.clip_length):
                source=int(clip_indices[batch_index,frame_index])
                available=valid_pixels[source]
                pixel_indices[batch_index,frame_index]=generator.choice(
                    available,config.points_per_frame,
                    replace=available.size<config.points_per_frame)
        coordinate=np.stack([
            (pixel_indices//columns)/max(rows-1,1),
            (pixel_indices%columns)/max(columns-1,1)],axis=-1).astype(np.float32)
        flattened=samples.reshape(samples.shape[0],-1,3)
        target=np.empty((*pixel_indices.shape,3),np.float32)
        for batch_index in range(config.clip_batch_size):
            target[batch_index]=flattened[
                clip_indices[batch_index,:,None],pixel_indices[batch_index]]
        return tuple(jax.device_put(value,device) for value in (
            descriptors[clip_indices],local_grids[clip_indices],coordinate,
            target,roles))

    cycle_arrays=np.stack([descriptors[value] for value in cycles])

    def sampled_cycles():
        selected=generator.integers(
            0,cycle_arrays.shape[0],size=config.clip_batch_size)
        return jax.device_put(np.ascontiguousarray(cycle_arrays[selected]),device)

    stage_specs=(
        ("color",config.color_pretrain_steps,
         config.color_pretrain_learning_rate),
        ("warp",config.warp_train_steps,config.warp_learning_rate),
        ("joint",config.joint_finetune_steps,config.joint_learning_rate),
    )
    global_step=0
    print(
        "direct_fit_s："
        f"trusted={trusted.size}，all={samples.shape[0]}，"
        f"clips={len(trusted_clips)}+{len(sequence_clips)}，"
        f"GRU={hidden_count}，warp=affine-s，"
        f"trunk={config.color_trunk_layers}x{config.color_trunk_width}，"
        f"heads=3x{config.color_head_layers}")
    for stage_name,count,rate in stage_specs:
        if count==0:
            continue
        # 各阶段的可训练子树和学习率不同；重新初始化 Adam 状态，避免上一阶段
        # 的动量越过“冻结颜色网络”的边界。
        moments=jax.tree.map(jnp.zeros_like,parameters)
        variances=jax.tree.map(jnp.zeros_like,parameters)
        print(f"direct_fit_s 阶段 {stage_name}: steps={count}, lr={rate:g}")
        stage_step=steps[stage_name]
        for local_step in range(1,count+1):
            global_step+=1
            parameters,moments,variances,loss,metrics,gradient_norm=stage_step(
                parameters,moments,variances,jnp.asarray(local_step,jnp.float32),
                training_batch(),sampled_cycles(),jnp.asarray(rate,jnp.float32))
            if local_step==1 or local_step%100==0 or local_step==count:
                loss_value,metric_values,norm_value=jax.device_get(
                    (loss,metrics,gradient_norm))
                print(
                    f"direct_fit_s {stage_name} {local_step}/{count}: "
                    f"loss={float(loss_value):.7f}, "
                    f"photo/id/temp/cycle/crop/rmse/untrusted_interval="
                    f"{np.asarray(metric_values).tolist()}, "
                    f"grad={float(norm_value):.5f}")
    assert global_step==total_steps

    host=jax.device_get(parameters)
    encoder_weights=tuple(np.asarray(value[0],np.float32)
                          for value in host["encoder"])
    encoder_biases=tuple(np.asarray(value[1],np.float32)
                         for value in host["encoder"])
    trunk_weights=tuple(np.asarray(value[0],np.float32)
                        for value in host["trunk"])
    trunk_biases=tuple(np.asarray(value[1],np.float32)
                       for value in host["trunk"])
    head_weights=tuple(tuple(np.asarray(value[0],np.float32) for value in head)
                       for head in host["heads"])
    head_biases=tuple(tuple(np.asarray(value[1],np.float32) for value in head)
                      for head in host["heads"])
    return DirectFitSTrainingResult(
        base_texture=base_texture,coordinate_frequencies=frequency_values,
        geometry_feature_mean=feature_mean,geometry_feature_scale=feature_scale,
        geometry_pca_components=pca_components,geometry_pca_scale=pca_scale,
        local_geometry_feature_mean=local_mean,
        local_geometry_feature_scale=local_scale,
        geometry_encoder_weights=encoder_weights,
        geometry_encoder_biases=encoder_biases,
        gru_input_weight=np.asarray(host["gru"][0],np.float32),
        gru_recurrent_weight=np.asarray(host["gru"][1],np.float32),
        gru_bias=np.asarray(host["gru"][2],np.float32),
        warp_weight=np.asarray(host["warp"][0],np.float32),
        warp_bias=np.asarray(host["warp"][1],np.float32),
        color_trunk_weights=trunk_weights,color_trunk_biases=trunk_biases,
        channel_head_weights=head_weights,channel_head_biases=head_biases)
