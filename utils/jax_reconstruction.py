"""实时曲面几何重建的 JAX/GPU 数值核心。"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .process import (_aggregate_edge_by_v, _resample_polyline, _undistort_pixels,
                      split_edge_segments)

Array = jax.Array
SURFACE_RECONSTRUCTION_PIPELINE_VERSION = "jax_surface_from_masks_v1"


def _solve_block_pentadiagonal_jax(
    local_diagonal: Array,
    rhs: Array,
    smooth_lambda: float,
) -> Array:
    """求解带二阶平滑项的 2x2 块五对角 SPD 系统。

    未显式构造/分解 (2N)x(2N) 稠密矩阵；块 LDLᵀ 的计算量随截面数 N
    线性增长。变量按每个截面的 [h,z] 交错排列。
    """
    count=local_diagonal.shape[0]
    dtype=local_diagonal.dtype
    eye=jnp.eye(2,dtype=dtype)
    zero=jnp.zeros((2,2),dtype=dtype)
    rows=jnp.arange(count-2,dtype=jnp.int32)
    penalty_diagonal=jnp.zeros((count,),dtype=dtype)
    penalty_diagonal=penalty_diagonal.at[rows].add(1)
    penalty_diagonal=penalty_diagonal.at[rows+1].add(4)
    penalty_diagonal=penalty_diagonal.at[rows+2].add(1)
    first_diagonal=jnp.zeros((count-1,),dtype=dtype)
    first_diagonal=first_diagonal.at[rows].add(-2)
    first_diagonal=first_diagonal.at[rows+1].add(-2)
    second_diagonal=jnp.ones((count-2,),dtype=dtype)
    diagonal=local_diagonal+smooth_lambda*penalty_diagonal[:,None,None]*eye
    first=jnp.zeros((count,2,2),dtype=dtype).at[1:].set(
        smooth_lambda*first_diagonal[:,None,None]*eye)
    second=jnp.zeros((count,2,2),dtype=dtype).at[2:].set(
        smooth_lambda*second_diagonal[:,None,None]*eye)
    indices=jnp.arange(count)

    def inverse_2x2(matrix):
        determinant=(matrix[...,0,0]*matrix[...,1,1]
                     -matrix[...,0,1]*matrix[...,1,0])
        return jnp.stack([
            jnp.stack([matrix[...,1,1],-matrix[...,0,1]],axis=-1),
            jnp.stack([-matrix[...,1,0],matrix[...,0,0]],axis=-1),
        ],axis=-2)/determinant[...,None,None]

    def right_solve(matrix,pivot):
        # 这里只会出现 2x2 SPD pivot；显式逆比通用 LU/QR kernel 快很多。
        return matrix@inverse_2x2(pivot)

    def factor(carry,values):
        pivot_minus_two,pivot_minus_one,previous_first=carry
        current_diagonal,current_first,current_second,index=values
        use_first=index>0
        use_second=index>1
        lower_second=jnp.where(
            use_second,right_solve(current_second,pivot_minus_two),zero)
        adjusted_first=(current_first-
                        lower_second@pivot_minus_two@previous_first.T)
        lower_first=jnp.where(
            use_first,right_solve(adjusted_first,pivot_minus_one),zero)
        pivot=(current_diagonal-
               lower_first@pivot_minus_one@lower_first.T-
               lower_second@pivot_minus_two@lower_second.T)
        return ((pivot_minus_one,pivot,lower_first),
                (pivot,lower_first,lower_second))

    initial=(eye,eye,zero)
    _,(pivots,lower_first,lower_second)=jax.lax.scan(
        factor,initial,(diagonal,first,second,indices))

    def forward(carry,values):
        value_minus_two,value_minus_one=carry
        current_rhs,current_first,current_second=values
        value=(current_rhs-current_first@value_minus_one-
               current_second@value_minus_two)
        return (value_minus_one,value),value

    _,forward_rhs=jax.lax.scan(
        forward,(jnp.zeros(2,dtype),jnp.zeros(2,dtype)),
        (rhs,lower_first,lower_second))
    diagonal_solution=jnp.einsum(
        "nij,nj->ni",inverse_2x2(pivots),forward_rhs)
    next_first=jnp.concatenate([lower_first[1:],zero[None]],axis=0)
    next_second=jnp.concatenate([lower_second[2:],zero[None],zero[None]],axis=0)

    def backward(carry,values):
        value_plus_two,value_plus_one=carry
        current_value,current_first,current_second=values
        value=(current_value-current_first.T@value_plus_one-
               current_second.T@value_plus_two)
        return (value_plus_one,value),value

    _,solution=jax.lax.scan(
        backward,(jnp.zeros(2,dtype),jnp.zeros(2,dtype)),
        (diagonal_solution,next_first,next_second),reverse=True)
    return solution


def _solve_smoothed_boundary_cg_jax(
    weights: Array,
    rhs: Array,
    smooth_lambda: float,
    *,
    iterations: int = 96,
) -> Array:
    """GPU 友好的 Jacobi-PCG，求解 (W+lambda D2ᵀD2)x=rhs。"""
    count=weights.shape[0]
    rows=jnp.arange(count-2,dtype=jnp.int32)
    penalty_diagonal=jnp.zeros((count,),rhs.dtype)
    penalty_diagonal=penalty_diagonal.at[rows].add(1)
    penalty_diagonal=penalty_diagonal.at[rows+1].add(4)
    penalty_diagonal=penalty_diagonal.at[rows+2].add(1)
    diagonal=weights+smooth_lambda*penalty_diagonal

    def matvec(values):
        second_difference=values[:-2]-2*values[1:-1]+values[2:]
        penalty=jnp.zeros_like(values)
        penalty=penalty.at[:-2].add(second_difference)
        penalty=penalty.at[1:-1].add(-2*second_difference)
        penalty=penalty.at[2:].add(second_difference)
        return weights[:,None]*values+smooth_lambda*penalty

    solution=rhs/diagonal[:,None]
    residual=rhs-matvec(solution)
    preconditioned=residual/diagonal[:,None]
    direction=preconditioned
    rz=jnp.sum(residual*preconditioned,axis=0)
    epsilon=jnp.asarray(1e-20,rhs.dtype)

    def iteration(_,state):
        current,residual,direction,rz=state
        product=matvec(direction)
        denominator=jnp.sum(direction*product,axis=0)
        alpha=rz/jnp.maximum(denominator,epsilon)
        current=current+direction*alpha
        residual=residual-product*alpha
        preconditioned=residual/diagonal[:,None]
        rz_new=jnp.sum(residual*preconditioned,axis=0)
        beta=rz_new/jnp.maximum(rz,epsilon)
        direction=preconditioned+direction*beta
        return current,residual,direction,rz_new

    solution,_,_,_=jax.lax.fori_loop(
        0,iterations,iteration,(solution,residual,direction,rz))
    return solution


def prepare_edge_curves(
    edges: Sequence[tuple[str | int,float,list[np.ndarray]]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray,np.ndarray]:
    """CPU 只整理 OpenCV 轮廓，返回固定形状的去畸变左右曲线。"""
    left_curves: list[np.ndarray]=[]
    right_curves: list[np.ndarray]=[]
    for _,u0,segments in edges:
        if not segments:
            continue
        left_raw,right_raw=split_edge_segments(segments,u0)
        left_ordered=_aggregate_edge_by_v(left_raw)
        right_ordered=_aggregate_edge_by_v(right_raw)
        if left_ordered.shape[0]<4 or right_ordered.shape[0]<4:
            continue
        try:
            left=_resample_polyline(
                _undistort_pixels(left_ordered,camera_matrix,distortion),sample_count)
            right=_resample_polyline(
                _undistort_pixels(right_ordered,camera_matrix,distortion),4*sample_count)
        except ValueError:
            continue
        left_curves.append(left.astype(np.float32))
        right_curves.append(right.astype(np.float32))
    if not left_curves:
        return (np.zeros((0,sample_count,2),np.float32),
                np.zeros((0,4*sample_count,2),np.float32))
    return np.stack(left_curves),np.stack(right_curves)


def _morphology_jax(masks: Array,kernel_size: int,operation: str) -> Array:
    """对 KxHxW mask 执行固定窗口二值形态学。"""
    if kernel_size<=1:
        return masks.astype(jnp.bool_)
    window=(1,kernel_size,kernel_size)
    strides=(1,1,1)
    values=masks.astype(jnp.uint8)
    if operation=="dilate":
        reduced=jax.lax.reduce_window(
            values,jnp.uint8(0),jax.lax.max,window,strides,"SAME")
    elif operation=="erode":
        reduced=jax.lax.reduce_window(
            values,jnp.uint8(1),jax.lax.min,window,strides,"SAME")
    else:
        raise ValueError(f"未知形态学操作: {operation}")
    return reduced>0


def refine_surface_masks_jax(
    masks: Array,
    *,
    close_kernel: int = 0,
    open_kernel: int = 0,
    blur_kernel: int = 0,
) -> Array:
    """在 GPU 上完成闭运算、开运算和高斯平滑阈值化。"""
    close_kernel=(close_kernel if close_kernel%2 else close_kernel+1)
    open_kernel=(open_kernel if open_kernel%2 else open_kernel+1)
    blur_kernel=(blur_kernel if blur_kernel%2 else blur_kernel+1)
    refined=jnp.asarray(masks,jnp.bool_)
    if refined.ndim!=3:
        raise ValueError("masks 必须是 KxHxW")
    if close_kernel>1:
        refined=_morphology_jax(refined,close_kernel,"dilate")
        refined=_morphology_jax(refined,close_kernel,"erode")
    if open_kernel>1:
        refined=_morphology_jax(refined,open_kernel,"erode")
        refined=_morphology_jax(refined,open_kernel,"dilate")
    if blur_kernel>1:
        radius=(blur_kernel-1)/2
        sigma=.3*(radius-1)+.8
        coordinate=jnp.arange(blur_kernel,dtype=jnp.float32)-radius
        gaussian=jnp.exp(-.5*(coordinate/sigma)**2)
        gaussian=gaussian/jnp.sum(gaussian)
        kernel=(gaussian[:,None,None,None]*
                gaussian[None,:,None,None])
        blurred=jax.lax.conv_general_dilated(
            refined[...,None].astype(jnp.float32),kernel,
            window_strides=(1,1),padding="SAME",
            dimension_numbers=("NHWC","HWIO","NHWC"))[...,0]
        refined=blurred>.5
    return refined


def _undistort_pixels_jax(points: Array,camera_matrix: Array,
                          distortion: Array,iterations: int = 5) -> Array:
    """固定迭代反解 OpenCV 五参数径向/切向畸变。"""
    distorted_x=(points[...,0]-camera_matrix[0,2])/camera_matrix[0,0]
    distorted_y=(points[...,1]-camera_matrix[1,2])/camera_matrix[1,1]
    k1,k2,p1,p2,k3=distortion[:5]

    def iteration(_,current):
        x,y=current
        radius2=x*x+y*y
        radial=1+k1*radius2+k2*radius2**2+k3*radius2**3
        delta_x=2*p1*x*y+p2*(radius2+2*x*x)
        delta_y=p1*(radius2+2*y*y)+2*p2*x*y
        safe_radial=jnp.where(jnp.abs(radial)>1e-8,radial,1.)
        return ((distorted_x-delta_x)/safe_radial,
                (distorted_y-delta_y)/safe_radial)

    x,y=jax.lax.fori_loop(
        0,iterations,iteration,(distorted_x,distorted_y))
    return jnp.stack([
        camera_matrix[0,0]*x+camera_matrix[0,2],
        camera_matrix[1,1]*y+camera_matrix[1,2]],axis=-1)


def _smooth_row_boundaries_jax(values: Array,valid_rows: Array,
                                window_size: int = 5) -> Array:
    weighted=values*jnp.asarray(valid_rows,values.dtype)
    window=(1,window_size)
    total=jax.lax.reduce_window(
        weighted,jnp.asarray(0.,values.dtype),jax.lax.add,
        window,(1,1),"SAME")
    count=jax.lax.reduce_window(
        valid_rows.astype(values.dtype),jnp.asarray(0.,values.dtype),jax.lax.add,
        window,(1,1),"SAME")
    return jnp.where(count>0,total/jnp.maximum(count,1),values)


def _resample_row_boundary_jax(values: Array,valid_rows: Array,
                               count: int) -> Array:
    """把逐行 x 边界插值为固定数量的亚像素 (x,y) 点。"""
    height=values.shape[1]
    rows=jnp.arange(height,dtype=values.dtype)
    first=jnp.min(jnp.where(valid_rows,rows[None],height),axis=1)
    last=jnp.max(jnp.where(valid_rows,rows[None],-1),axis=1)
    alpha=jnp.linspace(0.,1.,count,dtype=values.dtype)
    targets=first[:,None]+alpha[None]*(last-first)[:,None]
    row_grid=rows[None,None,:]
    valid_grid=valid_rows[:,None,:]
    lower=jnp.max(jnp.where(
        valid_grid&(row_grid<=targets[...,None]),row_grid,-1),axis=-1)
    upper=jnp.min(jnp.where(
        valid_grid&(row_grid>=targets[...,None]),row_grid,height),axis=-1)
    lower_index=jnp.clip(lower,0,height-1).astype(jnp.int32)
    upper_index=jnp.clip(upper,0,height-1).astype(jnp.int32)
    lower_x=jnp.take_along_axis(values,lower_index,axis=1)
    upper_x=jnp.take_along_axis(values,upper_index,axis=1)
    span=jnp.maximum(upper-lower,1)
    fraction=jnp.where(upper>lower,(targets-lower)/span,0)
    x=lower_x+fraction*(upper_x-lower_x)
    return jnp.stack([x,targets],axis=-1)


def prepare_edge_curves_from_masks_jax(
    masks: Array,
    camera_matrix: Array,
    distortion: Array,
    sample_count: int,
    center_band_d: float,
    *,
    close_kernel: int = 0,
    open_kernel: int = 0,
    blur_kernel: int = 0,
    min_v: int = 5,
) -> tuple[Array,Array,Array,Array]:
    """从设备端 mask 生成固定形状的无畸变左右边缘曲线。"""
    refined=refine_surface_masks_jax(
        masks,close_kernel=close_kernel,open_kernel=open_kernel,
        blur_kernel=blur_kernel)
    _,height,width=refined.shape
    columns=jnp.arange(width,dtype=jnp.int32)[None,None,:]
    left_index=jnp.min(jnp.where(refined,columns,width),axis=2)
    right_index=jnp.max(jnp.where(refined,columns,-1),axis=2)
    row_index=jnp.arange(height,dtype=jnp.int32)[None,:]
    valid_rows=((left_index<right_index)&(row_index>=min_v)&
                ((right_index-left_index)>2*center_band_d))
    left=_smooth_row_boundaries_jax(
        left_index.astype(jnp.float32),valid_rows)
    right=_smooth_row_boundaries_jax(
        right_index.astype(jnp.float32),valid_rows)
    left_curve=_resample_row_boundary_jax(left,valid_rows,sample_count)
    right_dense_curve=_resample_row_boundary_jax(
        right,valid_rows,4*sample_count)
    left_curve=_undistort_pixels_jax(
        left_curve,camera_matrix,distortion)
    right_dense_curve=_undistort_pixels_jax(
        right_dense_curve,camera_matrix,distortion)
    curve_valid=(jnp.sum(valid_rows,axis=1)>=4)&jnp.all(
        jnp.isfinite(left_curve),axis=(1,2))&jnp.all(
            jnp.isfinite(right_dense_curve),axis=(1,2))
    return refined,left_curve,right_dense_curve,curve_valid


def _intersect_pixels_with_x_plane_jax(pixels: Array,x_plane: float,
                                        inverse_camera: Array,rotation: Array,
                                        tx: float) -> Array:
    homogeneous=jnp.concatenate([pixels,jnp.ones((*pixels.shape[:-1],1),pixels.dtype)],axis=-1)
    rays_camera=homogeneous@inverse_camera.T
    camera_center=-(rotation.T@jnp.asarray([tx,0.,0.],pixels.dtype))
    rays_world=rays_camera@rotation
    scale=(x_plane-camera_center[0])/rays_world[...,0]
    points=camera_center+scale[...,None]*rays_world
    return points.at[...,0].set(x_plane)


def monotone_right_matches_jax(left_uv: Array,right_dense_uv: Array,
                               inverse_camera: Array,rotation: Array,
                               s1: float,s2: float,tx: float) -> tuple[Array,Array]:
    """沿截面用 JAX scan 求严格单调的左右边缘匹配。"""
    left_xyz=_intersect_pixels_with_x_plane_jax(
        left_uv,s2,inverse_camera,rotation,tx)
    right_xyz=_intersect_pixels_with_x_plane_jax(
        right_dense_uv,s1,inverse_camera,rotation,tx)
    count=left_uv.shape[0]; dense_count=right_dense_uv.shape[0]
    tau=jnp.linspace(0.,1.,count); xi=jnp.linspace(0.,1.,dense_count)
    dy=(left_xyz[:,None,1]-right_xyz[None,:,1])/2
    dz=(left_xyz[:,None,2]-right_xyz[None,:,2])/2
    dt=(tau[:,None]-xi[None,:])/.05
    cost=dy*dy+dz*dz+dt*dt
    cost=jnp.where(jnp.abs(tau[:,None]-xi[None,:])<=.08,cost,jnp.inf)
    ideal_step=(dense_count-1)/(count-1)
    max_step=max(2,int(np.ceil(2*ideal_step)))
    # prior_j 按从小到大排列，保持与原 Python 动态规划相同的平局选择。
    offsets=jnp.arange(max_step,0,-1,dtype=jnp.int32)
    columns=jnp.arange(dense_count,dtype=jnp.int32)
    prior=columns[:,None]-offsets[None,:]
    prior_valid=prior>=0
    prior_clipped=jnp.maximum(prior,0)
    step_penalty=.25*((offsets.astype(left_uv.dtype)-ideal_step)/ideal_step)**2
    initial=jnp.full((dense_count,),jnp.inf,left_uv.dtype).at[0].set(cost[0,0])

    def advance(previous,row_cost):
        candidates=previous[prior_clipped]+step_penalty[None,:]
        candidates=jnp.where(prior_valid,candidates,jnp.inf)
        choice=jnp.argmin(candidates,axis=1)
        best=jnp.take_along_axis(candidates,choice[:,None],axis=1)[:,0]
        pointer=jnp.take_along_axis(prior,choice[:,None],axis=1)[:,0]
        return row_cost+best,pointer

    accumulated,pointers=jax.lax.scan(advance,initial,cost[1:])
    indices=jnp.full((count,),-1,jnp.int32).at[-1].set(dense_count-1)

    def backtrack(step,current_indices):
        row=count-1-step
        previous=pointers[row-1,current_indices[row]]
        return current_indices.at[row-1].set(previous)

    indices=jax.lax.fori_loop(0,count-1,backtrack,indices)
    final_cost=accumulated[dense_count-1]
    valid=jnp.isfinite(final_cost)&jnp.all(indices>=0)
    return indices,valid


def _project_world_points_jax(points: Array,camera_matrix: Array,
                              rotation: Array,tx: float) -> tuple[Array,Array]:
    camera=points@rotation.T
    camera=camera.at[...,0].add(tx)
    depth=camera[...,2]
    safe_depth=jnp.where(jnp.abs(depth)<1e-6,1e-6,depth)
    normalized=camera[...,:2]/safe_depth[...,None]
    pixels=jnp.stack([
        camera_matrix[0,0]*normalized[...,0]+camera_matrix[0,2],
        camera_matrix[1,1]*normalized[...,1]+camera_matrix[1,2]],axis=-1)
    return pixels,depth


def _weighted_isotonic_nondecreasing_jax(values: Array,weights: Array) -> Array:
    """固定形状 PAVA：把一维序列投影到非递减加权最小二乘锥。"""
    count=values.shape[0]
    levels=jnp.zeros_like(values)
    block_weights=jnp.zeros_like(weights)
    block_lengths=jnp.zeros((count,),jnp.int32)

    def append(index,state):
        current_levels,current_weights,current_lengths,block_count=state
        current_levels=current_levels.at[block_count].set(values[index])
        current_weights=current_weights.at[block_count].set(weights[index])
        current_lengths=current_lengths.at[block_count].set(1)
        block_count=block_count+1

        def violates(merge_state):
            merge_levels,_,_,merge_count=merge_state
            return ((merge_count>1)&
                    (merge_levels[merge_count-2]>merge_levels[merge_count-1]))

        def merge(merge_state):
            merge_levels,merge_weights,merge_lengths,merge_count=merge_state
            left=merge_count-2; right=merge_count-1
            total_weight=merge_weights[left]+merge_weights[right]
            level=(merge_weights[left]*merge_levels[left]
                   +merge_weights[right]*merge_levels[right])/total_weight
            merge_levels=merge_levels.at[left].set(level)
            merge_weights=merge_weights.at[left].set(total_weight)
            merge_lengths=merge_lengths.at[left].add(merge_lengths[right])
            return merge_levels,merge_weights,merge_lengths,merge_count-1

        return jax.lax.while_loop(
            violates,merge,
            (current_levels,current_weights,current_lengths,block_count))

    levels,_,block_lengths,block_count=jax.lax.fori_loop(
        0,count,append,(levels,block_weights,block_lengths,jnp.int32(0)))
    block_is_used=jnp.arange(count,dtype=jnp.int32)<block_count
    block_ends=jnp.cumsum(jnp.where(block_is_used,block_lengths,0))
    positions=jnp.arange(count,dtype=jnp.int32)
    block_indices=jnp.sum(
        (positions[:,None]>=block_ends[None,:])&block_is_used[None,:],axis=1)
    return levels[block_indices]


def _project_convex_depth_jax(h: Array,z: Array,
                              direction: str,
                              smoothing_iterations: int = 4,
                              ) -> tuple[Array,Array]:
    """保持端点，将 z(h) 的斜率投影为单调序列。"""
    if direction not in ("increasing","decreasing"):
        raise ValueError("direction 必须是 increasing 或 decreasing")
    delta_h=jnp.diff(h)
    monotone_h=jnp.all(delta_h>1e-5)
    safe_delta_h=jnp.maximum(delta_h,1e-5)
    sign=1. if direction=="increasing" else -1.
    slopes=jnp.diff(z)/safe_delta_h
    projected=sign*_weighted_isotonic_nondecreasing_jax(
        sign*slopes,safe_delta_h)

    def smooth(_,current):
        padded=jnp.concatenate([current[:1],current,current[-1:]])
        return .25*padded[:-2]+.5*padded[1:-1]+.25*padded[2:]

    projected=jax.lax.fori_loop(
        0,smoothing_iterations,smooth,projected)
    # 平滑不会破坏单调性；统一平移斜率以精确保留末端深度。
    projected=projected+(
        (z[-1]-z[0])-jnp.sum(projected*safe_delta_h)
    )/jnp.sum(safe_delta_h)
    projected_z=jnp.concatenate([
        z[:1],z[0]+jnp.cumsum(projected*safe_delta_h)])
    slope_difference=sign*jnp.diff(projected)
    convex=jnp.all(slope_difference>=-2e-5)
    return projected_z,monotone_h&convex


def solve_smoothed_shared_curve_jax(left_uv: Array,right_uv: Array,
                                    camera_matrix: Array,rotation: Array,
                                    s1: float,s2: float,tx: float,
                                    smooth_lambda: float = 1.,
                                    curve_convexity: str = "none",
                                    ) -> tuple[Array,Array,Array,Array]:
    """仅由当前帧观测求共享 h/z 曲线，并施加可选凸性约束。"""
    if curve_convexity not in ("none","increasing","decreasing"):
        raise ValueError(
            "curve_convexity 必须是 none、increasing 或 decreasing")
    count=left_uv.shape[0]

    def side_equations(uv,x_plane):
        x=(uv[:,0]-camera_matrix[0,2])/camera_matrix[0,0]
        y=(uv[:,1]-camera_matrix[1,2])/camera_matrix[1,1]
        first=jnp.stack([rotation[0,1]-x*rotation[2,1],
                         rotation[0,2]-x*rotation[2,2]],axis=-1)
        first_target=-(rotation[0,0]-x*rotation[2,0])*x_plane-tx
        second=jnp.stack([rotation[1,1]-y*rotation[2,1],
                          rotation[1,2]-y*rotation[2,2]],axis=-1)
        second_target=-(rotation[1,0]-y*rotation[2,0])*x_plane
        return jnp.stack([first,second],axis=1),jnp.stack([first_target,second_target],axis=1)

    left_a,left_b=side_equations(left_uv,s2)
    right_a,right_b=side_equations(right_uv,s1)
    coefficients=jnp.concatenate([left_a,right_a],axis=1)
    targets=jnp.concatenate([left_b,right_b],axis=1)
    local_normal=jnp.einsum("nri,nrj->nij",coefficients,coefficients)
    local_rhs=jnp.einsum("nri,nr->ni",coefficients,targets)
    unconstrained_solution=_solve_block_pentadiagonal_jax(
        local_normal,local_rhs,smooth_lambda)

    def reprojection(curve_values):
        current_h=curve_values[:,0]; current_z=curve_values[:,1]
        left_xyz=jnp.stack(
            [jnp.full_like(current_h,s2),current_h,current_z],axis=-1)
        right_xyz=jnp.stack(
            [jnp.full_like(current_h,s1),current_h,current_z],axis=-1)
        left_projection,left_depth=_project_world_points_jax(
            left_xyz,camera_matrix,rotation,tx)
        right_projection,right_depth=_project_world_points_jax(
            right_xyz,camera_matrix,rotation,tx)
        squared=jnp.concatenate([
            jnp.sum((left_projection-left_uv)**2,axis=1),
            jnp.sum((right_projection-right_uv)**2,axis=1)])
        return jnp.sqrt(jnp.mean(squared)),left_depth,right_depth

    solution=unconstrained_solution
    h=solution[:,0]; raw_z=solution[:,1]
    unconstrained_rms,unconstrained_left_depth,unconstrained_right_depth=(
        reprojection(solution))
    if curve_convexity=="none":
        z=raw_z; rms=unconstrained_rms
        left_depth=unconstrained_left_depth
        right_depth=unconstrained_right_depth
        constraint_valid=jnp.asarray(True)
    else:
        z,convex=_project_convex_depth_jax(h,raw_z,curve_convexity)
        projected_curve=jnp.stack([h,z],axis=-1)
        rms,left_depth,right_depth=reprojection(projected_curve)
        # 硬凸性仍保留，但不再因相对无约束解的 RMS 增量拒绝整帧。
        constraint_valid=convex&jnp.isfinite(rms)
    valid=(jnp.all(jnp.isfinite(solution))&jnp.all(left_depth>0)&
           jnp.all(right_depth>0)&jnp.isfinite(rms)&constraint_valid&
           jnp.isfinite(unconstrained_rms))
    return h,z,rms,valid


def _smooth_boundary_error_jax(error: Array,smooth_lambda: float,
                               huber_delta: float,iterations: int = 4) -> Array:
    count=error.shape[0]
    if count<3 or smooth_lambda==0:
        return error

    def solve(weights: Array) -> Array:
        # 两个 UV 分量共享同一个五对角矩阵。PCG 的向量算子消除了沿
        # 120 个截面逐点 LDL 的串行依赖，更适合 GPU。
        rhs=weights[:,None]*error
        return _solve_smoothed_boundary_cg_jax(
            weights,rhs,smooth_lambda)

    estimate=solve(jnp.ones((count,),error.dtype))

    def iteration(_,current):
        residual_norm=jnp.linalg.norm(current-error,axis=1)
        weights=jnp.minimum(1.,huber_delta/jnp.maximum(residual_norm,1e-12))
        return solve(weights)

    return jax.lax.fori_loop(0,iterations,iteration,estimate)


def _resize_endpoint_aligned_jax(values: Array,rows: int,columns: int) -> Array:
    """对 BxRxCxK 网格双线性插值，并精确保留四条边界。"""
    source_rows=values.shape[1]; source_columns=values.shape[2]
    row_coordinate=jnp.linspace(
        0.,source_rows-1,rows,dtype=values.dtype)
    row_lower=jnp.floor(row_coordinate).astype(jnp.int32)
    row_upper=jnp.minimum(row_lower+1,source_rows-1)
    row_fraction=row_coordinate-row_lower
    row_values=(
        (1-row_fraction)[None,:,None,None]*values[:,row_lower,:,:]
        +row_fraction[None,:,None,None]*values[:,row_upper,:,:])
    column_coordinate=jnp.linspace(
        0.,source_columns-1,columns,dtype=values.dtype)
    column_lower=jnp.floor(column_coordinate).astype(jnp.int32)
    column_upper=jnp.minimum(column_lower+1,source_columns-1)
    column_fraction=column_coordinate-column_lower
    return (
        (1-column_fraction)[None,None,:,None]*row_values[:,:,column_lower,:]
        +column_fraction[None,None,:,None]*row_values[:,:,column_upper,:])


def resample_surface_batch_jax(
    xyz: Array,
    uv: Array,
    camera_depth: Array,
    *,
    surface_count: int,
    source_rows: int,
    target_rows: int,
    target_columns: int,
) -> tuple[Array,Array,Array,Array]:
    """把整体几何网格插值成独立的光场或高分辨率观测网格。"""
    source_columns=xyz.shape[1]
    xyz_batch=xyz.reshape(surface_count,source_rows,source_columns,3)
    uv_batch=uv.reshape(surface_count,source_rows,source_columns,2)
    depth_batch=camera_depth.reshape(
        surface_count,source_rows,source_columns,1)
    resized_xyz=_resize_endpoint_aligned_jax(
        xyz_batch,target_rows,target_columns)
    resized_uv=_resize_endpoint_aligned_jax(
        uv_batch,target_rows,target_columns)
    resized_depth=_resize_endpoint_aligned_jax(
        depth_batch,target_rows,target_columns)[...,0]
    surface_s,surface_t=jnp.meshgrid(
        jnp.linspace(0.,1.,target_rows,dtype=xyz.dtype),
        jnp.linspace(0.,1.,target_columns,dtype=xyz.dtype),indexing="ij")
    st_one=jnp.stack([surface_s,surface_t],axis=-1)
    resized_st=jnp.broadcast_to(
        st_one,(surface_count,target_rows,target_columns,2))
    flattened_rows=surface_count*target_rows
    return (
        resized_xyz.reshape(flattened_rows,target_columns,3),
        resized_uv.reshape(flattened_rows,target_columns,2),
        resized_st.reshape(flattened_rows,target_columns,2),
        resized_depth.reshape(flattened_rows,target_columns),
    )


def _distort_pixels_jax(points: Array,camera_matrix: Array,distortion: Array) -> Array:
    """OpenCV 五参数径向/切向畸变模型。"""
    x=(points[...,0]-camera_matrix[0,2])/camera_matrix[0,0]
    y=(points[...,1]-camera_matrix[1,2])/camera_matrix[1,1]
    k1,k2,p1,p2,k3=distortion[:5]
    radius2=x*x+y*y
    radial=1+k1*radius2+k2*radius2**2+k3*radius2**3
    distorted_x=x*radial+2*p1*x*y+p2*(radius2+2*x*x)
    distorted_y=y*radial+p1*(radius2+2*y*y)+2*p2*x*y
    return jnp.stack([camera_matrix[0,0]*distorted_x+camera_matrix[0,2],
                      camera_matrix[1,1]*distorted_y+camera_matrix[1,2]],axis=-1)


def build_surface_grid_jax(h: Array,z: Array,left_uv: Array,right_uv: Array,
                           camera_matrix: Array,distortion: Array,rotation: Array,
                           s1: float,s2: float,tx: float,n_fill: int,
                           boundary_smooth_lambda: float,
                           boundary_huber_delta: float,
                           return_boundary_errors: bool = False,
                           ) -> tuple[Array,...]:
    """只用当前帧边界在 GPU 上补全截面、修正 UV 并应用畸变。"""
    alpha=jnp.linspace(0.,1.,n_fill+2,dtype=h.dtype)
    left_xyz=jnp.stack([jnp.full_like(h,s2),h,z],axis=-1)
    right_xyz=jnp.stack([jnp.full_like(h,s1),h,z],axis=-1)
    xyz_grid=(1-alpha)[None,:,None]*left_xyz[:,None,:]+alpha[None,:,None]*right_xyz[:,None,:]
    projected,depth=_project_world_points_jax(
        xyz_grid.reshape(-1,3),camera_matrix,rotation,tx)
    projected=projected.reshape(h.shape[0],n_fill+2,2)
    left_observed_error=left_uv-projected[:,0]
    right_observed_error=right_uv-projected[:,-1]
    left_spatial_error=_smooth_boundary_error_jax(
        left_observed_error,boundary_smooth_lambda,boundary_huber_delta)
    right_spatial_error=_smooth_boundary_error_jax(
        right_observed_error,boundary_smooth_lambda,boundary_huber_delta)
    left_error=left_spatial_error
    right_error=right_spatial_error
    correction=(1-alpha)[None,:,None]*left_error[:,None,:]+alpha[None,:,None]*right_error[:,None,:]
    undistorted=projected+correction
    distorted=_distort_pixels_jax(undistorted,camera_matrix,distortion)
    surface_s,surface_t=jnp.meshgrid(
        jnp.linspace(0.,1.,h.shape[0],dtype=h.dtype),alpha,indexing="ij")
    st=jnp.stack([surface_s,surface_t],axis=-1)
    valid=(jnp.all(depth>0)&jnp.all(jnp.isfinite(xyz_grid))&
           jnp.all(jnp.isfinite(distorted)))
    result=(xyz_grid,distorted,st,
            depth.reshape(h.shape[0],n_fill+2),valid)
    if return_boundary_errors:
        return (*result,left_error,right_error)
    return result


def reconstruct_surface_batch_jax(left_curves: Array,right_dense_curves: Array,
                                  camera_matrix: Array,distortion: Array,rotation: Array,
                                  s1: float,s2: float,tx: float,n_fill: int,
                                  boundary_smooth_lambda: float,
                                  boundary_huber_delta: float,
                                  inverse_camera: Array | None = None,
                                  curve_convexity: str = "none",
                                  ) -> tuple[Array,Array,Array,Array,Array,Array]:
    """对每个表面仅用当前帧左右边界批量重建。"""
    if inverse_camera is None:
        inverse_camera=jnp.linalg.inv(camera_matrix)

    def reconstruct_one(left_uv,right_dense_uv):
        indices,matching_valid=monotone_right_matches_jax(
            left_uv,right_dense_uv,inverse_camera,rotation,s1,s2,tx)
        right_uv=right_dense_uv[jnp.maximum(indices,0)]
        h,z,rms,solve_valid=solve_smoothed_shared_curve_jax(
            left_uv,right_uv,camera_matrix,rotation,s1,s2,tx,1.,
            curve_convexity)
        xyz,uv,st,depth,grid_valid=build_surface_grid_jax(
            h,z,left_uv,right_uv,camera_matrix,distortion,rotation,
            s1,s2,tx,n_fill,boundary_smooth_lambda,boundary_huber_delta)
        return xyz,uv,st,depth,rms,matching_valid&solve_valid&grid_valid

    xyz,uv,st,depth,rms,valid=jax.vmap(reconstruct_one)(
        left_curves,right_dense_curves)
    rows=xyz.shape[0]*xyz.shape[1]
    return (xyz.reshape(rows,xyz.shape[2],3),
            uv.reshape(rows,uv.shape[2],2),
            st.reshape(rows,st.shape[2],2),
            depth.reshape(rows,depth.shape[2]),rms,valid)


def reconstruct_surface_from_masks_jax(
    raw_masks: Array,
    camera_matrix: Array,
    distortion: Array,
    inverse_camera: Array,
    rotation: Array,
    s1: float,
    s2: float,
    tx: float,
    sample_count: int,
    center_band_d: float,
    n_fill: int,
    boundary_smooth_lambda: float,
    boundary_huber_delta: float,
    curve_convexity: str = "none",
    *,
    close_kernel: int = 0,
    open_kernel: int = 0,
    blur_kernel: int = 0,
) -> tuple[Array,Array,Array,Array,Array,Array,Array]:
    """按实时路径从设备端 SAM mask 一次完成整体规则曲面重建。

    ``render_lightfield`` 和 ``get_surface`` 共用这一组合入口，避免两者在
    mask 后处理、边缘生成、动态规划配对或共享曲线求解上再次分叉。
    """
    refined,left_curves,right_dense_curves,edge_valid=(
        prepare_edge_curves_from_masks_jax(
            raw_masks,camera_matrix,distortion,sample_count,center_band_d,
            close_kernel=close_kernel,open_kernel=open_kernel,
            blur_kernel=blur_kernel))
    xyz,uv,st,depth,rms,reconstruction_valid=reconstruct_surface_batch_jax(
        left_curves,right_dense_curves,camera_matrix,distortion,rotation,
        s1,s2,tx,n_fill,boundary_smooth_lambda,boundary_huber_delta,
        inverse_camera,curve_convexity)
    return (refined,xyz,uv,st,depth,rms,
            reconstruction_valid&edge_valid)
