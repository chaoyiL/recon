"""实时局部形变重建的 JAX/GPU 频域预条件与矩阵自由求解核心。"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

Array=jax.Array


def classify_trusted_no_contact_jax(
    color_residual: Array,
    valid_mask: Array,
    baseline_center: Array,
    baseline_channel_scale: Array,
    baseline_valid_mask: Array,
    *,
    trusted_score_threshold: float,
    contact_guard_score_threshold: float,
    contact_guard_radius_pixels: int = 0,
    surface_edge_margin_pixels: int = 0,
) -> tuple[Array,Array]:
    """JAX 版保守可信无接触分类。"""
    colors=jnp.asarray(color_residual,jnp.float32)
    input_valid=jnp.asarray(valid_mask,jnp.bool_)
    center=jnp.asarray(baseline_center,jnp.float32)
    scale=jnp.asarray(baseline_channel_scale,jnp.float32)
    score=jnp.sqrt(jnp.sum(((colors-center)/scale)**2,axis=-1))
    input_valid=input_valid&jnp.isfinite(score)
    usable=input_valid&jnp.asarray(baseline_valid_mask,jnp.bool_)
    if surface_edge_margin_pixels>0:
        radius=surface_edge_margin_pixels
        size=2*radius+1
        padded=jnp.pad(usable,((radius,radius),(radius,radius)),
                       constant_values=False)
        usable=jax.lax.reduce_window(
            padded,jnp.asarray(True),jax.lax.bitwise_and,
            (size,size),(1,1),"VALID")
    guarded=input_valid&(score>=contact_guard_score_threshold)
    if contact_guard_radius_pixels>0:
        radius=contact_guard_radius_pixels
        size=2*radius+1
        guarded=jax.lax.reduce_window(
            guarded,jnp.asarray(False),jax.lax.bitwise_or,
            (size,size),(1,1),"SAME")
    trusted=(usable&(score<=trusted_score_threshold)&~guarded)
    return trusted,jnp.where(input_valid,score,0)


def _dst1_jax(values: Array,axis: int) -> Array:
    """未归一化 DST-I；用于规则网格零 Dirichlet Poisson 求解。"""
    count=values.shape[axis]
    zero_shape=list(values.shape)
    zero_shape[axis]=1
    zero=jnp.zeros(zero_shape,values.dtype)
    extension=jnp.concatenate(
        [zero,values,zero,-jnp.flip(values,axis=axis)],axis=axis)
    transformed=-jnp.fft.fft(extension,axis=axis).imag
    selection=[slice(None)]*values.ndim
    selection[axis]=slice(1,count+1)
    return transformed[tuple(selection)]


def _poisson_integrate_gradient_jax(
    coordinate_gradient: Array,
    valid_mask: Array,
) -> Array:
    """以一次频域 Poisson 解积分规则网格上的坐标梯度。"""
    gradient=jnp.asarray(coordinate_gradient,jnp.float32)
    valid=jnp.asarray(valid_mask,jnp.bool_)
    # 曲面几何的中心差分在最外圈无效；将最近的内部梯度外推到边界，
    # 避免零填充把真实接触形变错误地拉平。
    gradient=gradient.at[0].set(gradient[1])
    gradient=gradient.at[-1].set(gradient[-2])
    gradient=gradient.at[:,0].set(gradient[:,1])
    gradient=gradient.at[:,-1].set(gradient[:,-2])
    horizontal_valid=valid[:,:-1]&valid[:,1:]
    vertical_valid=valid[:-1]&valid[1:]
    horizontal=jnp.where(
        horizontal_valid,.5*(gradient[:,:-1,0]+gradient[:,1:,0]),0)
    vertical=jnp.where(
        vertical_valid,.5*(gradient[:-1,:,1]+gradient[1:,:,1]),0)
    # 对 min ||D d-g||² 的零 Dirichlet 正规方程：
    # (4I-neighbours)d = Dᵀg。
    rhs=(horizontal[1:-1,:-1]-horizontal[1:-1,1:]
         +vertical[:-1,1:-1]-vertical[1:,1:-1])
    rows,columns=rhs.shape
    row_frequency=jnp.arange(1,rows+1,dtype=rhs.dtype)[:,None]
    column_frequency=jnp.arange(1,columns+1,dtype=rhs.dtype)[None,:]
    eigenvalues=(4
                 -2*jnp.cos(jnp.pi*row_frequency/(rows+1))
                 -2*jnp.cos(jnp.pi*column_frequency/(columns+1)))
    transformed=_dst1_jax(_dst1_jax(rhs,0),1)/eigenvalues
    interior=_dst1_jax(_dst1_jax(transformed,0),1)/(
        4*(rows+1)*(columns+1))
    return jnp.pad(interior,((1,1),(1,1)))


def _spectral_initial_displacement_jax(
    slopes: Array,
    gamma: Array,
    k_matrix: Array,
    valid_mask: Array,
    *,
    iterations: int,
) -> Array:
    """用曲率固定点修正的频域 Poisson 积分生成当前帧初值。"""
    determinant=(k_matrix[...,0,0]*k_matrix[...,1,1]
                 -k_matrix[...,0,1]*k_matrix[...,1,0])
    invertible=jnp.abs(determinant)>1e-8
    safe_determinant=jnp.where(invertible,determinant,1)

    def update(_,displacement):
        corrected=slopes-gamma*displacement[...,None]
        coordinate_gradient=jnp.stack([
            (k_matrix[...,1,1]*corrected[...,0]
             -k_matrix[...,0,1]*corrected[...,1])/safe_determinant,
            (-k_matrix[...,1,0]*corrected[...,0]
             +k_matrix[...,0,0]*corrected[...,1])/safe_determinant,
        ],axis=-1)
        coordinate_gradient=jnp.where(
            invertible[...,None],coordinate_gradient,0)
        return _poisson_integrate_gradient_jax(
            coordinate_gradient,valid_mask)

    return jax.lax.fori_loop(
        0,iterations,update,jnp.zeros(slopes.shape[:2],slopes.dtype))


def _solve_shifted_poisson_dst_jax(
    rhs: Array,
    horizontal_scale: Array,
    vertical_scale: Array,
    diagonal_shift: Array,
) -> Array:
    """DST-I 直接求解各向异性零 Dirichlet Poisson 系统。"""
    rows,columns=rhs.shape
    row_frequency=jnp.arange(1,rows+1,dtype=rhs.dtype)[:,None]
    column_frequency=jnp.arange(1,columns+1,dtype=rhs.dtype)[None,:]
    eigenvalues=(diagonal_shift
                 +2*vertical_scale*(
                     1-jnp.cos(jnp.pi*row_frequency/(rows+1)))
                 +2*horizontal_scale*(
                     1-jnp.cos(jnp.pi*column_frequency/(columns+1))))
    transformed=_dst1_jax(_dst1_jax(rhs,0),1)/jnp.maximum(
        eigenvalues,jnp.asarray(1e-12,rhs.dtype))
    return _dst1_jax(_dst1_jax(transformed,0),1)/(
        4*(rows+1)*(columns+1))


def matrix_free_spectral_pcg_jax(
    matvec: Callable[[Array],Array],
    rmatvec: Callable[[Array],Array],
    target: Array,
    initial: Array,
    active_mask: Array,
    horizontal_scale: Array,
    vertical_scale: Array,
    diagonal_shift: Array,
    *,
    atol: float,
    btol: float,
    max_iterations: int,
) -> tuple[Array,Array,Array,Array]:
    """以四个奇偶子格频域 Poisson 逆作预条件的正规方程 PCG。"""
    rhs=rmatvec(target)

    def normal(values):
        return rmatvec(matvec(values))

    def precondition(values):
        result=jnp.zeros_like(values)
        # 中心差分的 AᵀA 分成四个步长为 2 的奇偶子格。分别做 DST
        # Poisson 直接解，能一次消除 LSMR 最慢的全局低频模态。
        for row_parity in (0,1):
            for column_parity in (0,1):
                selection=(slice(1+row_parity,-1,2),
                           slice(1+column_parity,-1,2))
                solved=_solve_shifted_poisson_dst_jax(
                    values[selection],horizontal_scale,vertical_scale,
                    diagonal_shift)
                result=result.at[selection].set(solved)
        return jnp.where(active_mask,result,0)

    norm_rhs=jnp.linalg.norm(rhs)
    zero_target=norm_rhs==0
    solution=jnp.where(
        zero_target,jnp.zeros_like(initial),jnp.asarray(initial,target.dtype))
    residual=rhs-normal(solution)
    preconditioned=precondition(residual)
    direction=preconditioned
    rz=jnp.sum(residual*preconditioned)
    tolerance=jnp.maximum(
        jnp.asarray(atol,target.dtype),
        jnp.asarray(btol,target.dtype)*norm_rhs)
    epsilon=jnp.asarray(1e-20,target.dtype)
    iteration=jnp.asarray(0,jnp.int32)

    def condition(state):
        _,current_residual,_,current_rz,current_iteration=state
        return ((current_iteration<max_iterations)
                &(jnp.linalg.norm(current_residual)>tolerance)
                &jnp.isfinite(current_rz)&(current_rz>epsilon))

    def advance(state):
        current,current_residual,current_direction,current_rz,current_iteration=state
        product=normal(current_direction)
        denominator=jnp.sum(current_direction*product)
        alpha=current_rz/jnp.maximum(denominator,epsilon)
        current=current+alpha*current_direction
        current_residual=current_residual-alpha*product
        current_preconditioned=precondition(current_residual)
        next_rz=jnp.sum(current_residual*current_preconditioned)
        beta=next_rz/jnp.maximum(current_rz,epsilon)
        current_direction=current_preconditioned+beta*current_direction
        return (current,current_residual,current_direction,next_rz,
                current_iteration+1)

    solution,residual,_,rz,iteration=jax.lax.while_loop(
        condition,advance,(solution,residual,direction,rz,iteration))
    converged=jnp.linalg.norm(residual)<=tolerance
    istop=jnp.where(converged,1,7).astype(jnp.int32)
    primal_residual=jnp.linalg.norm(target-matvec(solution))
    solution=jnp.where(zero_target,jnp.zeros_like(solution),solution)
    return solution,istop,iteration,primal_residual


def lookup_slopes_jax(
    color_residual: Array,
    slopes_lut: Array,
    variances_lut: Array,
    color_min: Array,
    color_max: Array,
    sigma_ref2: float,
    valid_mask: Array,
    *,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    trusted_no_contact_mask: Array | None = None,
    trusted_no_contact_confidence: float = 1.,
) -> tuple[Array,Array,Array]:
    """对完整 RGB LUT 作三线性查询，并在零色差附近平滑衰减坡度。"""
    colors=jnp.asarray(color_residual,jnp.float32)
    valid=(jnp.asarray(valid_mask,jnp.bool_)
           &jnp.all(jnp.isfinite(colors),axis=-1))
    trusted=(jnp.zeros(valid.shape,jnp.bool_)
             if trusted_no_contact_mask is None else
             jnp.asarray(trusted_no_contact_mask,jnp.bool_)&valid)
    size=slopes_lut.shape[0]
    normalized=jnp.clip((colors-color_min)/(color_max-color_min),0,1)
    coordinate=normalized*(size-1)
    lower=jnp.floor(coordinate).astype(jnp.int32)
    upper=jnp.minimum(lower+1,size-1)
    fraction=coordinate-lower
    slopes=jnp.zeros((*colors.shape[:-1],2),jnp.float32)
    corners=[]
    for red in (0,1):
        ir=jnp.where(red,upper[...,0],lower[...,0])
        wr=jnp.where(red,fraction[...,0],1-fraction[...,0])
        for green in (0,1):
            ig=jnp.where(green,upper[...,1],lower[...,1])
            wg=jnp.where(green,fraction[...,1],1-fraction[...,1])
            for blue in (0,1):
                ib=jnp.where(blue,upper[...,2],lower[...,2])
                wb=jnp.where(blue,fraction[...,2],1-fraction[...,2])
                weight=wr*wg*wb
                value=slopes_lut[ir,ig,ib]
                node_variance=variances_lut[ir,ig,ib]
                corners.append((weight,value,node_variance))
                slopes=slopes+weight[...,None]*value
    variance=jnp.zeros(colors.shape[:-1],jnp.float32)
    for weight,value,node_variance in corners:
        variance=variance+weight*(
            node_variance+jnp.sum((value-slopes)**2,axis=-1))
    confidence=1/(1+variance/jnp.asarray(sigma_ref2,jnp.float32))
    if zero_color_outer_radius>0:
        intensity=jnp.max(jnp.abs(colors),axis=-1)
        transition=jnp.clip(
            (intensity-zero_color_inner_radius)/
            (zero_color_outer_radius-zero_color_inner_radius),0,1)
        attenuation=transition**2*(3-2*transition)
        slopes=slopes*attenuation[...,None]
    slopes=jnp.where(trusted[...,None],0,slopes)
    confidence=jnp.where(
        trusted,jnp.maximum(confidence,trusted_no_contact_confidence),confidence)
    return (
        jnp.where(valid[...,None],slopes,0),
        jnp.where(valid,variance,0),
        jnp.where(valid,confidence,0),
    )


def _surface_geometry_jax(
    xyz: Array,valid_mask: Array,
) -> tuple[Array,...]:
    points=jnp.asarray(xyz,jnp.float32)
    valid=jnp.asarray(valid_mask,jnp.bool_)&jnp.all(jnp.isfinite(points),axis=-1)
    # lax.reduce_window 的 SAME padding 会使用 min 的单位元 1，导致外圈被
    # 错误地当作内部未知量；显式补零才能与零 Dirichlet 边界一致。
    padded_valid=jnp.pad(valid.astype(jnp.uint8),((1,1),(1,1)))
    interior=jax.lax.reduce_window(
        padded_valid,jnp.uint8(1),jax.lax.min,
        (3,3),(1,1),"VALID")>0
    xu=jnp.zeros_like(points).at[1:-1,1:-1].set(
        (points[1:-1,2:]-points[1:-1,:-2])/2)
    xv=jnp.zeros_like(points).at[1:-1,1:-1].set(
        (points[2:,1:-1]-points[:-2,1:-1])/2)
    xuu=jnp.zeros_like(points).at[1:-1,1:-1].set(
        points[1:-1,2:]-2*points[1:-1,1:-1]+points[1:-1,:-2])
    xvv=jnp.zeros_like(points).at[1:-1,1:-1].set(
        points[2:,1:-1]-2*points[1:-1,1:-1]+points[:-2,1:-1])
    xuv=jnp.zeros_like(points).at[1:-1,1:-1].set(
        (points[2:,2:]-points[2:,:-2]
         -points[:-2,2:]+points[:-2,:-2])/4)
    xu_norm=jnp.linalg.norm(xu,axis=-1)
    cross=jnp.cross(xu,xv); cross_norm=jnp.linalg.norm(cross,axis=-1)
    nonsingular=interior&(xu_norm>1e-9)&(cross_norm>1e-12)
    e1=xu/jnp.maximum(xu_norm[...,None],1e-12)
    normal=cross/jnp.maximum(cross_norm[...,None],1e-12)
    e2=jnp.cross(normal,e1)
    reverse=jnp.sum(e2*xv,axis=-1)<0
    e2=jnp.where(reverse[...,None],-e2,e2)
    normal=jnp.where(reverse[...,None],-normal,normal)
    g00=jnp.sum(xu*xu,axis=-1)
    g01=jnp.sum(xu*xv,axis=-1)
    g11=jnp.sum(xv*xv,axis=-1)
    determinant=g00*g11-g01*g01
    nonsingular=nonsingular&(determinant>1e-14)
    safe=jnp.maximum(determinant,1e-14)
    metric_inverse=jnp.stack([
        jnp.stack([g11/safe,-g01/safe],axis=-1),
        jnp.stack([-g01/safe,g00/safe],axis=-1)],axis=-2)
    second=jnp.stack([
        jnp.stack([jnp.sum(normal*xuu,axis=-1),
                   jnp.sum(normal*xuv,axis=-1)],axis=-1),
        jnp.stack([jnp.sum(normal*xuv,axis=-1),
                   jnp.sum(normal*xvv,axis=-1)],axis=-1)],axis=-2)
    jacobian=jnp.stack([xu,xv],axis=-1)
    frame=jnp.stack([e1,e2],axis=-1)
    k_matrix=jnp.einsum(
        "...ca,...cb,...bd->...ad",frame,jacobian,metric_inverse)
    return (valid,nonsingular,interior,normal,frame,jacobian,
            metric_inverse,second,k_matrix,jnp.sqrt(jnp.maximum(determinant,0)))


def _sym_ortho_jax(a: Array,b: Array) -> tuple[Array,Array,Array]:
    """数值稳定的对称 Givens 旋转。"""
    def b_zero(_):
        return jnp.sign(a),jnp.zeros_like(a),jnp.abs(a)

    def b_nonzero(_):
        def a_zero(__):
            return jnp.zeros_like(a),jnp.sign(b),jnp.abs(b)

        def both_nonzero(__):
            def b_larger(___):
                tau=a/b
                sine=jnp.sign(b)/jnp.sqrt(1+tau*tau)
                return sine*tau,sine,b/sine

            def a_larger(___):
                tau=b/a
                cosine=jnp.sign(a)/jnp.sqrt(1+tau*tau)
                return cosine,cosine*tau,a/cosine

            return jax.lax.cond(
                jnp.abs(b)>jnp.abs(a),b_larger,a_larger,operand=None)

        return jax.lax.cond(a==0,a_zero,both_nonzero,operand=None)

    return jax.lax.cond(b==0,b_zero,b_nonzero,operand=None)


class _LsmrState(NamedTuple):
    u: Array
    v: Array
    x: Array
    alpha: Array
    beta: Array
    iteration: Array
    zetabar: Array
    alphabar: Array
    rho: Array
    rhobar: Array
    cbar: Array
    sbar: Array
    h: Array
    hbar: Array
    betadd: Array
    betad: Array
    rhodold: Array
    tautildeold: Array
    thetatilde: Array
    zeta: Array
    d: Array
    norm_a2: Array
    max_rbar: Array
    min_rbar: Array
    norm_a: Array
    condition_a: Array
    norm_x: Array
    norm_r: Array
    norm_ar: Array
    istop: Array


def matrix_free_lsmr_jax(
    matvec: Callable[[Array],Array],
    rmatvec: Callable[[Array],Array],
    target: Array,
    initial: Array,
    *,
    atol: float,
    btol: float,
    max_iterations: int,
    condition_limit: float = 1e8,
) -> tuple[Array,Array,Array,Array]:
    """JAX 版 LSMR；只依赖 A(x)/Aᵀ(y)，支持设备端热启动。"""
    x=jnp.asarray(initial,target.dtype)
    norm_b=jnp.linalg.norm(target)
    u=target-matvec(x)
    beta=jnp.linalg.norm(u)
    u=jnp.where(beta>0,u/beta,jnp.zeros_like(u))
    v=rmatvec(u)
    alpha=jnp.linalg.norm(v)
    v=jnp.where(alpha>0,v/alpha,jnp.zeros_like(v))
    one=jnp.asarray(1.,target.dtype)
    zero=jnp.asarray(0.,target.dtype)
    state=_LsmrState(
        u=u,v=v,x=x,alpha=alpha,beta=beta,
        iteration=jnp.asarray(0,jnp.int32),zetabar=alpha*beta,
        alphabar=alpha,rho=one,rhobar=one,cbar=one,sbar=zero,
        h=v,hbar=jnp.zeros_like(x),betadd=beta,betad=zero,
        rhodold=one,tautildeold=zero,thetatilde=zero,zeta=zero,d=zero,
        norm_a2=alpha*alpha,max_rbar=zero,
        min_rbar=jnp.asarray(1e30,target.dtype),norm_a=alpha,
        condition_a=one,norm_x=zero,norm_r=beta,norm_ar=alpha*beta,
        istop=jnp.asarray(0,jnp.int32))
    ctol=(0. if condition_limit<=0 else 1/condition_limit)
    epsilon=jnp.finfo(target.dtype).eps

    def condition(current: _LsmrState) -> Array:
        return ((current.iteration<max_iterations)&(current.istop==0)
                &(norm_b>0)&(current.norm_ar>0))

    def iteration(current: _LsmrState) -> _LsmrState:
        number=current.iteration+1
        raw_u=matvec(current.v)-current.alpha*current.u
        beta_new=jnp.linalg.norm(raw_u)
        u_new=jnp.where(beta_new>0,raw_u/beta_new,current.u)
        raw_v=rmatvec(u_new)-beta_new*current.v
        alpha_new=jnp.linalg.norm(raw_v)
        v_new=jnp.where(alpha_new>0,raw_v/alpha_new,current.v)

        chat,shat,alphahat=_sym_ortho_jax(current.alphabar,zero)
        rho_old=current.rho
        cosine,sine,rho_new=_sym_ortho_jax(alphahat,beta_new)
        theta_new=sine*alpha_new
        alphabar_new=cosine*alpha_new
        rhobar_old=current.rhobar
        zeta_old=current.zeta
        theta_bar=current.sbar*rho_new
        rho_temp=current.cbar*rho_new
        cbar_new,sbar_new,rhobar_new=_sym_ortho_jax(
            current.cbar*rho_new,theta_new)
        zeta_new=cbar_new*current.zetabar
        zetabar_new=-sbar_new*current.zetabar
        hbar_new=(current.hbar*(-theta_bar*rho_new/(rho_old*rhobar_old))
                  +current.h)
        x_new=current.x+(zeta_new/(rho_new*rhobar_new))*hbar_new
        h_new=current.h*(-theta_new/rho_new)+v_new

        beta_acute=chat*current.betadd
        beta_check=-shat*current.betadd
        beta_hat=cosine*beta_acute
        betadd_new=-sine*beta_acute
        theta_tilde_old=current.thetatilde
        ctilde_old,stilde_old,rhotilde_old=_sym_ortho_jax(
            current.rhodold,theta_bar)
        theta_tilde_new=stilde_old*rhobar_new
        rhodold_new=ctilde_old*rhobar_new
        betad_new=-stilde_old*current.betad+ctilde_old*beta_hat
        tautilde_old_new=(zeta_old-theta_tilde_old*current.tautildeold)/rhotilde_old
        tau_d=(zeta_new-theta_tilde_new*tautilde_old_new)/rhodold_new
        d_new=current.d+beta_check*beta_check
        norm_r_new=jnp.sqrt(
            d_new+(betad_new-tau_d)**2+betadd_new*betadd_new)
        norm_a2_mid=current.norm_a2+beta_new*beta_new
        norm_a_new=jnp.sqrt(norm_a2_mid)
        norm_a2_new=norm_a2_mid+alpha_new*alpha_new
        max_rbar_new=jnp.maximum(current.max_rbar,rhobar_old)
        min_rbar_new=jnp.where(
            number>1,jnp.minimum(current.min_rbar,rhobar_old),current.min_rbar)
        condition_a_new=(
            jnp.maximum(max_rbar_new,rho_temp)
            /jnp.maximum(jnp.minimum(min_rbar_new,rho_temp),epsilon))
        norm_ar_new=jnp.abs(zetabar_new)
        norm_x_new=jnp.linalg.norm(x_new)
        test1=norm_r_new/jnp.maximum(norm_b,epsilon)
        test2=jnp.where(
            norm_a_new*norm_r_new>0,
            norm_ar_new/(norm_a_new*norm_r_new),jnp.inf)
        test3=1/jnp.maximum(condition_a_new,epsilon)
        scaled=test1/(1+norm_a_new*norm_x_new/jnp.maximum(norm_b,epsilon))
        relative_tolerance=(
            btol+atol*norm_a_new*norm_x_new/jnp.maximum(norm_b,epsilon))
        stop=jnp.asarray(0,jnp.int32)
        stop=jnp.where(number>=max_iterations,7,stop)
        stop=jnp.where(1+test3<=1,6,stop)
        stop=jnp.where(1+test2<=1,5,stop)
        stop=jnp.where(1+scaled<=1,4,stop)
        stop=jnp.where(test3<=ctol,3,stop)
        stop=jnp.where(test2<=atol,2,stop)
        stop=jnp.where(test1<=relative_tolerance,1,stop)
        return _LsmrState(
            u_new,v_new,x_new,alpha_new,beta_new,number,zetabar_new,
            alphabar_new,rho_new,rhobar_new,cbar_new,sbar_new,h_new,hbar_new,
            betadd_new,betad_new,rhodold_new,tautilde_old_new,
            theta_tilde_new,zeta_new,d_new,norm_a2_new,max_rbar_new,
            min_rbar_new,norm_a_new,condition_a_new,norm_x_new,norm_r_new,
            norm_ar_new,stop)

    result=jax.lax.while_loop(condition,iteration,state)
    solution=jnp.where(norm_b>0,result.x,jnp.zeros_like(result.x))
    return solution,result.istop,result.iteration,result.norm_r


def reconstruct_local_surface_jax(
    xyz: Array,
    color_residual_grid: Array,
    slopes_lut: Array,
    variances_lut: Array,
    color_min: Array,
    color_max: Array,
    sigma_ref2: float,
    valid_mask: Array,
    previous_displacement: Array,
    *,
    zero_color_inner_radius: float = 0.,
    zero_color_outer_radius: float = 0.,
    trusted_no_contact_mask: Array | None = None,
    trusted_no_contact_confidence: float = 1.,
    displacement_zero_lambda_per_mm2: float = 0.,
    lsmr_atol: float = 1e-4,
    lsmr_btol: float = 1e-4,
    lsmr_max_iterations: int | None = None,
    spectral_initialization_iterations: int = 0,
    linear_solver: str = "lsmr",
) -> tuple[Array,...]:
    """频域冷启动/上一帧优选后，以 GPU LSMR 精修法向位移。"""
    points=jnp.asarray(xyz,jnp.float32)
    colors=jnp.asarray(color_residual_grid,jnp.float32)
    valid=(jnp.asarray(valid_mask,jnp.bool_)
           &jnp.all(jnp.isfinite(points),axis=-1)
           &jnp.all(jnp.isfinite(colors),axis=-1))
    trusted_no_contact=(jnp.zeros(valid.shape,jnp.bool_)
                        if trusted_no_contact_mask is None else
                        jnp.asarray(trusted_no_contact_mask,jnp.bool_)&valid)
    slopes,_,confidence=lookup_slopes_jax(
        colors,slopes_lut,variances_lut,color_min,color_max,sigma_ref2,
        valid,
        zero_color_inner_radius=zero_color_inner_radius,
        zero_color_outer_radius=zero_color_outer_radius,
        trusted_no_contact_mask=trusted_no_contact,
        trusted_no_contact_confidence=trusted_no_contact_confidence)
    (valid,geometry_valid,interior,normal,frame,jacobian,metric_inverse,
     second,k_matrix,area)=_surface_geometry_jax(points,valid)
    tangent=jnp.einsum("...ca,...a->...c",frame,slopes)
    parameter=jnp.einsum(
        "...ab,...cb,...c->...a",metric_inverse,jacobian,tangent)
    shape_parameter=jnp.einsum("...ab,...b->...a",second,parameter)
    curvature=jnp.einsum(
        "...ca,...ab,...b->...c",jacobian,metric_inverse,shape_parameter)
    gamma=jnp.einsum("...ca,...c->...a",frame,curvature)
    equation_mask=geometry_valid&(confidence>0)
    active_mask=interior
    equation_weight=jnp.sqrt(jnp.maximum(area*confidence,0))*equation_mask
    zero_displacement_weight=(
        jnp.sqrt(jnp.maximum(
            area*displacement_zero_lambda_per_mm2,0))
        *trusted_no_contact*active_mask)
    equation_count=jnp.maximum(jnp.sum(equation_mask),1)
    closure_weight=jnp.sqrt(
        jnp.sum(equation_weight*equation_weight)/equation_count)
    closure_coefficients=[]
    for parity_i in (0,1):
        for parity_j in (0,1):
            row_indices=[
                index for index in range(1,points.shape[0]-1)
                if index%2==parity_i]
            column_indices=[
                index for index in range(1,points.shape[1]-1)
                if index%2==parity_j]
            coefficient=jnp.zeros(points.shape[:2],points.dtype)
            if len(row_indices)>=2 and len(column_indices)>=2:
                i0,i1=row_indices[:2]; j0,j1=column_indices[:2]
                wi=((i1)/(i1-i0),(-i0)/(i1-i0))
                wj=((j1)/(j1-j0),(-j0)/(j1-j0))
                for ai,i in enumerate((i0,i1)):
                    for aj,j in enumerate((j0,j1)):
                        coefficient=coefficient.at[i,j].set(wi[ai]*wj[aj])
            elif row_indices and column_indices:
                coefficient=coefficient.at[
                    row_indices[0],column_indices[0]].set(1)
            closure_coefficients.append(coefficient)
    closure_coefficients=jnp.stack(closure_coefficients)

    def neighbors(values: Array) -> tuple[Array,Array,Array,Array]:
        right=jnp.pad(values[:,1:],((0,0),(0,1)))
        left=jnp.pad(values[:,:-1],((0,0),(1,0)))
        down=jnp.pad(values[1:,:],((0,1),(0,0)))
        up=jnp.pad(values[:-1,:],((1,0),(0,0)))
        return right,left,down,up

    def matvec(displacement: Array) -> Array:
        active=jnp.where(active_mask,displacement,0)
        right,left,down,up=neighbors(active)
        gradient=(
            gamma*active[...,None]
            +.5*k_matrix[...,0]*(right-left)[...,None]
            +.5*k_matrix[...,1]*(down-up)[...,None])
        gradient=equation_weight[...,None]*gradient
        zero_displacement=(zero_displacement_weight*active)[...,None]
        closure_values=closure_weight*jnp.einsum(
            "kij,ij->k",closure_coefficients,active)
        closure_image=jnp.zeros(
            (*displacement.shape,4),displacement.dtype).at[0,0].set(
                closure_values)
        return jnp.concatenate(
            [gradient,zero_displacement,closure_image],axis=-1)

    def rmatvec(values: Array) -> Array:
        weighted=equation_weight[...,None]*values[...,:2]
        center=jnp.sum(weighted*gamma,axis=-1)
        horizontal=.5*jnp.sum(weighted*k_matrix[...,0],axis=-1)
        vertical=.5*jnp.sum(weighted*k_matrix[...,1],axis=-1)
        horizontal_adjoint=(
            jnp.pad(horizontal[:,:-1],((0,0),(1,0)))
            -jnp.pad(horizontal[:,1:],((0,0),(0,1))))
        vertical_adjoint=(
            jnp.pad(vertical[:-1,:],((1,0),(0,0)))
            -jnp.pad(vertical[1:,:],((0,1),(0,0))))
        zero_adjoint=zero_displacement_weight*values[...,2]
        closure_adjoint=closure_weight*jnp.einsum(
            "k,kij->ij",values[0,0,3:7],closure_coefficients)
        return jnp.where(
            active_mask,center+horizontal_adjoint+vertical_adjoint
            +zero_adjoint+closure_adjoint,0)

    target=jnp.concatenate([
        equation_weight[...,None]*slopes,
        jnp.zeros((*slopes.shape[:2],1),slopes.dtype),
        jnp.zeros((*slopes.shape[:2],4),slopes.dtype)],axis=-1)
    previous_initial=jnp.where(
        active_mask,jnp.asarray(previous_displacement,jnp.float32),0)
    if spectral_initialization_iterations>0:
        spectral_initial=_spectral_initial_displacement_jax(
            slopes,gamma,k_matrix,valid,
            iterations=int(spectral_initialization_iterations))
        spectral_initial=jnp.where(active_mask,spectral_initial,0)
        previous_residual=jnp.linalg.norm(target-matvec(previous_initial))
        spectral_residual=jnp.linalg.norm(target-matvec(spectral_initial))
        use_spectral=(jnp.isfinite(spectral_residual)
                      &(spectral_residual<=previous_residual))
        initial=jnp.where(use_spectral,spectral_initial,previous_initial)
    else:
        initial=previous_initial
    maximum=(max(200,4*points.shape[0]*points.shape[1])
             if lsmr_max_iterations is None else int(lsmr_max_iterations))
    if linear_solver=="lsmr":
        displacement,istop,iterations,residual_norm=matrix_free_lsmr_jax(
            matvec,rmatvec,target,initial,atol=lsmr_atol,btol=lsmr_btol,
            max_iterations=maximum)
    elif linear_solver=="spectral_pcg":
        weight_squared=equation_weight*equation_weight
        scale_count=jnp.maximum(jnp.sum(equation_mask),1)
        horizontal_scale=.25*jnp.sum(
            weight_squared*jnp.sum(k_matrix[...,0]**2,axis=-1))/scale_count
        vertical_scale=.25*jnp.sum(
            weight_squared*jnp.sum(k_matrix[...,1]**2,axis=-1))/scale_count
        diagonal_shift=jnp.sum(
            weight_squared*jnp.sum(gamma*gamma,axis=-1)
            +zero_displacement_weight*zero_displacement_weight)/scale_count
        displacement,istop,iterations,residual_norm=(
            matrix_free_spectral_pcg_jax(
                matvec,rmatvec,target,initial,active_mask,
                horizontal_scale,vertical_scale,diagonal_shift,
                atol=lsmr_atol,btol=lsmr_btol,max_iterations=maximum))
    else:
        raise ValueError(f"未知局部线性求解器: {linear_solver}")
    displacement=jnp.where(active_mask,displacement,0)
    displacement_vectors=displacement[...,None]*normal
    observed=(normal-tangent)/jnp.sqrt(
        1+jnp.sum(tangent*tangent,axis=-1,keepdims=True))
    output=points+displacement_vectors
    return (
        output,displacement,displacement_vectors,normal,slopes,observed,
        curvature,confidence,valid,valid&~active_mask,istop,iterations,
        residual_norm,jnp.sum(equation_mask),trusted_no_contact,
    )
