from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from utils.jax_reconstruction import (
    MATERIAL_UPDATE_FAILURE_NAMES,
    _select_material_surface_candidate_jax,
    initialize_material_surface_state_jax,
    material_curve_from_angles_jax,
    update_material_surface_state_jax,
)
from utils.material_surface import (
    MATERIAL_COORDINATE_MODE,
    MaterialSurfaceTemplate,
    canonical_st,
)


def _camera() -> tuple[np.ndarray,np.ndarray]:
    return (
        np.asarray([[120.,0.,160.],[0.,120.,120.],[0.,0.,1.]],np.float32),
        np.zeros(5,np.float32),
    )


def _grid(curve: np.ndarray,columns: int = 4) -> tuple[np.ndarray,np.ndarray]:
    x=np.linspace(-5.,5.,columns,dtype=np.float32)
    xyz=np.stack([
        np.broadcast_to(x[None],(curve.shape[0],columns)),
        np.broadcast_to(curve[:,0,None],(curve.shape[0],columns)),
        np.broadcast_to(curve[:,1,None],(curve.shape[0],columns)),
    ],axis=-1)
    k,_=_camera()
    uv=np.stack([
        k[0,0]*xyz[...,0]/xyz[...,2]+k[0,2],
        k[1,1]*xyz[...,1]/xyz[...,2]+k[1,2],
    ],axis=-1)
    return xyz.astype(np.float32),uv.astype(np.float32)


def _initialize(curve: np.ndarray,lengths: np.ndarray):
    xyz,uv=_grid(curve)
    k,distortion=_camera()
    return initialize_material_surface_state_jax(
        jnp.asarray(xyz),jnp.asarray(uv),jnp.asarray(.1),
        jnp.asarray(lengths),jnp.linspace(-5.,5.,4),jnp.asarray(k),
        jnp.asarray(distortion),jnp.eye(3),0.,
        initial_calibration_maximum_rms_px=2.,
        boundary_smooth_lambda=0.,boundary_huber_delta=1.)


def test_material_template_is_generated_exactly_from_dimensions():
    template=MaterialSurfaceTemplate(
        width_mm=22.,length_mm=55.,rows=12,columns=7,
        s_zero_endpoint="image_top",camera_sha256="a"*64,
        reconstruction_sha256="b"*64)
    same=MaterialSurfaceTemplate(
        width_mm=22.,length_mm=55.,rows=12,columns=7,
        s_zero_endpoint="image_top",camera_sha256="a"*64,
        reconstruction_sha256="b"*64)
    assert template.sha256==same.sha256
    assert np.array_equal(template.st,canonical_st(12,7))
    assert MATERIAL_COORDINATE_MODE=="configured_intrinsic_dimensions_v5"
    assert np.isclose(template.segment_lengths_mm.sum(),55.)
    np.testing.assert_allclose(template.x_coordinates_mm[[0,-1]],[-11.,11.])
    assert np.allclose(
        np.linalg.norm(np.diff(template.reference_curve_yz,axis=0),axis=1),
        template.segment_lengths_mm)


def test_image_bottom_can_be_material_s_zero():
    top=canonical_st(5,3,s_zero_endpoint="image_top")
    bottom=canonical_st(5,3,s_zero_endpoint="image_bottom")
    np.testing.assert_allclose(top[:,0,0],[0.,.25,.5,.75,1.])
    np.testing.assert_allclose(bottom[:,0,0],[1.,.75,.5,.25,0.])


def test_every_observed_silhouette_is_normalized_to_full_template_interval():
    rows=8
    lengths=np.full(rows-1,2.,np.float32)
    reference=np.stack([
        np.arange(rows,dtype=np.float32)*2.,
        np.full(rows,100.,np.float32)],axis=-1)
    state=_initialize(reference,lengths)
    # 这也可能是真实短材料，或是自遮挡后的可见子段；单目 mask 无法区分。
    observed=np.stack([
        np.linspace(3.,9.,rows,dtype=np.float32),
        np.full(rows,101.,np.float32)],axis=-1)
    observed_xyz,observed_uv=_grid(observed)
    k,distortion=_camera()
    updated=update_material_surface_state_jax(
        state,jnp.asarray(observed_xyz),jnp.asarray(observed_uv),jnp.asarray(.2),
        jnp.asarray(True),jnp.asarray(lengths),jnp.linspace(-5.,5.,4),
        jnp.asarray(k),jnp.asarray(distortion),jnp.eye(3),0.,
        match_confidence_scale_mm=2.,rms_confidence_scale_px=3.,
        confidence_floor=1e-6,bend_direction="none",
        boundary_smooth_lambda=0.,boundary_huber_delta=1.)
    actual=np.linalg.norm(np.diff(np.asarray(updated.curve_yz),axis=0),axis=1)
    np.testing.assert_allclose(actual,lengths,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(updated.observable_rows),np.ones(rows,bool))
    assert np.isclose(float(updated.visible_fraction),1.)


def test_current_observation_replaces_previous_shape_without_hysteresis():
    rows=10
    lengths=np.full(rows-1,1.5,np.float32)
    reference_angles=np.linspace(.05,.5,rows-1,dtype=np.float32)
    reference_curve=np.asarray(material_curve_from_angles_jax(
        jnp.asarray([0.,100.]),jnp.asarray(reference_angles),jnp.asarray(lengths)))
    state=_initialize(reference_curve,lengths)
    changed_angles=np.linspace(.7,1.1,rows-1,dtype=np.float32)
    changed_curve=np.asarray(material_curve_from_angles_jax(
        jnp.asarray([2.,101.]),jnp.asarray(changed_angles),jnp.asarray(lengths)))
    changed_xyz,changed_uv=_grid(changed_curve)
    k,distortion=_camera()
    updated=update_material_surface_state_jax(
        state,jnp.asarray(changed_xyz),jnp.asarray(changed_uv),jnp.asarray(.1),
        jnp.asarray(True),jnp.asarray(lengths),jnp.linspace(-5.,5.,4),
        jnp.asarray(k),jnp.asarray(distortion),jnp.eye(3),0.,
        match_confidence_scale_mm=2.,rms_confidence_scale_px=3.,
        confidence_floor=1e-6,bend_direction="none",
        boundary_smooth_lambda=0.,boundary_huber_delta=1.)
    np.testing.assert_allclose(
        np.asarray(updated.curve_yz),changed_curve,rtol=1e-5,atol=1e-5)


_SELECTION_LIMITS={
    "maximum_rms_px":4.,
    "minimum_confidence":.05,
    "maximum_uv_triangle_width_px":64,
    "maximum_uv_triangle_height_px":32,
}


def test_low_confidence_rejects_calibration_but_advances_tracking():
    rows=6
    lengths=np.ones(rows-1,np.float32)
    curve=np.stack([
        np.linspace(0.,5.,rows),np.full(rows,100.)],axis=-1).astype(np.float32)
    state=_initialize(curve,lengths)
    candidate=replace(
        state,curve_yz=state.curve_yz+3.,
        matching_confidence=jnp.asarray(1e-6,jnp.float32))
    selected,tracking_accepted,calibration_accepted,flags=(
        _select_material_surface_candidate_jax(
            state,candidate,observation_ok=jnp.asarray(True),
            **_SELECTION_LIMITS))
    assert bool(tracking_accepted)
    assert not bool(calibration_accepted)
    assert bool(selected.tracking_accepted)
    assert bool(np.asarray(flags)[
        MATERIAL_UPDATE_FAILURE_NAMES.index("confidence_too_low")])
    np.testing.assert_array_equal(
        np.asarray(selected.curve_yz),np.asarray(candidate.curve_yz))


def test_structurally_invalid_online_candidate_is_rolled_back_atomically():
    rows=6
    lengths=np.ones(rows-1,np.float32)
    curve=np.stack([
        np.linspace(0.,5.,rows),np.full(rows,100.)],axis=-1).astype(np.float32)
    previous=_initialize(curve,lengths)
    candidate=replace(
        previous,xyz=jnp.full_like(previous.xyz,jnp.nan),
        uv=jnp.full_like(previous.uv,jnp.nan),
        camera_depth=-jnp.ones_like(previous.camera_depth),
        valid=jnp.asarray(False),tracking_accepted=jnp.asarray(False),
        matching_confidence=jnp.asarray(1e-6,jnp.float32))

    @jax.jit
    def select(old,new):
        return _select_material_surface_candidate_jax(
            old,new,observation_ok=jnp.asarray(True),**_SELECTION_LIMITS)

    selected,tracking_accepted,calibration_accepted,flags=select(
        previous,candidate)
    assert not bool(tracking_accepted)
    assert not bool(calibration_accepted)
    assert not bool(selected.tracking_accepted)
    assert bool(selected.valid)
    np.testing.assert_array_equal(np.asarray(selected.xyz),np.asarray(previous.xyz))
    assert bool(np.asarray(flags)[MATERIAL_UPDATE_FAILURE_NAMES.index("nonfinite")])
    assert bool(np.asarray(flags)[
        MATERIAL_UPDATE_FAILURE_NAMES.index("nonpositive_depth")])


def test_high_rms_rejects_calibration_but_advances_tracking():
    rows=6
    lengths=np.ones(rows-1,np.float32)
    curve=np.stack([
        np.linspace(0.,5.,rows),np.full(rows,100.)],axis=-1).astype(np.float32)
    previous=_initialize(curve,lengths)
    candidate=replace(
        previous,xyz=previous.xyz.at[...,1].add(.25),
        reprojection_rms_px=jnp.asarray(4.01,jnp.float32),
        matching_confidence=jnp.asarray(1.,jnp.float32))
    selected,tracking_accepted,calibration_accepted,flags=(
        _select_material_surface_candidate_jax(
            previous,candidate,observation_ok=jnp.asarray(True),
            **_SELECTION_LIMITS))
    assert bool(tracking_accepted)
    assert not bool(calibration_accepted)
    np.testing.assert_array_equal(np.asarray(selected.xyz),np.asarray(candidate.xyz))
    assert bool(np.asarray(flags)[MATERIAL_UPDATE_FAILURE_NAMES.index("rms_too_high")])


def test_oversized_uv_triangle_is_rejected_before_rasterization():
    rows=6
    lengths=np.ones(rows-1,np.float32)
    curve=np.stack([
        np.linspace(0.,5.,rows),np.full(rows,100.)],axis=-1).astype(np.float32)
    previous=_initialize(curve,lengths)
    huge_uv=np.asarray(previous.uv).copy()
    huge_uv[1,1,0]+=1000.
    candidate=replace(
        previous,uv=jnp.asarray(huge_uv),
        matching_confidence=jnp.asarray(1.,jnp.float32))
    selected,tracking_accepted,calibration_accepted,flags=(
        _select_material_surface_candidate_jax(
            previous,candidate,observation_ok=jnp.asarray(True),
            **_SELECTION_LIMITS))
    assert not bool(tracking_accepted)
    assert not bool(calibration_accepted)
    assert not bool(selected.tracking_accepted)
    assert bool(np.asarray(flags)[
        MATERIAL_UPDATE_FAILURE_NAMES.index("uv_triangle_too_large")])
    np.testing.assert_array_equal(np.asarray(selected.uv),np.asarray(previous.uv))
