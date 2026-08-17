import unittest
import tempfile
import contextlib
import io
from pathlib import Path
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from utils.gpu_residual_fit import (_adaptive_channel_weights,
                                    fit_direct_geometry_conditioned_field_gpu,
                                    fit_residual_correction_model_gpu)
from utils.lightfield import (LightFieldModel, bounded_mixing_matrix,
                              bgr_to_linear_rgb, bgr_to_linear_rgb_jax,
                              bspline_basis, build_canonical_residual_sample_jax,
                              direct_background_field_chunked,
                              direct_background_field_jax,
                              direct_geometry_descriptor_jax,
                              geometry_cache_background_field_jax,
                              erode_mask_jax, evaluate_rgb_bspline,
                              fit_uniform_huber_residual_correction_scores_jax,
                              fit_uniform_residual_correction_scores_jax,
                              fit_residual_m_scores_jax,
                              fit_startup_direct_bsession_model,
                              fit_startup_residual_bsession_model,
                              point_set_to_grid, surface_normals, diffuse_surface_fields,
                              irls_gain_bias, physical_background,
                              rasterize_attributes_jax, rgb_bspline_field,
                              sample_linear_rgb_jax, sample_residual_correction_jax,
                              _interpolate_samples,
                              _direct_light_fields,
                              _surface_diffusion_geometry,
                              _light_source_boundaries, _light_source_inward_directions,
                              _resample_curve, parse_light_source_layout,
                              sample_image_mask_to_canonical_jax,
                              sample_unsaturated_mask)


def make_direct_model(
    session_correction: np.ndarray,*,base_texture: np.ndarray | None = None,
    decoder_bias: np.ndarray | None = None,
) -> LightFieldModel:
    descriptor_rows=4
    descriptor_count=6+10*descriptor_rows
    return LightFieldModel.direct_fit(
        session_correction,
        base_texture=(np.full((7,5,3),.5,np.float32)
                      if base_texture is None else base_texture),
        coordinate_frequencies=np.asarray([1.],np.float32),
        geometry_feature_mean=np.zeros(descriptor_count,np.float32),
        geometry_feature_scale=np.ones(descriptor_count,np.float32),
        geometry_pca_components=np.zeros((descriptor_count,2),np.float32),
        geometry_pca_scale=np.ones(2,np.float32),
        local_geometry_feature_mean=np.zeros(15,np.float32),
        local_geometry_feature_scale=np.ones(15,np.float32),
        geometry_encoder_weights=(
            np.zeros((descriptor_count,2),np.float32),),
        geometry_encoder_biases=(np.zeros(2,np.float32),),
        decoder_weights=(np.zeros((25,3),np.float32),),
        decoder_biases=((np.asarray([-.5,-1.,-1.5],np.float32)
                         if decoder_bias is None else decoder_bias),),
        geometry_descriptor_rows=descriptor_rows)


def make_direct_3_model(
    session_correction: np.ndarray,*,base_texture: np.ndarray | None = None,
    decoder_bias: np.ndarray | None = None,
) -> LightFieldModel:
    descriptor_rows=4
    descriptor_count=6+10*descriptor_rows
    biases=(np.asarray([-.5,-1.,-1.5],np.float32)
            if decoder_bias is None else np.asarray(decoder_bias,np.float32))
    return LightFieldModel.direct_fit_3(
        session_correction,
        base_texture=(np.full((7,5,3),.5,np.float32)
                      if base_texture is None else base_texture),
        coordinate_frequencies=np.asarray([1.],np.float32),
        geometry_feature_mean=np.zeros(descriptor_count,np.float32),
        geometry_feature_scale=np.ones(descriptor_count,np.float32),
        geometry_pca_components=np.zeros((descriptor_count,2),np.float32),
        geometry_pca_scale=np.ones(2,np.float32),
        local_geometry_feature_mean=np.zeros(15,np.float32),
        local_geometry_feature_scale=np.ones(15,np.float32),
        geometry_encoder_weights=(
            np.zeros((descriptor_count,2),np.float32),),
        geometry_encoder_biases=(np.zeros(2,np.float32),),
        channel_decoder_weights=tuple(
            (np.zeros((25,1),np.float32),) for _ in range(3)),
        channel_decoder_biases=tuple(
            (np.asarray([bias],np.float32),) for bias in biases),
        geometry_descriptor_rows=descriptor_rows)

class LightFieldTest(unittest.TestCase):
    def test_adaptive_channel_weights_follow_rmse_and_stop_gradient(self):
        mean_squared_error=jnp.asarray([1.,4.,16.],jnp.float32)
        weights=np.asarray(_adaptive_channel_weights(
            mean_squared_error,.5))
        np.testing.assert_allclose(weights.mean(),1.,atol=1e-6)
        self.assertTrue(weights[0]<weights[1]<weights[2])
        self.assertGreaterEqual(float(weights.min()),.5)
        self.assertLessEqual(float(weights.max()),2.)
        np.testing.assert_allclose(np.asarray(_adaptive_channel_weights(
            mean_squared_error,0.)),np.ones(3),atol=1e-7)
        gradient=jax.grad(lambda value:jnp.sum(
            _adaptive_channel_weights(value,.5)))(mean_squared_error)
        np.testing.assert_array_equal(np.asarray(gradient),np.zeros(3))

    def test_direct_static_base_preserves_native_high_frequency(self):
        rows,columns=8,6
        y,x=np.meshgrid(np.arange(rows),np.arange(columns),indexing="ij")
        checker=((x+y)%2).astype(np.float32)
        base=np.stack([
            .2+.5*checker,.15+.35*(1-checker),.1+.4*checker],axis=-1)
        model=make_direct_model(
            np.zeros((3,4,4),np.float32),base_texture=base,
            decoder_bias=np.zeros(3,np.float32))
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1).astype(np.float32)
        predicted=np.asarray(direct_background_field_jax(
            (rows,columns),xyz,model))
        np.testing.assert_allclose(predicted,base,atol=2e-6)

    def test_direct_background_field_chunked_matches_full_field(self):
        rows,columns=9,7
        y,x=np.meshgrid(np.arange(rows),np.arange(columns),indexing="ij")
        base=np.stack([
            .25+.5*((x+y)%2),.2+.3*((x*y)%2),.15+.4*((x+2*y)%2)
        ],axis=-1).astype(np.float32)
        model=make_direct_3_model(
            np.zeros((3,4,4),np.float32),base_texture=base,
            decoder_bias=np.zeros(3,np.float32))
        xyz=np.stack([
            x.astype(np.float32),y.astype(np.float32),
            np.full_like(x,80,np.float32)],axis=-1)
        full=np.asarray(direct_background_field_jax((rows,columns),xyz,model))
        chunked=direct_background_field_chunked(
            (rows,columns),xyz,model,chunk_size=11)
        np.testing.assert_allclose(chunked,full,atol=1e-6)

    def test_v15_direct_model_is_rejected_as_pre_full_jax_cache(self):
        model=make_direct_model(np.zeros((3,8,4),np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"direct.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            raw["format_version"]=15
            raw.pop("direct_reconstruction_pipeline",None)
            path.write_text(yaml.safe_dump(raw),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"完整实时 JAX"):
                LightFieldModel.load(path)

    def test_v16_direct_model_is_rejected_as_pre_base_delta_split(self):
        model=make_direct_model(np.zeros((3,8,4),np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"direct.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            raw["format_version"]=16
            path.write_text(yaml.safe_dump(raw),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"静态 B 与几何 delta B"):
                LightFieldModel.load(path)

    def test_gpu_canonical_residual_sampling_preserves_saturated_linear_grid(self):
        rows,columns=6,8
        y,x=np.meshgrid(np.arange(rows),np.arange(columns),indexing="ij")
        residual=np.stack([x/20,y/20,(x+y)/40],axis=-1).astype(np.float32)
        # 规范背景需要学习相机实际裁剪平台；即使整帧为 255 也不能删样本。
        frame=np.full((rows,columns,3),255,np.uint8)
        uv=np.asarray([[[0.,0.],[columns-1.,0.]],
                       [[0.,rows-1.],[columns-1.,rows-1.]]],np.float32)
        sampled,valid=build_canonical_residual_sample_jax(
            jnp.asarray(residual),jnp.asarray(frame),
            jnp.ones((rows,columns),bool),jnp.asarray(uv),(rows,columns),
            saturation_threshold=250,erode_pixels=0)
        np.testing.assert_allclose(np.asarray(sampled),residual,atol=1e-6)
        np.testing.assert_array_equal(np.asarray(valid),np.ones((rows,columns),bool))

    def test_canonical_erosion_does_not_expand_internal_saturation(self):
        rows,columns=7,9
        residual=np.ones((rows,columns,3),np.float32)
        frame=np.zeros((rows,columns,3),np.uint8)
        frame[3,4]=255
        uv=np.asarray([[[0.,0.],[columns-1.,0.]],
                       [[0.,rows-1.],[columns-1.,rows-1.]]],np.float32)
        sampled,valid=build_canonical_residual_sample_jax(
            jnp.asarray(residual),jnp.asarray(frame),
            jnp.ones((rows,columns),bool),jnp.asarray(uv),(rows,columns),
            saturation_threshold=1,erode_pixels=1)
        valid=np.asarray(valid)
        self.assertTrue(valid[3,4])
        np.testing.assert_allclose(np.asarray(sampled)[3,4],1.)

    def test_image_exclusion_mask_uses_same_canonical_uv_mapping(self):
        rows,columns=7,9
        source=np.zeros((rows,columns),bool)
        source[2:5,3:7]=True
        uv=np.asarray([[[0.,0.],[columns-1.,0.]],
                       [[0.,rows-1.],[columns-1.,rows-1.]]],np.float32)
        sampled=sample_image_mask_to_canonical_jax(
            jnp.asarray(source),jnp.asarray(uv),(rows,columns))
        np.testing.assert_array_equal(np.asarray(sampled),source)

    def test_bounded_mixing_matrix_is_identifiable_and_diagonally_sharp(self):
        matrix=bounded_mixing_matrix(jnp.asarray([
            [2.,3.,-2.],[-1.,0.,1.],[4.,-3.,-2.]],jnp.float32),.2)
        values=np.asarray(matrix)
        np.testing.assert_allclose(values.sum(axis=1),np.ones(3),atol=1e-6)
        self.assertTrue(np.all(values>=0))
        self.assertTrue(np.all(np.diag(values)>=.8))

    def test_bspline_partition_of_unity(self):
        x = jnp.linspace(0, 1, 101)
        basis = bspline_basis(x, 6)
        np.testing.assert_allclose(np.asarray(basis.sum(1)), np.ones(101), atol=1e-5)
        self.assertTrue(bool((basis >= 0).all()))

    def test_rgb_bspline_field_preserves_constant_coefficients(self):
        coefficients=jnp.ones((3,8,4))*jnp.asarray([.2,-.1,.05])[:,None,None]
        field=rgb_bspline_field((17,9),coefficients)
        expected=np.broadcast_to(np.asarray([.2,-.1,.05]),(17,9,3))
        np.testing.assert_allclose(np.asarray(field),expected,atol=1e-6)
        np.testing.assert_allclose(evaluate_rgb_bspline(np.asarray(coefficients),(17,9)),
                                   expected,atol=1e-6)

    def test_flat_grid_normals_are_consistent(self):
        y, x = jnp.meshgrid(jnp.arange(4.), jnp.arange(5.), indexing="ij")
        xyz = jnp.stack([x, y, jnp.ones_like(x) * 10], -1)
        normals = surface_normals(xyz)
        np.testing.assert_allclose(np.asarray(normals[...,2]), np.ones_like(x), atol=1e-6)

    def test_point_set_restores_grid_order(self):
        data = {
            "xyz": np.array([[1,1,0],[0,0,0],[1,0,0],[0,1,0]], np.float32),
            "uv": np.array([[1,1],[0,0],[1,0],[0,1]], np.float32),
            "st": np.array([[1,1],[0,0],[0,1],[1,0]], np.float32),
            "cross_section_index": np.array([1,0,0,1]),
            "cross_section_alpha": np.array([1.,0.,1.,0.]),
            "camera_depth": np.array([2.,1.,1.,2.]),
        }
        xyz, uv, st, depth = point_set_to_grid(data)
        np.testing.assert_array_equal(xyz[..., :2], uv)
        np.testing.assert_array_equal(st,[[[0,0],[0,1]],[[1,0],[1,1]]])
        np.testing.assert_array_equal(depth,[[1.,1.],[2.,2.]])

    def test_irls_gain_bias_rejects_local_outlier(self):
        ramp=jnp.linspace(.1,.7,100).reshape(10,10)
        physical=jnp.stack([ramp,.8*ramp+.03,.6*ramp+.08],axis=-1)
        expected_gain=jnp.asarray([1.1,.9,1.05]); expected_bias=jnp.asarray([.02,-.01,.03])
        observed=physical*expected_gain+expected_bias
        observed = observed.at[3:6,3:6].set(1)
        gain,bias,weights=irls_gain_bias(
            observed,physical,jnp.zeros(3),jnp.ones(3)*.03,iterations=6,
            lambda_gain=.01,lambda_bias=.01,max_gain_deviation=.25,
            max_bias_deviation=.05)
        np.testing.assert_allclose(np.asarray(gain),np.asarray(expected_gain),atol=3e-3)
        np.testing.assert_allclose(np.asarray(bias),np.asarray(expected_bias),atol=2e-3)
        self.assertLess(float(weights[4,4]), float(weights[0,0]))

        residual=np.asarray(observed-gain*physical-bias)
        distance=np.sqrt(np.sum((residual/.03)**2,axis=-1)+1e-12)
        scale=max(np.median(distance)/1.5381722544550522,1e-6)
        z=distance/(4.685*scale+1e-6)
        expected=np.where(z<1,(1-z*z)**2,0)
        np.testing.assert_allclose(np.asarray(weights),expected,atol=1e-6)

    def test_irls_gain_and_bias_are_bounded_around_supplied_priors(self):
        ramp=jnp.linspace(.1,.5,25).reshape(5,5)
        physical=jnp.stack([ramp,ramp,ramp],axis=-1)
        observed=physical+.2
        gain_prior=jnp.asarray([.85,1.1,1.2])
        gain,bias,_=irls_gain_bias(
            observed,physical,jnp.zeros(3),jnp.ones(3)*.03,iterations=3,
            lambda_gain=.01,lambda_bias=.01,max_gain_deviation=0,
            max_bias_deviation=.03,gain_prior=gain_prior)
        np.testing.assert_allclose(np.asarray(gain),gain_prior,atol=1e-7)
        np.testing.assert_allclose(np.asarray(bias),np.ones(3)*.03,atol=1e-7)

    def test_irls_gain_bias_excludes_pixels_outside_projection(self):
        ramp=jnp.linspace(.1,.7,100).reshape(10,10)
        physical=jnp.stack([ramp,.8*ramp,.6*ramp],axis=-1)
        expected_gain=jnp.asarray([1.12,.91,1.04])
        expected_bias=jnp.asarray([.015,-.008,.02])
        observed=physical*expected_gain+expected_bias
        valid=np.ones((10,10),bool); valid[:4]=False
        observed=observed.at[:4].set(jnp.asarray([1.,0.,1.]))
        gain,bias,weights=irls_gain_bias(
            observed,physical,jnp.zeros(3),jnp.ones(3)*.03,iterations=6,
            lambda_gain=.01,lambda_bias=.01,max_gain_deviation=.25,
            max_bias_deviation=.05,valid_mask=jnp.asarray(valid))
        np.testing.assert_allclose(np.asarray(gain),expected_gain,atol=4e-3)
        np.testing.assert_allclose(np.asarray(bias),expected_bias,atol=2e-3)
        np.testing.assert_array_equal(np.asarray(weights[:4]),0)

    def test_model_format_enforces_zero_dark_baseline(self):
        layout=parse_light_source_layout({
            "R":["left","right"],"G":"left","B":"bottom"})
        model=LightFieldModel(jnp.ones((4,2)),jnp.ones((4,6)),jnp.zeros(3),
                              jnp.ones(4)*.5,jnp.ones(4)*20,jnp.eye(3),
                              jnp.zeros((3,8,4)),jnp.zeros((1,3,8,4)),
                              layout)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["format_version"],17)
            self.assertEqual(raw["background_method"],"physical_residual")
            self.assertEqual(raw["delta_axes"],["x","normal"])
            np.testing.assert_allclose(np.asarray(raw["scatter_ratio"]),np.ones(4)*.5)
            np.testing.assert_allclose(np.asarray(raw["mixing_matrix"]),np.eye(3))
            self.assertEqual(raw["light_source_layout"],
                             {"R":["left","right"],"G":"left","B":"bottom"})
            self.assertEqual(raw["dark_bias_mode"],"fixed_zero")
            self.assertNotIn("baseline_bspline_coefficients",raw)
            self.assertEqual(raw["residual_correction_mode"],"difference_only")
            self.assertEqual(raw["base_coefficient_mode"],"free")
            self.assertEqual(raw["residual_m_basis"],"raw_offline")
            loaded=LightFieldModel.load(path)
            np.testing.assert_array_equal(np.asarray(loaded.bias),np.zeros(3))
            self.assertEqual(loaded.source_layout,layout)
            self.assertEqual(loaded.delta.shape,(4,2))

            raw["format_version"]=8
            path.write_text(yaml.safe_dump(raw),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"版本已经过期"):
                LightFieldModel.load(path)

    def test_direct_fit_model_round_trip_omits_physical_parameters(self):
        b=np.zeros((3,8,4),np.float32)
        model=make_direct_model(b)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"direct.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded=LightFieldModel.load(path)
        self.assertEqual(raw["format_version"],18)
        self.assertEqual(raw["background_method"],"direct_fit")
        self.assertEqual(
            raw["residual_correction_mode"],
            "static_base_plus_geometry_delta_plus_additive_session")
        self.assertEqual(raw["base_coefficient_mode"],"direct_static_texture")
        self.assertEqual(raw["direct_session_correction_mode"],"additive_bspline")
        self.assertEqual(raw["direct_base_mode"],"robust_full_resolution_texture")
        self.assertEqual(raw["direct_delta_mode"],"additive_logit")
        self.assertEqual(np.asarray(raw["direct_base_texture"]).shape,(7,5,3))
        self.assertEqual(raw["direct_decoder_skip_mode"],"input_every_layer")
        self.assertEqual(
            raw["direct_reconstruction_pipeline"],"jax_surface_from_masks_v1")
        self.assertEqual(np.asarray(
            raw["direct_geometry_pca_components"]).shape,(46,2))
        self.assertNotIn("delta_mm",raw)
        self.assertNotIn("bspline_coefficients",raw)
        self.assertNotIn("residual_m_bspline_coefficients",raw)
        self.assertEqual(loaded.background_method,"direct_fit")
        np.testing.assert_array_equal(np.asarray(loaded.residual_b_coefficients),b)
        self.assertEqual(loaded.residual_m_coefficients.shape,(0,3,8,4))
        expected=1/(1+np.exp(-np.asarray([-.5,-1.,-1.5])))
        y,x=np.meshgrid(np.arange(6.),np.arange(3.),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(direct_background_field_jax((7,5),xyz,loaded)),
            np.broadcast_to(expected,(7,5,3)),atol=1e-6)

    def test_direct_fit_3_round_trip_uses_independent_scalar_decoders(self):
        b=np.zeros((3,8,4),np.float32)
        base=np.stack([
            np.full((7,5),.2,np.float32),
            np.full((7,5),.4,np.float32),
            np.full((7,5),.6,np.float32)],axis=-1)
        model=make_direct_3_model(b,base_texture=base)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"direct_3.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded=LightFieldModel.load(path)
        self.assertEqual(raw["format_version"],19)
        self.assertEqual(raw["background_method"],"direct_fit_3")
        self.assertEqual(raw["direct_base_channel_mode"],"independent_huber")
        self.assertEqual(raw["direct_channel_decoder_order"],["R","G","B"])
        self.assertNotIn("direct_spatial_mode_bspline_coefficients",raw)
        self.assertNotIn("direct_mode_score_weights",raw)
        self.assertNotIn("direct_decoder_weights",raw)
        self.assertEqual(len(raw["direct_channel_decoder_weights"]),3)
        self.assertEqual(loaded.background_method,"direct_fit_3")
        self.assertIsNone(loaded.direct_decoder_weights)
        self.assertEqual(len(loaded.direct_channel_decoder_weights),3)
        y,x=np.meshgrid(np.arange(6.),np.arange(3.),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1).astype(np.float32)
        expected=1/(1+np.exp(-(
            np.log(base[0,0])-np.log1p(-base[0,0])
            +np.asarray([-.5,-1.,-1.5]))))
        np.testing.assert_allclose(
            np.asarray(direct_background_field_jax((7,5),xyz,loaded)),
            np.broadcast_to(expected,(7,5,3)),atol=1e-6)

    def test_geometry_cache_round_trip_and_anchor_interpolation(self):
        curve_coefficients=4
        descriptor_count=4*curve_coefficients
        anchors=np.zeros((2,3,4,4),np.float32)
        anchors[1,2]=.1
        descriptor_mean=np.zeros(descriptor_count,np.float32)
        descriptor_scale=np.ones(descriptor_count,np.float32)
        pca_components=np.zeros((descriptor_count,1),np.float32)
        # 第一个中心线 Z 控制点决定此合成测试的一维几何键。
        pca_components[2,0]=1
        model=LightFieldModel.geometry_cache(
            np.zeros((3,4,4),np.float32),
            base_texture=np.full((7,5,3),.4,np.float32),
            anchor_coefficients=anchors,
            descriptor_mean=descriptor_mean,descriptor_scale=descriptor_scale,
            pca_components=pca_components,pca_scale=np.ones(1,np.float32),
            anchor_keys=np.asarray([[100.],[110.]],np.float32),
            curve_coefficients=curve_coefficients,
            interpolation_neighbors=2,distance_epsilon=1e-4)
        y,x=np.meshgrid(np.arange(6.),np.arange(3.),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*105],axis=-1).astype(np.float32)
        expected=np.full((7,5,3),.4,np.float32); expected[...,2]=.45
        np.testing.assert_allclose(np.asarray(
            geometry_cache_background_field_jax((7,5),xyz,model)),
            expected,atol=2e-4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cache.yaml"
            model.save(path)
            raw=yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded=LightFieldModel.load(path)
        self.assertEqual(raw["format_version"],21)
        self.assertEqual(raw["background_method"],"geometry_cache")
        self.assertEqual(
            raw["geometry_cache_mode"],
            "nearest_anchor_convex_interpolation")
        self.assertEqual(loaded.geometry_cache_anchor_coefficients.shape,
                         (2,3,4,4))

    def test_direct_neural_field_changes_with_global_geometry(self):
        descriptor_rows=4; descriptor_count=6+10*descriptor_rows
        feature_mean=np.zeros(descriptor_count,np.float32)
        feature_mean[2]=100
        encoder_weight=np.zeros((descriptor_count,1),np.float32)
        encoder_weight[2,0]=1
        decoder_weight=np.zeros((24,3),np.float32)
        decoder_weight[6]=np.asarray([1.,-.5,.25],np.float32)
        model=LightFieldModel.direct_fit(
            np.zeros((3,4,4),np.float32),
            base_texture=np.full((7,5,3),.5,np.float32),
            coordinate_frequencies=np.asarray([1.],np.float32),
            geometry_feature_mean=feature_mean,
            geometry_feature_scale=np.ones(descriptor_count,np.float32),
            geometry_pca_components=np.zeros(
                (descriptor_count,2),np.float32),
            geometry_pca_scale=np.ones(2,np.float32),
            local_geometry_feature_mean=np.zeros(15,np.float32),
            local_geometry_feature_scale=np.ones(15,np.float32),
            geometry_encoder_weights=(encoder_weight,),
            geometry_encoder_biases=(np.zeros(1,np.float32),),
            decoder_weights=(decoder_weight,),
            decoder_biases=(np.zeros(3,np.float32),),
            geometry_descriptor_rows=descriptor_rows)
        y,x=np.meshgrid(np.arange(8.),np.arange(5.),indexing="ij")
        first=np.stack([x,y,np.ones_like(x)*99],axis=-1).astype(np.float32)
        second=first.copy(); second[...,2]=101
        a=np.asarray(direct_background_field_jax((7,5),first,model))
        b=np.asarray(direct_background_field_jax((7,5),second,model))
        self.assertGreater(float(np.max(np.abs(a-b))),.1)
        self.assertEqual(direct_geometry_descriptor_jax(first,4).ndim,1)

    def test_direct_neural_field_has_deterministic_pca_geometry_skip(self):
        descriptor_rows=4; descriptor_count=6+10*descriptor_rows
        feature_mean=np.zeros(descriptor_count,np.float32); feature_mean[2]=100
        pca_components=np.zeros((descriptor_count,2),np.float32)
        pca_components[2,0]=1
        decoder_weight=np.zeros((24,3),np.float32)
        # 6 维坐标和 1 维学习 latent 之后是 PCA 直连。
        decoder_weight[7,0]=1
        model=LightFieldModel.direct_fit(
            np.zeros((3,4,4),np.float32),
            base_texture=np.full((7,5,3),.5,np.float32),
            coordinate_frequencies=np.asarray([1.],np.float32),
            geometry_feature_mean=feature_mean,
            geometry_feature_scale=np.ones(descriptor_count,np.float32),
            geometry_pca_components=pca_components,
            geometry_pca_scale=np.ones(2,np.float32),
            local_geometry_feature_mean=np.zeros(15,np.float32),
            local_geometry_feature_scale=np.ones(15,np.float32),
            geometry_encoder_weights=(
                np.zeros((descriptor_count,1),np.float32),),
            geometry_encoder_biases=(np.zeros(1,np.float32),),
            decoder_weights=(decoder_weight,),
            decoder_biases=(np.zeros(3,np.float32),),
            geometry_descriptor_rows=descriptor_rows)
        y,x=np.meshgrid(np.arange(8.),np.arange(5.),indexing="ij")
        first=np.stack([x,y,np.ones_like(x)*99],axis=-1).astype(np.float32)
        second=first.copy(); second[...,2]=101
        a=np.asarray(direct_background_field_jax((7,5),first,model))
        b=np.asarray(direct_background_field_jax((7,5),second,model))
        self.assertGreater(float(np.max(np.abs(a-b))),.4)

    def test_direct_neural_field_uses_local_geometry_through_dense_skip(self):
        descriptor_rows=4; descriptor_count=6+10*descriptor_rows
        first_weight=np.zeros((24,2),np.float32)
        output_weight=np.zeros((26,3),np.float32)
        # 完整条件为 [6 坐标, 1 latent, 2 PCA, 15 局部]；索引 11 是局部 z。
        output_weight[2+11,0]=4
        model=LightFieldModel.direct_fit(
            np.zeros((3,4,4),np.float32),
            base_texture=np.full((8,5,3),.5,np.float32),
            coordinate_frequencies=np.asarray([1.],np.float32),
            geometry_feature_mean=np.zeros(descriptor_count,np.float32),
            geometry_feature_scale=np.ones(descriptor_count,np.float32),
            geometry_pca_components=np.zeros(
                (descriptor_count,2),np.float32),
            geometry_pca_scale=np.ones(2,np.float32),
            local_geometry_feature_mean=np.zeros(15,np.float32),
            local_geometry_feature_scale=np.ones(15,np.float32),
            geometry_encoder_weights=(
                np.zeros((descriptor_count,1),np.float32),),
            geometry_encoder_biases=(np.zeros(1,np.float32),),
            decoder_weights=(first_weight,output_weight),
            decoder_biases=(np.zeros(2,np.float32),np.zeros(3,np.float32)),
            geometry_descriptor_rows=descriptor_rows)
        y,x=np.meshgrid(
            np.linspace(0,1,8),np.linspace(0,1,5),indexing="ij")
        xyz=np.stack([x,y,.3*y*y],axis=-1).astype(np.float32)
        field=np.asarray(direct_background_field_jax((8,5),xyz,model))
        self.assertGreater(float(field[-1,:,0].mean()-field[0,:,0].mean()),.2)
        np.testing.assert_allclose(field[...,1:],.5,atol=1e-6)

    def test_residual_correction_never_changes_physical_render(self):
        y,x=jnp.meshgrid(jnp.linspace(-2,2,4),jnp.linspace(-3,3,5),indexing="ij")
        xyz=jnp.stack([x,y,jnp.ones_like(x)*100],axis=-1)
        common=(jnp.zeros((3,2)),jnp.ones((3,6)),jnp.zeros(3),
                jnp.zeros(3),jnp.ones(3)*3,jnp.eye(3))
        zero=LightFieldModel(*common,jnp.zeros((3,4,4)),jnp.zeros((2,3,4,4)))
        nonzero=LightFieldModel(*common,jnp.ones((3,4,4))*.3,jnp.ones((2,3,4,4))*.1)
        first=physical_background(xyz,zero,integration_nodes=8)
        second=physical_background(xyz,nonzero,integration_nodes=8)
        np.testing.assert_allclose(np.asarray(first),np.asarray(second),atol=1e-7)

    @unittest.skipUnless(
        any(device.platform=="gpu" for device in jax.devices()),
        "需要可用的 JAX GPU",
    )
    def test_direct_geometry_conditioned_network_fits_bending_field(self):
        rows,columns=12,9
        y,x=np.meshgrid(
            np.linspace(0,1,rows),np.linspace(0,1,columns),indexing="ij")
        amplitudes=np.linspace(-1,1,7,dtype=np.float32)
        surfaces=np.stack([
            np.stack([20*x,40*y,100+amplitude*(y-.5)**2],axis=-1)
            for amplitude in amplitudes]).astype(np.float32)
        fields=np.stack([np.stack([
            .25+.18*y+.04*amplitude,
            .18+.15*x-.03*amplitude,
            .12+.10*x*y+.02*amplitude],axis=-1)
            for amplitude in amplitudes]).astype(np.float32)
        (base_texture,frequencies,feature_mean,feature_scale,pca_components,pca_scale,
         local_mean,local_scale,
         encoder_weights,encoder_biases,decoder_weights,decoder_biases)=(
            fit_direct_geometry_conditioned_field_gpu(
                fields,np.ones(fields.shape[:3],bool),surface_xyz=surfaces,
                device=jax.devices("gpu")[0],frequencies=(1.,2.),
                geometry_descriptor_rows=4,geometry_encoder_width=16,
                geometry_encoder_layers=1,geometry_latent_dimensions=6,
                geometry_pca_dimensions=3,
                decoder_width=24,decoder_layers=2,steps=100,batch_size=768,
                frame_batch_size=4,learning_rate=.01,huber_delta=.04,
                base_huber_iterations=2,
                spatial_difference_points_per_frame=96,
                geometry_difference_neighbor_count=3,
                geometry_difference_points_per_pair=96,seed=3))
        model=LightFieldModel.direct_fit(
            np.zeros((3,4,4),np.float32),
            base_texture=base_texture,
            coordinate_frequencies=frequencies,
            geometry_feature_mean=feature_mean,
            geometry_feature_scale=feature_scale,
            geometry_pca_components=pca_components,
            geometry_pca_scale=pca_scale,
            local_geometry_feature_mean=local_mean,
            local_geometry_feature_scale=local_scale,
            geometry_encoder_weights=encoder_weights,
            geometry_encoder_biases=encoder_biases,
            decoder_weights=decoder_weights,decoder_biases=decoder_biases,
            geometry_descriptor_rows=4)
        prediction=np.stack([np.asarray(direct_background_field_jax(
            (rows,columns),surface,model)) for surface in surfaces])
        self.assertEqual(frequencies.shape,(2,))
        self.assertEqual(len(encoder_weights),2)
        self.assertEqual(len(decoder_weights),3)
        self.assertLess(float(np.sqrt(np.mean((prediction-fields)**2))),.035)
        predicted_difference=prediction[1:]-prediction[:-1]
        target_difference=fields[1:]-fields[:-1]
        self.assertLess(float(np.sqrt(np.mean(
            (predicted_difference-target_difference)**2))),.025)

    @unittest.skipUnless(
        any(device.platform=="gpu" for device in jax.devices()),
        "需要可用的 JAX GPU",
    )
    def test_direct_fit_3_training_returns_three_scalar_decoders(self):
        rows,columns=6,5
        y,x=np.meshgrid(
            np.linspace(0,1,rows),np.linspace(0,1,columns),indexing="ij")
        amplitudes=np.linspace(-1,1,5,dtype=np.float32)
        surfaces=np.stack([
            np.stack([10*x,20*y,100+amplitude*(y-.5)**2],axis=-1)
            for amplitude in amplitudes]).astype(np.float32)
        fields=np.stack([np.stack([
            .2+.05*amplitude+.03*x,
            .35-.04*amplitude+.02*y,
            .5+.02*amplitude+.01*x*y],axis=-1)
            for amplitude in amplitudes]).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/"direct_3.npz"
            result=fit_direct_geometry_conditioned_field_gpu(
                fields,np.ones(fields.shape[:3],bool),surface_xyz=surfaces,
                validation_fields=fields[-2:],
                validation_valid=np.ones(fields[-2:].shape[:3],bool),
                validation_surface_xyz=surfaces[-2:],
                checkpoint_path=checkpoint,
                device=jax.devices("gpu")[0],frequencies=(1.,),
                geometry_descriptor_rows=4,geometry_encoder_width=8,
                geometry_encoder_layers=1,geometry_latent_dimensions=3,
                geometry_pca_dimensions=2,decoder_width=10,decoder_layers=1,
                steps=10,batch_size=120,frame_batch_size=3,
                learning_rate=.01,huber_delta=.04,base_huber_iterations=1,
                adaptive_channel_weight_strength=.5,
                spatial_difference_points_per_frame=24,
                geometry_difference_neighbor_count=2,
                geometry_difference_points_per_pair=24,seed=7,
                validation_interval=5,validation_points_per_frame=16,
                early_stopping_patience=5,early_stopping_min_steps=0,
                separate_channel_decoders=True)
            with np.load(checkpoint,allow_pickle=False) as saved:
                self.assertEqual(int(saved["checkpoint_format_version"]),4)
                self.assertEqual(int(saved["channel_decoder_count"]),3)
                self.assertIn("decoder_2_weight_1",saved.files)
        (*common,encoder_weights,encoder_biases,
         channel_weights,channel_biases)=result
        self.assertEqual(len(channel_weights),3)
        self.assertEqual(len(channel_biases),3)
        self.assertTrue(all(decoder[-1].shape[-1]==1
                            for decoder in channel_weights))
        model=LightFieldModel.direct_fit_3(
            np.zeros((3,4,4),np.float32),base_texture=common[0],
            coordinate_frequencies=common[1],
            geometry_feature_mean=common[2],geometry_feature_scale=common[3],
            geometry_pca_components=common[4],geometry_pca_scale=common[5],
            local_geometry_feature_mean=common[6],
            local_geometry_feature_scale=common[7],
            geometry_encoder_weights=encoder_weights,
            geometry_encoder_biases=encoder_biases,
            channel_decoder_weights=channel_weights,
            channel_decoder_biases=channel_biases,
            geometry_descriptor_rows=4)
        predicted=np.asarray(direct_background_field_jax(
            (rows,columns),surfaces[0],model))
        self.assertEqual(predicted.shape,(rows,columns,3))
        self.assertTrue(np.isfinite(predicted).all())

    @unittest.skipUnless(
        any(device.platform=="gpu" for device in jax.devices()),
        "需要可用的 JAX GPU",
    )
    def test_direct_training_saves_best_checkpoint_and_early_stops(self):
        rows,columns=8,6
        y,x=np.meshgrid(
            np.linspace(0,1,rows),np.linspace(0,1,columns),indexing="ij")
        amplitudes=np.linspace(-1,1,6,dtype=np.float32)
        surfaces=np.stack([
            np.stack([10*x,20*y,100+amplitude*(y-.5)**2],axis=-1)
            for amplitude in amplitudes]).astype(np.float32)
        fields=np.stack([np.stack([
            .2+.1*y+.02*amplitude,.15+.08*x-.01*amplitude,
            .1+.06*x*y],axis=-1) for amplitude in amplitudes]).astype(np.float32)
        # 四个训练帧的有效集合互不相交，用于确认差分权重为零时不会
        # 仍然要求几何帧对存在共同有效 observation 点。
        training_valid=np.zeros(fields[:4].shape[:3],bool)
        for frame_index in range(4):
            training_valid[frame_index,:,frame_index]=True
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/"direct.best_ckpt.npz"
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                fit_direct_geometry_conditioned_field_gpu(
                    fields[:4],training_valid,
                    surface_xyz=surfaces[:4],
                    validation_fields=fields[4:],
                    validation_valid=np.ones(fields[4:].shape[:3],bool),
                    validation_surface_xyz=surfaces[4:],
                    checkpoint_path=checkpoint,device=jax.devices("gpu")[0],
                    frequencies=(1.,),geometry_descriptor_rows=4,
                    geometry_encoder_width=8,geometry_encoder_layers=1,
                    geometry_latent_dimensions=4,geometry_pca_dimensions=2,
                    decoder_width=12,
                    decoder_layers=1,steps=30,batch_size=128,
                    frame_batch_size=2,learning_rate=.01,
                    base_huber_iterations=1,
                    spatial_difference_points_per_frame=24,
                    geometry_difference_weight=0,
                    geometry_difference_neighbor_count=2,
                    geometry_difference_points_per_pair=24,
                    validation_interval=5,validation_points_per_frame=24,
                    early_stopping_patience=2,early_stopping_min_steps=10,
                    early_stopping_min_delta=1.,seed=5)
            with np.load(checkpoint) as data:
                self.assertEqual(int(data["checkpoint_format_version"]),3)
                best_step=int(data["step"])
                self.assertIn(best_step,(5,10,15))
                self.assertEqual(data["validation_rmse_rgb"].shape,(3,))
                self.assertEqual(
                    data["validation_difference_rmse_rgb"].shape,(3,))
                self.assertEqual(
                    data["validation_spatial_difference_rmse_rgb"].shape,(3,))
                self.assertEqual(data["direct_base_texture"].shape,(rows,columns,3))
                self.assertIn("encoder_moment_weight_0",data.files)
                self.assertIn("decoder_variance_bias_1",data.files)
                self.assertEqual(data["geometry_pca_components"].shape,(46,2))
            self.assertIn("direct early stopping: step=15",output.getvalue())
            self.assertIn(
                f"恢复最佳参数：step={best_step}/15",output.getvalue())

    @unittest.skipUnless(
        any(device.platform=="gpu" for device in jax.devices()),
        "需要可用的 JAX GPU",
    )
    def test_gpu_chunked_residual_model_handles_tail_batches_and_pixels(self):
        rng=np.random.default_rng(18); count,rows,columns=7,12,9
        y,x=np.meshgrid(
            np.linspace(-1,1,rows),np.linspace(-1,1,columns),indexing="ij")
        mean=np.stack([.03*y,.02*x,-.02*y],axis=-1)
        modes=np.stack([
            np.stack([.025*x,np.zeros_like(x),-.015*x],axis=-1),
            np.stack([np.zeros_like(y),.02*y,.012*y],axis=-1),
        ])
        scores=rng.normal(size=(count,2)); scores-=scores.mean(axis=0)
        residuals=(mean[None]+np.einsum(
            "nk,khwc->nhwc",scores,modes)).astype(np.float32)
        surface_xyz=np.stack([
            np.broadcast_to(10*x,(count,rows,columns)),
            np.broadcast_to(50*y,(count,rows,columns)),
            (100+3*scores[:,0,None,None]*y[None]**2
             +2*scores[:,1,None,None]*y[None]**3),
        ],axis=-1).astype(np.float32)
        fitted_b,fitted_m,fitted_scores=fit_residual_correction_model_gpu(
            residuals,np.ones((count,rows,columns),bool),
            surface_xyz=surface_xyz,row_coefficients=5,
            column_coefficients=4,m_count=2,huber_delta=.08,
            smooth_lambda=1e-5,magnitude_lambda=1e-7,
            outer_weight=1,outer_fraction=0,b_max_deviation=.2,
            m_max_deviation=.1,curvature_feature_count=2,
            curvature_curve_coefficients=6,
            curvature_smooth_lambda=1e-6,
            curvature_regression_lambda=1e-6,
            sample_batch_size=3,pixel_chunk_size=17,
            scale_sample_pixels=31,device=jax.devices("gpu")[0])
        fitted_b_field=evaluate_rgb_bspline(fitted_b,(rows,columns))
        fitted_m_fields=np.stack([
            evaluate_rgb_bspline(item,(rows,columns)) for item in fitted_m])
        reconstructed=fitted_b_field[None]+np.einsum(
            "nck,khwc->nhwc",fitted_scores,fitted_m_fields)
        self.assertEqual(fitted_scores.shape,(count,3,2))
        self.assertLess(
            float(np.sqrt(np.mean((residuals-reconstructed)**2))),.012)

    def test_startup_model_reparameterizes_raw_ms_against_bsession(self):
        count,rows,columns=10,24,18
        y,x=np.meshgrid(np.linspace(-1,1,rows),np.linspace(-1,1,columns),indexing="ij")
        offline=np.stack([.02*y,.01*x,-.015*y],axis=-1)
        bsession=np.stack([.025*x,-.018*y,.012*x],axis=-1)
        first=np.stack([.012*y*x,.02*y*x,-.01*y*x],axis=-1)
        second=np.stack([
            .01*(x*x-1/3),-.012*(y*y-1/3),.008*(x*x-1/3)],axis=-1)
        scores=np.stack([
            np.linspace(-1.5,1.5,count),
            np.asarray([-1,1,-1,1,-1,1,-1,1,-1,1])],axis=1)
        residuals=(offline[None]+bsession[None]
                   +np.einsum("nk,khwc->nhwc",scores,np.stack([first,second])))
        offline_coefficients=np.zeros((3,8,6),np.float32)
        # 这些场对三次 B 样条是可表示的；先单独拟合离线 B 系数。
        from utils.lightfield import _fit_rgb_bspline
        offline_coefficients=_fit_rgb_bspline(
            offline,np.ones((rows,columns)),8,6,1e-6,1e-8)
        offline_ms=_fit_rgb_bspline(
            np.stack([first,second]),np.ones((rows,columns)),8,6,1e-6,1e-8)
        fields,training_scores,channel_huber,diagnostics=(
            fit_startup_residual_bsession_model(
            residuals,np.ones((count,rows,columns),bool),offline_coefficients,
            offline_ms,
            huber_delta=.08,smooth_lambda=1e-6,magnitude_lambda=1e-8,
            outer_weight=1,outer_fraction=0,bsession_prior_lambda=1e-8,
            bsession_max_field_deviation=.3))
        field_values=np.stack([
            evaluate_rgb_bspline(item,(rows,columns)) for item in fields])
        reconstructed=np.einsum("nck,khwc->nhwc",training_scores,field_values)
        self.assertEqual(training_scores.shape,(10,3,3))
        self.assertEqual(channel_huber.shape,(3,))
        self.assertTrue(np.all((channel_huber>=.04)&(channel_huber<=.16)))
        self.assertLess(float(np.sqrt(np.mean((residuals-reconstructed)**2))),.004)
        self.assertEqual(set(diagnostics),{
            "raw_rmse_rgb","bsession_rmse_rgb","bsession_m_rmse_rgb",
            "cross_frame_floor_rmse_rgb","bspline_spatial_miss_rmse_rgb"})
        for channel in range(3):
            b=field_values[0,...,channel].reshape(-1)
            for mode in field_values[1:,...,channel]:
                self.assertLess(abs(float(b@mode.reshape(-1))),1e-6)
        self.assertTrue(np.all(
            diagnostics["bsession_m_rmse_rgb"]<diagnostics["bsession_rmse_rgb"]))

    def test_startup_orthogonalization_removes_collinear_bsession_from_m(self):
        count,rows,columns=6,20,14
        y,x=np.meshgrid(
            np.linspace(-1,1,rows),np.linspace(-1,1,columns),indexing="ij")
        field=np.stack([.03*y,.025*x,.02*(x+y)],axis=-1)
        residuals=np.broadcast_to(field,(count,rows,columns,3)).copy()
        from utils.lightfield import _fit_rgb_bspline
        coefficients=_fit_rgb_bspline(
            field,np.ones((rows,columns)),8,6,1e-7,1e-9)
        fields,_,_,_=fit_startup_residual_bsession_model(
            residuals,np.ones((count,rows,columns),bool),coefficients,
            coefficients[None],huber_delta=.08,smooth_lambda=1e-7,
            magnitude_lambda=1e-9,outer_weight=1,outer_fraction=0,
            bsession_prior_lambda=1.,bsession_max_field_deviation=.3)
        session_m=evaluate_rgb_bspline(fields[1],(rows,columns))
        self.assertLess(float(np.sqrt(np.mean(session_m**2))),1e-5)

    def test_uniform_residual_scores_recover_fixed_basis_coefficients(self):
        rows,columns=20,14
        y,x=np.meshgrid(np.linspace(-1,1,rows),np.linspace(-1,1,columns),indexing="ij")
        mean=np.stack([.04*y,.03*x,-.02*y],axis=-1).astype(np.float32)
        modes=np.stack([
            np.stack([.01*y*x,.02*y*x,-.015*y*x],axis=-1),
            np.stack([.012*x,-.01*y,.008*x],axis=-1),
        ]).astype(np.float32)
        expected=np.asarray([
            [1.04,-.45,.25],[.98,.3,-.35],[1.02,-.2,.4]],np.float32)
        fields=np.concatenate([mean[None],modes],axis=0)
        residual=np.einsum("ck,khwc->hwc",expected,fields)
        actual=fit_uniform_residual_correction_scores_jax(
            jnp.asarray(residual),jnp.asarray(mean),jnp.asarray(modes),
            jnp.ones((rows,columns)))
        np.testing.assert_allclose(np.asarray(actual),expected,atol=1e-5)

    def test_uniform_residual_scores_use_equal_valid_pixel_weights(self):
        target_values=np.asarray([.1,.2,2.,4.],np.float32)
        residual=np.broadcast_to(target_values[None,:,None],(1,4,3)).copy()
        mean=np.ones((1,4,3),np.float32)
        modes=np.zeros((0,1,4,3),np.float32)
        valid=np.asarray([[1.,1.,1.,0.]],np.float32)
        scores=fit_uniform_residual_correction_scores_jax(
            jnp.asarray(residual),jnp.asarray(mean),jnp.asarray(modes),
            jnp.asarray(valid))
        expected=np.mean(target_values[:3])
        np.testing.assert_allclose(
            np.asarray(scores),np.full((3,1),expected),atol=1e-6)

    def test_uniform_huber_reduces_local_outlier_pull(self):
        target_values=np.asarray([.1,.1,.1,2.],np.float32)
        residual=np.broadcast_to(target_values[None,:,None],(1,4,3)).copy()
        bsession=np.ones((1,4,3),np.float32)
        modes=np.zeros((0,1,4,3),np.float32)
        valid=np.ones((1,4),np.float32)
        uniform=fit_uniform_residual_correction_scores_jax(
            jnp.asarray(residual),jnp.asarray(bsession),jnp.asarray(modes),
            jnp.asarray(valid))
        robust=fit_uniform_huber_residual_correction_scores_jax(
            jnp.asarray(residual),jnp.asarray(bsession),jnp.asarray(modes),
            jnp.asarray(valid),huber_delta=.05,iterations=8)
        self.assertGreater(float(np.asarray(uniform)[0,0]),.5)
        np.testing.assert_allclose(
            np.asarray(robust),np.full((3,1),.1+.05/3),atol=2e-3)

    def test_direct_startup_fits_only_additive_session(self):
        count,rows,columns=6,18,12
        y,x=np.meshgrid(
            np.linspace(-1,1,rows),np.linspace(-1,1,columns),indexing="ij")
        session=np.stack([.015*y,-.012*x,.01*y],axis=-1)
        samples=np.broadcast_to(session[None],(count,rows,columns,3)).copy()
        samples[0,8:10,5:7]+=.2
        offline_session=np.zeros((3,7,5),np.float32)
        b_coefficients,actual_scores,_,diagnostics=(
            fit_startup_direct_bsession_model(
                samples,np.ones((count,rows,columns),bool),offline_session,
                huber_delta=.05,smooth_lambda=1e-7,
                magnitude_lambda=1e-9,outer_weight=1,outer_fraction=0,
                bsession_prior_lambda=1e-8,
                session_correction_bounds=(-.1,.1)))
        fitted=evaluate_rgb_bspline(b_coefficients,(rows,columns))
        self.assertEqual(actual_scores.shape,(count,3,0))
        self.assertTrue(np.all((b_coefficients>=-.1)&(b_coefficients<=.1)))
        self.assertLess(float(np.sqrt(np.mean((session-fitted)**2))),2e-3)
        self.assertTrue(np.all(
            diagnostics["bsession_rmse_rgb"]<diagnostics["raw_rmse_rgb"]))

    def test_physical_rgb_light_source_layout(self):
        xyz=jnp.arange(4*5*3,dtype=jnp.float32).reshape(4,5,3)
        normals=xyz+1000
        layout=parse_light_source_layout({"R":"right","G":"left","B":"bottom"})
        edges,edge_normals=_light_source_boundaries(xyz,normals,layout)
        np.testing.assert_array_equal(edges[0],xyz[:,-1])   # R: 右
        np.testing.assert_array_equal(edges[1],xyz[:,0])    # G: 左
        np.testing.assert_array_equal(edges[2],xyz[-1])     # B: 下
        np.testing.assert_array_equal(edge_normals[0],normals[:,-1])
        np.testing.assert_array_equal(edge_normals[1],normals[:,0])
        np.testing.assert_array_equal(edge_normals[2],normals[-1])

    def test_light_source_layout_is_configurable(self):
        layout=parse_light_source_layout({"R":"top","G":"right","B":"left"})
        xyz=jnp.arange(3*4*3,dtype=jnp.float32).reshape(3,4,3)
        edges,_=_light_source_boundaries(xyz,xyz,layout)
        np.testing.assert_array_equal(edges[0],xyz[0])
        np.testing.assert_array_equal(edges[1],xyz[:,-1])
        np.testing.assert_array_equal(edges[2],xyz[:,0])
        shared=parse_light_source_layout({
            "R":["left","right"],"G":"left","B":"bottom"})
        self.assertEqual(shared[0],("left","right"))
        shared_edges,_=_light_source_boundaries(xyz,xyz,shared)
        self.assertEqual(len(shared_edges),4)
        np.testing.assert_array_equal(shared_edges[0],xyz[:,0])
        np.testing.assert_array_equal(shared_edges[1],xyz[:,-1])
        np.testing.assert_array_equal(shared_edges[2],xyz[:,0])
        with self.assertRaisesRegex(ValueError,"重复配置"):
            parse_light_source_layout({
                "R":["left","left"],"G":"right","B":"bottom"})

    def test_same_color_sources_have_independent_parameters_and_sum(self):
        y,x=jnp.meshgrid(
            jnp.linspace(-2,2,4),jnp.linspace(-3,3,5),indexing="ij")
        xyz=jnp.stack([x,y,jnp.ones_like(x)*100],axis=-1)
        layout=parse_light_source_layout({
            "R":["left","right"],"G":"top","B":"bottom"})
        beta=jnp.asarray([
            [1.,1.,1.,1.],[2.,2.,2.,2.],
            [3.,3.,3.,3.],[4.,4.,4.,4.]],jnp.float32)
        model=LightFieldModel(
            jnp.tile(jnp.asarray([[0.,2.]],jnp.float32),(4,1)),
            beta,jnp.zeros(3),jnp.zeros(4),
            jnp.ones(4),jnp.eye(3),jnp.zeros((3,4,4)),
            jnp.zeros((1,3,4,4)),layout)
        direct=_direct_light_fields(xyz,model,8,.05)
        physical=physical_background(xyz,model,integration_nodes=8)
        np.testing.assert_allclose(
            np.asarray(physical[...,0]),
            np.asarray(direct[...,0]+direct[...,1]),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(physical[...,1]),np.asarray(direct[...,2]),
            rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(physical[...,2]),np.asarray(direct[...,3]),
            rtol=1e-6,atol=1e-7)
        self.assertGreater(float(jnp.max(jnp.abs(
            direct[...,0]-direct[...,1]))),0)

    def test_edge_normals_use_edge_arclength_plan(self):
        t=jnp.linspace(0,1,20)
        edge=jnp.stack([jnp.zeros_like(t),100*t,10*jnp.sin(jnp.pi*t*t)],axis=-1)
        normals=jnp.stack([jnp.zeros_like(t),jnp.sin(t*t),jnp.cos(t*t)],axis=-1)
        _,_,_,index,fraction=_resample_curve(edge,11)
        sampled=_interpolate_samples(normals,index,fraction)
        expected=normals[index-1]*(1-fraction)+normals[index]*fraction
        np.testing.assert_allclose(sampled,expected,atol=1e-7)

    def test_local_x_points_inward_instead_of_along_strip(self):
        y,x=jnp.meshgrid(jnp.arange(4.),jnp.arange(5.),indexing="ij")
        xyz=jnp.stack([x,y,jnp.zeros_like(x)],axis=-1)
        layout=parse_light_source_layout({"R":"right","G":"left","B":"bottom"})
        inward=_light_source_inward_directions(xyz,layout)
        np.testing.assert_allclose(inward[0],jnp.tile(jnp.asarray([-1.,0.,0.]),(4,1)))
        np.testing.assert_allclose(inward[1],jnp.tile(jnp.asarray([1.,0.,0.]),(4,1)))
        np.testing.assert_allclose(inward[2],jnp.tile(jnp.asarray([0.,-1.,0.]),(5,1)))

    def test_surface_diffusion_preserves_constant_fields(self):
        y,x=jnp.meshgrid(jnp.arange(4.),jnp.arange(5.),indexing="ij")
        xyz=jnp.stack([x,y,jnp.zeros_like(x)],axis=-1)
        direct=jnp.ones((4,5,3))*jnp.asarray([.2,.4,.8])
        scattered=diffuse_surface_fields(xyz,direct,jnp.asarray([1.,2.,4.]),
                                         cg_tolerance=1e-6,cg_max_iterations=50)
        np.testing.assert_allclose(np.asarray(scattered),np.asarray(direct),atol=1e-5)

    def test_surface_diffusion_spreads_impulse_and_preserves_area_integral(self):
        y,x=jnp.meshgrid(jnp.arange(5.),jnp.arange(5.),indexing="ij")
        xyz=jnp.stack([x,y,jnp.zeros_like(x)],axis=-1)
        direct=jnp.zeros((5,5,3)).at[2,2,0].set(1)
        scattered=diffuse_surface_fields(xyz,direct,jnp.ones(3),
                                         cg_tolerance=1e-7,cg_max_iterations=100)
        area,*_=_surface_diffusion_geometry(xyz)
        self.assertLess(float(scattered[2,2,0]),1.)
        self.assertGreater(float(scattered[2,1,0]),0.)
        np.testing.assert_allclose(
            np.asarray(jnp.sum(area*direct[...,0].reshape(-1))),
            np.asarray(jnp.sum(area*scattered[...,0].reshape(-1))),atol=1e-5)

    def test_surface_diffusion_length_gradient_matches_executed_solver(self):
        y,x=jnp.meshgrid(jnp.arange(4.),jnp.arange(5.),indexing="ij")
        xyz=jnp.stack([x,y,jnp.zeros_like(x)],axis=-1)
        direct=jnp.zeros((4,5,3)).at[1,2,0].set(1).at[2,1,1].set(.7)
        def objective(length):
            lengths=jnp.asarray([length,length,1.])
            scattered=diffuse_surface_fields(xyz,direct,lengths,
                                             cg_tolerance=1e-6,cg_max_iterations=40)
            return jnp.sum(scattered**2)
        length=jnp.asarray(2.)
        automatic=float(jax.grad(objective)(length))
        step=.01
        finite=float((objective(length+step)-objective(length-step))/(2*step))
        self.assertAlmostEqual(automatic,finite,places=3)

    def test_jax_rasterizer_outputs_valid_depth_selected_image(self):
        uv=np.asarray([[[1,1],[5,1]],[[1,5],[5,5]]],np.float32)
        depth=np.ones((2,2),np.float32)*3
        rgb=np.ones((2,2,3),np.float32)*.4
        image,valid,overflow=rasterize_attributes_jax(
            jnp.asarray(uv),jnp.asarray(depth),jnp.asarray(rgb),(8,8),
            triangle_chunk=2,max_triangle_width=8,max_triangle_height=8)
        self.assertFalse(bool(overflow))
        self.assertTrue(bool(valid[3,3]))
        np.testing.assert_allclose(np.asarray(image[3,3]),.4,atol=1e-6)

    def test_jax_rasterizer_z_buffer_keeps_front_fold(self):
        # 第二个网格单元折回并覆盖第一个；蓝色端深度更小，必须覆盖红色后表面。
        uv=np.asarray([[[1,1],[6,1],[1,1]],[[1,6],[6,6],[1,6]]],np.float32)
        depth=np.asarray([[5,5,1],[5,5,1]],np.float32)
        rgb=np.zeros((2,3,3),np.float32)
        rgb[:,0]=[1,0,0]; rgb[:,1]=[.5,0,.5]; rgb[:,2]=[0,0,1]
        image,valid,overflow=rasterize_attributes_jax(
            jnp.asarray(uv),jnp.asarray(depth),jnp.asarray(rgb),(8,8),
            triangle_chunk=4,max_triangle_width=8,max_triangle_height=8)
        self.assertFalse(bool(overflow))
        self.assertTrue(bool(valid[3,2]))
        self.assertGreater(float(image[3,2,2]),float(image[3,2,0]))

    def test_jax_rasterizer_fuses_attributes(self):
        uv=np.asarray([[[1,1],[5,1]],[[1,5],[5,5]]],np.float32)
        depth=np.ones((2,2),np.float32)*3
        rgb=np.arange(12,dtype=np.float32).reshape(2,2,3)/12
        coordinates=np.asarray([[[0,0],[0,1]],[[1,0],[1,1]]],np.float32)
        attributes=np.concatenate([rgb,coordinates],axis=-1)
        actual,valid,overflow=rasterize_attributes_jax(
            jnp.asarray(uv),jnp.asarray(depth),jnp.asarray(attributes),(8,8),
            triangle_chunk=2,max_triangle_width=8,max_triangle_height=8)
        self.assertFalse(bool(overflow))
        self.assertEqual(int(np.asarray(valid).sum()),16)
        self.assertEqual(actual.shape[-1],5)
        np.testing.assert_allclose(np.asarray(actual[3,3,3:]),[.625,.625],atol=1e-6)

    def test_jax_rasterizer_reports_capacity_overflow(self):
        uv=jnp.asarray([[[0,0],[7,0]],[[0,7],[7,7]]],jnp.float32)
        attributes=jnp.ones((2,2,1),jnp.float32)
        _,_,overflow=rasterize_attributes_jax(
            uv,jnp.ones((2,2)),attributes,(8,8),triangle_chunk=2,
            max_triangle_width=4,max_triangle_height=4)
        self.assertTrue(bool(overflow))

    def test_jax_frame_linearization_and_sampling_match_cpu(self):
        image=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3)
        uv=np.asarray([[[1.25,2.5],[4.5,3.25]]],np.float32)
        linear=np.asarray(bgr_to_linear_rgb_jax(jnp.asarray(image)))
        np.testing.assert_allclose(linear,bgr_to_linear_rgb(image),atol=1e-7)
        sampled=np.asarray(sample_linear_rgb_jax(jnp.asarray(linear),jnp.asarray(uv)))
        expected=np.stack([
            cv2.remap(linear[...,channel],uv[...,0],uv[...,1],cv2.INTER_LINEAR)
            for channel in range(3)],axis=-1)
        np.testing.assert_allclose(sampled,expected,atol=2e-4)

    def test_jax_residual_fit_recovers_scores_and_texture_sampling(self):
        rows,columns=12,9
        y,x=np.meshgrid(np.linspace(0,1,rows),np.linspace(0,1,columns),indexing="ij")
        mean=np.stack([.02*y,.01*x,-.01*y],axis=-1).astype(np.float32)
        modes=np.stack([mean*.5,np.flip(mean,axis=1)]).astype(np.float32)
        expected_scores=np.asarray([.7,-.3],np.float32)
        residual=mean+np.einsum("k,khwc->hwc",expected_scores,modes)
        weight=np.ones((rows,columns),np.float32)
        jax_scores=fit_residual_m_scores_jax(
            jnp.asarray(residual),jnp.asarray(mean),jnp.asarray(modes),jnp.asarray(weight),
            .04,0.,3,6)
        np.testing.assert_allclose(np.asarray(jax_scores),expected_scores,atol=2e-4)
        coordinates=jnp.stack(jnp.meshgrid(jnp.linspace(0,1,rows),
            jnp.linspace(0,1,columns),indexing="ij"),axis=-1)
        sampled_mean,sampled_modes=sample_residual_correction_jax(
            coordinates,jnp.asarray(mean),jnp.asarray(modes),jnp.ones((rows,columns),bool))
        np.testing.assert_allclose(np.asarray(sampled_mean),mean,atol=1e-6)
        np.testing.assert_allclose(np.asarray(sampled_modes),modes,atol=1e-6)

    def test_jax_mask_erosion(self):
        mask=jnp.zeros((7,7),bool).at[1:6,1:6].set(True)
        eroded=np.asarray(erode_mask_jax(mask,1))
        expected=np.zeros((7,7),bool); expected[2:5,2:5]=True
        np.testing.assert_array_equal(eroded,expected)

    def test_saturation_mask_checks_all_bilinear_neighbors(self):
        image=np.zeros((4,4,3),np.uint8)+100
        image[1,1,0]=250
        uv=np.asarray([[[.5,.5],[2.2,2.2],[-.1,1.]]],np.float32)
        mask=sample_unsaturated_mask(image,uv,250)
        np.testing.assert_array_equal(mask,[[False,True,False]])

    def test_saturation_threshold_is_strict(self):
        image=np.zeros((3,3,3),np.uint8)+249
        uv=np.asarray([[[.5,.5]]],np.float32)
        self.assertTrue(sample_unsaturated_mask(image,uv,250)[0,0])
        image[0,0,2]=250
        self.assertFalse(sample_unsaturated_mask(image,uv,250)[0,0])

if __name__ == "__main__": unittest.main()
