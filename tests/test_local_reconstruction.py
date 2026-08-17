import tempfile
import unittest
from types import ModuleType
from unittest.mock import Mock,patch
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from recon import (NormalCalibration,classify_trusted_no_contact,
                   fit_no_contact_residual_model,
                   load_local_reconstruction_settings,main,lookup_slopes,
                   parse_no_contact_constraints,
                   reconstruct_local_surface,
                   sample_full_resolution_residual)
from utils.jax_local_reconstruction import (classify_trusted_no_contact_jax,
                                            lookup_slopes_jax,
                                            reconstruct_local_surface_jax)


def linear_slope_calibration() -> NormalCalibration:
    axes=np.asarray([-1.,1.],np.float32)
    red,green,blue=np.meshgrid(axes,axes,axes,indexing="ij")
    slopes=np.stack([red,green],axis=-1)
    return NormalCalibration(
        slopes=slopes,variances=np.ones((2,2,2),np.float32)*1e-4,
        color_min=np.full(3,-1.,np.float32),
        color_max=np.full(3,1.,np.float32),sigma_ref2=1e-4)


class LocalReconstructionTest(unittest.TestCase):
    def test_main_dispatches_to_realtime_pipeline(self):
        render_module=ModuleType("render_lightfield")
        render_module.main=Mock()
        with patch.dict("sys.modules",{"render_lightfield":render_module}):
            main()
        render_module.main.assert_called_once_with()

    def test_standalone_settings_are_loaded_from_config_with_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"config.yaml"
            config.write_text(
                "local_reconstruction:\n"
                "  input_file: data/input.npz\n"
                "  calibration_file: models/norm.npz\n"
                "  output_file: output/result.npz\n"
                "  lsmr_atol: 1.0e-8\n"
                "  lsmr_btol: 2.0e-8\n"
                "  lsmr_max_iterations: 321\n"
                "  zero_color_protection:\n"
                "    inner_radius: 0.02\n"
                "    outer_radius: 0.05\n"
                "  no_contact_constraints:\n"
                "    enabled: true\n"
                "    slope_confidence: 0.9\n"
                "    displacement_zero_lambda_per_mm2: 0.4\n",
                encoding="utf-8")
            settings=load_local_reconstruction_settings(
                config,output_override="override.npz",lsmr_atol_override=3e-9)
        self.assertEqual(settings.input_file,root/"data/input.npz")
        self.assertEqual(settings.calibration_file,root/"models/norm.npz")
        self.assertEqual(settings.output_file,Path("override.npz"))
        self.assertEqual(settings.lsmr_atol,3e-9)
        self.assertEqual(settings.lsmr_btol,2e-8)
        self.assertEqual(settings.lsmr_max_iterations,321)
        self.assertEqual(settings.zero_color_inner_radius,.02)
        self.assertEqual(settings.zero_color_outer_radius,.05)
        self.assertTrue(settings.no_contact_constraints_enabled)
        self.assertEqual(settings.trusted_no_contact_confidence,.9)
        self.assertEqual(settings.displacement_zero_lambda_per_mm2,.4)

    def test_standalone_output_defaults_next_to_cli_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"config.yaml"
            config.write_text(
                "local_reconstruction:\n"
                "  calibration_file: models/norm.npz\n",
                encoding="utf-8")
            settings=load_local_reconstruction_settings(
                config,input_override="captures/frame.npz")
        self.assertEqual(settings.input_file,Path("captures/frame.npz"))
        self.assertEqual(
            settings.output_file,
            Path("captures/frame_local_reconstruction.npz"))

    def test_standalone_selects_lut_for_configured_background_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"config.yaml"
            config.write_text(
                "lightfield:\n"
                "  background:\n"
                "    method: direct_fit\n"
                "    model_files:\n"
                "      physical_residual: models/physical.yaml\n"
                "      direct_fit: models/direct.yaml\n"
                "local_reconstruction:\n"
                "  calibration_files:\n"
                "    physical_residual: lut/physical.npz\n"
                "    direct_fit: lut/direct.npz\n",
                encoding="utf-8")
            settings=load_local_reconstruction_settings(
                config,input_override="captures/frame.npz")
        self.assertEqual(settings.calibration_file,root/"lut/direct.npz")

    def test_standalone_missing_input_points_to_realtime_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/"config.yaml"
            config.write_text(
                "local_reconstruction:\n"
                "  calibration_file: models/norm.npz\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"uv run recon.py"):
                load_local_reconstruction_settings(config)

    def test_full_resolution_color_is_bilinearly_sampled_at_native_uv(self):
        y,x=np.meshgrid(np.arange(6),np.arange(8),indexing="ij")
        image=np.stack([x,y,x+2*y],axis=-1).astype(np.float32)
        uv=np.asarray([[[1.5,2.5],[4.25,1.75]]],np.float32)
        sampled,valid=sample_full_resolution_residual(
            image,uv,np.ones(image.shape[:2],np.uint8)*255)
        np.testing.assert_allclose(sampled[0,0],[1.5,2.5,6.5],atol=1e-6)
        np.testing.assert_allclose(sampled[0,1],[4.25,1.75,7.75],atol=1e-6)
        np.testing.assert_array_equal(valid,np.ones((1,2),bool))

    def test_full_resolution_sampling_preserves_negative_channels(self):
        image=np.asarray(
            [[[-1.,.2,.3],[-.5,.4,.5]],[[.1,-.2,.6],[.3,-.4,.7]]],
            np.float32)
        sampled,valid=sample_full_resolution_residual(
            image,np.asarray([[[0.,0.],[1.,1.]]],np.float32))
        np.testing.assert_allclose(
            sampled,np.asarray([[[-1.,.2,.3],[.3,-.4,.7]]],np.float32))
        np.testing.assert_array_equal(valid,np.ones((1,2),bool))

    def test_trilinear_lut_propagates_linear_slopes(self):
        colors=np.asarray([[[.2,-.4,.7],[-.8,.6,-.2]]],np.float32)
        slopes,variance,confidence=lookup_slopes(
            colors,linear_slope_calibration())
        np.testing.assert_allclose(slopes,colors[...,:2],atol=1e-6)
        self.assertTrue(np.all(variance>=1e-4))
        self.assertTrue(np.all((confidence>0)&(confidence<=1)))

    def test_zero_color_protection_smoothly_releases_lut_slopes(self):
        colors=np.asarray([
            [0.,0.,0.],[.025,.01,0.],[.05,.01,0.],[.1,.01,0.]],
            np.float32)
        slopes,_,_=lookup_slopes(
            colors,linear_slope_calibration(),
            zero_color_inner_radius=0.,zero_color_outer_radius=.05)
        weights=np.asarray([0.,.5,1.,1.],np.float32)
        np.testing.assert_allclose(
            slopes,colors[:,:2]*weights[:,None],atol=1e-6)

    def test_jax_zero_color_protection_matches_cpu(self):
        colors=np.asarray([
            [.005,-.01,0.],[.025,.01,0.],[.04,-.02,.01]],np.float32)
        calibration=linear_slope_calibration()
        expected=lookup_slopes(
            colors,calibration,zero_color_inner_radius=.01,
            zero_color_outer_radius=.05)
        actual=lookup_slopes_jax(
            jnp.asarray(colors),jnp.asarray(calibration.slopes),
            jnp.asarray(calibration.variances),
            jnp.asarray(calibration.color_min),
            jnp.asarray(calibration.color_max),calibration.sigma_ref2,
            jnp.ones(colors.shape[:-1],jnp.bool_),
            zero_color_inner_radius=.01,zero_color_outer_radius=.05)
        for cpu,gpu in zip(expected,actual,strict=True):
            np.testing.assert_allclose(cpu,np.asarray(gpu),atol=1e-6)

    def test_startup_baseline_classifies_trusted_no_contact_and_guard(self):
        rng=np.random.default_rng(4)
        samples=rng.normal(0,.002,(10,21,21,3)).astype(np.float32)
        valid=np.ones(samples.shape[:3],bool)
        model=fit_no_contact_residual_model(
            samples,valid,minimum_channel_scale=.003)
        colors=model.center.copy()
        colors[10,10,0]+=6*model.channel_scale[0]
        trusted,score=classify_trusted_no_contact(
            colors,np.ones((21,21),bool),model,
            trusted_score_threshold=2.5,
            contact_guard_score_threshold=5.,
            contact_guard_radius_pixels=4,
            surface_edge_margin_pixels=2)
        self.assertTrue(trusted[5,5])
        self.assertFalse(trusted[10,10])
        self.assertFalse(trusted[10,7])
        self.assertFalse(trusted[0,0])
        self.assertGreater(score[10,10],5.)

        jax_trusted,jax_score=classify_trusted_no_contact_jax(
            jnp.asarray(colors),jnp.ones((21,21),jnp.bool_),
            jnp.asarray(model.center),jnp.asarray(model.channel_scale),
            jnp.asarray(model.valid_mask),trusted_score_threshold=2.5,
            contact_guard_score_threshold=5.,contact_guard_radius_pixels=4,
            surface_edge_margin_pixels=2)
        np.testing.assert_array_equal(trusted,np.asarray(jax_trusted))
        np.testing.assert_allclose(score,np.asarray(jax_score),atol=1e-6)

    def test_no_contact_constraint_config_is_parsed(self):
        settings=parse_no_contact_constraints({
            "no_contact_constraints":{
                "trusted_score_threshold":2.5,
                "contact_guard_score_threshold":5.,
                "contact_guard_radius_pixels":20,
                "surface_edge_margin_pixels":4,
                "slope_confidence":.9,
                "displacement_zero_lambda_per_mm2":.25,
            }})
        self.assertEqual(settings.trusted_score_threshold,2.5)
        self.assertEqual(settings.contact_guard_radius_pixels,20)
        self.assertEqual(settings.surface_edge_margin_pixels,4)
        self.assertEqual(settings.slope_confidence,.9)
        self.assertEqual(settings.displacement_zero_lambda_per_mm2,.25)

    def test_trusted_no_contact_forces_lut_slope_to_zero(self):
        colors=np.asarray([[[.2,-.1,0.],[.3,.15,0.]]],np.float32)
        trusted=np.asarray([[True,False]])
        slopes,_,confidence=lookup_slopes(
            colors,linear_slope_calibration(),
            trusted_no_contact_mask=trusted,
            trusted_no_contact_confidence=.9)
        np.testing.assert_array_equal(slopes[0,0],0)
        np.testing.assert_allclose(slopes[0,1],colors[0,1,:2],atol=1e-6)
        self.assertAlmostEqual(float(confidence[0,0]),.9,places=6)

    def test_flat_surface_recovers_known_zero_boundary_displacement(self):
        rows,columns=17,19
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float64),
            np.arange(columns,dtype=np.float64),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        expected=.18*np.sin(np.pi*y/(rows-1))*np.sin(np.pi*x/(columns-1))
        p=np.zeros_like(expected); q=np.zeros_like(expected)
        p[1:-1,1:-1]=(expected[1:-1,2:]-expected[1:-1,:-2])/2
        q[1:-1,1:-1]=(expected[2:,1:-1]-expected[:-2,1:-1])/2
        colors=np.stack([p,q,np.zeros_like(p)],axis=-1).astype(np.float32)
        result=reconstruct_local_surface(
            xyz,colors,linear_slope_calibration(),
            valid_mask=np.ones((rows,columns),bool),
            lsmr_atol=1e-11,lsmr_btol=1e-11)
        np.testing.assert_allclose(
            result.displacement[1:-1,1:-1],expected[1:-1,1:-1],atol=1e-4)
        np.testing.assert_allclose(result.displacement[[0,-1]],0,atol=1e-9)
        np.testing.assert_allclose(result.displacement[:,[0,-1]],0,atol=1e-9)
        np.testing.assert_allclose(
            result.xyz_out[...,2],100+expected,atol=1e-4)

    def test_calibration_round_trip(self):
        model=linear_slope_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"normal_lut.npz"
            model.save(
                path,sphere_radius_mm=np.asarray(5.,np.float32),
                residual_method=np.asarray("uniform"),
                background_method=np.asarray("direct_fit"),
                background_model_sha256=np.asarray("a"*64))
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(int(data["format_version"]),3)
                self.assertEqual(str(data["color_residual_mode"]),"signed")
                legacy={key:np.asarray(data[key]) for key in data.files
                        if key!="color_residual_mode"}
            legacy["format_version"]=np.asarray(1,np.int32)
            legacy_path=Path(directory)/"legacy_normal_lut.npz"
            np.savez_compressed(legacy_path,**legacy)
            with self.assertRaisesRegex(ValueError,"单边正色差"):
                NormalCalibration.load(legacy_path)
            loaded=NormalCalibration.load(path)
        np.testing.assert_array_equal(loaded.slopes,model.slopes)
        np.testing.assert_array_equal(loaded.variances,model.variances)
        self.assertEqual(loaded.residual_method,"uniform")
        self.assertEqual(loaded.background_method,"direct_fit")
        self.assertEqual(loaded.background_model_sha256,"a"*64)

    def test_normal_calibration_accepts_direct_fit_3_metadata(self):
        source=linear_slope_calibration()
        model=NormalCalibration(
            slopes=source.slopes,variances=source.variances,
            color_min=source.color_min,color_max=source.color_max,
            sigma_ref2=source.sigma_ref2,background_method="direct_fit_3",
            background_model_sha256="b"*64)
        self.assertEqual(model.background_method,"direct_fit_3")

    def test_zero_color_keeps_curved_reference_surface_unchanged(self):
        rows,columns=13,15
        y,angle=np.meshgrid(
            np.linspace(-4,4,rows),np.linspace(-.35,.35,columns),
            indexing="ij")
        radius=30.
        xyz=np.stack([
            radius*np.sin(angle),y,100+radius*(1-np.cos(angle))],axis=-1)
        result=reconstruct_local_surface(
            xyz,np.zeros_like(xyz,np.float32),linear_slope_calibration())
        np.testing.assert_allclose(result.displacement,0,atol=1e-8)
        np.testing.assert_allclose(result.xyz_out,xyz,atol=1e-6)

    def test_jax_lsmr_recovers_flat_surface_and_warm_starts(self):
        rows,columns=17,19
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        expected=.18*np.sin(np.pi*y/(rows-1))*np.sin(np.pi*x/(columns-1))
        p=np.zeros_like(expected); q=np.zeros_like(expected)
        p[1:-1,1:-1]=(expected[1:-1,2:]-expected[1:-1,:-2])/2
        q[1:-1,1:-1]=(expected[2:,1:-1]-expected[:-2,1:-1])/2
        colors=np.stack([p,q,np.zeros_like(p)],axis=-1)
        calibration=linear_slope_calibration()

        @jax.jit
        def solve(previous):
            return reconstruct_local_surface_jax(
                jnp.asarray(xyz),jnp.asarray(colors),
                jnp.asarray(calibration.slopes),
                jnp.asarray(calibration.variances),
                jnp.asarray(calibration.color_min),
                jnp.asarray(calibration.color_max),calibration.sigma_ref2,
                jnp.ones((rows,columns),jnp.bool_),previous,
                lsmr_atol=1e-7,lsmr_btol=1e-7,
                lsmr_max_iterations=5000)

        cold=solve(jnp.zeros((rows,columns),jnp.float32))
        warm=solve(cold[1])
        np.testing.assert_allclose(
            np.asarray(cold[1]),expected,atol=8e-4)
        self.assertLess(int(warm[11]),int(cold[11]))
        np.testing.assert_allclose(np.asarray(warm[1]),expected,atol=8e-4)

    def test_jax_lsmr_zero_target_clears_previous_frame(self):
        rows,columns=7,9
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        calibration=linear_slope_calibration()
        result=reconstruct_local_surface_jax(
            jnp.asarray(xyz),jnp.zeros_like(jnp.asarray(xyz)),
            jnp.asarray(calibration.slopes),
            jnp.asarray(calibration.variances),
            jnp.asarray(calibration.color_min),
            jnp.asarray(calibration.color_max),calibration.sigma_ref2,
            jnp.ones((rows,columns),jnp.bool_),
            jnp.ones((rows,columns),jnp.float32),
            lsmr_atol=1e-6,lsmr_btol=1e-6,
            lsmr_max_iterations=1000)
        np.testing.assert_array_equal(np.asarray(result[1]),0)

    def test_no_contact_prior_is_soft_and_keeps_full_surface_domain(self):
        rows,columns=17,19
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        colors=np.zeros_like(xyz,np.float32)
        colors[4:13,9:14,0]=.1
        trusted=np.zeros((rows,columns),bool)
        trusted[2:-2,2:8]=True
        calibration=linear_slope_calibration()
        cpu=reconstruct_local_surface(
            xyz,colors,calibration,trusted_no_contact_mask=trusted,
            displacement_zero_lambda_per_mm2=.25,
            lsmr_atol=1e-9,lsmr_btol=1e-9,lsmr_max_iterations=5000)
        gpu=reconstruct_local_surface_jax(
            jnp.asarray(xyz),jnp.asarray(colors),
            jnp.asarray(calibration.slopes),jnp.asarray(calibration.variances),
            jnp.asarray(calibration.color_min),jnp.asarray(calibration.color_max),
            calibration.sigma_ref2,jnp.ones((rows,columns),jnp.bool_),
            jnp.zeros((rows,columns),jnp.float32),
            trusted_no_contact_mask=jnp.asarray(trusted),
            displacement_zero_lambda_per_mm2=.25,
            lsmr_atol=1e-7,lsmr_btol=1e-7,lsmr_max_iterations=5000)
        np.testing.assert_array_equal(cpu.slopes[trusted],0)
        np.testing.assert_array_equal(np.asarray(gpu[4])[trusted],0)
        np.testing.assert_array_equal(np.asarray(gpu[14]),trusted)
        # D0 仍只是曲面几何外边界，统计掩膜不裁切未知量域。
        self.assertEqual(np.count_nonzero(cpu.boundary_mask),
                         2*rows+2*(columns-2))
        np.testing.assert_allclose(
            cpu.displacement,np.asarray(gpu[1]),atol=2e-3)
        self.assertGreater(np.max(np.abs(cpu.displacement)),1e-3)

    def test_all_trusted_no_contact_clears_dirty_color_and_previous(self):
        rows,columns=7,9
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        calibration=linear_slope_calibration()
        result=reconstruct_local_surface_jax(
            jnp.asarray(xyz),jnp.ones_like(jnp.asarray(xyz))*.3,
            jnp.asarray(calibration.slopes),jnp.asarray(calibration.variances),
            jnp.asarray(calibration.color_min),jnp.asarray(calibration.color_max),
            calibration.sigma_ref2,jnp.ones((rows,columns),jnp.bool_),
            jnp.ones((rows,columns),jnp.float32),
            trusted_no_contact_mask=jnp.ones((rows,columns),jnp.bool_),
            displacement_zero_lambda_per_mm2=.25,
            lsmr_atol=1e-6,lsmr_btol=1e-6,lsmr_max_iterations=100,
            spectral_initialization_iterations=2,
            linear_solver="spectral_pcg")
        np.testing.assert_array_equal(np.asarray(result[1]),0)
        np.testing.assert_array_equal(np.asarray(result[4]),0)
        np.testing.assert_array_equal(np.asarray(result[14]),True)

    def test_spectral_initialization_handles_a_cold_frame_change(self):
        rows,columns=65,67
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,np.ones_like(x)*100],axis=-1)
        expected=.8*np.sin(np.pi*y/(rows-1))*np.sin(
            np.pi*x/(columns-1))
        p=np.zeros_like(expected); q=np.zeros_like(expected)
        p[1:-1,1:-1]=(expected[1:-1,2:]-expected[1:-1,:-2])/2
        q[1:-1,1:-1]=(expected[2:,1:-1]-expected[:-2,1:-1])/2
        colors=np.stack([p,q,np.zeros_like(p)],axis=-1)
        calibration=linear_slope_calibration()

        @jax.jit
        def solve(previous):
            return reconstruct_local_surface_jax(
                jnp.asarray(xyz),jnp.asarray(colors),
                jnp.asarray(calibration.slopes),
                jnp.asarray(calibration.variances),
                jnp.asarray(calibration.color_min),
                jnp.asarray(calibration.color_max),calibration.sigma_ref2,
                jnp.ones((rows,columns),jnp.bool_),previous,
                lsmr_atol=1e-6,lsmr_btol=1e-6,
                lsmr_max_iterations=30,
                spectral_initialization_iterations=2,
                linear_solver="spectral_pcg")

        cold=solve(jnp.zeros((rows,columns),jnp.float32))
        wrong_history=solve(jnp.full((rows,columns),-1.,jnp.float32))
        np.testing.assert_allclose(
            np.asarray(cold[1]),expected,atol=5e-4)
        np.testing.assert_allclose(
            np.asarray(wrong_history[1]),np.asarray(cold[1]),atol=1e-7)
        self.assertLessEqual(int(cold[11]),30)


if __name__=="__main__":
    unittest.main()
