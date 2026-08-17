import tempfile
import unittest
from pathlib import Path

from utils.config import (ConfigError,parse_background_method,
                          parse_direct_fit_config,
                          parse_geometry_cache_config,
                          parse_reconstruction_config,
                          resolve_background_model_path,resolve_method_path)


class ReconstructionGridConfigTest(unittest.TestCase):
    def test_background_method_selects_separate_model_and_lut_paths(self):
        lightfield={
            "background":{
                "method":"direct_fit",
                "model_files":{
                    "physical_residual":"models/physical.yaml",
                    "direct_fit":"models/direct.yaml",
                },
            },
        }
        local={"calibration_files":{
            "physical_residual":"lut/physical.npz",
            "direct_fit":"lut/direct.npz",
        }}
        base=Path("/tmp/config-base")
        method=parse_background_method(lightfield)
        self.assertEqual(method,"direct_fit")
        self.assertEqual(
            resolve_background_model_path(lightfield,method=method,base=base),
            base/"models/direct.yaml")
        self.assertEqual(resolve_method_path(
            local,method=method,mapping_key="calibration_files",
            legacy_key="calibration_file",base=base,
            section_name="local_reconstruction"),base/"lut/direct.npz")

    def test_direct_fit_3_selects_own_paths_and_can_reuse_direct_settings(self):
        lightfield={
            "background":{
                "method":"direct_fit_3",
                "model_files":{
                    "physical_residual":"models/physical.yaml",
                    "direct_fit":"models/direct.yaml",
                    "direct_fit_3":"models/direct_3.yaml",
                },
            },
            "direct_fit":{"neural_field":{"decoder_width":77}},
        }
        local={"calibration_files":{
            "physical_residual":"lut/physical.npz",
            "direct_fit":"lut/direct.npz",
            "direct_fit_3":"lut/direct_3.npz",
        }}
        base=Path("/tmp/config-base")
        method=parse_background_method(lightfield)
        self.assertEqual(method,"direct_fit_3")
        self.assertEqual(parse_direct_fit_config(lightfield).decoder_width,77)
        self.assertEqual(
            resolve_background_model_path(lightfield,method=method,base=base),
            base/"models/direct_3.yaml")
        self.assertEqual(resolve_method_path(
            local,method=method,mapping_key="calibration_files",
            legacy_key="calibration_file",base=base,
            section_name="local_reconstruction"),base/"lut/direct_3.npz")

    def test_geometry_cache_selects_independent_paths_and_parameters(self):
        lightfield={
            "background":{"method":"geometry_cache","model_files":{
                "geometry_cache":"models/cache.yaml"}},
            "geometry_cache":{
                "descriptor":{"curve_coefficients":10,"pca_dimensions":6,
                              "huber_delta_mm":.3},
                "anchors":{"count":12,"neighbor_count":5,
                           "interpolation_neighbor_count":3,
                           "interpolation_distance_power":1.5,
                           "interpolation_distance_epsilon":.002,
                           "background_huber_delta":.03,
                           "background_huber_iterations":4,
                           "fit_batch_size":3},
                "sample_filter":{"saturation_threshold":254,
                                 "erode_pixels":2},
                "session_correction_max_deviation":.2}}
        parsed=parse_geometry_cache_config(lightfield)
        self.assertEqual(parse_background_method(lightfield),"geometry_cache")
        self.assertEqual(parsed.anchor_count,12)
        self.assertEqual(parsed.anchor_neighbor_count,5)
        self.assertEqual(parsed.interpolation_neighbor_count,3)
        self.assertEqual(parsed.descriptor_curve_coefficients,10)
        self.assertAlmostEqual(parsed.session_correction_max_deviation,.2)
        self.assertEqual(resolve_background_model_path(
            lightfield,method="geometry_cache",base="/tmp"),
            Path("/tmp/models/cache.yaml"))

    def test_legacy_background_config_defaults_to_physical_method(self):
        lightfield={"model_file":"models/legacy.yaml"}
        method=parse_background_method(lightfield)
        self.assertEqual(method,"physical_residual")
        self.assertEqual(resolve_background_model_path(
            lightfield,method=method,base="/tmp"),
            Path("/tmp/models/legacy.yaml"))

    def test_direct_fit_network_and_geometry_config_are_validated(self):
        parsed=parse_direct_fit_config({})
        self.assertEqual(parsed.coordinate_frequencies,(1.,2.,4.,8.,16.,32.))
        self.assertEqual(parsed.geometry_latent_dimensions,96)
        self.assertEqual(parsed.geometry_pca_dimensions,32)
        self.assertEqual(parsed.decoder_layers,5)
        self.assertEqual(parsed.base_huber_iterations,5)
        self.assertEqual(parsed.adaptive_channel_weight_strength,0.)
        self.assertEqual(parsed.spatial_difference_weight,1.)
        self.assertEqual(parsed.spatial_difference_validation_weight,1.)
        self.assertEqual(parsed.spatial_difference_points_per_frame,1024)
        self.assertAlmostEqual(parsed.geometry_difference_weight,.25)
        self.assertAlmostEqual(
            parsed.geometry_difference_validation_weight,.25)
        self.assertEqual(parsed.geometry_difference_neighbor_count,16)
        self.assertEqual(parsed.geometry_difference_points_per_pair,512)
        self.assertEqual(parsed.validation_interval,100)
        self.assertEqual(parsed.validation_frame_count,64)
        self.assertEqual(parsed.validation_points_per_frame,512)
        self.assertEqual(parsed.early_stopping_patience,10)
        self.assertEqual(parsed.early_stopping_min_steps,1500)
        self.assertAlmostEqual(parsed.early_stopping_min_delta,5e-5)
        self.assertEqual(parsed.sample_saturation_threshold,255)
        self.assertEqual(parsed.sample_erode_pixels,2)
        self.assertAlmostEqual(parsed.session_correction_max_deviation,.15)
        configured=parse_direct_fit_config({"direct_fit":{
            "neural_field":{
                "frequencies":[1,3],"geometry_descriptor_rows":12,
                "geometry_encoder_width":48,"geometry_latent_dimensions":7,
                "geometry_pca_dimensions":5,
                "decoder_width":72,"frame_batch_size":3,
                "base_huber_iterations":3,
                "adaptive_channel_weight_strength":.6,
                "spatial_difference_weight":.6,
                "spatial_difference_validation_weight":.8,
                "spatial_difference_points_per_frame":144,
                "geometry_difference_weight":.4,
                "geometry_difference_validation_weight":.7,
                "geometry_difference_neighbor_count":6,
                "geometry_difference_points_per_pair":96,
                "validation_interval":20,"validation_frame_count":9,
                "validation_points_per_frame":128,
                "early_stopping_patience":4,"early_stopping_min_steps":800,
                "early_stopping_min_delta":.0002},
            "sample_filter":{"saturation_threshold":253,"erode_pixels":1},
            "session_correction_max_deviation":.08,
        }})
        self.assertEqual(configured.coordinate_frequencies,(1.,3.))
        self.assertEqual(configured.geometry_encoder_width,48)
        self.assertEqual(configured.geometry_descriptor_rows,12)
        self.assertEqual(configured.geometry_latent_dimensions,7)
        self.assertEqual(configured.geometry_pca_dimensions,5)
        self.assertEqual(configured.decoder_width,72)
        self.assertEqual(configured.base_huber_iterations,3)
        self.assertAlmostEqual(configured.adaptive_channel_weight_strength,.6)
        self.assertAlmostEqual(configured.spatial_difference_weight,.6)
        self.assertAlmostEqual(
            configured.spatial_difference_validation_weight,.8)
        self.assertEqual(configured.spatial_difference_points_per_frame,144)
        self.assertAlmostEqual(configured.geometry_difference_weight,.4)
        self.assertAlmostEqual(
            configured.geometry_difference_validation_weight,.7)
        self.assertEqual(configured.geometry_difference_neighbor_count,6)
        self.assertEqual(configured.validation_interval,20)
        self.assertEqual(configured.validation_frame_count,9)
        self.assertEqual(configured.early_stopping_patience,4)
        self.assertEqual(configured.early_stopping_min_steps,800)
        self.assertEqual(configured.sample_saturation_threshold,253)
        self.assertEqual(configured.sample_erode_pixels,1)
        with self.assertRaisesRegex(ConfigError,"frequencies"):
            parse_direct_fit_config({"direct_fit":{
                "neural_field":{"frequencies":[1,0]}}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_config({"direct_fit":{"b_coefficient_bounds":[0,1]}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_config({"direct_fit":{
                "neural_field":{"smooth_lambda":0}}})
        with self.assertRaisesRegex(ConfigError,"不大于 255"):
            parse_direct_fit_config({"direct_fit":{
                "sample_filter":{"saturation_threshold":256}}})
        with self.assertRaisesRegex(ConfigError,"不能大于 steps"):
            parse_direct_fit_config({"direct_fit":{"neural_field":{
                "steps":10,"early_stopping_min_steps":11}}})
        with self.assertRaisesRegex(ConfigError,"不大于 1"):
            parse_direct_fit_config({"direct_fit":{"neural_field":{
                "adaptive_channel_weight_strength":1.1}}})

    def calibration(self,directory: str) -> Path:
        path=Path(directory)/"camera.yaml"
        path.write_text(
            "camera_matrix: [[400, 0, 320], [0, 400, 240], [0, 0, 1]]\n"
            "distortion_coefficients: [0, 0, 0, 0, 0]\n",encoding="utf-8")
        return path

    def test_grid_sizes_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            result=parse_reconstruction_config({
                "calibration_file":str(calibration),
                "geometry_grid":{"rows":80,"columns":12},
                "lightfield_grid":{"rows":120,"columns":52},
                "observation_grid":{"rows":360,"columns":102},
                "residual_coefficient_grid":{"rows":32,"columns":16},
                "residual_texture_grid":{"rows":256,"columns":128},
                "curve_convexity":"increasing",
            },config_path=Path(directory)/"config.yaml")
        self.assertEqual((result.geometry_rows,result.geometry_columns),(80,12))
        self.assertEqual((result.sample_count,result.pair_fill_count),(80,10))
        self.assertEqual((result.lightfield_rows,result.lightfield_columns),(120,52))
        self.assertEqual((result.observation_rows,result.observation_columns),(360,102))
        self.assertEqual(
            (result.residual_coefficient_rows,result.residual_coefficient_columns),
            (32,16))
        self.assertEqual(
            (result.residual_texture_rows,result.residual_texture_columns),
            (256,128))
        self.assertEqual(result.curve_convexity,"increasing")

    def test_legacy_grid_fields_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            result=parse_reconstruction_config({
                "calibration_file":str(calibration),
                "sample_count":24,"pair_fill_count":8,
            },config_path=Path(directory)/"config.yaml")
        self.assertEqual((result.geometry_rows,result.geometry_columns),(24,10))
        self.assertEqual((result.lightfield_rows,result.lightfield_columns),(24,10))
        self.assertEqual((result.observation_rows,result.observation_columns),(24,10))
        self.assertEqual(
            (result.residual_coefficient_rows,result.residual_coefficient_columns),
            (24,10))
        self.assertEqual(
            (result.residual_texture_rows,result.residual_texture_columns),
            (256,128))

    def test_grid_requires_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(ConfigError,"同时配置 rows 和 columns"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "geometry_grid":{"rows":80},
                },config_path=Path(directory)/"config.yaml")

    def test_observation_grid_cannot_be_smaller_than_coefficient_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(
                    ConfigError,"observation_grid 不能小于"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "observation_grid":{"rows":20,"columns":10},
                    "residual_coefficient_grid":{"rows":24,"columns":12},
                },config_path=Path(directory)/"config.yaml")

    def test_curve_convexity_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(ConfigError,"curve_convexity 必须"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "curve_convexity":"sometimes",
                },config_path=Path(directory)/"config.yaml")

    def test_removed_temporal_prior_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(ConfigError,"未知字段.*temporal_prior"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "temporal_prior":{"enabled":True},
                },config_path=Path(directory)/"config.yaml")


if __name__=="__main__":
    unittest.main()
