import tempfile
import unittest
from pathlib import Path

from utils.config import (ConfigError,load_config,parse_background_method,
                          parse_direct_fit_3_config,
                          parse_direct_fit_s_config,
                          parse_reconstruction_config,
                          resolve_background_model_path,resolve_method_path)


class ReconstructionGridConfigTest(unittest.TestCase):
    def test_config_extends_recursively_merges_nested_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"base.yaml").write_text(
                "top:\n  keep: 1\n  replace: base\n",encoding="utf-8")
            (root/"child.yaml").write_text(
                "extends: base.yaml\ntop:\n  replace: child\nadded: true\n",
                encoding="utf-8")
            loaded=load_config(root/"child.yaml")
        self.assertEqual(loaded,{"top":{"keep":1,"replace":"child"},
                                 "added":True})

    def test_repository_configs_isolate_physical_residual_settings(self):
        root=Path(__file__).resolve().parents[1]
        direct=load_config(root/"config.yaml")
        physical=load_config(root/"config_physical_residual.yaml")
        self.assertEqual(parse_background_method(direct["lightfield"]),
                         "direct_fit_s")
        self.assertEqual(parse_background_method(physical["lightfield"]),
                         "physical_residual")
        self.assertNotIn("light_source_layout",direct["lightfield"])
        self.assertNotIn("integration_nodes",direct["lightfield"])
        self.assertIn("light_source_layout",physical["lightfield"])
        self.assertIn("integration_nodes",physical["lightfield"])
        self.assertNotIn("delta_initial_mm",direct["lightfield"]["calibration"])

    def test_removed_direct_fit_method_and_paths_are_rejected(self):
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
        with self.assertRaises(ConfigError):
            parse_background_method(lightfield)
        with self.assertRaises(ConfigError):
            resolve_method_path(
                local,method="direct_fit",mapping_key="calibration_files",
                legacy_key="calibration_file",base=base,
                section_name="local_reconstruction")

    def test_direct_fit_3_selects_own_paths_and_settings(self):
        lightfield={
            "background":{
                "method":"direct_fit_3",
                "model_files":{
                    "physical_residual":"models/physical.yaml",
                    "direct_fit_3":"models/direct_3.yaml",
                },
            },
            "direct_fit_3":{"neural_field":{"decoder_width":77}},
        }
        local={"calibration_files":{
            "physical_residual":"lut/physical.npz",
            "direct_fit_3":"lut/direct_3.npz",
        }}
        base=Path("/tmp/config-base")
        method=parse_background_method(lightfield)
        self.assertEqual(method,"direct_fit_3")
        self.assertEqual(parse_direct_fit_3_config(lightfield).decoder_width,77)
        self.assertEqual(
            resolve_background_model_path(lightfield,method=method,base=base),
            base/"models/direct_3.yaml")
        self.assertEqual(resolve_method_path(
            local,method=method,mapping_key="calibration_files",
            legacy_key="calibration_file",base=base,
            section_name="local_reconstruction"),base/"lut/direct_3.npz")

    def test_removed_geometry_cache_method_is_rejected(self):
        lightfield={"background":{"method":"geometry_cache"}}
        with self.assertRaises(ConfigError):
            parse_background_method(lightfield)

    def test_legacy_background_config_defaults_to_physical_method(self):
        lightfield={"model_file":"models/legacy.yaml"}
        method=parse_background_method(lightfield)
        self.assertEqual(method,"physical_residual")
        self.assertEqual(resolve_background_model_path(
            lightfield,method=method,base="/tmp"),
            Path("/tmp/models/legacy.yaml"))

    def test_direct_fit_network_and_geometry_config_are_validated(self):
        parsed=parse_direct_fit_3_config({})
        self.assertEqual(parsed.coordinate_frequencies,(1.,2.,4.,8.,16.,32.))
        self.assertEqual(parsed.geometry_latent_dimensions,96)
        self.assertEqual(parsed.geometry_pca_dimensions,32)
        self.assertEqual(parsed.decoder_layers,5)
        self.assertEqual(parsed.base_huber_iterations,5)
        self.assertEqual(parsed.adaptive_channel_weight_strength,0.)
        self.assertEqual(parsed.spatial_difference_weight,1.)
        self.assertEqual(parsed.spatial_difference_points_per_frame,1024)
        self.assertAlmostEqual(parsed.geometry_difference_weight,.25)
        self.assertEqual(parsed.geometry_difference_neighbor_count,16)
        self.assertEqual(parsed.geometry_difference_points_per_pair,512)
        self.assertEqual(parsed.validation_interval,100)
        self.assertEqual(parsed.validation_frame_count,64)
        self.assertEqual(parsed.validation_points_per_frame,512)
        self.assertEqual(parsed.early_stopping_patience,10)
        self.assertEqual(parsed.early_stopping_min_steps,1500)
        self.assertAlmostEqual(parsed.early_stopping_min_delta,5e-5)
        self.assertEqual(parsed.sample_erode_pixels,2)
        self.assertAlmostEqual(parsed.session_correction_max_deviation,.15)
        configured=parse_direct_fit_3_config({"direct_fit_3":{
            "neural_field":{
                "frequencies":[1,3],"geometry_descriptor_rows":12,
                "geometry_encoder_width":48,"geometry_latent_dimensions":7,
                "geometry_pca_dimensions":5,
                "decoder_width":72,"frame_batch_size":3,
                "base_huber_iterations":3,
                "adaptive_channel_weight_strength":.6,
                "spatial_difference_weight":.6,
                "spatial_difference_points_per_frame":144,
                "geometry_difference_weight":.4,
                "geometry_difference_neighbor_count":6,
                "geometry_difference_points_per_pair":96,
                "validation_interval":20,"validation_frame_count":9,
                "validation_points_per_frame":128,
                "early_stopping_patience":4,"early_stopping_min_steps":800,
                "early_stopping_min_delta":.0002},
            "sample_filter":{"erode_pixels":1},
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
        self.assertEqual(configured.spatial_difference_points_per_frame,144)
        self.assertAlmostEqual(configured.geometry_difference_weight,.4)
        self.assertEqual(configured.geometry_difference_neighbor_count,6)
        self.assertEqual(configured.validation_interval,20)
        self.assertEqual(configured.validation_frame_count,9)
        self.assertEqual(configured.early_stopping_patience,4)
        self.assertEqual(configured.early_stopping_min_steps,800)
        self.assertEqual(configured.sample_erode_pixels,1)
        with self.assertRaisesRegex(ConfigError,"frequencies"):
            parse_direct_fit_3_config({"direct_fit_3":{
                "neural_field":{"frequencies":[1,0]}}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_3_config({"direct_fit_3":{"b_coefficient_bounds":[0,1]}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_3_config({"direct_fit_3":{
                "neural_field":{"smooth_lambda":0}}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_3_config({"direct_fit_3":{
                "sample_filter":{"saturation_threshold":256}}})
        with self.assertRaisesRegex(ConfigError,"不能大于 steps"):
            parse_direct_fit_3_config({"direct_fit_3":{"neural_field":{
                "steps":10,"early_stopping_min_steps":11}}})
        with self.assertRaisesRegex(ConfigError,"不大于 1"):
            parse_direct_fit_3_config({"direct_fit_3":{"neural_field":{
                "adaptive_channel_weight_strength":1.1}}})

    def test_direct_fit_s_sequence_and_runtime_config_are_validated(self):
        parsed=parse_direct_fit_s_config({"direct_fit_s":{
            "neural_field":{"gru_hidden_dimensions":128,
                            "appearance_descriptor_rows":16,
                            "appearance_descriptor_columns":8,
                            "appearance_pca_dimensions":24},
            "measurement":{"full_visibility_threshold":.96,
                           "minimum_visible_fraction":.3,
                           "prior_logit_limit":3.5},
            "training":{"clip_length":8,"warmup_frames":12,
                        "clip_batch_size":2,
                        "color_train_steps":100,
                        "color_checkpoint_interval":20,
                        "color_monitor_batch_count":3,
                        "synthetic_min_visible_fraction":.4},
            "loss":{"warp_distillation_weight":2.5,
                    "frame_quality_full_confidence":.3,
                    "minimum_frame_quality_weight":.1,
                    "appearance_supervision_weight":.02,
                    "appearance_score_clip":2.5},
            "sequence":{"maximum_sequence_gap":5}}})
        self.assertEqual(parsed.gru_hidden_dimensions,128)
        self.assertEqual(parsed.appearance_pca_dimensions,24)
        self.assertEqual(parsed.clip_length,8)
        self.assertEqual(parsed.warmup_frames,12)
        self.assertEqual(parsed.color_checkpoint_interval,20)
        self.assertEqual(parsed.color_monitor_batch_count,3)
        self.assertAlmostEqual(parsed.measurement_prior_logit_limit,3.5)
        self.assertAlmostEqual(parsed.warp_distillation_weight,2.5)
        self.assertAlmostEqual(parsed.frame_quality_full_confidence,.3)
        self.assertAlmostEqual(parsed.minimum_frame_quality_weight,.1)
        self.assertAlmostEqual(parsed.appearance_supervision_weight,.02)
        self.assertAlmostEqual(parsed.appearance_score_clip,2.5)
        self.assertEqual(parsed.maximum_sequence_gap,5)
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "runtime":{"online_gain_bias_enabled":False}}})
        with self.assertRaisesRegex(ConfigError,"minimum_visible_fraction"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "measurement":{"full_visibility_threshold":.8,
                               "minimum_visible_fraction":.9}}})
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "loss":{"minimum_crop_weight":.02}}})
        with self.assertRaisesRegex(ConfigError,"warp_distillation_weight"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "loss":{"warp_distillation_weight":-1}}})
        with self.assertRaisesRegex(ConfigError,"帧质量"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "loss":{"minimum_frame_quality_weight":1.1}}})
        with self.assertRaisesRegex(ConfigError,"color_checkpoint_interval"):
            parse_direct_fit_s_config({"direct_fit_s":{
                "training":{"color_train_steps":10,
                            "color_checkpoint_interval":11}}})

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
        self.assertAlmostEqual(result.side_edge_exclusion_ratio,.02)
        self.assertAlmostEqual(result.material_surface.width_mm,22.)
        self.assertAlmostEqual(result.material_surface.length_mm,55.)
        self.assertEqual(result.material_surface.s_zero_endpoint,"image_top")
        self.assertAlmostEqual(result.s1,11.)
        self.assertAlmostEqual(result.s2,-11.)
        self.assertAlmostEqual(
            result.material_surface.calibration_maximum_rms_px,4.)
        self.assertAlmostEqual(
            result.material_surface.calibration_minimum_confidence,.05)

    def test_material_dimensions_and_s_zero_endpoint_are_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            configured=parse_reconstruction_config({
                "calibration_file":str(calibration),
                "material_surface":{
                    "width_mm":24.,
                    "length_mm":60.,
                    "s_zero_endpoint":"image_bottom",
                    "calibration_maximum_rms_px":3.5,
                    "calibration_minimum_confidence":.1,
                },
            },config_path=Path(directory)/"config.yaml")
        self.assertAlmostEqual(configured.material_surface.width_mm,24.)
        self.assertAlmostEqual(configured.material_surface.length_mm,60.)
        self.assertEqual(
            configured.material_surface.s_zero_endpoint,"image_bottom")
        self.assertAlmostEqual(configured.s1,12.)
        self.assertAlmostEqual(configured.s2,-12.)
        self.assertAlmostEqual(configured.material_template.st[0,0,0],1.)
        self.assertAlmostEqual(configured.material_template.st[-1,0,0],0.)
        self.assertAlmostEqual(
            configured.material_surface.calibration_maximum_rms_px,3.5)
        self.assertAlmostEqual(
            configured.material_surface.calibration_minimum_confidence,.1)

    def test_material_dimensions_and_s_zero_endpoint_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            for values,pattern in (
                ({"width_mm":0.},"width_mm"),
                ({"length_mm":-1.},"length_mm"),
                ({"s_zero_endpoint":"left"},"s_zero_endpoint"),
            ):
                with self.assertRaisesRegex(ConfigError,pattern):
                    parse_reconstruction_config({
                        "calibration_file":str(calibration),
                        "material_surface":values,
                    },config_path=Path(directory)/"config.yaml")

    def test_material_update_failure_thresholds_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            for values,pattern in (
                ({"calibration_maximum_rms_px":0.},"正数"),
                ({"calibration_minimum_confidence":1e-7},"confidence_floor"),
                ({"calibration_minimum_confidence":1.},"confidence_floor"),
            ):
                with self.assertRaisesRegex(ConfigError,pattern):
                    parse_reconstruction_config({
                        "calibration_file":str(calibration),
                        "material_surface":values,
                    },config_path=Path(directory)/"config.yaml")

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

    def test_side_edge_exclusion_ratio_is_configurable_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            configured=parse_reconstruction_config({
                "calibration_file":str(calibration),
                "side_edge_exclusion_ratio":.075,
            },config_path=Path(directory)/"config.yaml")
            self.assertAlmostEqual(configured.side_edge_exclusion_ratio,.075)
            for invalid in (-.01,.5,True):
                with self.assertRaisesRegex(ConfigError,"位于"):
                    parse_reconstruction_config({
                        "calibration_file":str(calibration),
                        "side_edge_exclusion_ratio":invalid,
                    },config_path=Path(directory)/"config.yaml")

    def test_removed_temporal_prior_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(ConfigError,"未知字段.*temporal_prior"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "temporal_prior":{"enabled":True},
                },config_path=Path(directory)/"config.yaml")

    def test_removed_length_based_full_observation_switch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            with self.assertRaisesRegex(
                    ConfigError,"未知字段.*full_observation"):
                parse_reconstruction_config({
                    "calibration_file":str(calibration),
                    "material_surface":{
                        "full_observation_length_tolerance_ratio":.03},
                },config_path=Path(directory)/"config.yaml")

    def test_removed_endpoint_matching_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration=self.calibration(directory)
            for name,value in (
                    ("startup_length_tolerance_ratio",.05),
                    ("temporal_blend",.2)):
                with self.assertRaisesRegex(ConfigError,f"未知字段.*{name}"):
                    parse_reconstruction_config({
                        "calibration_file":str(calibration),
                        "material_surface":{name:value},
                    },config_path=Path(directory)/"config.yaml")


if __name__=="__main__":
    unittest.main()
