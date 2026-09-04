import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cv2
import jax
import numpy as np
from calibrate_lightfield import (_iter_physical_batch_indices,
                                  _cached_reconstruction_failure,
                                  _cached_observation_pose,
                                  _reconstruction_failure_path,
                                  _refine_calibration_masks,
                                  _write_reconstruction_failure,
                                  _write_material_initialization_failure,
                                  _observation_metadata,
                                  _split_calibration_indices,
                                  extract_video_frames,
                                  reconstruct_all_observations,
                                  save_calibration_observation)
from utils.material_surface import MaterialSurfaceState
from utils.jax_reconstruction import MATERIAL_UPDATE_FAILURE_NAMES
from utils.process import build_reconstruction_point_set
from utils.surface_mask import MaskRefineConfig,refine_mask

class LightFieldCalibrationSampleTest(unittest.TestCase):
    def test_calibration_mask_refine_exactly_matches_sam_preview(self):
        raw=np.zeros((1,100,120),np.uint8)
        cv2.fillPoly(
            raw[0],[np.asarray([[18,8],[102,8],[84,72],[36,72]],np.int32)],1)
        raw[0,1,1]=1
        raw=raw.astype(bool)
        tensor=MagicMock()
        tensor.detach.return_value.cpu.return_value=raw
        config=MaskRefineConfig(enabled=True)
        actual=_refine_calibration_masks(tensor,config)
        expected=refine_mask(raw[0],config)
        np.testing.assert_array_equal(actual[0],expected)
        count,_=cv2.connectedComponents(actual[0].astype(np.uint8))
        self.assertEqual(count-1,1)

    def test_validation_split_seeds_images_and_spans_video_timeline(self):
        paths=[Path(f"image_{index}.png") for index in range(5)]
        paths += [Path(f"video_001_clip_frame_{index:08d}.png")
                  for index in range(6)]
        training,validation=_split_calibration_indices(
            paths,independent_image_count=5,validation_fraction=.34,seed=11)
        repeated=_split_calibration_indices(
            paths,independent_image_count=5,validation_fraction=.34,seed=11)
        np.testing.assert_array_equal(training,repeated[0])
        np.testing.assert_array_equal(validation,repeated[1])
        # 视频验证帧均匀覆盖时间轴；训练仍包含首尾弯曲状态。
        self.assertEqual(set(validation[-2:]),{6,9})
        self.assertIn(5,training)
        self.assertIn(10,training)
        self.assertGreaterEqual(training.size,2)

    def test_zero_validation_fraction_keeps_all_samples_for_training(self):
        paths=[Path(f"image_{index}.png") for index in range(4)]
        training,validation=_split_calibration_indices(paths,4,0.,0)
        np.testing.assert_array_equal(training,np.arange(4))
        self.assertEqual(validation.size,0)

    def test_physical_batches_cover_each_epoch_without_dropping_tail(self):
        batches=list(_iter_physical_batch_indices(
            sample_count=10,batch_size=3,update_count=4,seed=7))
        self.assertEqual([epoch for epoch,_ in batches],[1,1,1,1])
        self.assertEqual([len(indices) for _,indices in batches],[3,3,3,1])
        np.testing.assert_array_equal(
            np.sort(np.concatenate([indices for _,indices in batches])),
            np.arange(10))

    def test_physical_batch_shuffle_is_seeded_and_continues_next_epoch(self):
        first=list(_iter_physical_batch_indices(7,4,4,19))
        second=list(_iter_physical_batch_indices(7,4,4,19))
        self.assertEqual([epoch for epoch,_ in first],[1,1,2,2])
        for (_,left),(_,right) in zip(first,second,strict=True):
            np.testing.assert_array_equal(left,right)

    def test_video_is_read_in_order_and_selected_frames_are_saved_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frames=[np.full((8,10,3),value,np.uint8) for value in range(5)]
            capture=MagicMock()
            capture.isOpened.return_value=True
            capture.read.side_effect=[*((True,frame) for frame in frames),(False,None)]
            with patch("calibrate_lightfield.cv2.VideoCapture",return_value=capture):
                paths=extract_video_frames(
                    [root/"input.mp4"],root/"video_frames",frame_step=2)

            self.assertEqual(len(paths),3)
            self.assertIn("frame_00000000",paths[0].name)
            self.assertIn("frame_00000002",paths[1].name)
            self.assertIn("frame_00000004",paths[2].name)
            self.assertEqual(int(cv2.imread(str(paths[1]))[0,0,0]),2)
            capture.release.assert_called_once()

    def test_video_frame_limit_stops_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            capture=MagicMock()
            capture.isOpened.return_value=True
            capture.read.side_effect=[
                (True,np.zeros((4,4,3),np.uint8)) for _ in range(5)
            ]
            with patch("calibrate_lightfield.cv2.VideoCapture",return_value=capture):
                paths=extract_video_frames(
                    [root/"input.mp4"],root/"video_frames",
                    max_frames_per_file=2)
            self.assertEqual(len(paths),2)
            self.assertEqual(capture.read.call_count,2)
            capture.release.assert_called_once()

    def test_existing_video_frames_are_reused_without_redecoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frames_dir=root/"video_frames"
            frames_dir.mkdir()
            existing=frames_dir/"video_001_input_frame_00000000.png"
            cv2.imwrite(str(existing),np.full((4,4,3),7,np.uint8))
            capture=MagicMock()
            with patch("calibrate_lightfield.cv2.VideoCapture",return_value=capture):
                paths=extract_video_frames(
                    [root/"input.mp4"],frames_dir,reuse_existing=True)
            self.assertEqual(paths,[existing])
            capture.isOpened.assert_not_called()

    def test_each_image_becomes_one_spatial_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=root/"frame.png"
            frame=np.full((480,640,3),128,np.uint8)
            point_set,_,_=build_reconstruction_point_set(
                np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
                np.asarray([[1.,0.,100.],[1.,1.,100.]]),
                np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
                np.zeros(5),np.zeros(3),0.,n_fill=1)
            output=save_calibration_observation(image,frame,point_set,root/"samples",root/"maps")
            with np.load(output) as data:
                self.assertEqual(data["xyz"].shape,(2,3,3)); self.assertEqual(data["rgb"].shape,(2,3,3))
                self.assertEqual(data["st"].shape,(2,3,2))
                self.assertEqual(data["camera_depth"].shape,(2,3))
                np.testing.assert_array_equal(data["valid_mask"],np.ones((2,3),bool))
                self.assertEqual(int(data["saturation_threshold"]),250)
                self.assertFalse(bool(
                    data["original_saturation_filter_enabled"]))
                expected=((128/255+.055)/1.055)**2.4
                np.testing.assert_allclose(data["rgb"],expected,atol=1e-6)
                self.assertEqual(str(data["source_image"]),str(image))
                self.assertTrue(Path(str(data["source_surface_map"])).exists())

    def test_observation_keeps_original_saturation_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=root/"saturated.png"
            frame=np.full((480,640,3),255,np.uint8)
            point_set,_,_=build_reconstruction_point_set(
                np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
                np.asarray([[1.,0.,100.],[1.,1.,100.]]),
                np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
                np.zeros(5),np.zeros(3),0.,n_fill=1)
            output=save_calibration_observation(
                image,frame,point_set,root/"samples",root/"maps")
            with np.load(output) as data:
                np.testing.assert_array_equal(
                    data["valid_mask"],np.ones((2,3),bool))
                self.assertFalse(bool(
                    data["original_saturation_filter_enabled"]))

    def test_independent_images_can_share_material_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); images=[]
            for index in range(3):
                path=root/f"frame_{index}.png"
                cv2.imwrite(str(path),np.zeros((480,640,3),np.uint8)); images.append(path)
            point_set,_,_=build_reconstruction_point_set(
                np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
                np.asarray([[1.,0.,100.],[1.,1.,100.]]),
                np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
                np.zeros(5),np.zeros(3),0.,n_fill=1)
            segmenter=MagicMock()
            mask_tensor=MagicMock()
            mask_tensor.detach.return_value.cpu.return_value=np.zeros(
                (1,4,4),bool)
            segmenter.segment_tensors.return_value=(
                ("surface",),mask_tensor,MagicMock())
            reconstruction=SimpleNamespace(K=np.eye(3),distortion_coefficients=np.zeros(5),
                s1=1.,s2=-1.,sample_count=2,pair_fill_count=1,
                geometry_rows=2,geometry_columns=3,curve_convexity="increasing",
                side_edge_exclusion_ratio=.02,
                uv_boundary_smooth_lambda=10.,uv_boundary_huber_delta_px=2.,
                material_surface=SimpleNamespace(
                    width_mm=2.,length_mm=1.,s_zero_endpoint="image_top",
                    bend_direction="increasing",
                    initial_calibration_maximum_rms_px=4.,
                    calibration_maximum_rms_px=4.,
                    calibration_minimum_confidence=.05,
                    match_confidence_scale_mm=8.,
                    rms_confidence_scale_px=4.,confidence_floor=1e-6,
                ))
            reconstructor=MagicMock()
            reconstructor.calibrated=True
            reconstructor.rotation_vector=np.zeros(3)
            reconstructor.tx=0.
            xyz=point_set.xyz.reshape(2,3,3)
            uv=point_set.uv.reshape(2,3,2)
            st=point_set.st.reshape(2,3,2)
            depth=point_set.camera_depth.reshape(2,3)
            state=MaterialSurfaceState(
                curve_yz=xyz[:,0,1:3],angles_rad=np.zeros(1,np.float32),
                xyz=xyz,uv=uv,camera_depth=depth,
                visible=np.ones((2,3),bool),observable_rows=np.ones(2,bool),
                left_uv_error=np.zeros((2,2),np.float32),
                right_uv_error=np.zeros((2,2),np.float32),valid=np.asarray(True),
                tracking_accepted=np.asarray(True),
                reprojection_rms_px=np.asarray(.2,np.float32),
                visible_fraction=np.asarray(1.,np.float32),
                matching_confidence=np.asarray(1.,np.float32))
            reconstructed=(np.ones((1,4,4),bool),state,st,
                           np.asarray([True]),np.asarray(True),np.asarray(True),
                           xyz,uv,
                           np.asarray([.2],np.float32),
                           np.zeros(len(MATERIAL_UPDATE_FAILURE_NAMES),bool))
            failed_state=replace(
                state,tracking_accepted=np.asarray(True),
                reprojection_rms_px=np.asarray(7.,np.float32),
                matching_confidence=np.asarray(.001,np.float32))
            failed_flags=np.zeros(len(MATERIAL_UPDATE_FAILURE_NAMES),bool)
            failed_flags[
                MATERIAL_UPDATE_FAILURE_NAMES.index("rms_too_high")]=True
            failed_flags[
                MATERIAL_UPDATE_FAILURE_NAMES.index("confidence_too_low")]=True
            failed=(np.ones((1,4,4),bool),failed_state,st,
                    np.asarray([True]),np.asarray(True),np.asarray(False),xyz,uv,
                    np.asarray([.2],np.float32),failed_flags)
            reconstruction_call=MagicMock(
                side_effect=[reconstructed,failed,reconstructed])
            material_template=SimpleNamespace(
                st=st,segment_lengths_mm=np.ones(1,np.float32),
                x_coordinates_mm=np.asarray([-1.,0.,1.],np.float32),
                reference_curve_yz=xyz[:,0,1:3],
                reference_angles_rad=np.zeros(1,np.float32),
                width_mm=2.,length_mm=1.,s_zero_endpoint="image_top",
                total_length_mm=1.,
                sha256="c"*64)
            reconstruction.material_template=material_template
            config={"get_surface":{"segmentation":{
                        "mode":"liteseg","liteseg":{
                            "model_dir":"model","device":"cpu"}},
                    "prompts":{"surface":{"positive":[[1,1]]}},
                    "reconstruction":{}},
                    "calibration":{"output":"camera.yaml"},
                    "lightfield":{"device":"cpu"}}
            with patch("calibrate_lightfield.SurfaceSegmenter",return_value=segmenter), \
                 patch("calibrate_lightfield.parse_prompts",return_value={
                     "surface":{"positive":[(1.,1.)],"negative":[]}}), \
                 patch("calibrate_lightfield.parse_mask_refine",return_value=
                       SimpleNamespace(enabled=False)), \
                 patch("calibrate_lightfield.parse_reconstruction_config",return_value=reconstruction), \
                 patch("calibrate_lightfield.EdgeReconstructor",return_value=reconstructor), \
                 patch("calibrate_lightfield.choose_device",
                       return_value=jax.devices()[0]), \
                 patch("calibrate_lightfield.reference_material_surface_state_jax",
                       return_value=state), \
                 patch("calibrate_lightfield.reconstruct_material_surface_with_diagnostics_from_masks_jax",
                       reconstruction_call), \
                 patch("calibrate_lightfield.jax.jit",side_effect=lambda fn:fn):
                diagnostic_dir=root/"normal_material_failures"
                update_diagnostic_dir=root/"normal_material_update_failures"
                update_diagnostic_dir.mkdir(parents=True)
                stale_accepted=(update_diagnostic_dir/
                                "frame_0_material_update_failure.jpg")
                stale_accepted.write_bytes(b"stale")
                outputs=reconstruct_all_observations(
                    images,config,root,root/"samples",root/"maps",
                    material_initialization_diagnostic_dir=diagnostic_dir,
                    material_update_failure_diagnostic_dir=
                        update_diagnostic_dir,
                    independent_material_sequence_id="normal_images")
                repeated_outputs=reconstruct_all_observations(
                    images,config,root,root/"samples",root/"maps",
                    material_initialization_diagnostic_dir=diagnostic_dir,
                    material_update_failure_diagnostic_dir=
                        update_diagnostic_dir,
                    independent_material_sequence_id="normal_images")
            self.assertEqual(len(outputs),2)
            self.assertEqual(repeated_outputs,outputs)
            self.assertTrue(diagnostic_dir.is_dir())
            self.assertTrue((update_diagnostic_dir/
                             "frame_1_material_update_failure.jpg").is_file())
            self.assertFalse(stale_accepted.exists())
            failure_record=_reconstruction_failure_path(
                root/"samples",images[1])
            self.assertTrue(failure_record.is_file())
            with np.load(root/"samples"/"frame_0.npz") as data:
                reconstruction_signature=str(data["reconstruction_signature"])
            self.assertEqual(
                _cached_reconstruction_failure(
                    failure_record,images[1],reconstruction_signature,
                    "increasing",250,False),
                ("rms_too_high","confidence_too_low"))
            segmenter.reset.assert_not_called()
            self.assertEqual(segmenter.segment_tensors.call_count,3)
            self.assertEqual(reconstruction_call.call_count,3)
            self.assertFalse(bool(
                reconstruction_call.call_args_list[0].kwargs["initialized"]))
            self.assertTrue(bool(
                reconstruction_call.call_args_list[1].kwargs["initialized"]))
            self.assertTrue(bool(
                reconstruction_call.call_args_list[2].kwargs["initialized"]))
            self.assertEqual(
                reconstruction_call.call_args.kwargs["bend_direction"],
                "increasing")
            self.assertEqual(
                reconstruction_call.call_args.kwargs[
                    "maximum_uv_triangle_width_px"],64)

    def test_material_initialization_failure_image_draws_both_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=np.full((120,160,3),64,np.uint8)
            mask=np.zeros((1,120,160),bool)
            mask[0,30:101,45:116]=True
            template_uv=np.asarray([
                [[35.,15.],[125.,15.]],
                [[35.,105.],[125.,105.]],
            ],np.float32)
            observed_uv=np.asarray([
                [[45.,30.],[115.,30.]],
                [[45.,100.],[115.,100.]],
            ],np.float32)
            observed_xyz=np.asarray([
                [[-1.,0.,10.],[1.,0.,10.]],
                [[-1.,7.,10.],[1.,7.,10.]],
            ],np.float32)
            output,length=_write_material_initialization_failure(
                root/"diagnostics"/"failed.jpg",frame,mask,template_uv,
                observed_xyz,observed_uv,template_length_mm=10.,
                observed_rms_px=2.,maximum_rms_px=4.,
                reconstruction_valid=True)
            self.assertTrue(output.exists())
            self.assertAlmostEqual(length,7.)
            diagnostic=cv2.imread(str(output),cv2.IMREAD_COLOR)
            self.assertIsNotNone(diagnostic)
            assert diagnostic is not None
            self.assertEqual(diagnostic.shape,frame.shape)
            self.assertFalse(np.array_equal(diagnostic,frame))

    def test_old_or_wrong_convexity_observation_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=root/"frame.png"
            frame=np.full((480,640,3),128,np.uint8)
            cv2.imwrite(str(image),frame)
            point_set,_,_=build_reconstruction_point_set(
                np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
                np.asarray([[1.,0.,100.],[1.,1.,100.]]),
                np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
                np.zeros(5),np.zeros(3),0.,n_fill=1)
            old=save_calibration_observation(
                image,frame,point_set,root/"old",root/"old_maps")
            self.assertIsNone(_cached_observation_pose(
                old,image,"signature","increasing",(2,3),250,False))

            metadata=_observation_metadata(
                image,"signature","increasing",np.asarray([.1,.2,.3]),
                4.,np.asarray([.5]),raw_observed_xyz=np.asarray(
                    point_set.xyz,np.float32).reshape(2,3,3))
            current=save_calibration_observation(
                image,frame,point_set,root/"current",root/"current_maps",
                reconstruction_metadata=metadata)
            pose=_cached_observation_pose(
                current,image,"signature","increasing",(2,3),250,False)
            self.assertIsNotNone(pose)
            assert pose is not None
            np.testing.assert_allclose(pose[0],[.1,.2,.3])
            self.assertEqual(pose[1],4.)
            self.assertIsNone(_cached_observation_pose(
                current,image,"signature","decreasing",(2,3),250,False))
            with np.load(root/"current_maps"/"frame_uv_xyz.npz") as data:
                self.assertEqual(str(data["curve_convexity"]),"increasing")

    def test_failure_cache_requires_matching_source_and_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/"frame.png"
            cv2.imwrite(str(image),np.zeros((8,10,3),np.uint8))
            output=_reconstruction_failure_path(root/"samples",image)
            _write_reconstruction_failure(
                output,image,"signature","increasing",
                stage="material_calibration_admission",
                failure_reasons=("rms_too_high",),
                saturation_threshold=250,
                filter_original_saturation=False,
                sequence_id="video_001_test",sequence_frame_index=3,
                tracking_committed=True,reprojection_rms_px=5.)
            self.assertEqual(
                _cached_reconstruction_failure(
                    output,image,"signature","increasing",250,False),
                ("rms_too_high",))
            self.assertIsNone(_cached_reconstruction_failure(
                output,image,"different","increasing",250,False))
            cv2.imwrite(str(image),np.ones((8,10,3),np.uint8))
            self.assertIsNone(_cached_reconstruction_failure(
                output,image,"signature","increasing",250,False))

if __name__=="__main__": unittest.main()
