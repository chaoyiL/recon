import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cv2
import jax
import numpy as np
from calibrate_lightfield import (_iter_physical_batch_indices,
                                  _cached_observation_pose,
                                  _observation_metadata,
                                  _split_calibration_indices,
                                  extract_video_frames,
                                  reconstruct_all_observations,
                                  save_calibration_observation)
from utils.process import build_reconstruction_point_set

class LightFieldCalibrationSampleTest(unittest.TestCase):
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

    def test_independent_images_reset_sam_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); images=[]
            for index in range(2):
                path=root/f"frame_{index}.png"
                cv2.imwrite(str(path),np.zeros((480,640,3),np.uint8)); images.append(path)
            point_set,_,_=build_reconstruction_point_set(
                np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
                np.asarray([[1.,0.,100.],[1.,1.,100.]]),
                np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
                np.zeros(5),np.zeros(3),0.,n_fill=1)
            segmenter=MagicMock()
            segmenter.segment_tensors.return_value=(
                ("surface",),MagicMock(),MagicMock())
            reconstruction=SimpleNamespace(K=np.eye(3),distortion_coefficients=np.zeros(5),
                s1=1.,s2=-1.,sample_count=2,pair_fill_count=1,
                geometry_rows=2,geometry_columns=3,curve_convexity="increasing",
                uv_boundary_smooth_lambda=10.,uv_boundary_huber_delta_px=2.)
            reconstructor=MagicMock()
            reconstructor.calibrated=True
            reconstructor.rotation_vector=np.zeros(3)
            reconstructor.tx=0.
            xyz=point_set.xyz.reshape(2,3,3)
            uv=point_set.uv.reshape(2,3,2)
            st=point_set.st.reshape(2,3,2)
            depth=point_set.camera_depth.reshape(2,3)
            reconstructed=(np.ones((1,4,4),bool),xyz,uv,st,depth,
                           np.asarray([.2],np.float32),np.asarray([True]))
            reconstruction_call=MagicMock(return_value=reconstructed)
            config={"get_surface":{"model":"mock","prompts":{"surface":{"positive":[[1,1]]}},
                    "reconstruction":{},"center_band_d":1},
                    "calibration":{"output":"camera.yaml"},
                    "lightfield":{"device":"cpu"}}
            with patch("calibrate_lightfield.SurfaceSegmenter",return_value=segmenter), \
                 patch("calibrate_lightfield.parse_prompts",return_value={
                     "surface":{"positive":[(1.,1.)],"negative":[]}}), \
                 patch("calibrate_lightfield.parse_mask_refine",return_value=SimpleNamespace(
                     enabled=False,close_kernel=0,open_kernel=0,blur_kernel=0,
                     keep_largest=True)), \
                 patch("calibrate_lightfield.parse_reconstruction_config",return_value=reconstruction), \
                 patch("calibrate_lightfield.EdgeReconstructor",return_value=reconstructor), \
                 patch("calibrate_lightfield.choose_device",
                       return_value=jax.devices()[0]), \
                 patch("calibrate_lightfield.torch_tensor_to_jax",
                       return_value=np.zeros((1,4,4),bool)), \
                 patch("calibrate_lightfield.reconstruct_surface_from_masks_jax",
                       reconstruction_call), \
                 patch("calibrate_lightfield.jax.jit",side_effect=lambda fn:fn):
                outputs=reconstruct_all_observations(images,config,root,root/"samples",root/"maps")
            self.assertEqual(len(outputs),2)
            self.assertEqual(segmenter.reset.call_count,2)
            self.assertEqual(reconstruction_call.call_count,2)
            self.assertEqual(
                reconstruction_call.call_args.kwargs["curve_convexity"],
                "increasing")

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
                4.,np.asarray([.5]))
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

if __name__=="__main__": unittest.main()
