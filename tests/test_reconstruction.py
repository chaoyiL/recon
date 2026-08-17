import unittest
from unittest.mock import patch

import cv2
import jax.numpy as jnp
import numpy as np

import utils.process as process_module
from utils.jax_reconstruction import (_undistort_pixels_jax,
                                      _project_convex_depth_jax,
                                      _smooth_boundary_error_jax,
                                      build_surface_grid_jax,
                                      monotone_right_matches_jax,
                                      prepare_edge_curves_from_masks_jax,
                                      reconstruct_surface_batch_jax,
                                      reconstruct_surface_from_masks_jax,
                                      resample_surface_batch_jax,
                                      solve_smoothed_shared_curve_jax)
from utils.process import (
    EdgeReconstructor,
    _monotone_right_matches,
    _rotation,
    _solve_smoothed_shared_curve,
    build_reconstruction_point_set,
)


class EdgeReconstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.asarray(
            [[414.0, 0.0, 315.0], [0.0, 413.0, 233.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros(5, dtype=np.float64)

    def test_joint_reconstruction_keeps_paired_yz_identical(self) -> None:
        sample_count = 16
        dense_count = 240
        h = np.linspace(-60.0, 60.0, dense_count)
        z = 260.0 + 8.0 * np.sin(np.linspace(0.0, np.pi, dense_count))
        left_xyz = np.column_stack([np.full(dense_count, -11.0), h, z])
        right_xyz = np.column_stack([np.full(dense_count, 11.0), h, z])
        rotation = np.deg2rad([0.0, -1.0, 0.8])
        translation = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        left_uv, _ = cv2.projectPoints(
            left_xyz, rotation, translation, self.K, self.distortion
        )
        right_uv, _ = cv2.projectPoints(
            right_xyz, rotation, translation, self.K, self.distortion
        )
        pixels = np.concatenate([left_uv.reshape(-1, 2), right_uv.reshape(-1, 2)])
        segments = [np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)]

        result = EdgeReconstructor(
            self.K,
            self.distortion,
            11.0,
            -11.0,
            sample_count=sample_count,
        ).process(segments, float(np.mean(pixels[:, 0])))

        np.testing.assert_allclose(result.left_xyz[:, 1:], result.right_xyz[:, 1:])
        self.assertEqual(result.left_uv.shape, (sample_count, 2))
        self.assertEqual(result.right_uv.shape, (sample_count, 2))
        self.assertTrue(np.all(np.diff(result.right_uv[:, 1]) >= 0.0))
        self.assertTrue(np.isfinite(result.reprojection_rms_px))
        self.assertLess(result.reprojection_rms_px, 2.0)

    def test_edges_and_fills_share_one_projected_uv_xyz_point_set(self) -> None:
        left = np.asarray([[-1.0, 2.0, 3.0], [-1.0, 4.0, 5.0]])
        right = np.asarray([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
        point_set, line_points, indices = build_reconstruction_point_set(
            left,
            right,
            self.K,
            self.distortion,
            np.zeros(3),
            0.0,
            n_fill=1,
        )

        np.testing.assert_allclose(
            point_set.xyz,
            [
                [-1.0, 2.0, 3.0],
                [0.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [-1.0, 4.0, 5.0],
                [0.0, 4.0, 5.0],
                [1.0, 4.0, 5.0],
            ],
        )
        np.testing.assert_array_equal(
            point_set.is_edge,
            [True, False, True, True, False, True],
        )
        np.testing.assert_allclose(point_set.uv, point_set.undistorted_uv)
        np.testing.assert_allclose(
            point_set.st,
            [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
             [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        )
        self.assertEqual(point_set.xyz.shape[0], point_set.uv.shape[0])
        self.assertEqual(point_set.xyz.shape[0], point_set.undistorted_uv.shape[0])
        np.testing.assert_array_equal(indices, [[0, 1], [2, 3]])
        np.testing.assert_allclose(line_points[0::2], left)
        np.testing.assert_allclose(line_points[1::2], right)

    def test_observed_boundaries_correct_uv_and_interpolate_across_surface(self) -> None:
        y = np.linspace(-2.0, 2.0, 5)
        left = np.column_stack([np.full(5, -1.0), y, np.full(5, 100.0)])
        right = np.column_stack([np.full(5, 1.0), y, np.full(5, 100.0)])
        projected, _, _ = build_reconstruction_point_set(
            left, right, self.K, self.distortion, np.zeros(3), 0.0, n_fill=2
        )
        projected_uv = projected.uv.reshape(5, 4, 2)
        left_error = np.asarray([2.0, -1.0])
        right_error = np.asarray([-1.0, 3.0])
        observed_left = projected_uv[:, 0] + left_error
        observed_right = projected_uv[:, -1] + right_error

        corrected, _, _ = build_reconstruction_point_set(
            left,
            right,
            self.K,
            self.distortion,
            np.zeros(3),
            0.0,
            n_fill=2,
            observed_left_uv=observed_left,
            observed_right_uv=observed_right,
            uv_boundary_smooth_lambda=10.0,
            uv_boundary_huber_delta_px=2.0,
        )
        corrected_uv = corrected.uv.reshape(5, 4, 2)
        alpha = np.linspace(0.0, 1.0, 4)
        expected_error = (
            (1.0 - alpha)[:, None] * left_error
            + alpha[:, None] * right_error
        )
        np.testing.assert_allclose(corrected_uv[:, 0], observed_left, atol=1e-9)
        np.testing.assert_allclose(corrected_uv[:, -1], observed_right, atol=1e-9)
        np.testing.assert_allclose(
            corrected_uv - projected_uv,
            np.broadcast_to(expected_error, corrected_uv.shape),
            atol=1e-9,
        )
        np.testing.assert_allclose(corrected.xyz, projected.xyz)

    def test_boundary_correction_requires_both_observed_sides(self) -> None:
        left = np.asarray([[-1.0, 0.0, 100.0], [-1.0, 1.0, 100.0]])
        right = np.asarray([[1.0, 0.0, 100.0], [1.0, 1.0, 100.0]])
        with self.assertRaisesRegex(ValueError, "必须同时提供"):
            build_reconstruction_point_set(
                left,
                right,
                self.K,
                self.distortion,
                np.zeros(3),
                0.0,
                observed_left_uv=np.zeros((2, 2)),
            )

    def test_external_pose_is_optimized_only_on_first_valid_frame(self) -> None:
        count = 160
        h = np.linspace(-50.0, 50.0, count)
        z = np.full(count, 250.0)
        left_xyz = np.column_stack([np.full(count, -11.0), h, z])
        right_xyz = np.column_stack([np.full(count, 11.0), h, z])
        left_uv, _ = cv2.projectPoints(
            left_xyz, np.zeros(3), np.zeros(3), self.K, self.distortion
        )
        right_uv, _ = cv2.projectPoints(
            right_xyz, np.zeros(3), np.zeros(3), self.K, self.distortion
        )
        pixels = np.concatenate([left_uv.reshape(-1, 2), right_uv.reshape(-1, 2)])
        segments = [np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)]
        reconstructor = EdgeReconstructor(
            self.K, self.distortion, 11.0, -11.0, sample_count=12
        )

        with patch(
            "utils.process._optimize_shared_curve",
            wraps=process_module._optimize_shared_curve,
        ) as optimize:
            reconstructor.process(segments, float(np.mean(pixels[:, 0])))
            first_frame_calls = optimize.call_count
            reconstructor.process(segments, float(np.mean(pixels[:, 0])))

        self.assertTrue(reconstructor.calibrated)
        self.assertEqual(first_frame_calls, 2)
        self.assertEqual(optimize.call_count, first_frame_calls)

    def test_second_difference_term_smooths_linear_solution(self) -> None:
        rng = np.random.default_rng(3)
        count = 30
        h_true = np.linspace(-40.0, 40.0, count)
        z_true = 240.0 + 5.0 * np.sin(np.linspace(0.0, np.pi, count))
        left_xyz = np.column_stack([np.full(count, -11.0), h_true, z_true])
        right_xyz = np.column_stack([np.full(count, 11.0), h_true, z_true])
        left_uv, _ = cv2.projectPoints(
            left_xyz, np.zeros(3), np.zeros(3), self.K, self.distortion
        )
        right_uv, _ = cv2.projectPoints(
            right_xyz, np.zeros(3), np.zeros(3), self.K, self.distortion
        )
        left_uv = left_uv.reshape(-1, 2) + rng.normal(0.0, 0.8, (count, 2))
        right_uv = right_uv.reshape(-1, 2) + rng.normal(0.0, 0.8, (count, 2))

        h_raw, z_raw, _ = _solve_smoothed_shared_curve(
            left_uv,
            right_uv,
            self.K,
            11.0,
            -11.0,
            np.zeros(3),
            0.0,
            smooth_lambda=0.0,
        )
        h_smooth, z_smooth, _ = _solve_smoothed_shared_curve(
            left_uv,
            right_uv,
            self.K,
            11.0,
            -11.0,
            np.zeros(3),
            0.0,
            smooth_lambda=10.0,
        )

        raw_curvature = np.linalg.norm(np.diff(h_raw, n=2)) + np.linalg.norm(
            np.diff(z_raw, n=2)
        )
        smooth_curvature = np.linalg.norm(np.diff(h_smooth, n=2)) + np.linalg.norm(
            np.diff(z_smooth, n=2)
        )
        self.assertLess(smooth_curvature, raw_curvature)

    def test_jax_monotone_matching_equals_cpu_dynamic_program(self) -> None:
        count=24; dense_count=4*count
        h=np.linspace(-45.,45.,dense_count)
        z=250.+6*np.sin(np.linspace(0,np.pi,dense_count))
        left_xyz=np.column_stack([np.full(dense_count,-11.),h,z])
        right_xyz=np.column_stack([np.full(dense_count,11.),h,z])
        rotation_vector=np.deg2rad([0.,-1.,.8])
        translation=np.asarray([-1.,0.,0.])
        left_uv,_=cv2.projectPoints(
            left_xyz,rotation_vector,translation,self.K,self.distortion)
        right_uv,_=cv2.projectPoints(
            right_xyz,rotation_vector,translation,self.K,self.distortion)
        sample=np.linspace(0,dense_count-1,count)
        left=np.column_stack([
            np.interp(sample,np.arange(dense_count),left_uv[:,0,axis])
            for axis in range(2)])
        right=right_uv.reshape(-1,2)
        expected=_monotone_right_matches(
            left,right,self.K,11.,-11.,rotation_vector,-1.)
        actual,valid=monotone_right_matches_jax(
            jnp.asarray(left,np.float32),jnp.asarray(right,np.float32),
            jnp.asarray(np.linalg.inv(self.K),jnp.float32),
            jnp.asarray(_rotation(rotation_vector),jnp.float32),11.,-11.,-1.)
        self.assertTrue(bool(valid))
        np.testing.assert_array_equal(np.asarray(actual),expected)

    def test_jax_mask_path_produces_fixed_gpu_edge_curves(self) -> None:
        mask=np.zeros((1,64,80),bool)
        for row in range(8,58):
            mask[0,row,10+row//20:70-row//30]=True
        refined,left,right,valid=prepare_edge_curves_from_masks_jax(
            jnp.asarray(mask),jnp.asarray(self.K,np.float32),
            jnp.zeros(5,jnp.float32),16,5.,close_kernel=3,
            open_kernel=3,blur_kernel=5)
        self.assertEqual(refined.shape,(1,64,80))
        self.assertEqual(left.shape,(1,16,2))
        self.assertEqual(right.shape,(1,64,2))
        self.assertTrue(bool(valid[0]))
        self.assertTrue(np.all(np.diff(np.asarray(left[0,:,1]))>=0))
        self.assertTrue(np.all(np.asarray(left[0,:,0])<np.asarray(right[0,::4,0])))

    def test_jax_mask_path_rejects_narrow_or_empty_surface(self) -> None:
        masks=np.zeros((2,32,40),bool)
        masks[0,8:24,18:21]=True
        _,_,_,valid=prepare_edge_curves_from_masks_jax(
            jnp.asarray(masks),jnp.asarray(self.K,np.float32),
            jnp.zeros(5,jnp.float32),8,5.)
        np.testing.assert_array_equal(np.asarray(valid),[False,False])

    def test_shared_realtime_mask_entry_matches_explicit_jax_pipeline(self) -> None:
        mask=np.zeros((1,64,80),bool)
        for row in range(8,58):
            mask[0,row,10+row//20:70-row//30]=True
        masks=jnp.asarray(mask)
        camera=jnp.asarray(self.K,np.float32)
        distortion=jnp.zeros(5,jnp.float32)
        inverse=jnp.asarray(np.linalg.inv(self.K),np.float32)
        rotation=jnp.eye(3,dtype=jnp.float32)
        refined,left,right,edge_valid=prepare_edge_curves_from_masks_jax(
            masks,camera,distortion,16,5.,close_kernel=3,
            open_kernel=3,blur_kernel=5)
        explicit=(*reconstruct_surface_batch_jax(
            left,right,camera,distortion,rotation,11.,-11.,0.,4,10.,2.,
            inverse),)
        shared=reconstruct_surface_from_masks_jax(
            masks,camera,distortion,inverse,rotation,11.,-11.,0.,16,5.,4,
            10.,2.,close_kernel=3,open_kernel=3,blur_kernel=5)
        np.testing.assert_array_equal(np.asarray(shared[0]),np.asarray(refined))
        for actual,expected in zip(shared[1:6],explicit[:5],strict=True):
            np.testing.assert_allclose(np.asarray(actual),np.asarray(expected))
        np.testing.assert_array_equal(
            np.asarray(shared[6]),np.asarray(explicit[5]&edge_valid))

    def test_jax_undistortion_matches_opencv_five_iteration_result(self) -> None:
        distortion=np.asarray([.0639,-.683,-.00215,.00812,1.393],np.float32)
        points=np.asarray([[20.,20.],[315.,233.],[620.,460.]],np.float32)
        expected=cv2.undistortPoints(
            points.reshape(-1,1,2),self.K,distortion,P=self.K).reshape(-1,2)
        actual=_undistort_pixels_jax(
            jnp.asarray(points),jnp.asarray(self.K,np.float32),
            jnp.asarray(distortion))
        np.testing.assert_allclose(np.asarray(actual),expected,atol=2e-3)

    def test_jax_smoothed_solve_matches_cpu_geometry(self) -> None:
        count=30
        h=np.linspace(-40.,40.,count)
        z=240.+5*np.sin(np.linspace(0,np.pi,count))
        left_xyz=np.column_stack([np.full(count,-11.),h,z])
        right_xyz=np.column_stack([np.full(count,11.),h,z])
        left_uv,_=cv2.projectPoints(
            left_xyz,np.zeros(3),np.zeros(3),self.K,self.distortion)
        right_uv,_=cv2.projectPoints(
            right_xyz,np.zeros(3),np.zeros(3),self.K,self.distortion)
        left=left_uv.reshape(-1,2); right=right_uv.reshape(-1,2)
        expected_h,expected_z,expected_rms=_solve_smoothed_shared_curve(
            left,right,self.K,11.,-11.,np.zeros(3),0.,smooth_lambda=1.)
        actual_h,actual_z,actual_rms,valid=solve_smoothed_shared_curve_jax(
            jnp.asarray(left,np.float32),jnp.asarray(right,np.float32),
            jnp.asarray(self.K,np.float32),jnp.eye(3,dtype=jnp.float32),
            11.,-11.,0.,1.)
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(np.asarray(actual_h),expected_h,atol=.02)
        np.testing.assert_allclose(np.asarray(actual_z),expected_z,atol=.02)
        self.assertAlmostEqual(float(actual_rms),expected_rms,places=3)

    def test_convex_depth_projection_preserves_endpoints_and_slope_sign(self) -> None:
        h=jnp.asarray([0.,1.,2.,4.,7.,8.],jnp.float32)
        z=jnp.asarray([10.,11.,10.5,13.,12.,16.],jnp.float32)
        projected,valid=_project_convex_depth_jax(h,z,"increasing")
        projected=np.asarray(projected)
        slopes=np.diff(projected)/np.diff(np.asarray(h))
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(projected[[0,-1]],[10.,16.],atol=1e-5)
        self.assertTrue(np.all(np.diff(slopes)>=-2e-5))

    def test_convex_depth_projection_rejects_nonmonotone_h(self) -> None:
        projected,valid=_project_convex_depth_jax(
            jnp.asarray([0.,1.,.5,2.],jnp.float32),
            jnp.asarray([10.,11.,12.,13.],jnp.float32),"increasing")
        self.assertFalse(bool(valid))
        self.assertTrue(np.all(np.isfinite(np.asarray(projected))))

    def test_jax_surface_grid_matches_cpu_projection_and_correction(self) -> None:
        count=12
        h=np.linspace(-20.,20.,count).astype(np.float32)
        z=np.full(count,220.,np.float32)
        left=np.column_stack([np.full(count,-11.),h,z])
        right=np.column_stack([np.full(count,11.),h,z])
        left_uv,_=cv2.projectPoints(
            left,np.zeros(3),np.zeros(3),self.K,self.distortion)
        right_uv,_=cv2.projectPoints(
            right,np.zeros(3),np.zeros(3),self.K,self.distortion)
        left_uv=left_uv.reshape(-1,2); right_uv=right_uv.reshape(-1,2)
        expected,_,_=build_reconstruction_point_set(
            left,right,self.K,self.distortion,np.zeros(3),0.,n_fill=4,
            observed_left_uv=left_uv,observed_right_uv=right_uv,
            uv_boundary_smooth_lambda=10.,uv_boundary_huber_delta_px=2.)
        left_undistorted=cv2.undistortPoints(
            left_uv.reshape(-1,1,2),self.K,self.distortion,P=self.K).reshape(-1,2)
        right_undistorted=cv2.undistortPoints(
            right_uv.reshape(-1,1,2),self.K,self.distortion,P=self.K).reshape(-1,2)
        xyz,uv,st,depth,valid=build_surface_grid_jax(
            jnp.asarray(h),jnp.asarray(z),jnp.asarray(left_undistorted,np.float32),
            jnp.asarray(right_undistorted,np.float32),jnp.asarray(self.K,np.float32),
            jnp.asarray(self.distortion,np.float32),jnp.eye(3,dtype=jnp.float32),
            11.,-11.,0.,4,10.,2.)
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(np.asarray(xyz).reshape(-1,3),expected.xyz,atol=1e-4)
        np.testing.assert_allclose(np.asarray(uv).reshape(-1,2),expected.uv,atol=1e-3)
        np.testing.assert_allclose(np.asarray(st).reshape(-1,2),expected.st,atol=1e-6)
        np.testing.assert_allclose(np.asarray(depth).reshape(-1),expected.camera_depth,atol=1e-4)

    def test_boundary_irls_banded_solve_matches_dense_reference(self) -> None:
        rng=np.random.default_rng(12)
        error=rng.normal(0,.5,(31,2)).astype(np.float32)
        error[15]+=[7.,-5.]
        smooth_lambda=8.; huber_delta=1.2
        second=np.zeros((error.shape[0]-2,error.shape[0]),np.float64)
        for row in range(second.shape[0]):
            second[row,row:row+3]=[1.,-2.,1.]
        regularizer=smooth_lambda*(second.T@second)
        estimate=np.linalg.solve(np.eye(error.shape[0])+regularizer,error)
        for _ in range(4):
            norm=np.linalg.norm(estimate-error,axis=1)
            weights=np.minimum(1.,huber_delta/np.maximum(norm,1e-12))
            estimate=np.linalg.solve(
                np.diag(weights)+regularizer,weights[:,None]*error)
        actual=_smooth_boundary_error_jax(
            jnp.asarray(error),smooth_lambda,huber_delta)
        np.testing.assert_allclose(np.asarray(actual),estimate,atol=2e-5)

    def test_surface_grids_can_be_resampled_independently(self) -> None:
        rows,columns=4,5
        y,x=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([x,y,100+x+2*y],axis=-1)
        uv=xyz[...,:2]*3
        depth=xyz[...,2]
        resized_xyz,resized_uv,resized_st,resized_depth=(
            resample_surface_batch_jax(
                jnp.asarray(xyz),jnp.asarray(uv),jnp.asarray(depth),
                surface_count=1,source_rows=rows,
                target_rows=9,target_columns=11))
        self.assertEqual(resized_xyz.shape,(9,11,3))
        self.assertEqual(resized_uv.shape,(9,11,2))
        self.assertEqual(resized_st.shape,(9,11,2))
        self.assertEqual(resized_depth.shape,(9,11))
        np.testing.assert_allclose(np.asarray(resized_xyz)[0,0],xyz[0,0])
        np.testing.assert_allclose(np.asarray(resized_xyz)[-1,-1],xyz[-1,-1])
        np.testing.assert_allclose(np.asarray(resized_st)[[0,-1],0,0],[0,1])
        np.testing.assert_allclose(np.asarray(resized_st)[0,[0,-1],1],[0,1])


if __name__ == "__main__":
    unittest.main()
