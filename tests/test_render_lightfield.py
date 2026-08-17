import unittest
import cv2
import jax.numpy as jnp
import numpy as np
import torch
from render_lightfield import (evaluate_startup_stability,
                               mask_original_frame, point_set_grid,
                               torch_tensor_to_jax)
from utils.lightfield import (positive_residual_bgr,signed_difference_bgr,
                              signed_residual_bgr,signed_residual_bgr_jax)
from utils.process import (build_colored_surface_mesh,
                           build_reconstruction_point_set,
                           SurfaceMeshVisualizer)


class RuntimeInMemorySurfaceTest(unittest.TestCase):
    def test_torch_to_jax_dlpack_preserves_tensor_without_numpy_bridge(self):
        source=torch.arange(12,dtype=torch.int32).reshape(3,4)
        shared=torch_tensor_to_jax(source)
        np.testing.assert_array_equal(np.asarray(shared),np.arange(12).reshape(3,4))

    def test_startup_stability_accepts_quiet_window(self):
        offsets=np.asarray([-.008,-.004,0.,.004,.008])
        samples=(np.ones((5,2,3,3),np.float32)*.1
                 +offsets[:,None,None,None])
        parameter_offsets=offsets/8
        gains=np.ones((5,3)) + parameter_offsets[:,None]
        biases=np.ones((5,3))*.2 + parameter_offsets[:,None]
        result=evaluate_startup_stability(
            samples,np.ones((5,2,3),bool),gains,biases,
            residual_field_rmse_threshold=.01,
            gain_range_threshold=.01,bias_range_threshold=.01,
            minimum_valid_overlap=.9)
        self.assertTrue(result["stable"])
        self.assertEqual(result["valid_overlap"],1.)

    def test_startup_stability_rejects_transient_window(self):
        samples=np.ones((5,2,3,3),np.float32)*.1
        samples[0,...,1]=-.1
        gains=np.ones((5,3)); gains[0,1]=.8
        biases=np.ones((5,3))*.2; biases[0,1]=-.1
        result=evaluate_startup_stability(
            samples,np.ones((5,2,3),bool),gains,biases,
            residual_field_rmse_threshold=.005,
            gain_range_threshold=.01,bias_range_threshold=.01,
            minimum_valid_overlap=.9)
        self.assertFalse(result["stable"])
        self.assertGreater(result["residual_field_rmse_rgb"][1],.005)
        self.assertGreater(result["gain_range_rgb"][1],.01)
        self.assertGreater(result["bias_range_rgb"][1],.01)

    def test_original_frame_only_keeps_mask_union(self):
        frame=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
        first=np.zeros((4,5),bool); first[1,1]=True
        second=np.zeros((4,5),bool); second[2,3]=True
        masked=mask_original_frame(frame,{"first":first,"second":second})
        np.testing.assert_array_equal(masked[1,1],frame[1,1])
        np.testing.assert_array_equal(masked[2,3],frame[2,3])
        self.assertEqual(np.count_nonzero(masked[np.logical_not(first|second)]),0)

    def test_reconstruction_point_set_becomes_grid_without_file(self):
        point_set,_,_=build_reconstruction_point_set(
            np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
            np.asarray([[1.,0.,100.],[1.,1.,100.]]),
            np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
            np.zeros(5),np.zeros(3),0.,n_fill=2)
        xyz,uv,st,depth=point_set_grid(point_set)
        self.assertEqual(xyz.shape,(2,4,3))
        self.assertEqual(uv.shape,(2,4,2))
        self.assertEqual(st.shape,(2,4,2))
        np.testing.assert_allclose(st[0,:,0],0.)
        np.testing.assert_allclose(st[1,:,0],1.)
        np.testing.assert_allclose(
            st[:,:,1],np.broadcast_to(np.linspace(0.,1.,4),(2,4)))
        self.assertEqual(depth.shape,(2,4))
        np.testing.assert_allclose(depth,100.)

    def test_deformed_surface_grid_becomes_depth_colored_triangle_mesh(self):
        grid=np.arange(3*4*3,dtype=np.float64).reshape(3,4,3)
        depth=np.linspace(-2,2,12,dtype=np.float64).reshape(3,4)
        vertices,triangles,colors=build_colored_surface_mesh(
            grid,depth,depth_range_mm=2.)
        np.testing.assert_array_equal(vertices,grid.reshape(-1,3))
        self.assertEqual(triangles.shape,(2*(3-1)*(4-1),3))
        self.assertEqual(colors.shape,(12,3))
        self.assertTrue(np.all((colors>=0)&(colors<=1)))
        self.assertFalse(np.allclose(colors[0],colors[-1]))

    def test_surface_mesh_omits_triangles_touching_invalid_vertices(self):
        grid=np.zeros((2,2,3),np.float64)
        valid=np.asarray([[True,True],[True,False]])
        _,triangles,colors=build_colored_surface_mesh(
            grid,np.zeros((2,2)),valid_mask=valid)
        self.assertEqual(triangles.shape,(0,3))
        np.testing.assert_allclose(colors[-1],[.15,.15,.15])

    def test_deformation_colors_restore_linear_turbo_palette(self):
        grid=np.zeros((2,3,3),np.float64)
        depth=np.asarray([[-2.,0.,2.],[-2.,0.,2.]])
        _,_,colors=build_colored_surface_mesh(
            grid,depth,depth_range_mm=2.)
        indices=np.asarray([[0,128,255]],np.uint8)
        expected=cv2.applyColorMap(
            indices,cv2.COLORMAP_TURBO)[...,::-1].reshape(-1,3)/255.
        np.testing.assert_allclose(colors[:3],expected,atol=0)
        self.assertGreater(np.ptp(colors[:3]),.8)

    def test_surface_visualizer_rejects_invalid_display_parameters(self):
        with self.assertRaisesRegex(ValueError,"show_coordinate_frame"):
            SurfaceMeshVisualizer(show_coordinate_frame=1)

    def test_signed_difference_uses_gray_for_zero_and_black_outside(self):
        original=np.full((2,2,3),128,np.uint8)
        srgb=128/255
        linear=((srgb+.055)/1.055)**2.4
        rendered=np.full((2,2,3),linear,np.float32)
        valid=np.asarray([[255,0],[255,255]],np.uint8)
        difference=signed_difference_bgr(original,rendered,valid,gain=2)
        np.testing.assert_allclose(difference[0,0],[128,128,128],atol=1)
        np.testing.assert_array_equal(difference[0,1],[0,0,0])

    def test_signed_residual_display_preserves_positive_and_negative_direction(self):
        residual=np.asarray(
            [[[-.1,0,0],[0,0,0],[.1,0,0],[.2,.2,.2]]],np.float32)
        valid=np.asarray([[True,True,True,False]])
        cpu=signed_residual_bgr(residual,valid,gain=1)
        gpu=np.asarray(signed_residual_bgr_jax(
            jnp.asarray(residual),jnp.asarray(valid),gain=1))
        np.testing.assert_array_equal(gpu,cpu)
        np.testing.assert_allclose(cpu[0,1],[128,128,128],atol=1)
        # 输出是 BGR：正红色差使 R 高于灰点，负红色差使 R 低于灰点。
        self.assertLess(int(cpu[0,0,2]),int(cpu[0,1,2]))
        self.assertGreater(int(cpu[0,2,2]),int(cpu[0,1,2]))
        np.testing.assert_array_equal(cpu[0,3],[0,0,0])

    def test_positive_red_difference_is_red_in_bgr_display(self):
        original=np.zeros((1,1,3),np.uint8); original[0,0,2]=255
        difference=signed_difference_bgr(original,np.zeros((1,1,3),np.float32),
                                         np.ones((1,1),np.uint8)*255,gain=1)
        self.assertGreater(int(difference[0,0,2]),int(difference[0,0,0]))

    def test_positive_residual_clamps_negative_and_maps_zero_to_black(self):
        residual=np.asarray([[[-.1,0,0],[0,0,0],[.1,0,0],[.2,0,0]]],
                            np.float32)
        image=positive_residual_bgr(
            residual,np.ones((1,4),np.uint8)*255,gain=1)
        np.testing.assert_array_equal(image[0,0],[0,0,0])
        np.testing.assert_array_equal(image[0,1],[0,0,0])
        self.assertGreater(int(image[0,2,2]),0)
        self.assertGreater(int(image[0,3,2]),int(image[0,2,2]))

if __name__=="__main__": unittest.main()
