import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from calibrate_norm import (
    build_normal_lut,
    dilate_manual_contact,
    fit_normal_calibration_residual_session,
    manual_contact_selection,
    save_manual_verification,
    save_pq_rgb_mapping,
    sphere_slope_samples,
)
from manual_norm_regions import (
    ManualEllipse,
    ellipse_mask,
    eroded_ellipse_mask,
    load_manual_ellipses,
    save_manual_ellipses,
)


class NormalCalibrationTest(unittest.TestCase):
    def synthetic_surface(self):
        height,width=160,200
        y,x=np.meshgrid(np.arange(height),np.arange(width),indexing="ij")
        scale=.05
        xyz_image=np.stack([
            (x-width/2)*scale,(y-height/2)*scale,
            np.ones_like(x)*100],axis=-1).astype(np.float32)
        sy,sx=np.meshgrid(
            np.linspace(-(height/2)*scale,(height/2)*scale,20),
            np.linspace(-(width/2)*scale,(width/2)*scale,25),indexing="ij")
        surface=np.stack([sx,sy,np.ones_like(sx)*100],axis=-1).astype(np.float32)
        valid=np.ones((height,width),bool)
        ellipse=ManualEllipse(True,103.,77.,40.,32.,0.,width,height)
        residual=np.zeros((height,width,3),np.float32)
        residual[ellipse_mask((height,width),ellipse)]=np.asarray([.11,-.08,.09])
        return residual,valid,xyz_image,surface,ellipse

    def test_manual_ellipse_masks_and_persistence(self):
        ellipse=ManualEllipse(True,60.,50.,24.,15.,12.,120,100)
        outer=ellipse_mask((100,120),ellipse)
        inner=eroded_ellipse_mask((100,120),ellipse,3)
        self.assertTrue(outer[50,60])
        self.assertTrue(np.all(~inner|outer))
        self.assertLess(np.count_nonzero(inner),np.count_nonzero(outer))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ellipses.yaml"
            save_manual_ellipses(path,{"image-a":ellipse})
            loaded=load_manual_ellipses(path)
        self.assertEqual(loaded["image-a"],ellipse)

    def test_manual_selection_is_the_only_sample_region(self):
        residual,valid,xyz_image,surface,ellipse=self.synthetic_surface()
        selection=manual_contact_selection(
            ellipse,valid,valid,xyz_image,surface,5.,2)
        self.assertTrue(selection.accepted)
        np.testing.assert_array_equal(
            selection.contact_mask,ellipse_mask(valid.shape,ellipse))
        self.assertTrue(np.all(~selection.sample_mask|selection.contact_mask))
        np.testing.assert_allclose(selection.center_xyz[:2],[.15,-.15],atol=.06)
        colors,slopes=sphere_slope_samples(
            residual,xyz_image,selection,5.)
        self.assertGreater(colors.shape[0],100)
        self.assertTrue(np.all(colors[:,0]>0))
        self.assertTrue(np.all(colors[:,1]<0))
        self.assertTrue(np.isfinite(slopes).all())

    def test_manual_skip_never_generates_lut_samples(self):
        residual,valid,xyz_image,surface,ellipse=self.synthetic_surface()
        skipped=ManualEllipse(
            False,ellipse.center_x,ellipse.center_y,ellipse.semi_axis_x,
            ellipse.semi_axis_y,ellipse.angle_degrees,
            ellipse.image_width,ellipse.image_height)
        selection=manual_contact_selection(
            skipped,valid,valid,xyz_image,surface,5.,2)
        self.assertFalse(selection.accepted)
        with self.assertRaisesRegex(ValueError,"人工跳过"):
            sphere_slope_samples(residual,xyz_image,selection,5.)

    def test_manual_contact_dilation_is_local(self):
        mask=np.zeros((31,31),bool); mask[15,15]=True
        expanded=dilate_manual_contact(mask,2)
        self.assertTrue(expanded[15,15])
        self.assertFalse(expanded[0,0])
        self.assertGreater(np.count_nonzero(expanded),1)
        self.assertLess(np.count_nonzero(expanded),40)

    def test_manual_verification_image_is_written(self):
        residual,valid,xyz_image,surface,ellipse=self.synthetic_surface()
        selection=manual_contact_selection(
            ellipse,valid,valid,xyz_image,surface,5.,2)
        frame=np.full(residual.shape,80,np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path=save_manual_verification(
                Path(directory)/"verification.png",frame,residual,selection)
            image=cv2.imread(str(path))
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[:2],(frame.shape[0],2*frame.shape[1]))

    def test_normal_calibration_session_uses_runtime_bsession_parameters(self):
        residuals=np.zeros((3,8,6,3),np.float32)
        valid=np.ones((3,8,6),bool)
        model=SimpleNamespace(
            residual_b_coefficients=np.zeros((3,5,4),np.float32),
            residual_m_coefficients=np.zeros((2,3,5,4),np.float32))
        expected=(np.zeros((3,3,5,4),np.float32),
                  np.zeros((3,3,3),np.float32),np.ones(3,np.float32),{})
        lightfield_cfg={
            "calibration":{
                "lambda_residual_smooth":.02,
                "lambda_residual_magnitude":.003,
                "residual_outer_weight":.4,
                "residual_outer_fraction":.1,
                "residual_bsession_prior_strength":.006,
                "residual_bsession_max_field_deviation":.45,
            },
            "runtime":{
                "residual_score_huber_delta":.07,
                "residual_channel_huber_ratio_min":.6,
                "residual_channel_huber_ratio_max":1.8,
            },
        }
        with patch(
                "calibrate_norm.fit_startup_residual_bsession_model",
                return_value=expected) as fit:
            actual=fit_normal_calibration_residual_session(
                residuals,valid,model,lightfield_cfg)
        self.assertIs(actual,expected)
        args=fit.call_args.args; kwargs=fit.call_args.kwargs
        self.assertIs(args[0],residuals); self.assertIs(args[1],valid)
        self.assertEqual(kwargs,{
            "huber_delta":.07,"smooth_lambda":.02,
            "magnitude_lambda":.003,"outer_weight":.4,
            "outer_fraction":.1,"bsession_prior_lambda":.006,
            "bsession_max_field_deviation":.45,
            "channel_huber_ratio_min":.6,
            "channel_huber_ratio_max":1.8,
        })

    def test_direct_normal_session_only_uses_additive_correction_bounds(self):
        residuals=np.zeros((3,8,6,3),np.float32)
        valid=np.ones((3,8,6),bool)
        model=SimpleNamespace(
            background_method="direct_fit",
            residual_b_coefficients=np.zeros((3,5,4),np.float32),
            residual_m_coefficients=np.zeros((0,3,5,4),np.float32))
        b=np.ones((3,5,4),np.float32)*.2
        scores=np.zeros((3,3,0),np.float32)
        expected=(b,scores,np.ones(3,np.float32),{})
        lightfield_cfg={
            "direct_fit":{"session_correction_max_deviation":.12},
            "calibration":{
                "lambda_residual_smooth":.02,
                "lambda_residual_magnitude":.003,
                "residual_outer_weight":.4,
                "residual_outer_fraction":.1,
                "residual_bsession_prior_strength":.006,
                "residual_bsession_max_field_deviation":.45,
            },
            "runtime":{
                "residual_score_huber_delta":.07,
                "residual_channel_huber_ratio_min":.6,
                "residual_channel_huber_ratio_max":1.8,
            },
        }
        with patch("calibrate_norm.fit_startup_direct_bsession_model",
                   return_value=expected) as fit:
            fields,actual_scores,channel_huber,diagnostics=(
                fit_normal_calibration_residual_session(
                    residuals,valid,model,lightfield_cfg))
        np.testing.assert_array_equal(fields[0],b)
        self.assertEqual(fields[1:].shape,(0,3,5,4))
        self.assertIs(actual_scores,scores)
        self.assertIs(channel_huber,expected[2])
        self.assertIs(diagnostics,expected[3])
        self.assertEqual(fit.call_args.kwargs["session_correction_bounds"],
                         (-.12,.12))

    def test_manual_samples_build_lut(self):
        residual,valid,xyz_image,surface,ellipse=self.synthetic_surface()
        selection=manual_contact_selection(
            ellipse,valid,valid,xyz_image,surface,5.,2)
        colors,slopes=sphere_slope_samples(
            residual,xyz_image,selection,5.)
        colors=colors.copy()
        phase=np.linspace(-1,1,colors.shape[0],dtype=np.float32)
        colors[:,0]+=phase*.02; colors[:,1]-=phase*.01
        model=build_normal_lut(
            colors,slopes,size=16,maximum_rms_angle_degrees=15.)
        self.assertEqual(model.slopes.shape,(16,16,16,2))
        self.assertTrue(np.isfinite(model.slopes).all())
        self.assertGreater(np.count_nonzero(model.original_valid),0)

    def test_pq_rgb_mapping_visualization_is_written(self):
        p,q=np.meshgrid(
            np.linspace(-1,1,21),np.linspace(-1,1,21),indexing="xy")
        slopes=np.stack([p.ravel(),q.ravel()],axis=-1).astype(np.float32)
        colors=np.stack([
            .1*p.ravel(),.1*q.ravel(),np.full(p.size,-.025)],axis=-1).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=save_pq_rgb_mapping(
                Path(directory)/"pq_rgb.png",colors,slopes,resolution=192)
            image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertGreater(image.shape[0],192)
        self.assertGreater(image.shape[1],192)

    def test_pq_rgb_mapping_hides_display_outlier_but_still_writes(self):
        slopes=np.zeros((401,2),np.float32)
        slopes[:400,0]=np.linspace(-1,1,400,dtype=np.float32)
        slopes[-1]=[500.,-400.]
        colors=np.zeros((401,3),np.float32)
        messages=[]
        original_put_text=cv2.putText

        def record_text(image,text,*args,**kwargs):
            messages.append(text)
            return original_put_text(image,text,*args,**kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch("calibrate_norm.cv2.putText",side_effect=record_text):
            path=save_pq_rgb_mapping(
                Path(directory)/"pq_rgb.png",colors,slopes,resolution=192,
                display_percentile=99.5)
            self.assertTrue(path.exists())
        self.assertTrue(any("hidden=1" in text for text in messages))
        self.assertTrue(any("raw-max=500.000" in text for text in messages))


if __name__=="__main__":
    unittest.main()
