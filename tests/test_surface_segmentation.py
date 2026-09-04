import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from get_surface import parse_mask_refine
from utils.config import ConfigError

from utils.surface_mask import MaskRefineConfig
from utils.liteseg_surface import preprocess_frame
from utils.surface_segmentation import (
    SurfaceSegmentationBackend,masks_to_numpy,refine_masks_numpy,
    parse_surface_segmentation_config)


class SurfaceSegmentationConfigTest(unittest.TestCase):
    def test_mask_refine_only_accepts_largest_fill_switch(self):
        self.assertFalse(parse_mask_refine({"enabled":False}).enabled)
        self.assertTrue(parse_mask_refine(None).enabled)
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_mask_refine({"minimum_intersection_angle_deg":24.25})

    def test_fast_liteseg_preprocess_matches_explicit_normalization(self):
        rng=np.random.default_rng(20260831)
        frame=rng.integers(0,256,size=(7,9,3),dtype=np.uint8)
        mean=np.full((1,1,3),.5,np.float32)
        std=np.full((1,1,3),.5,np.float32)
        actual,original=preprocess_frame(
            frame,input_shape=(11,13),mean=mean,std=std)
        resized=cv2.resize(frame,(13,11),interpolation=cv2.INTER_LINEAR)
        expected=(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)
                  /255.-mean)/std
        expected=np.ascontiguousarray(expected.transpose(2,0,1)[None])
        self.assertEqual(original,(7,9))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-7)

    def test_liteseg_config_resolves_model_relative_to_main_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path=Path(directory)/"config.yaml"
            parsed=parse_surface_segmentation_config({
                "segmentation":{
                    "mode":"liteseg",
                    "liteseg":{
                        "model_dir":"models/surface",
                        "device":"cpu",
                        "frame_interval":2,
                        "foreground_class":1,
                        "label":"surface",
                    },
                },
            },config_path=config_path)
        self.assertEqual(parsed.mode,"liteseg")
        self.assertEqual(parsed.liteseg_model_dir,
                         config_path.parent/"models/surface")
        self.assertEqual(parsed.frame_interval,2)

    def test_non_liteseg_runtime_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"只允许 liteseg"):
            parse_surface_segmentation_config({
                "segmentation":{"mode":"direct_fit"}},
                config_path="config.yaml")

    def test_sam2_runtime_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"自动标注教师"):
            parse_surface_segmentation_config({
                "segmentation":{
                    "mode":"sam2",
                    "sam2":{"model":"facebook/sam2.1-hiera-base-plus"},
                }},
                config_path="config.yaml")

    def test_liteseg_backend_returns_single_named_mask(self):
        prediction=SimpleNamespace(mask=np.ones((3,4),np.bool_))
        with patch("utils.liteseg_surface.PaddleSegPredictor") as predictor_type:
            predictor_type.return_value.predict.return_value=prediction
            config=parse_surface_segmentation_config({
                "segmentation":{
                    "mode":"liteseg",
                    "liteseg":{"model_dir":"model","device":"cpu"},
                },
            },config_path="/tmp/config.yaml")
            backend=SurfaceSegmentationBackend(
                config,prompts={"surface":{"positive":[(1.,1.)]}},
                mask_refine=MaskRefineConfig())
            labels,masks=backend.segment_tensors(
                np.zeros((3,4,3),np.uint8))
        self.assertEqual(labels,("surface",))
        np.testing.assert_array_equal(masks_to_numpy(masks),
                                      np.ones((1,3,4),np.bool_))

    def test_numpy_refine_uses_exact_largest_external_contour_semantics(self):
        raw=np.zeros((1,40,50),np.bool_)
        raw[0,5:35,10:40]=True
        raw[0,18:22,23:27]=False
        raw[0,1:3,1:3]=True
        refined=refine_masks_numpy(
            raw,MaskRefineConfig(enabled=True))
        component_count=cv2.connectedComponents(
            refined[0].astype(np.uint8),8)[0]-1
        self.assertEqual(component_count,1)
        self.assertTrue(refined[0,20,25])
        self.assertFalse(refined[0,1,1])


if __name__=="__main__":
    unittest.main()
