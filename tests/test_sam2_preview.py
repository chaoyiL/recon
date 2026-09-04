import unittest

import numpy as np

from sam2_preview import (draw_sam2_prompt_mask_overlay,
                          parse_preview_mask_refine,
                          parse_preview_prompts)
from utils.config import ConfigError


class Sam2PreviewTest(unittest.TestCase):
    def test_preview_reads_shared_largest_fill_switch(self):
        self.assertFalse(
            parse_preview_mask_refine({"enabled":False}).enabled)
        with self.assertRaisesRegex(ConfigError,"未知字段"):
            parse_preview_mask_refine({
                "minimum_intersection_angle_deg":23.5})

    def test_overlay_draws_mask_boundary_and_prompt_points(self):
        frame=np.full((80,120,3),40,np.uint8)
        mask=np.zeros((80,120),bool)
        mask[20:60,30:90]=True
        prompts={"surface":{
            "positive":[(45.,35.)],"negative":[(100.,65.)]}}
        overlay=draw_sam2_prompt_mask_overlay(
            frame,prompts,{"surface":mask})
        self.assertEqual(overlay.shape,frame.shape)
        self.assertEqual(overlay.dtype,np.uint8)
        self.assertFalse(np.array_equal(overlay[40,60],frame[40,60]))
        self.assertFalse(np.array_equal(overlay[20,60],frame[20,60]))
        self.assertFalse(np.array_equal(overlay[35,45],frame[35,45]))
        self.assertFalse(np.array_equal(overlay[65,100],frame[65,100]))
        np.testing.assert_array_equal(overlay[5,5],frame[5,5])

    def test_overlay_rejects_wrong_mask_shape(self):
        with self.assertRaisesRegex(ValueError,"mask 尺寸"):
            draw_sam2_prompt_mask_overlay(
                np.zeros((20,30,3),np.uint8),
                {"surface":{"positive":[(5.,5.)]}},
                {"surface":np.zeros((19,30),bool)})

    def test_prompt_parser_rejects_empty_positive_points(self):
        with self.assertRaisesRegex(ConfigError,"positive"):
            parse_preview_prompts({"surface":{"negative":[[1,2]]}})


if __name__=="__main__":
    unittest.main()
