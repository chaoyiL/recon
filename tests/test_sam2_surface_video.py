import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.sam2_surface import MaskRefineConfig, SurfaceSegmenter


class _Inputs(dict):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.pixel_values = torch.zeros(1, 3, 8, 8)
        self.original_sizes = torch.tensor([[height, width]])

    def to(self, _device):
        return self


class _Session:
    def __init__(self) -> None:
        self.processed_frames = {}
        self.output_dict_per_obj = {
            0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        }
        self.frames_tracked_per_obj = {0: {}}
        self.reset_count = 0

    def reset_inference_session(self) -> None:
        self.reset_count += 1


class _Processor:
    def __init__(self) -> None:
        self.sessions: list[_Session] = []
        self.add_calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        raise AssertionError("patched instance should be used")

    def init_video_session(self, **_kwargs):
        session = _Session()
        self.sessions.append(session)
        return session

    def add_inputs_to_inference_session(self, **kwargs):
        self.add_calls.append(kwargs)

    def __call__(self, images, **_kwargs):
        return _Inputs(images.height, images.width)

    def post_process_masks(self, masks, **_kwargs):
        return [masks[0]]


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_maskmem=3,max_object_pointers_in_encoder=16)
        self.num_maskmem=3
        self.frame_indices: list[int] = []
        self.memory_encoder_calls: list[bool] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, *, inference_session, frame, frame_idx,
                 run_mem_encoder=True):
        del frame
        self.frame_indices.append(frame_idx)
        self.memory_encoder_calls.append(run_mem_encoder)
        inference_session.processed_frames[frame_idx] = torch.zeros(1)
        inference_session.output_dict_per_obj[0]["non_cond_frame_outputs"][frame_idx] = {}
        inference_session.frames_tracked_per_obj[0][frame_idx] = {}
        return SimpleNamespace(
            object_ids=[1],
            pred_masks=torch.ones(1, 1, 8, 8),
            frame_idx=frame_idx,
        )


class SurfaceSegmenterVideoTest(unittest.TestCase):
    def test_device_tensor_api_keeps_fixed_mask_stack(self) -> None:
        processor = _Processor()
        model = _Model()
        with (
            patch(
                "utils.sam2_surface.Sam2VideoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch(
                "utils.sam2_surface.Sam2VideoModel.from_pretrained",
                return_value=model,
            ),
        ):
            segmenter = SurfaceSegmenter(
                model_id="fake",device="cpu",
                mask_refine=MaskRefineConfig(enabled=False))
        frame=np.zeros((8,8,3),np.uint8)
        labels,masks,frame_tensor=segmenter.segment_tensors(
            frame,{"surface":{"positive":[(3.,3.)]}})
        self.assertEqual(labels,("surface",))
        self.assertEqual(tuple(masks.shape),(1,8,8))
        self.assertEqual(masks.dtype,torch.bool)
        self.assertEqual(tuple(frame_tensor.shape),(8,8,3))
        self.assertEqual(frame_tensor.dtype,torch.uint8)

    def test_prompts_are_added_once_and_recent_history_is_bounded(self) -> None:
        processor = _Processor()
        model = _Model()
        with (
            patch(
                "utils.sam2_surface.Sam2VideoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch(
                "utils.sam2_surface.Sam2VideoModel.from_pretrained",
                return_value=model,
            ),
        ):
            segmenter = SurfaceSegmenter(
                model_id="fake",
                device="cpu",
                mask_refine=MaskRefineConfig(enabled=False),
            )

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        prompts = {"surface": {"positive": [(3.0, 3.0)]}}
        for _ in range(5):
            result = segmenter.segment(frame, prompts)

        self.assertEqual(model.frame_indices, [0, 1, 2, 3, 4])
        self.assertEqual(len(processor.sessions), 1)
        self.assertEqual(len(processor.add_calls), 1)
        self.assertEqual(set(processor.sessions[0].processed_frames), {2, 3, 4})
        self.assertIn("surface", result)
        self.assertIsInstance(result["surface"], np.ndarray)
        self.assertEqual(result["surface"].shape, frame.shape[:2])
        self.assertEqual(result["surface"].dtype, np.bool_)

    def test_prompt_change_resets_video_session(self) -> None:
        processor = _Processor()
        model = _Model()
        with (
            patch(
                "utils.sam2_surface.Sam2VideoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch(
                "utils.sam2_surface.Sam2VideoModel.from_pretrained",
                return_value=model,
            ),
        ):
            segmenter = SurfaceSegmenter(
                model_id="fake",
                device="cpu",
                mask_refine=MaskRefineConfig(enabled=False),
            )

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        segmenter.segment(frame, {"surface": {"positive": [(2.0, 2.0)]}})
        first_session = processor.sessions[0]
        segmenter.segment(frame, {"surface": {"positive": [(3.0, 2.0)]}})

        self.assertEqual(first_session.reset_count, 1)
        self.assertEqual(len(processor.sessions), 2)
        self.assertEqual(model.frame_indices, [0, 0])
        self.assertEqual(len(processor.add_calls), 2)

    def test_single_memory_frame_skips_unused_memory_encoder(self) -> None:
        processor=_Processor(); model=_Model()
        with (
            patch(
                "utils.sam2_surface.Sam2VideoProcessor.from_pretrained",
                return_value=processor),
            patch(
                "utils.sam2_surface.Sam2VideoModel.from_pretrained",
                return_value=model),
        ):
            segmenter=SurfaceSegmenter(
                model_id="fake",device="cpu",memory_frames=1,
                mask_refine=MaskRefineConfig(enabled=False))

        frame=np.zeros((8,8,3),np.uint8)
        prompts={"surface":{"positive":[(3.,3.)]}}
        for _ in range(3):
            segmenter.segment_tensors(frame,prompts)

        self.assertEqual(model.num_maskmem,1)
        self.assertEqual(model.config.num_maskmem,1)
        self.assertEqual(model.config.max_object_pointers_in_encoder,1)
        self.assertEqual(model.memory_encoder_calls,[True,False,False])
        self.assertEqual(set(processor.sessions[0].processed_frames),{2})


if __name__ == "__main__":
    unittest.main()
