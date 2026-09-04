from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml


TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from common import (  # noqa: E402
    DEFAULT_SETTINGS_PATH,
    DatasetSample,
    SettingsError,
    discover_videos,
    read_manifest,
    split_samples_by_video,
    validate_dataset_samples,
    write_paddleseg_split,
    load_settings,
)
from auto_label import label_video  # noqa: E402
from inference import (  # noqa: E402
    locate_model_files,
    output_to_label_map,
    parse_fixed_input_shape,
    parse_normalize,
    preprocess_frame,
)
from realtime import refine_prediction_mask  # noqa: E402
from train import build_paddleseg_config, dump_yaml, repair_deploy_model_name  # noqa: E402
from utils.surface_mask import MaskRefineConfig  # noqa: E402


def sample(group: str, index: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"{group}-{index}",
        source_video=f"/{group}.avi",
        source_frame=index,
        timestamp_seconds=float(index),
        image=f"images/{group}/{index}.png",
        mask=f"masks/{group}/{index}.png",
        width=8,
        height=6,
        mask_area_ratio=0.5,
    )


class CommonTests(unittest.TestCase):
    def test_default_settings_are_loaded_from_main_config(self) -> None:
        path,settings=load_settings()
        self.assertEqual(path,DEFAULT_SETTINGS_PATH.resolve())
        self.assertIn("training",settings)
        self.assertIn("labeling",settings)

    def test_group_split_never_leaks_a_video(self) -> None:
        values = [sample(group, index) for group in ("a", "b", "c") for index in range(4)]
        train, val, mode = split_samples_by_video(values, val_ratio=0.25, seed=7)
        self.assertEqual(mode, "source_video")
        self.assertFalse(
            {item.source_video for item in train} & {item.source_video for item in val})
        self.assertTrue(train)
        self.assertTrue(val)

    def test_single_video_uses_temporal_tail(self) -> None:
        values = [sample("only", index) for index in range(10)]
        train, val, mode = split_samples_by_video(values, val_ratio=0.2, seed=1)
        self.assertEqual(mode, "temporal_tail")
        self.assertEqual([item.source_frame for item in val], [8, 9])
        self.assertEqual(len(train), 8)

    def test_discover_videos_expands_directory_and_glob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.avi").touch()
            (root / "b.mp4").touch()
            (root / "ignored.txt").touch()
            values = discover_videos([str(root), str(root / "*.avi")], base=root)
            self.assertEqual([path.name for path in values], ["a.avi", "b.mp4"])

    def test_dataset_validation_and_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [sample("a", 0), sample("a", 1), sample("b", 0)]
            for item in values:
                image_path = root / item.image
                mask_path = root / item.mask
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), np.zeros((6, 8, 3), np.uint8))
                mask = np.zeros((6, 8), np.uint8)
                mask[:, 2:6] = 1
                cv2.imwrite(str(mask_path), mask)
            self.assertEqual(validate_dataset_samples(root, values), (6, 8))
            write_paddleseg_split(root, values[:2], values[2:], mode="source_video", seed=3)
            self.assertEqual(len((root / "train.txt").read_text().splitlines()), 2)
            split = json.loads((root / "split.json").read_text())
            self.assertEqual(split["val_count"], 1)

    def test_dataset_validation_rejects_255_binary_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = sample("a", 0)
            (root / item.image).parent.mkdir(parents=True)
            (root / item.mask).parent.mkdir(parents=True)
            cv2.imwrite(str(root / item.image), np.zeros((6, 8, 3), np.uint8))
            cv2.imwrite(str(root / item.mask), np.full((6, 8), 255, np.uint8))
            with self.assertRaises(SettingsError):
                validate_dataset_samples(root, [item])


class TrainingConfigTests(unittest.TestCase):
    def test_t1_config_has_three_auxiliary_losses_and_no_yaml_alias(self) -> None:
        config = build_paddleseg_config(
            dataset_dir=Path("/dataset"),
            height=480,
            width=640,
            iterations=100,
            batch_size=2,
            learning_rate=0.005,
        )
        self.assertEqual(config["model"]["backbone"]["type"], "STDC1")
        self.assertEqual(len(config["loss"]["types"]), 3)
        dumped = dump_yaml(config)
        self.assertNotIn("&id", dumped)
        self.assertNotIn("*id", dumped)

    def test_repair_deploy_for_paddle3_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.json").write_text("{}")
            (root / "model.pdiparams").write_bytes(b"params")
            (root / "deploy.yaml").write_text(
                yaml.safe_dump({"Deploy": {"model": "model.pdmodel", "params": "model.pdiparams"}}))
            repair_deploy_model_name(root)
            deploy = yaml.safe_load((root / "deploy.yaml").read_text())["Deploy"]
            self.assertEqual(deploy["model"], "model.json")
            model, params = locate_model_files(root, deploy)
            self.assertEqual(model.name, "model.json")
            self.assertEqual(params.name, "model.pdiparams")


class InferenceTests(unittest.TestCase):
    def test_realtime_refine_keeps_largest_component_and_fills_holes(self) -> None:
        raw = np.zeros((100, 120), np.uint8)
        cv2.fillPoly(
            raw,[np.asarray([[18,8],[102,8],[84,72],[36,72]],np.int32)],1)
        raw[30:34,58:62] = 0
        raw[1:3, 1:3] = 1
        refined = refine_prediction_mask(
            raw.astype(bool),MaskRefineConfig(enabled=True),
        )
        component_count = cv2.connectedComponents(refined.astype(np.uint8), 8)[0] - 1
        self.assertEqual(component_count, 1)
        self.assertTrue(refined[31, 60])
        self.assertFalse(refined[1, 1])

    def test_preprocess_matches_paddleseg_rgb_normalize(self) -> None:
        frame = np.zeros((2, 3, 3), np.uint8)
        frame[..., 0] = 255  # BGR blue becomes RGB channel 2.
        mean = np.asarray([0.5, 0.5, 0.5], np.float32).reshape(1, 1, 3)
        std = np.asarray([0.5, 0.5, 0.5], np.float32).reshape(1, 1, 3)
        tensor, original = preprocess_frame(
            frame, input_shape=(4, 6), mean=mean, std=std)
        self.assertEqual(original, (2, 3))
        self.assertEqual(tensor.shape, (1, 3, 4, 6))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), -1.0)
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), 1.0)

    def test_output_decoder_accepts_argmax_and_logits(self) -> None:
        argmax = np.asarray([[[0, 1], [1, 0]]], np.int32)
        np.testing.assert_array_equal(output_to_label_map(argmax), argmax[0])
        logits = np.asarray([[[[2, 0]], [[0, 3]]]], np.float32)
        np.testing.assert_array_equal(output_to_label_map(logits), [[0, 1]])

    def test_deploy_parsers(self) -> None:
        deploy = {
            "input_shape": [1, 3, 480, 640],
            "transforms": [{"type": "Normalize", "mean": [0.5], "std": [0.25]}],
        }
        self.assertEqual(parse_fixed_input_shape(deploy), (480, 640))
        mean, std = parse_normalize(deploy)
        self.assertEqual(mean.shape, (1, 1, 3))
        self.assertTrue(np.all(std == 0.25))


class AutoLabelTests(unittest.TestCase):
    def test_sam_memory_advances_every_frame_but_only_samples_are_saved(self) -> None:
        frames = [np.full((8, 10, 3), index, np.uint8) for index in range(6)]

        class FakeCapture:
            def __init__(self) -> None:
                self.index = 0

            def isOpened(self) -> bool:
                return True

            def get(self, key: int) -> float:
                return 4.0 if key == cv2.CAP_PROP_FPS else 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.index >= len(frames):
                    return False, None
                frame = frames[self.index]
                self.index += 1
                return True, frame

            def release(self) -> None:
                pass

        class FakeSegmenter:
            def __init__(self) -> None:
                self.seen: list[int] = []
                self.reset_count = 0

            def reset(self) -> None:
                self.reset_count += 1

            def segment(self, frame: np.ndarray, prompts: object) -> dict[str, np.ndarray]:
                self.seen.append(int(frame[0, 0, 0]))
                mask = np.zeros(frame.shape[:2], np.bool_)
                mask[:, 2:7] = True
                return {"surface": mask}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segmenter = FakeSegmenter()
            with patch("auto_label.cv2.VideoCapture", return_value=FakeCapture()):
                accepted, rejected = label_video(
                    Path("sequence.avi"),
                    dataset_dir=root,
                    segmenter=segmenter,
                    prompts={"surface": {"positive": [(1.0, 1.0)]}},
                    prompt_labels=["surface"],
                    sample_fps=2.0,
                    minimum_area=0.1,
                    maximum_area=0.9,
                    save_previews=False,
                    existing_ids=set(),
                    resume=False,
                    max_samples=0,
                )
            samples = read_manifest(root)

        self.assertEqual((accepted, rejected), (3, 0))
        self.assertEqual(segmenter.seen, list(range(6)))
        self.assertEqual(segmenter.reset_count, 2)
        self.assertEqual([item.source_frame for item in samples], [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
