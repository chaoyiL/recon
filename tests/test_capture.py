import tempfile
import unittest
from pathlib import Path
from capture import capture_output_path, recording_fps, video_output_path


class CaptureOutputPathTest(unittest.TestCase):
    def test_image_file_is_used_directly(self):
        path = Path("result.png")
        self.assertEqual(capture_output_path(path), path)

    def test_directory_gets_unique_png_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "calibration"
            first = capture_output_path(target)
            self.assertEqual(first.parent, target)
            self.assertEqual(first.suffix, ".png")
            first.touch()
            second = capture_output_path(target)
            self.assertNotEqual(first, second)

    def test_video_file_is_used_directly(self):
        path = Path("recording.avi")
        self.assertEqual(video_output_path(path), path)

    def test_video_directory_gets_unique_lossless_avi_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "videos"
            first = video_output_path(target)
            self.assertEqual(first.parent, target)
            self.assertEqual(first.suffix, ".avi")
            first.touch()
            second = video_output_path(target)
            self.assertNotEqual(first, second)

    def test_invalid_camera_fps_uses_fallback(self):
        self.assertEqual(recording_fps(0.0), 30.0)
        self.assertEqual(recording_fps(float("nan")), 30.0)
        self.assertEqual(recording_fps(60.0), 60.0)


if __name__ == "__main__":
    unittest.main()
