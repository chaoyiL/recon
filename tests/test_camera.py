import unittest
from unittest.mock import MagicMock, patch

import cv2

from utils.camera import open_camera
from utils.config import ConfigError, parse_camera_config


class CameraControlTest(unittest.TestCase):
    def test_camera_config_requires_manual_white_balance_temperature(self):
        config=parse_camera_config({
            "device":4,"exposure":170,"white_balance_temperature":4600,
            "width":None,"height":None})
        self.assertEqual(config.white_balance_temperature,4600.)
        with self.assertRaisesRegex(ConfigError,"white_balance_temperature"):
            parse_camera_config({
                "device":4,"exposure":170,"width":None,"height":None})

    @patch("utils.camera.cv2.VideoCapture")
    def test_open_camera_locks_and_verifies_white_balance(self,video_capture):
        cap=MagicMock()
        cap.isOpened.return_value=True
        cap.set.return_value=True
        cap.get.side_effect=lambda prop: {
            cv2.CAP_PROP_AUTO_WB:0.,cv2.CAP_PROP_WB_TEMPERATURE:4600.,
        }.get(prop,0.)
        video_capture.return_value=cap

        actual=open_camera(4,170,4600,None,None)

        self.assertIs(actual,cap)
        cap.set.assert_any_call(cv2.CAP_PROP_AUTO_WB,0)
        cap.set.assert_any_call(cv2.CAP_PROP_WB_TEMPERATURE,4600)

    @patch("utils.camera.cv2.VideoCapture")
    def test_open_camera_rejects_ignored_manual_white_balance(self,video_capture):
        cap=MagicMock()
        cap.isOpened.return_value=True
        cap.set.return_value=True
        cap.get.side_effect=lambda prop: {
            cv2.CAP_PROP_AUTO_WB:1.,cv2.CAP_PROP_WB_TEMPERATURE:4600.,
        }.get(prop,0.)
        video_capture.return_value=cap

        with self.assertRaisesRegex(RuntimeError,"自动白平衡未成功关闭"):
            open_camera(4,170,4600,None,None)
        cap.release.assert_called_once()


if __name__=="__main__":
    unittest.main()
