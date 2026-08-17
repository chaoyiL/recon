"""共享的相机初始化。"""

from __future__ import annotations

import math

import cv2


def open_camera(
    device: int,
    exposure: float,
    white_balance_temperature: float,
    width: int | None,
    height: int | None,
) -> cv2.VideoCapture:
    """打开 V4L2 相机并锁定曝光、白平衡及可选分辨率。"""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机 device={device}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        # V4L2: 1=手动曝光，3=自动曝光。
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        if not cap.set(cv2.CAP_PROP_AUTO_WB, 0):
            raise RuntimeError("相机驱动不支持或拒绝关闭自动白平衡")
        if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance_temperature):
            raise RuntimeError(
                f"相机驱动不支持或拒绝设置白平衡色温 {white_balance_temperature} K")
        actual_auto_wb=float(cap.get(cv2.CAP_PROP_AUTO_WB))
        actual_temperature=float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
        if not math.isfinite(actual_auto_wb) or actual_auto_wb<0 or actual_auto_wb>.5:
            raise RuntimeError(
                f"自动白平衡未成功关闭，驱动回读值={actual_auto_wb}")
        tolerance=max(250.,.1*float(white_balance_temperature))
        if (
            not math.isfinite(actual_temperature)
            or actual_temperature<=0
            or abs(actual_temperature-white_balance_temperature)>tolerance
        ):
            raise RuntimeError(
                "手动白平衡设置未生效："
                f"请求={white_balance_temperature} K，回读={actual_temperature} K")
    except Exception:
        cap.release()
        raise
    return cap
