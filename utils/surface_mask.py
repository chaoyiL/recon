"""曲面分割后端共用的提示类型与二值 mask 后处理。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

import cv2
import numpy as np


Point: TypeAlias = tuple[float, float]
PromptGroup: TypeAlias = Mapping[str, Sequence[Point]]
Prompts: TypeAlias = Mapping[str | int, PromptGroup]


@dataclass(frozen=True)
class MaskRefineConfig:
    """二值 mask 后处理开关；算法只保留最大区域并填洞。"""

    enabled: bool = True


def largest_filled_mask(mask: np.ndarray) -> np.ndarray:
    """只保留最大外轮廓并填充内部孔洞，不修改轮廓形状。"""
    binary=np.asarray(mask,dtype=np.bool_)
    if binary.ndim!=2:
        raise ValueError("mask 必须是二维数组")
    contours,_=cv2.findContours(
        binary.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(binary)
    contour=max(contours,key=cv2.contourArea)
    filled=np.zeros_like(binary,dtype=np.uint8)
    cv2.drawContours(filled,[contour],-1,1,thickness=cv2.FILLED)
    return filled.astype(np.bool_)


def refine_mask(
    mask: np.ndarray,
    config: MaskRefineConfig,
) -> np.ndarray:
    """只保留最大外轮廓并填洞；不修正任何圆角或端点。"""
    if not config.enabled:
        return np.asarray(mask, dtype=np.bool_)
    return largest_filled_mask(mask)
