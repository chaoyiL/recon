"""独立实时显示 SAM2 prompt 位置与分割 mask 范围。"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.camera import open_camera
from utils.config import ConfigError, load_config, parse_camera_config
from utils.sam2_surface import (MaskRefineConfig,Prompts,SurfaceSegmenter,
                                refine_mask)


DEFAULT_CONFIG_PATH=Path(__file__).with_name("config.yaml")
WINDOW_NAME="SAM2 prompt + mask"


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="实时显示 SAM2 prompt 位置与提取出的 mask 范围")
    parser.add_argument(
        "--config",default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置文件，默认 {DEFAULT_CONFIG_PATH}")
    return parser.parse_args()


def _parse_points(
    value: object,label: str | int,point_type: str,
) -> list[tuple[float,float]]:
    if not isinstance(value,Sequence) or isinstance(value,(str,bytes)):
        raise ConfigError(f"label {label!r} 的 {point_type} 必须是点列表")
    result=[]
    for raw_point in value:
        if not isinstance(raw_point,Sequence) \
                or isinstance(raw_point,(str,bytes)) or len(raw_point)!=2:
            raise ConfigError(f"label {label!r} 的点必须是 [x, y]")
        try:
            point=(float(raw_point[0]),float(raw_point[1]))
        except (TypeError,ValueError) as error:
            raise ConfigError(
                f"label {label!r} 的点坐标必须是数字") from error
        if not np.isfinite(point).all():
            raise ConfigError(f"label {label!r} 的点坐标必须是有限数")
        result.append(point)
    return result


def parse_preview_prompts(value: object) -> Prompts:
    if not isinstance(value,Mapping) or not value:
        raise ConfigError("get_surface.prompts 必须是非空字典")
    prompts: dict[str | int,dict[str,list[tuple[float,float]]]]={}
    for label,raw_group in value.items():
        if not isinstance(label,(str,int)) or isinstance(label,bool):
            raise ConfigError("prompt label 必须是字符串或整数")
        if not isinstance(raw_group,Mapping):
            raise ConfigError(f"label {label!r} 的配置必须是字典")
        unknown=set(raw_group)-{"positive","negative"}
        if unknown:
            raise ConfigError(
                f"label {label!r} 包含未知字段: {sorted(unknown)}")
        positive=_parse_points(raw_group.get("positive",[]),label,"positive")
        negative=_parse_points(raw_group.get("negative",[]),label,"negative")
        if not positive:
            raise ConfigError(f"label {label!r} 至少需要一个 positive 点")
        prompts[label]={"positive":positive,"negative":negative}
    return prompts


def parse_preview_mask_refine(value: object) -> MaskRefineConfig:
    if value is None:
        return MaskRefineConfig()
    if not isinstance(value,Mapping):
        raise ConfigError("get_surface.mask_refine 必须是字典或 null")
    known={"enabled"}
    unknown=set(value)-known
    if unknown:
        raise ConfigError(
            f"get_surface.mask_refine 包含未知字段: {sorted(unknown)}")
    enabled=value.get("enabled",MaskRefineConfig.enabled)
    if not isinstance(enabled,bool):
        raise ConfigError("mask_refine.enabled 必须是布尔值")
    return MaskRefineConfig(enabled=enabled)


def _label_color(label: str | int) -> tuple[int,int,int]:
    digest=hashlib.sha256(repr(label).encode("utf-8")).digest()
    return tuple(80+channel%176 for channel in digest[:3])


def draw_sam2_prompt_mask_overlay(
    frame: np.ndarray,
    prompts: Prompts,
    results: Mapping[str | int,np.ndarray],
    *,
    mask_alpha: float=.28,
) -> np.ndarray:
    """叠加 mask 填充/轮廓，以及带坐标的正负 prompt。"""
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 \
            or frame.ndim!=3 or frame.shape[2]!=3:
        raise ValueError("frame 必须是 HxWx3 的 uint8 BGR 图像")
    if not np.isfinite(mask_alpha) or not 0<=mask_alpha<=1:
        raise ValueError("mask_alpha 必须在 [0, 1] 范围内")
    height,width=frame.shape[:2]
    visualization=frame.copy()
    color_layer=frame.copy()
    contours_by_label: dict[str | int,list[np.ndarray]]={}
    for label,raw_mask in results.items():
        mask=np.asarray(raw_mask,dtype=np.bool_)
        if mask.shape!=(height,width):
            raise ValueError(f"label {label!r} 的 mask 尺寸与相机帧不一致")
        color_layer[mask]=_label_color(label)
        contours,_=cv2.findContours(
            mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours_by_label[label]=contours
    if results and mask_alpha>0:
        visualization=cv2.addWeighted(
            visualization,1-mask_alpha,color_layer,mask_alpha,0)
    for label,contours in contours_by_label.items():
        if contours:
            cv2.drawContours(
                visualization,contours,-1,_label_color(label),2,cv2.LINE_AA)

    for label,group in prompts.items():
        color=_label_color(label)
        for point_type,points in (
            ("+",group.get("positive",())),
            ("-",group.get("negative",())),
        ):
            for x,y in points:
                point=(round(x),round(y))
                if point_type=="+":
                    cv2.circle(visualization,point,6,color,-1,cv2.LINE_AA)
                    cv2.circle(
                        visualization,point,8,(255,255,255),2,cv2.LINE_AA)
                else:
                    cv2.drawMarker(
                        visualization,point,color,cv2.MARKER_TILTED_CROSS,
                        16,3,cv2.LINE_AA)
                    cv2.circle(
                        visualization,point,9,(255,255,255),1,cv2.LINE_AA)
                text=f"{point_type}{label} ({x:.0f},{y:.0f})"
                origin=(
                    min(max(point[0]+11,2),max(width-180,2)),
                    min(max(point[1]-9,16),max(height-4,16)))
                cv2.putText(
                    visualization,text,origin,cv2.FONT_HERSHEY_SIMPLEX,
                    .45,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(
                    visualization,text,origin,cv2.FONT_HERSHEY_SIMPLEX,
                    .45,color,1,cv2.LINE_AA)
    return visualization


def _load_settings(config_path: Path) -> tuple[Any,Mapping[str,Any],Prompts,
                                                      MaskRefineConfig]:
    config=load_config(config_path)
    if not isinstance(config,Mapping):
        raise ConfigError("配置根节点必须是字典")
    camera=parse_camera_config(config.get("camera"))
    surface=config.get("get_surface")
    if not isinstance(surface,Mapping):
        raise ConfigError("get_surface 必须是字典")
    prompts=parse_preview_prompts(surface.get("prompts"))
    refine=parse_preview_mask_refine(surface.get("mask_refine"))
    return camera,surface,prompts,refine


def main() -> None:
    args=parse_args()
    config_path=Path(args.config).expanduser().resolve()
    try:
        camera,surface,prompts,mask_refine=_load_settings(config_path)
        segmentation=surface.get("segmentation",{})
        if not isinstance(segmentation,Mapping):
            raise ConfigError("get_surface.segmentation 必须是字典")
        sam2=segmentation.get("sam2",{})
        if not isinstance(sam2,Mapping):
            raise ConfigError("get_surface.segmentation.sam2 必须是字典")
        model_id=sam2.get("model",surface.get("model"))
        compile_model=sam2.get(
            "torch_compile",surface.get("torch_compile",True))
        memory_frames=sam2.get(
            "memory_frames",surface.get("sam_memory_frames",4))
        frame_interval=sam2.get(
            "frame_interval",surface.get("sam_frame_interval",1))
        if not isinstance(model_id,str) or not model_id:
            raise ConfigError("get_surface.model 必须是非空字符串")
        if not isinstance(compile_model,bool):
            raise ConfigError("get_surface.torch_compile 必须是布尔值")
        if not isinstance(memory_frames,int) or isinstance(memory_frames,bool) \
                or memory_frames<1:
            raise ConfigError("get_surface.sam_memory_frames 必须为正整数")
        if not isinstance(frame_interval,int) or isinstance(frame_interval,bool) \
                or frame_interval<1:
            raise ConfigError("get_surface.sam_frame_interval 必须为正整数")
    except (ConfigError,OSError,ValueError) as error:
        print(f"配置错误: {error}",file=sys.stderr)
        raise SystemExit(2) from error

    print(f"加载 SAM2: {model_id}")
    segmenter=SurfaceSegmenter(
        model_id=model_id,mask_refine=mask_refine,
        compile_model=compile_model,memory_frames=memory_frames)
    cap=open_camera(
        camera.device,camera.exposure,camera.white_balance_temperature,
        camera.width,camera.height)
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    print("圆点=positive prompt，叉号=negative prompt；按 q 或 Esc 退出。")
    frame_number=0
    fps=0.
    masks: dict[str | int,np.ndarray]={}
    try:
        while True:
            started=time.perf_counter()
            ok,frame=cap.read()
            if not ok or frame is None:
                raise RuntimeError("读取相机帧失败")
            if not masks or frame_number%frame_interval==0:
                labels,mask_tensor,_=segmenter.segment_tensors(frame,prompts)
                mask_values=np.asarray(
                    mask_tensor.detach().cpu(),dtype=np.bool_)
                masks={
                    label:np.ascontiguousarray(
                        refine_mask(mask_values[index],mask_refine),
                        dtype=np.bool_)
                    for index,label in enumerate(labels)}
            frame_number+=1
            instantaneous_fps=1/max(time.perf_counter()-started,1e-9)
            fps=instantaneous_fps if fps==0 else .85*fps+.15*instantaneous_fps
            display=draw_sam2_prompt_mask_overlay(frame,prompts,masks)
            fps_text=f"FPS: {fps:.1f}  SAM interval: {frame_interval}"
            cv2.putText(
                display,fps_text,(12,30),cv2.FONT_HERSHEY_SIMPLEX,
                .65,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(
                display,fps_text,(12,30),cv2.FONT_HERSHEY_SIMPLEX,
                .65,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME,display)
            if cv2.waitKey(1)&0xff in (ord("q"),27):
                break
    finally:
        cap.release()
        del segmenter
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
