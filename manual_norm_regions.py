"""法向标定球的人工椭圆标注、持久化与 OpenCV 键盘编辑器。"""
from __future__ import annotations

from dataclasses import asdict,dataclass,replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.lightfield import signed_residual_bgr


@dataclass(frozen=True)
class ManualEllipse:
    use_for_lut: bool
    center_x: float
    center_y: float
    semi_axis_x: float
    semi_axis_y: float
    angle_degrees: float = 0.
    image_width: int = 0
    image_height: int = 0

    @classmethod
    def from_mapping(cls,raw: dict) -> "ManualEllipse":
        values=dict(raw)
        known={field for field in cls.__dataclass_fields__}
        unknown=set(values)-known
        if unknown:
            raise ValueError("人工椭圆包含未知字段: "+", ".join(sorted(unknown)))
        result=cls(**values)
        if result.image_width<1 or result.image_height<1 \
                or result.semi_axis_x<=0 or result.semi_axis_y<=0:
            raise ValueError("人工椭圆图像尺寸和半轴必须为正数")
        return result

    def to_mapping(self) -> dict:
        return asdict(self)


def annotation_key(image_path: str | Path) -> str:
    return str(Path(image_path).expanduser().resolve())


def load_manual_ellipses(path: str | Path) -> dict[str,ManualEllipse]:
    source=Path(path).expanduser()
    if not source.exists():
        return {}
    raw=yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if raw.get("version")!=1 or not isinstance(raw.get("regions"),dict):
        raise ValueError(f"人工椭圆文件格式无效: {source}")
    return {
        str(key):ManualEllipse.from_mapping(value)
        for key,value in raw["regions"].items()
    }


def save_manual_ellipses(
    path: str | Path,regions: dict[str,ManualEllipse],
) -> Path:
    output=Path(path).expanduser(); output.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "version":1,
        "coordinate_semantics":"source_image_pixels; axes_are_semi_axes",
        "regions":{
            key:regions[key].to_mapping() for key in sorted(regions)
        },
    }
    temporary=output.with_suffix(output.suffix+".tmp")
    temporary.write_text(
        yaml.safe_dump(payload,allow_unicode=True,sort_keys=False),encoding="utf-8")
    temporary.replace(output)
    return output


def default_manual_ellipse(
    valid_mask: np.ndarray,previous: ManualEllipse | None = None,
) -> ManualEllipse:
    valid=np.asarray(valid_mask,np.bool_); height,width=valid.shape
    if previous is not None:
        return replace(previous,image_width=width,image_height=height)
    rows,columns=np.nonzero(valid)
    if rows.size:
        left,right=float(columns.min()),float(columns.max())
        top,bottom=float(rows.min()),float(rows.max())
        center_x=(left+right)/2; center_y=(top+bottom)/2
        axis_x=max(8.,(right-left)*.10); axis_y=max(8.,(bottom-top)*.10)
    else:
        center_x=(width-1)/2; center_y=(height-1)/2
        axis_x=max(8.,width*.10); axis_y=max(8.,height*.10)
    return ManualEllipse(
        True,center_x,center_y,axis_x,axis_y,0.,width,height)


def ellipse_mask(shape: tuple[int,int],ellipse: ManualEllipse) -> np.ndarray:
    height,width=shape
    mask=np.zeros((height,width),np.uint8)
    center=(int(np.rint(ellipse.center_x)),int(np.rint(ellipse.center_y)))
    axes=(max(1,int(np.rint(ellipse.semi_axis_x))),
          max(1,int(np.rint(ellipse.semi_axis_y))))
    cv2.ellipse(mask,center,axes,float(ellipse.angle_degrees),0,360,1,-1,cv2.LINE_8)
    return mask>0


def eroded_ellipse_mask(
    shape: tuple[int,int],ellipse: ManualEllipse,margin_pixels: int,
) -> np.ndarray:
    if margin_pixels<0:
        raise ValueError("人工椭圆内缩像素数不能为负数")
    inner=replace(
        ellipse,
        semi_axis_x=max(1.,ellipse.semi_axis_x-margin_pixels),
        semi_axis_y=max(1.,ellipse.semi_axis_y-margin_pixels))
    return ellipse_mask(shape,inner)


def _draw_editor_canvas(
    frame_bgr: np.ndarray,residual_linear_rgb: np.ndarray,
    valid_mask: np.ndarray,ellipse: ManualEllipse,label: str,index: int,total: int,
) -> np.ndarray:
    frame=np.asarray(frame_bgr,np.uint8); valid=np.asarray(valid_mask,np.bool_)
    masked=np.zeros_like(frame); masked[valid]=frame[valid]
    residual=signed_residual_bgr(residual_linear_rgb,valid,gain=2.)
    height,width=frame.shape[:2]
    center=(int(np.rint(ellipse.center_x)),int(np.rint(ellipse.center_y)))
    axes=(max(1,int(np.rint(ellipse.semi_axis_x))),
          max(1,int(np.rint(ellipse.semi_axis_y))))
    for panel,offset in ((masked,0),(residual,width)):
        cv2.ellipse(panel,center,axes,ellipse.angle_degrees,0,360,
                    (0,255,0),2,cv2.LINE_AA)
        cv2.drawMarker(panel,center,(255,0,255),cv2.MARKER_CROSS,18,2,cv2.LINE_AA)
    canvas=np.concatenate([masked,residual],axis=1)
    footer=np.zeros((112,canvas.shape[1],3),np.uint8)
    lines=[
        f"[{index}/{total}] {label}",
        f"center=({ellipse.center_x:.1f},{ellipse.center_y:.1f})  "
        f"semi-axes=({ellipse.semi_axis_x:.1f},{ellipse.semi_axis_y:.1f})  "
        f"angle={ellipse.angle_degrees:.1f} deg",
        "Arrows: center 1px | I/J/K/L: center 10px | a/d: axis-X -/+ | w/s: axis-Y +/-",
        "Uppercase A/D/W/S: axis 10px | q/e: rotate | ENTER: use | x: skip | r: reset | ESC: save+stop",
    ]
    for row,line in enumerate(lines):
        cv2.putText(footer,line,(10,22+row*26),cv2.FONT_HERSHEY_SIMPLEX,.55,
                    (230,230,230),1,cv2.LINE_AA)
    return np.concatenate([canvas,footer],axis=0)


def edit_manual_ellipse(
    frame_bgr: np.ndarray,residual_linear_rgb: np.ndarray,
    valid_mask: np.ndarray,initial: ManualEllipse,
    *,label: str,index: int,total: int,
) -> ManualEllipse | None:
    """键盘编辑单帧椭圆；返回 None 表示 ESC 中止并保留已保存进度。"""
    frame=np.asarray(frame_bgr,np.uint8); height,width=frame.shape[:2]
    current=replace(initial,image_width=width,image_height=height,use_for_lut=True)
    reset=current
    window="manual normal-calibration ellipse"
    position=[current.center_x,current.center_y]

    def mouse(event,x,y,_flags,_parameter):
        if event==cv2.EVENT_LBUTTONDOWN:
            position[0]=float(x%width); position[1]=float(min(y,height-1))

    try:
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window,mouse)
    except cv2.error as error:
        raise RuntimeError(
            "无法打开人工椭圆编辑窗口；请在具有图形桌面的终端运行 calibrate-norm") from error
    left={81,2424832,65361}; up={82,2490368,65362}
    right={83,2555904,65363}; down={84,2621440,65364}
    while True:
        current=replace(current,center_x=position[0],center_y=position[1])
        cv2.imshow(window,_draw_editor_canvas(
            frame,residual_linear_rgb,valid_mask,current,label,index,total))
        key=cv2.waitKeyEx(0)
        if key in (10,13):
            cv2.destroyWindow(window)
            return replace(current,use_for_lut=True)
        if key in (27,):
            cv2.destroyWindow(window); return None
        if key in (ord("x"),ord("X")):
            cv2.destroyWindow(window)
            return replace(current,use_for_lut=False)
        if key in left: position[0]-=1
        elif key in right: position[0]+=1
        elif key in up: position[1]-=1
        elif key in down: position[1]+=1
        elif key==ord("j"): position[0]-=10
        elif key==ord("l"): position[0]+=10
        elif key==ord("i"): position[1]-=10
        elif key==ord("k"): position[1]+=10
        elif key==ord("a"): current=replace(
            current,semi_axis_x=max(1.,current.semi_axis_x-1))
        elif key==ord("d"): current=replace(
            current,semi_axis_x=current.semi_axis_x+1)
        elif key==ord("w"): current=replace(
            current,semi_axis_y=current.semi_axis_y+1)
        elif key==ord("s"): current=replace(
            current,semi_axis_y=max(1.,current.semi_axis_y-1))
        elif key==ord("A"): current=replace(
            current,semi_axis_x=max(1.,current.semi_axis_x-10))
        elif key==ord("D"): current=replace(
            current,semi_axis_x=current.semi_axis_x+10)
        elif key==ord("W"): current=replace(
            current,semi_axis_y=current.semi_axis_y+10)
        elif key==ord("S"): current=replace(
            current,semi_axis_y=max(1.,current.semi_axis_y-10))
        elif key==ord("q"): current=replace(
            current,angle_degrees=current.angle_degrees-1)
        elif key==ord("e"): current=replace(
            current,angle_degrees=current.angle_degrees+1)
        elif key==ord("Q"): current=replace(
            current,angle_degrees=current.angle_degrees-5)
        elif key==ord("E"): current=replace(
            current,angle_degrees=current.angle_degrees+5)
        elif key in (ord("r"),ord("R")):
            current=reset; position[:]=[reset.center_x,reset.center_y]
        position[0]=float(np.clip(position[0],0,width-1))
        position[1]=float(np.clip(position[1],0,height-1))

