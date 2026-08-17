# 标定球法向标定与局部重建

## 1. 标定前配置

在 `config.yaml` 的 `normal_calibration` 中至少填写：

```yaml
normal_calibration:
  sphere_radius_mm: 5.0       # 示例；必须替换为实际半径
  images: assets/cali_norm_pics/*.png

lightfield:
  background:
    method: physical_residual # 或 direct_fit / direct_fit_3
```

标定图片只需要包含标定球压入状态。`physical_residual` 计算
`camera_linear - rendered - fitted(B, M)`；`direct_fit` 与 `direct_fit_3` 计算
`camera_linear - fitted(B + deltaB + Bsession)`。其中 `direct_fit_3` 共享几何 encoder，
但 R/G/B 各自拟合静态 B，并由三个标量 decoder 独立预测 delta B。三种方法都不读取
或生成配对的无接触参考图。
曝光、白平衡、相机增益和灯光必须与光场标定及在线使用保持一致。

## 2. 运行标定

```bash
uv run calibrate-norm --config config.yaml
```

程序对每张图独立执行 SAM2 分割、整体曲面重建、全分辨率色差计算和自动接触圆
检测。圆拟合在重建平面的毫米坐标中完成，而不是在透视像素坐标中完成。
检测前只保留有效域中面积最大的单个接触面，并按
`detection.surface_erode_pixels` 额外向内腐蚀；背景 median/MAD 和双阈值连通域
都只在腐蚀后的面内计算。当前使用 `5 MAD` 高阈值、`0.65` 低阈值比例，避免
侧边与投影边界残差生成高能量假圆。

每张图都会在 `normal_calibration.verification_dir` 中产生
`*_verification.png`：

- 橙色：色差检测得到的候选压入区域；
- 青色：鲁棒圆拟合得到的完整接触范围；
- 绿色：剔除最外侧指定像素后，真正用于 LUT 的区域；
- 紫色十字：拟合圆心；
- 左图顶部：接受/拒绝、半径、估计压入深度、内点率、圆周覆盖和拟合误差；
- 右图：同一帧的有符号净化色差。

`detection_report.csv` 汇总所有图片。被拒绝的图片不会进入 LUT。标定结果按背景
方法保存到 `normal_calibration.output_files` 对应路径，包含 `64^3` 坡度表、方差
表、原始有效节点、样本数、颜色范围、背景方法、背景模型哈希和标定球元数据。

建议准备约 **30--50 张最终通过自动检验的标定图**。可先按接触面的规则网格选择
约 20--25 个不同位置，每个位置使用两个不同但不过深的压入量；中心、四角和靠近
灯带的区域都应覆盖。`minimum_accepted_images: 20` 只是防止明显不足的硬下限，
不是推荐采集量。若检验图频繁拒绝、不同位置的接触斑大小非常接近，或 LUT 原始
有效节点数继续随新图片明显增长，应继续补拍。

## 3. 实时局部重建

标定成功后修改：

```yaml
local_reconstruction:
  enabled: true
  calibration_files:
    physical_residual: assets/normal_calibration/normal_lut.npz
    direct_fit: assets/normal_calibration/normal_lut_direct_fit.npz
    direct_fit_3: assets/normal_calibration/normal_lut_direct_fit_3.npz
  residual_method: uniform_huber
  depth_color_range_mm: 2.0
  deformation_geometry_gain: 3.0
  show_coordinate_frame: false
  show_surface_mesh: true
```

然后运行实时入口（省略 `--image` 时直接读取相机）：

```bash
uv run recon.py --config config.yaml
```

局部求解直接使用整体重建的原生规则 `xyz` 网格。每个网格点的颜色输入是在 GPU
上从全分辨率净化色差图按该点 `uv` 双线性采样得到的，不会先缩小色差图。单图
模式额外保存 `*_local_reconstruction.npz`，其中包括最终点云、法向位移、参考
法向、坡度、观测法向、曲率修正、置信度和求解器诊断。

`uniform` 在差分有效域内等权拟合，`uniform_huber` 再以 Huber IRLS 抑制局部
异常。物理路径拟合 Bsession/M 的全部分数；纯拟合路径直接使用几何条件神经场和
冻结的低频会话修正。净化后的线性 RGB 残差保留正负符号，再按整体曲面原生 `uv`
采样给法向 LUT；通道变暗和变亮都参与坡度映射。修改背景方法或残差方法后必须重新运行
`calibrate-norm`，旧的单边正色差 LUT 也必须重建。

实时模式使用 Open3D 将规则整体曲面三角化为连续三维平面。窗口关闭默认光照，
直接显示无高光、无反光的顶点色；颜色恢复为对比更强的线性 Turbo 色表来表示
有符号法向深度。`depth_color_range_mm: 2.0` 表示色表覆盖 `-2` 到 `+2 mm`，
超出范围截断。

`deformation_geometry_gain: 3.0` 仅把 Open3D 中沿参考法向的几何位移显示为三倍，
求解结果、日志和保存文件仍为真实毫米值。`show_coordinate_frame: false` 隐藏容易
分散注意力的亮色坐标轴。
`show_surface_mesh: false` 或 `--no-display` 会禁用该窗口。

也可以对已经导出的 NPZ 独立运行数值求解：

```bash
uv run recon-local \
  --input frame_data.npz \
  --calibration assets/normal_calibration/normal_lut.npz \
  --output local_reconstruction.npz
```

`recon-local` 是离线诊断工具，只有它需要 `--input`。`--output` 可省略；省略时
会在输入文件旁生成 `*_local_reconstruction.npz`。实时入口不读取这两个参数。

输入文件可以直接包含 `xyz` 和 `color_residual_grid`；也可以包含 `xyz`、`uv`、
`residual_linear_rgb` 和可选的 `residual_valid`，由 `recon.py` 完成全分辨率 UV
采样。
