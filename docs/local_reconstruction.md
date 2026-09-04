# 标定球法向标定与局部重建

## 1. 配置固定材料模板

光场、法向 LUT 和实时局部重建必须使用同一组材料尺寸。模板不需要相机采集，直接由
`reconstruction.material_surface.width_mm`、`length_mm`、`s_zero_endpoint` 和几何
网格确定性生成。尺寸或端点方向变化后，必须依次重新生成光场模型和法向 LUT；
运行时会比较模板 SHA-256，拒绝混用旧资产。

## 2. 标定前配置

主 `config.yaml` 固定使用 `direct_fit_3`。其中至少填写：

```yaml
normal_calibration:
  sphere_radius_mm: 5.0       # 示例；必须替换为实际半径
  images: assets/cali_norm_pics/*.png

lightfield:
  background:
    method: direct_fit_3
```

若要运行物理光场链路，不修改主配置，改为给命令传入
`config_physical_residual.yaml`。该文件通过顶层 `extends: config.yaml` 继承相机、分割、
几何和公共局部重建参数，并集中保存所有 `physical_residual` 专属参数。

标定图片只需要包含标定球压入状态。`physical_residual` 计算
`camera_linear - rendered - fitted(B, M)`；`direct_fit_3` 计算
`camera_linear - fitted(B + deltaB + Bsession)`。后者共享几何 encoder，
但 R/G/B 各自拟合静态 B，并由三个标量 decoder 独立预测 delta B。两种方法都不读取
或生成配对的无接触参考图。
曝光、白平衡、相机增益和灯光必须与光场标定及在线使用保持一致。

## 3. 运行标定

```bash
uv run calibrate-norm --config config.yaml
# 物理链路：uv run calibrate-norm --config config_physical_residual.yaml
```

程序使用 `get_surface.segmentation.liteseg` 对每张独立图片执行 PP-LiteSeg
分割、当前轮廓到固定尺寸模板的统一归一化、全分辨率色差计算和自动接触圆检测。
程序不识别物理端边，也不区分完整轮廓与自遮挡轮廓；单目二值 mask 无法提供这种
材料身份信息。因此标定图片必须由采集流程保证不存在纵向自遮挡。圆拟合在重建
平面的毫米坐标中完成，而不是在透视像素坐标中完成。
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

`detection_report.csv` 汇总所有图片。被拒绝的图片不会进入 LUT。标定结果保存到所选
配置的 `normal_calibration.output`，包含 `64^3` 坡度表、方差
表、原始有效节点、样本数、颜色范围、背景方法、背景模型哈希和标定球元数据。

建议准备约 **30--50 张最终通过自动检验的标定图**。可先按接触面的规则网格选择
约 20--25 个不同位置，每个位置使用两个不同但不过深的压入量；中心、四角和靠近
灯带的区域都应覆盖。`minimum_accepted_images: 20` 只是防止明显不足的硬下限，
不是推荐采集量。若检验图频繁拒绝、不同位置的接触斑大小非常接近，或 LUT 原始
有效节点数继续随新图片明显增长，应继续补拍。

## 4. 实时局部重建

标定成功后修改：

```yaml
local_reconstruction:
  enabled: true
  calibration_file: assets/normal_calibration/normal_lut_direct_fit_3.npz
  residual_method: uniform_huber
  depth_color_range_mm: 2.0
  deformation_geometry_gain: 3.0
  show_coordinate_frame: false
  show_surface_mesh: true
  render_style: gray_points
  gray_color: [0.35, 0.35, 0.35]
  point_size: 1.0
  grid_line_width: 1.0
  projection: orthographic
```

然后运行实时入口（省略 `--image` 时直接读取相机）：

```bash
uv run recon.py --config config.yaml
# 物理链路：uv run recon.py --config config_physical_residual.yaml
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

实时模式的 Open3D 窗口可通过 `render_style` 切换三种显示方法：

- `depth_mesh` 将规则曲面三角化，使用深色背景和 Turbo 顶点色显示有符号法向深度；
- `gray_points` 在白底上显示全部有效规则顶点，适合论文式高密度点云图；
- `gray_grid` 在白底上连接相邻行列顶点，不添加三角形对角线。

三种模式都关闭默认光照。`gray_color` 使用 `[0,1]` RGB，`point_size` 和
`grid_line_width` 分别控制点云与栅格线宽；`projection: orthographic` 通过 Legacy
Open3D 的最小视场角生成近正交视图。只有 `depth_mesh` 使用
`depth_color_range_mm`：值 `2.0` 表示色表覆盖 `-2` 到 `+2 mm`，超出范围截断。

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
