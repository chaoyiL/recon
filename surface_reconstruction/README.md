# 单目柔性曲面重建

通过纸片上的十字标记与已标定的单目摄像头，估计纸片的三维曲面。适用于柔性视触觉传感器的曲面重建实验。

程序支持图片、视频文件和摄像头输入。相机与纸片的相对位置可变化，初始纸片可弯曲，不依赖固定平面参考帧。当前版本为研究原型，面向平滑弯曲和有限扭转；三维绝对精度尚需实物真值验证。

## 1. 实验条件

| 项目 | 当前配置 |
|---|---|
| 相机 | 已标定单目相机，640×480 图像 |
| 标记模板 | 8 行×6 列，共 48 个十字，按行编号 0～47 |
| 标记间距 | 横向 3 mm、纵向 6 mm，覆盖 15×42 mm |
| 表面假设 | 近似不可拉伸，标记不相对纸面滑动 |
| 初始化 | 需要完整、清晰、可排序的网格及可检测的纸张内框；不要求放平 |
| 持续重建 | 当前 48 点模板至少需要 32 个有效观测，且覆盖足够区域 |

相机移动不改变相机内参；更换镜头、变焦、改变图像裁剪或成像模式后，必须重新检查标定。输入尺寸不匹配时程序报错，不自动缩放。

## 2. 安装与快速运行

已验证环境：**Windows x64、Python 3.13**。在项目根目录打开终端，创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 固定了本次验证使用的 NumPy、SciPy、OpenCV 版本。已有兼容环境可以直接使用，无须重复安装。不需要 MATLAB、GPU 或模型权重。


打开摄像头：

```powershell
.\.venv\Scripts\python.exe run_reconstruction.py --video 0
```

Windows 也可双击 `start.bat`，默认打开配置中的摄像头。启动器优先使用项目内 `.venv`，否则使用当前环境的 `python`，不含特定电脑路径，不自动安装依赖。

预览左侧是实际标记检测与编号，右侧是去除整体姿态后的曲面。按 **Esc** 退出，按 **R** 重新初始化编号。视频文件按计算速度处理，预览并非原速播放器。

## 3. 项目结构

```text
surface_reconstruction/
├── run_reconstruction.py    # 唯一 Python 入口：输入、调度、结果保存
├── start.bat                # Windows 便捷启动
├── config.json              # 相机、数据路径与求解参数
├── requirements.txt         # 三项运行依赖及验证版本
├── markers.py               # 区域检测、十字中心检测、网格排序
├── tracking.py              # 二维初始化、连续编号跟踪与局部找回
├── surface.py               # 三维曲面拟合与姿态对齐
├── display.py               # 标记叠加、曲面渲染、OBJ 导出
├── tests/
│   └── test_reconstruction.py
├── data/
│   ├── camera_params.npz    # camera_matrix、dist_coeffs
│   ├── template_points.csv  # id,x_mm,y_mm
└── README.md
```

运行时自动创建 `output/` 和 `marker_detection/`。交付包不包含虚拟环境、本机 `.runtime`、编辑器配置、缓存和完整实验输出。原始照片、标定、模板及视频示例包含在包内，解压后可按上述步骤复现。

## 4. 算法流程

1. **建立二维编号。** 检测十字中心，对完整网格做拓扑检查和排序；采用平滑图像映射检查少量多余候选点，初始化不假设纸片平整。
2. **持续跟踪。** 使用前后向光流与一对一检测匹配维持编号。搜索区域随纸片移动；局部找回必须匹配到真实检测中心，不把预测位置当作观测。
3. **表示三维曲面。** 使用双三次参数曲面 `Q(x,y)`，16 个三维系数共 48 个标量同时描述位置、姿态与平滑变形，不限定单一弯曲轴。
4. **联合优化。** 用鲁棒非线性最小二乘拟合图像投影，并约束 `|∂Q/∂x|≈1`、`|∂Q/∂y|≈1`、`∂Q/∂x·∂Q/∂y≈0`，限制局部拉伸和剪切。另有弱高阶系数正则项以抑制无观测支持的起伏。
5. **利用连续帧。** 首帧尝试平面及多个弯曲方向的初值；后续先估计前一曲面在当前相机下的刚体姿态，再施加较弱的形状连续性约束。竞争形状各自使用对应先验，不锁定固定相机平面。
6. **质量检查。** 少量明显误匹配剔除后重算；几何或投影误差不合格时不输出三维结果。连续失败后重新初始化，并增加片段编号。

曲面采用差分度量约束，没有把纸面弧长直接作为弯曲后两点间的直线距离。实现借鉴等距形变与模板重建原理，并非某篇论文的逐式复现。

## 5. 配置与命令行

`config.json` 中的数据路径相对配置文件所在目录解析；命令行输入文件路径相对当前工作目录解析。建议在项目根目录运行。

| 配置项 | 含义 |
|---|---|
| `camera_index` | 无输入参数时使用的摄像头编号 |
| `image_size` | 标定对应的图像宽、高 |
| `camera_file` / `template_file` | 标定与物理模板路径 |
| `pixel_sigma` | 图像残差尺度，单位 px |
| `metric_sigma` | 局部等距残差尺度，无量纲 |
| `temporal_sigma_mm` | 运动对齐后的形状先验尺度，越大约束越弱 |
| `max_nfev` / `initialization_nfev` | 跟踪与初始化的求解预算 |
| `max_reprojection_px` | 拒绝三维估计的投影 RMS 阈值 |
| `max_metric_error` | 局部长度/正交性误差的拒绝阈值 |

主要参数：`--image` 与 `--video` 二选一；`--preview` 显示视频预览；`--no-preview` 关闭摄像头预览；`--max-frames N` 限制处理帧数；`--no-record` 仅保存数据和诊断、不保存视频；`--out` 指定输出目录；`--row-direction` 指定初始行方向（auto/down/up/left/right）。完整说明可运行 `python run_reconstruction.py --help`。

## 6. 输出与坐标定义

图片重建结果保存到 `output/<输入名>/`：`surface.png`、`surface.obj`、`reconstructed_points.csv`、`frame_metrics.csv`、`diagnostics.json`。检测图片单独保存到 `marker_detection/<同名目录>/detected_markers.png`。

视频结果包括 `reconstruction.avi`、`frame_points.csv`、`frame_metrics.csv` 和 `diagnostics.json`；单独的识别录像位于 `marker_detection/<同名目录>/detected_markers.avi`。默认不会覆盖已有输出，而是新建带时间后缀的目录。

| 数据字段 | 含义 |
|---|---|
| `frame`, `time_s` | 帧编号与时间；文件视频按帧率计算，相机使用采集计时 |
| `segment`, `id` | 连续跟踪片段与片段内编号；跨片段物理身份不保证一致 |
| `observed` | 当前是否有通过检查的实际标记观测 |
| `point_source` | `observed_fit`：观测拟合；`model_inferred`：缺点处模型推断；`unavailable`：无三维估计 |
| `u_px`, `v_px` | 图像坐标，单位 px，未观测时留空 |
| `X_mm`, `Y_mm`, `Z_mm` | 相机坐标，X 向图像右、Y 向图像下、Z 向相机前方 |
| `X_local_mm`, `Y_local_mm`, `Z_local_mm` | 去除最佳拟合整体平移与旋转后的形状坐标 |
| `confidence` | 跟踪启发式分数，不是三维精度概率 |

local 坐标原点为当前点云中心，X/Y 对齐物理模板，+Z 由 X×Y 决定。**local Z 不是相对固定平面的绝对位移，也不是相对初始帧的形变量。** OBJ 使用相机坐标，单位 mm；显示曲面和点间网格均属于插值结果。

`estimated` 表示通过当前一致性检查；`ambiguous` 表示存在相近代价的不同形状；`budget_limited` 表示达到求解预算但残差尚满足阈值。`waiting_for_grid`、`insufficient_observations`、`fit_rejected` 分别表示等待初始化、观测不足、拟合拒绝。失败帧三维坐标留空，界面不重复显示旧曲面。

相机采用“只取最新帧”，处理不及时会丢弃旧帧。录像按输入帧率编码已处理帧，发生丢帧时播放时长会缩短；分析实验时间应使用 CSV 中的时间戳，不能只依据输出录像时长。

## 7. 验证结果与复现

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


## 8. 已知限制与常见问题

- **方向或编号歧义：** 相同十字存在 180° 身份对称性，重新初始化后不能保证相同 ID 是同一物理标记。需要跨片段身份一致时，应增加不对称或编码标记。
- **反光、模糊和遮挡：** 可能漏点或误匹配。检查识别叠加图；初始化时应展示完整网格，避免高光覆盖十字。极端透视、强折叠或重叠可能无法排序。
- **模型范围：** 单个双三次曲面不能精确表达尖锐折痕、复杂褶皱或强自遮挡。被遮挡点的模型推断不能当作测量值。
- **精度解释：** 小投影误差和小度量误差不等于小三维误差；候选搜索也不保证找到全部歧义解。实物绝对精度、动态延迟和极限形变范围尚未完成受控验证。
- **打不开相机：** 检查占用情况，尝试 `--video 1`，确认设备可输出 640×480。
- **依赖或启动失败：** 按安装步骤创建 Python 3.13 环境；不要把另一台电脑的虚拟环境或 `.runtime` 当作可移植安装包。

## 9. 方法参考

- Perriollat 等（2008），*Monocular Template-based Reconstruction of Inextensible Surfaces*：项目初始问题与不可伸展模板重建参考。
- [Bartoli 与 Collins（CVPR 2013）](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Bartoli_Template-Based_Isometric_Deformable_2013_CVPR_paper.pdf)：差分等距约束与模板重建。
- [SciPy least_squares 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)：鲁棒非线性最小二乘求解。
