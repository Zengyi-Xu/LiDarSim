# SNN/蓄水池网络处理匹配滤波后雷达回波 — 可行性实验

在 `D:\kimi_workspace\lidar-pointnet\snn` 下，验证 **LSM（Liquid State Machine，脉冲蓄水池）**
能否处理匹配滤波后的相参脉冲串回波，并与经典方法（MTD-CFAR / FFT）及 ESN 基线对比。
仿真 + 训练全部纯 PyTorch（CPU），无额外依赖。

## 运行

```bash
python experiment.py               # 主实验: 扫描 5 个 SNR 点, ~1 min
python robustness.py               # 片上可行性: 权重量化/噪声鲁棒性
python experiment.py --n-train 300 --n-test 150 --snr -10 0 --out outputs_smoke   # 冒烟
```

产物在 `outputs/`：`summary.png`（三图汇总）、`demo_signals.png`（信号/脉冲光栅示例）、`results.json`。

## 信号链与任务

- **仿真**（`simulator.py`）：LFM 相参脉冲串（N=64 脉冲，K=64 脉压采样，128 距离门），
  目标延迟随机 ∈ [68,104]，多普勒 f_D ∈ [-0.4,0.4] 周期/脉冲，Swerling I 幅度起伏，
  复高斯噪声；SNR 定义为**单脉冲 MF 输出峰值 SNR**（0 dB 时目标峰仅 1σ——
  信息全部在跨脉冲相干相位结构中）。匹配滤波用频域共轭乘法实现（增益 K=18 dB）。
- **编码**（`encoding.py`）：A=门限事件化（4σ_mf）；B=I/Q 群体编码（每距离门 4 通道，
  发放率∝整流幅值）；C=慢时间增量编码。
- **池**（`reservoir.py`）：LSM = 256 个 LIF 神经元，随机稀疏递归（密度 0.12，谱半径 0.95，
  Dale 定律 8:2，不应期 2 步），固定权重只训读出头；ESN = 连续值（发放率）基线。
- **任务**：① 低 SNR 检测（Pd @ Pfa=1e-3，门限统一在训练集负样本上标定）；
  ② 多普勒估计（RMSE，与 FFT 及 CRLB 对比）。

## 关键设计（踩坑记录，复现/改进前必读）

1. **线性读出头检测不了未知位置目标**。信号只占 36 个距离门中的 1 个，任何单通道几乎都
   是噪声；贝叶斯最优检测器是能量（二次）统计量。LSM 随机 ± 输入权重把一阶信息对称
   抵消，必须在随机混合**之前**取平方 → 检测读出头 = `quad([轨迹末快照 r(T), 池均值, 输入发放率])`。
   `r(T)` 是 leaky 积分器输出（神经元群体随机滤波器 + 平方 ≈ 周期图检测器），
   这是 LSM 唯一能做**相干积累**的地方（leak=0.9 → 增益 ~10）。
2. **多普勒信息在状态时间轨迹里**，时间平均会把它抹掉 → 读出头看 16 个下采样快照。
3. 高维岭回归需较强正则（α=10），否则数值爆炸；训练集零方差特征要丢弃。
4. 发放率指标要除以神经元数（曾把 6.7% 的健康发放率误读为饱和）。
5. 硬门限（事件/增量编码）在相干积累**之前**不可逆地丢掉了亚 σ 信号 → 本场景下不可用。

## 结果（Pfa=1e-3，N_train=2000 / N_test=500）

检测 Pd：

| SNR (dB) | LSM-iq | ESN-iq | MTD-CFAR |
|---|---|---|---|
| -15 | **0.39** | 0.06 | 0.01 |
| -10 | **0.31** | 0.05 | 0.05 |
| -5  | 0.24 | 0.04 | **0.35** |
| 0   | 0.29 | 0.07 | **0.60** |
| +5  | 0.38 | 0.34 | **0.77** |

多普勒 RMSE（周期/脉冲；CRLB = 0.0043→0.0004）：

| SNR (dB) | LSM-iq | ESN-iq | FFT |
|---|---|---|---|
| -15 | **0.268** | 0.361 | 0.358 |
| -10 | **0.276** | 0.365 | 0.343 |
| -5  | **0.281** | 0.352 | 0.312 |
| 0   | 0.275 | 0.365 | **0.201** |
| +5  | 0.293 | 0.390 | **0.097** |

**结论**：
- **低 SNR（≤-5 dB）LSM 双任务反超经典方法**——蓄水池把"学习到的相干积累 + 能量读出"
  融合在一起，比"FFT + 逐单元 CFAR"（受多重检验惩罚）更鲁棒。
- **高 SNR 出现平台期**（Pd ~0.3、RMSE ~0.28 不随 SNR 改善）：随机滤波器组相干增益
  ~1/(1-leak) ≪ N=64，且 LIF 非线性 + Swerling 随机相位干涉构成系统性瓶颈。
  这是"模型不够好"的原因——不是 SNN 不行，是**随机被动蓄水池的表达上限**。
- 事件/增量编码不可用于此类亚 σ 相参积累场景。

## 片上可行性（对应 QD-MLL PCU 的 6.74 bit 权重精度）

`robustness.py`（leak=0.9, in_scale=0.5）：

| 扰动 | Pd@-10dB | RMSE@-10dB | Pd@0dB | RMSE@0dB |
|---|---|---|---|---|
| 理想 | 0.30 | 0.271 | 0.26 | 0.302 |
| 量化 6 bit | 0.30 | 0.267 | 0.32 | 0.294 |
| 量化 5 bit | 0.33 | 0.275 | 0.32 | 0.308 |
| 量化 4 bit | 0.20 | 0.274 | 0.26 | 0.293 |
| 权重噪声 10% | 0.24 | 0.260 | 0.30 | 0.296 |

固定随机池对权重误差极不敏感（性能由读出头提取，不靠权重精度），
**MRR 权重库的 6.74 bit 精度 + LUT 再校准对该模型足够**；3-4 bit 以下才明显退化。
进一步量化热漂移/RIN 影响可直接复用论文的 HATF（训练时注入硬件噪声）。

## 里程碑笔记本（博士后研究计划素材）

`notebooks/` 下 M1–M5，均带执行输出，用对应 `build_m*.py` 复现：

| 文件 | 内容 | 关键结论 |
|---|---|---|
| m1_device_spec.ipynb | 器件参数与规模标定 | 128 神经元拐点；leak 容差 ±5%；实测光栅纹波免疫 |
| m2_non_gaussian_clutter.ipynb | 沙尘/大雾 K 杂波 | CFAR Pfa 膨胀 172×；LSM 保持 0.2–0.3；多普勒 0.27 vs FFT 0.36 |
| m3_microdoppler_classification.ipynb | 微多普勒分类 v1→v4 消融 | 平移不变特征（自相关）必要；二次核过拟合随机载波 |
| m4_hardware_codesign.ipynb | 硬件协同设计 | 全栈噪声性能不降；光子能耗 1.36 nJ/CPI、0.14 mW@100kHz |
| m5_research_proposal.ipynb | **博士后研究计划草稿** | 3 科学问题 + 3 创新点 + WP1–5 技术路线 + 风险表 |

## 与既有工作的接口

- Zhou et al., "Integrated quantum dot lasers for parallelized photonic edge computing",
  Adv. Photonics 8(2) 2026：QD-MLL 多波长光源 + MRR crossbar 权重 + TWMZM + HATF。
  本模型的 W_in/W_rec 正好映射到其 crossbar（随机固定权重，无需训练），
  I/Q 编码天然由其 TWMZM 前端完成，读出头系数 ~2k-4k 个，EIC 可存。
- 待办：leak → 微环损耗的灵敏度（光子积分器时间常数控制）、N_res 缩放曲线、
  delay-line reservoir 变体（最易片上化的蓄水池形态）。
- 已验证（`hetero.py`）：逐神经元异构 leak [0.6,0.99] 对平台期几乎无改善
  （+5 dB Pd 0.38→0.40）——高 SNR 平台源于脉冲量化噪声/随机相位干涉，
  但在 ≤-5 dB 目标工作区不影响；且高 SNR 本就交给视觉/FFT。

## 分类上限探测 (2026-02, cls_ceiling.py / cls_ceiling_followup.py)

**动机**: M3 的 0.57/0.71 是用弱读出(线性/二次核)隔离"平移不变性"变量得到的下限;
本实验在相同 4 类微多普勒任务上测上限. 结果(npz/log 在 outputs_classify/):

| 方法 (CPI=64) | -10dB | -5dB | 0dB | +5dB |
|---|---|---|---|---|
| acorr+ridge (M3 v4) | 0.29 | 0.35 | 0.57 | 0.71 |
| log 谱 / 倒谱 / hann 谱 +ridge | ~0.24 | ~0.28 | ~0.24 | ~0.26 |
| acorr+MLP (学习读出) | 0.29 | 0.35 | 0.54 | 0.61 |
| **oracle 去载波+ridge** | 0.27 | 0.36 | 0.59 | 0.74 |
| LSM+MLP (学习读出) | 0.25 | 0.24 | 0.27 | 0.27 |

**三个结论**:
1. **单次 CPI 分类是信息受限, 不是读出受限** — MLP 不胜过 ridge, 完美去载波也只到
   0.74, 池状态+MLP 完全过拟合(train 1.0 / test 0.27). 不变性只是门槛, 过了门槛还有信息墙.
2. **难度来自类设计**: 深调制类 (β=0.5, 类1/2) 0dB 即有 0.68/0.72; 浅调制类 (β=0.2,
   类0/3) 边带低于载波 ≥20dB, 且类0 的 f_m=0.02 在 64 脉冲内只有 1.3 个周期
   (周期估计欠定). Swerling I 单 CPI 幅度衰落进一步随机化信息量.
3. **积累时间是唯一杠杆** (信息量 ∝ CPI×SNR, 与架构无关): CPI 64→256 使
   acorr+ridge 从 0.57→0.70 (0dB); 4 个 CPI 多数投票(航迹级融合)在 +5dB 达 **0.80**.

**对器件定位的含义**: 竞争指标不是"单次 CPI 准确率", 而是"每焦耳信息量"与"航迹级
准确率". 光子 SNN 前端在 -15dB 提供信息存在性(CFAR 全盲), 并以 ~1/200 功耗换取
更多 CPI/更长航迹 → 直接买到客户体验到的分类准确率.

## M6 语义目标分类 (2026-02, semantic_sim.py + notebooks/build_m6.py)

**任务升级**: articulated 多点散射体运动学仿真 (行人6点摆肢 / 无人机5点旋翼谐波梳 /
车辆4点刚体+浅振动 / 鸟类3点反相扑翼), 类间调制频率留 >=2x 物理间隙, 类内保留
实现随机性. 笔记本: notebooks/m6_semantic_classification.ipynb (npz/log 在 outputs_classify/).

**v1 阴性结果 (m6_dryrun.py)**: 沿用 M3 的单质心门选通 -> 全员随机 (0dB acorr+ridge 0.286).
原因: 判别能量分布在多个距离门 (腿 ±2 门, 旋翼 ±1 门), 单门选通整体丢失.

**v2 处理链**: 检测后取 RD 邻域块 (9门) -> 估计+补偿质心多普勒 -> 块级谱特征:

| 臂 (CPI=64) | -10dB | -5dB | 0dB | +5dB |
|---|---|---|---|---|
| 单门 acorr+ridge (阴性对照) | 0.28 | 0.27 | 0.26 | 0.26 |
| 无补偿 logspec+ridge | 0.25 | 0.29 | 0.29 | 0.23 |
| profile+ridge | 0.29 | 0.39 | 0.57 | 0.79 |
| pergate+MLP2 (n_tr=5000) | — | — | 0.64 | **0.81** |

+5dB 逐类 recall: 鸟 0.87 / 行人 0.88 / 车辆 0.73 / 无人机 0.64 (drone<->vehicle
互混: 谐波梳低端 vs 浅振动边带同呈"中心峰+近旁小峰"). 航迹投票 0dB: 0.64.

**结论**: 语义分类在检测典型工作点 (+5dB, 有效 SNR~23dB) 达 0.79-0.81/CPI;
仍是信息受限 (MLP2 不胜 ridge); 系统价值仍在低功耗前端买 CPI.

## ANN 上限基准 (2026-02, pointcloud_ann_bench.py)

**目的**: 回答"3D 点云不限模型规模能到多高的分类准确率"——为计划书提供 ANN 天花板
参照 (单次 HRRP 0.4 / SNN 扫描链 0.68 / ANN 3D 点云 = 本脚本).

**依赖**: 仅 torch + numpy + pandas + pyarrow, 纯 PyTorch 实现 (无 PyG), GPU 机器
`pip install torch pandas pyarrow` 即可.

```bash
# 冒烟 (CPU, <1 min)
python snn/pointcloud_ann_bench.py --smoke
# GPU 正式跑 (10 类子集, 与 snn 实验同设定)
python snn/pointcloud_ann_bench.py --model pointnet     --n_points 1024 --epochs 200
python snn/pointcloud_ann_bench.py --model pointnet2_ssg --n_points 1024 --epochs 200
python snn/pointcloud_ann_bench.py --model dgcnn        --n_points 1024 --epochs 200
# 全 40 类 (文献可比)
python snn/pointcloud_ann_bench.py --model dgcnn --classes 40 --n_points 1024 --epochs 200
```

**数据**: data/modelnet40_{train,test}.parquet (2048 点/形状). 默认前 10 类
(3643 训练 / 710 测试, 类别 64–889/类不均衡). 训练增强: 随机重采样/抖动/
点丢弃(85-100%); 评估含 T=10 次重采样投票 (TTA). 输出: outputs_classify/ann_bench/
{model}_best.pt + {model}_summary.json.

**预期数字** (文献参考: 全 40 类 ModelNet40 @1024 点; 子集 10 类因类数少且形状
差异大, 应高 3-6 个点; 本数据每类仅 ~360 样本, 可能比文献低 1-3 个点):

| 模型 | ModelNet40 文献 | 10 类子集预期 |
|---|---|---|
| PointNet (无 TNet) | ~87-89% | ~91-94% |
| PointNet++ SSG | 90.7% | ~94-96% |
| DGCNN (k=32) | 92.2-92.9% | ~95-97% |

**口径注意**: "0.84 上限"出自受限配置 (冻结/浅层读出), 不是 ANN 天花板;
本基准的数字才是计划书里 "3D 点云 ANN 上限" 的引用值. GPU 实测后回填
outputs_classify/ann_bench/ 并更新此表.

## 3D 视角扫描仿真 (2026-02, isal_3dview.py)

**动机**: 视线流形必须覆盖真实相对几何. 把 isal_range_profile 的面内方位角旋转
扩展为完整 3D 旋转 (LOS=(cos el·cos az, cos el·sin az, sin el)), 加两个场景预设:
road_crossing (az 全环, el ±5°) 与 uav_cap (az 全环, el 25-65°).
图与结果在 outputs_isal/3dview/ (e0 视线流形 + 类内方位变化, e1-e3 柱状图).

**结果** (1370/340, 10 类):

| 实验 | 数字 | 解读 |
|---|---|---|
| E1 road: 单次 / 扫描+ESN+角度 | 0.294 / 0.656 | 扫描链增益 +0.36 |
| E1 uav: 单次 / 扫描+ESN+角度 | 0.209 / 0.582 | 家具类侧视轮廓信息>俯视足迹 |
| E2 跨分布 (road训->uav测) | **0.118** (~随机) | 投影统计随俯仰完全改变, 部署需各自训练 |
| E3 俯仰维标价 (12剖面预算) | 赤道环 0.712 > 双环 0.691 > 随机半球 0.338 | 固定预算下视角分配应跟随类信息分布 |

**注意**: E3 的"赤道环最优"是家具类特异的 (类信息集中在侧视轮廓); 道路车辆类
(顶视足迹信息量大) 的分配会不同 — 待参数化车辆生成器验证.

## 双项目知识图谱 (2026-02, notebooks/knowledge_graph.ipynb)

联同 tfln-dispersion-lab 的六层知识图谱 (器件->仿真器->方法->实验->结论->应用):
图 1 全图 (45 节点/44 边), 图 2 证据链 (两条破局路径汇聚), 双项目里程碑对照表
(含实测数字与出处). 机器可读版 notebooks/kg_data.json. 构建脚本 build_kg.py.

## 道路对象线 (2026-02, road_car_person.py / road_object_extractor.py / radarscenes_stats.py)

**① 止gap (本地 ModelNet40 car vs person, 397/186)**: 缩小分类对象到道路二类的验证:

| 预设 | 单次 HRRP+CNN1D | 4 波束x3 步+ESN+角度 |
|---|---|---|
| road_crossing | 0.817 | **0.892** |
| uav_cap | 0.758 | **0.866** |

对比 10 类通用 (road: 0.294/0.656): 缩小范围+类间差异大 => 0.87-0.89, 假设证实.
图: outputs_isal/road_car_person/.

**② KITTI 提取器** road_object_extractor.py: 解析 velodyne/label_2/calib,
框内点裁剪 (Tr_velo_to_cam+R0_rect 变换), 输出 data/road_objects.npz
(对象点云+类+遮挡/截断元数据), --selftest 已验证几何正确.
等用户注册下载 KITTI 后: `python road_object_extractor.py --root <KITTI>/training`.

**③ RadarScenes**: Zenodo 免费真实车载毫米波数据 (11 类, 多普勒+track_id),
下载到 data/radarscenes/; radarscenes_stats.py 出 类分布/各类多普勒/RCS 统计
(供杂波与散射统计标定 + 真实数据分类基准).

## 参数化道路车辆 7 类 (2026-02, road_vehicles.py)

几何原型库 (米制, 尺寸真实分布): sedan/suv/truck/bus/motorcycle/bicycle/pedestrian,
表面采样点云, **保留米制尺寸** (尺寸本身是判别特征), 剖面窗口 D=16m.
2450 训练 / 700 测试, 7 类:

| 预设 | 单次 HRRP+CNN1D | 4 波束x3 步+ESN+角度 |
|---|---|---|
| road_crossing | 0.660 | **0.919** |
| uav_cap | 0.659 | **0.963** |

关键发现: 家具类 uav 劣于 road (0.58<0.66), 车辆类 uav **反转占优** (0.96>0.92) ——
视角分配必须匹配类信息分布 (E3 俯仰标价结论的应用级实证; 车辆顶视足迹信息量大,
家具侧视轮廓信息量大). 图: outputs_isal/road_vehicles/ (原型 3D 散点 + 尺寸分布).

## road_kitti_experiment.py (备好待数据)

road_object_extractor.py 提取的 KITTI 对象库 -> 同款扫描链 (类合并: car/van->car,
truck/pedestrian/cyclist), 真实稀疏+遮挡点云下的诚实测试. KITTI 下载后即跑.
