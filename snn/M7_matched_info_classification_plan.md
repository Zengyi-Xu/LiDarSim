# M7 实验计划：信息量匹配条件下的分类对比（proposal 预备数据）

> 状态：待跑（当前机器无独立显卡，纯 CPU 可跑小规模，正式数字建议 GPU）
> 目的：为 KGFP Concept Note 修订提供"严格公平对比 + 90% 级 headline"的预备数据
> 原则：**比较必须在信息量匹配的输入上进行**——光栅前端输出什么格式，就喂什么格式。

## 背景（为什么要做这个实验）

- M2 的 0.40 vs 0.676 不公平：基线拿全精度浮点坐标，蓄水池拿有损脉冲编码 + 岭回归。
- M3a 已证：同数据换交叉熵读出 → 0.84–0.89，反超基线。
- 本实验把"编码 × 读出"两个变量分离，给出可写进 proposal 的诚实数字。

## 实验矩阵

统一设置：ModelNet40 前 10 类，256 点，训练 150/类、测试 40/类（与 M2 相同划分，种子 0/1/2 三次重复）。

| 编号 | 输入编码 | 读出 | 回答的问题 |
|---|---|---|---|
| E1 | range-echo（96 bin × 3 扇区） | 岭回归 | 复现 M2 下界（对照） |
| E2 | range-echo | MLP（交叉熵） | **主数字**：物理合理格式 + 训练读出能到多少 |
| E3 | 坐标速率编码 | MLP | 脉冲编码路线的上限 |
| E4 | 原始浮点坐标 | MLP | 信息量天花板（对照 E2，量化编码损失） |
| E5 | 原始浮点坐标 | 岭回归 | = M2 基线的加强版对照 |

消融（在 E2 最优配置上做）：

- A1 编码分辨率：range-echo bin 数 48 / 96 / 192
- A2 池规模：256 / 512 / 1024 神经元
- A3 读出容量：线性 / MLP-1 隐层(256) / MLP-2 隐层
- A4 训练量：75 / 150 / 300 每类（学习曲线，proposal 里说明数据效率）

鲁棒性（复用 robustness.py 的模式，在 E2 配置上加）：

- R1 权重量化 4–8 bit（对应 MRR crossbar 6.74 bit）
- R2 输入噪声 / spike 时间抖动（对应光栅 ripple → spike 抖动 0.1 ps 量级）

## 判定标准

- **Headline**：E2 ≥ 0.90（3 种子均值）→ 写进 proposal："on grating-native echo format, the photonic chain achieves >0.90 accuracy with a trained readout"
- 若 E2 在 0.85–0.90：报实际值，proposal 用 road car/person 0.89 结果做主数字
- 若 E2 < 0.85：说明 range-echo 编码本身信息不足，回到编码设计（增加角度通道/多视角），不硬凑数字

## 计算预算

| 实验 | CPU 估计 | GPU 估计 |
|---|---|---|
| E1–E5 主矩阵 | ~2–4 h（池 512、3 种子） | <10 min |
| 消融 A1–A4 | ~3–5 h | ~20 min |
| 鲁棒性 R1–R2 | ~1 h | 几分钟 |

CPU 冒烟先行：每配置 1 种子、池 256，确认管线无误后再全量。

## 产出

- `outputs_m7/`：results.json（均值±std）、混淆矩阵、E2 vs E4 对比图
- `notebooks/m7_matched_info_classification.ipynb`（build_m7.py 生成，沿用 M1–M6 体例）
- 回填本 README 的 M7 节 + tfln-dispersion-lab/lab-note.ipynb 对应节
- 供 Concept Note 引用的两句话结论 + 一个图
