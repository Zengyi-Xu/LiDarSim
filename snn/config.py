"""SNN 处理匹配滤波后雷达回波: 实验配置"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------- 雷达回波仿真 ----------
N_PULSES = 64          # 相干脉冲数 (慢时间维)
M_BINS = 128           # 每个 PRI 的快时间采样数 = 距离门数
PULSE_LEN = 64         # LFM 脉宽采样数 K, 匹配滤波增益 ~10*log10(K) = 18 dB
F_D_MAX = 0.4          # 归一化多普勒上界 (周期/脉冲), 避开奈奎斯特边缘

# ---------- 实验 ----------
SNR_LIST_DB = [-15, -10, -5, 0, 5]   # 匹配滤波输出峰值 SNR (单脉冲, dB, Swerling 平均)
N_TRAIN = 2000
N_TEST = 500
P_FA = 1e-3                          # 检测任务目标虚警率

# ---------- 编码 ----------
EVENT_THRESH = 4.0   # 方案A: 门限 (单位: MF 输出噪声标准差)
IQ_CLIP = 6.0        # 方案B: I/Q 归一化幅度截断
IQ_MAX_RATE = 0.8    # 方案B: 最大发放率
DELTA_THRESH = 3.0   # 方案C: 慢时间增量门限

# ---------- LSM 蓄水池 ----------
N_RES = 256
LEAK_V = 0.55            # 膜电位泄漏 (每个 PRI)
V_TH = 1.0
REC_DENSITY = 0.12
SPECTRAL_RADIUS = 0.95
IN_FAN = 32              # 每个蓄水池神经元连接的输入通道数 (C/16)
IN_SCALE = 0.3           # 输入权重幅度 (随机符号)
EXC_FRAC = 0.8           # 兴奋神经元比例 (Dale 定律), 抑制权重 ×2
REFRACTORY = 2           # 发放后不应期 (步), 防止池自持续饱和

# ---------- ESN 基线 ----------
ESN_RES = 256
ESN_LEAK = 0.2
ESN_IN_SCALE = 1.0       # 输入权重 ~ N(0, (in_scale/sqrt(n_in))^2)
ESN_REC_DENSITY = 0.12
ESN_SPECTRAL_RADIUS = 0.95

RIDGE_ALPHA = 1e-3
