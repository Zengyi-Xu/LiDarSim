# RUNBOOK：3060 机器上跑 KITTI 全量分类实验

> 目标平台：11800H + RTX 3060。新 Kimi session 里直接按本文件执行即可。

## 一次性环境准备

```bash
git clone git@github.com:Zengyi-Xu/LiDarSim.git lidar-pointnet
cd lidar-pointnet
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 或 cu118
pip install numpy pandas pyarrow matplotlib scikit-learn
python -c "import torch; print(torch.cuda.is_available())"   # 必须输出 True
```

把 `road_objects.npz`（82 MB，从旧机器 `lidar-pointnet/data/` 拷贝）放到 `lidar-pointnet/data/` 下。
（如拿不到，可用旧机器 KITTI 原始包重跑 `python snn/road_object_extractor.py --root <KITTI>/training --out ../data/road_objects.npz` 重新生成，约 3 分钟。）

## 开跑

```bash
cd lidar-pointnet/snn
python road_kitti_experiment.py            # 全量预算（每类75%训练，~21.5k train / ~7.2k test）
python road_kitti_experiment.py --n-tr-per 300   # 旧的小预算对照（可选）
```

- 代码自动检测 CUDA（`isal_range_profile.DEVICE`）；`--cpu` 可强制 CPU 调试
- 预计 10–20 分钟（GPU）；结果：`outputs_isal/road_kitti/results.json` + `acc_bars.png`
- 对照基准（小预算 1200 train）：road_crossing 单次 0.657 / 扫描链 0.766；uav_cap 0.419 / 0.631

## 跑完后

把 `outputs_isal/road_kitti/results.json` 的数字汇报即可：两场景 × 两臂（单次 HRRP+CNN1D / 4波束×3步+ESN+角度）共 4 个数。
重点看：① 全量预算下 road_crossing 扫描链是否 >0.85；② 扫描链增益是否保持 +0.1 以上。

## 数值等价性

`make_dataset_fast`（torch 批量版）与原 numpy 版已在本机 CPU 校验：
corr=0.999991，小规模端到端 0.699/0.770 vs 0.701/0.770（噪声内一致）。
校验脚本：`python snn/validate_profiles_fast.py`（GPU 机器上可重跑确认）。
