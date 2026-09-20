"""自动下载 ModelNet40 HDF5 格式数据集

优先使用斯坦福官方源; 官方源不可达时自动回退到 hf-mirror 的
ModelNet40-2048 parquet 数据, 并转换为标准 HDF5 布局, 保证
utils/dataset.py 无需任何修改即可加载。
"""
import os
import zipfile
import urllib.request
from pathlib import Path

URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
MIRROR_PARQUET = {
    "train": "https://hf-mirror.com/datasets/jxie/modelnet40-2048/resolve/main/data/train-00000-of-00001-4fece5076596e98a.parquet",
    "test": "https://hf-mirror.com/datasets/jxie/modelnet40-2048/resolve/main/data/test-00000-of-00001-baa2ae7c6a5df7e0.parquet",
}
DATA_ROOT = Path(__file__).parent / "data"
TARGET_DIR = DATA_ROOT / "modelnet40_ply_hdf5_2048"


def download():
    DATA_ROOT.mkdir(exist_ok=True)

    if TARGET_DIR.exists():
        print("Dataset already exists.")
        return

    # 优先尝试斯坦福官方源
    try:
        _download_official()
        return
    except Exception as e:
        print(f"Official source failed ({e}), trying HF mirror...")

    # 回退: hf-mirror parquet -> 转换为标准 HDF5 布局
    _download_from_mirror()


def _download_official():
    zip_path = DATA_ROOT / "modelnet40.zip"
    print(f"Downloading ModelNet40 from {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_ROOT)

    zip_path.unlink()
    print("Done. Dataset ready at:", TARGET_DIR)


def _download_from_mirror():
    import h5py
    import numpy as np
    import pyarrow.parquet as pq

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for split, url in MIRROR_PARQUET.items():
        parquet_path = DATA_ROOT / f"modelnet40_{split}.parquet"
        if not parquet_path.exists():
            print(f"Downloading {split} parquet from mirror ...")
            urllib.request.urlretrieve(url, parquet_path)

        print(f"Converting {split} parquet -> HDF5 ...")
        table = pq.read_table(parquet_path)
        data = np.asarray(table.column('inputs').to_pylist(), dtype=np.float32)  # (N, 2048, 3)
        labels = table.column('label').to_numpy()                                # (N,)

        # 按官方布局分片: train 5 个文件, test 2 个文件
        n_shards = 5 if split == 'train' else 2
        for i, idx in enumerate(np.array_split(np.arange(len(data)), n_shards)):
            out = TARGET_DIR / f"ply_data_{split}{i}.h5"
            with h5py.File(out, 'w') as f:
                f['data'] = data[idx]
                f['label'] = labels[idx].reshape(-1, 1)

    print("Done. Dataset ready at:", TARGET_DIR)


if __name__ == '__main__':
    download()
