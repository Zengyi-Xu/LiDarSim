# -*- coding: utf-8 -*-
"""修复 road_kitti/results.json 键名乱码：中文键 -> UTF-8 英文键（V6 附带任务）。"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "outputs_isal" / "road_kitti" / "results.json"
DST = HERE / "outputs_isal" / "road_kitti" / "results_fixed.json"

KEYMAP = {"单次HRRP+CNN1D": "single_hrrp_cnn1d",
          "4波束x3步+ESN+角度": "scan_chain_esn_angle"}

raw = SRC.read_bytes()
try:
    d = json.loads(raw.decode("utf-8"))
    enc = "utf-8"
except UnicodeDecodeError:
    d = json.loads(raw.decode("gbk"))
    enc = "gbk"

fixed = {}
for scen, row in d.items():
    fixed[scen] = {KEYMAP.get(k, k): v for k, v in row.items()}

DST.write_text(json.dumps({"encoding_source": enc, "results": fixed},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print("source encoding:", enc)
print(json.dumps(fixed, indent=1))
