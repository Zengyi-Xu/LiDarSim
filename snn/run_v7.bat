@echo off
cd /d C:\Users\Xuzen\Documents\kimi\workspace\lidar-pointnet\snn
set KMP_DUPLICATE_LIB_OK=TRUE
for %%B in (300 750 1500 3000) do (
  echo ===== budget n-tr-per %%B =====
  C:\Users\Xuzen\anaconda3\python.exe road_kitti_verify.py --seeds 0 1 2 --n-tr-per %%B --tag budget%%B
)
echo ALL_BUDGETS_DONE
