@echo off
cd /d C:\Users\Xuzen\Documents\kimi\workspace\lidar-pointnet\snn
set KMP_DUPLICATE_LIB_OK=TRUE
C:\Users\Xuzen\anaconda3\python.exe road_vehicles_verify.py --seeds 0 1 2 > v8_4class.log 2>&1
