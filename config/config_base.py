import os
from pathlib import Path

# 获取项目根目录 (假设此文件在 config/ 目录下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 尝试获取环境变量 KALMAN_DATA_ROOT_AKS，如果未设置则提供一个默认路径
KALMAN_DATA_ROOT_AKS = os.environ.get(
    "KALMAN_DATA_ROOT_AKS", 
    "D:/KalmanStockData/AKS"
)

# 统一的数据目录
DATA_DIR = Path(KALMAN_DATA_ROOT_AKS)
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 你可以在这里添加其他“不需要频繁变动”的系统级常量
