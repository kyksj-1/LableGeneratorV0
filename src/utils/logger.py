import logging
import sys
from pathlib import Path
import os

# 确保能导入 config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config_base import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    """
    获取一个标准化的 logger 实例
    既输出到控制台，也输出到文件
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler
    log_file = LOG_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 提供一个默认的全局 logger
logger = get_logger("System")
