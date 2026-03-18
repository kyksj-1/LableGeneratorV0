import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import config_base

class ConfigManager:
    """
    动态配置管理器，负责读取 config.yaml，并结合 config_base 提供统一接口
    """
    def __init__(self):
        self.config_path = config_base.PROJECT_ROOT / "config" / "config.yaml"
        self._config = self._load_yaml()

    def _load_yaml(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"解析 yaml 配置失败: {e}")
                return {}

    @property
    def data_loading(self) -> dict:
        return self._config.get("data_loading", {})

    @property
    def raw_data_dir(self) -> Path:
        return config_base.RAW_DATA_DIR

# 全局单例
config = ConfigManager()
