"""
配置管理模块
提供全局配置加载和访问接口
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """全局配置管理类"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None):
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 解析路径（相对路径转绝对路径）
        self._resolve_paths()
        
        logger.info(f"配置文件加载成功: {config_path}")
    
    def _resolve_paths(self):
        """将相对路径转换为绝对路径"""
        root_dir = Path(self._config['project']['root_dir'])
        
        def resolve_dict(d: dict, parent_key: str = ''):
            for key, value in d.items():
                if isinstance(value, dict):
                    resolve_dict(value, f"{parent_key}.{key}" if parent_key else key)
                elif isinstance(value, str) and ('/' in value or '\\' in value):
                    if not os.path.isabs(value):
                        d[key] = str(root_dir / value)
        
        if 'paths' in self._config:
            resolve_dict(self._config['paths'])
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        支持点号分隔的嵌套键，如 'paths.raw_data.gee'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项（运行时修改）"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None):
        """保存配置到文件"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置文件已保存: {config_path}")
    
    @property
    def config(self) -> Dict:
        """返回完整配置字典"""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.set(key, value)


# 全局配置实例
cfg = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return cfg


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(key: str) -> Path:
    """获取路径配置并确保目录存在"""
    path = cfg.get(key)
    if path is None:
        raise ValueError(f"路径配置不存在: {key}")
    return ensure_dir(path)
