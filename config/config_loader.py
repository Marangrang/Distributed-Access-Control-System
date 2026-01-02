"""
Configuration loader with validation and environment variable substitution
"""
import os
import re
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigLoader:
    """Load and validate configuration from YAML with env var substitution"""

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]+))?\}')

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Substitute environment variables
        self._config = self._substitute_env_vars(raw_config)

        # Validate required fields
        self._validate_config()

        return self._config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string(obj)
        else:
            return obj

    def _substitute_string(self, value: str) -> Any:
        """Substitute ${VAR:default} patterns with environment variables"""
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.getenv(var_name)
            if env_value is None:
                if default_value is None:
                    raise ValueError(f"Required environment variable not set: {var_name}")
                return default_value
            return env_value

        result = self.ENV_VAR_PATTERN.sub(replacer, value)

        # Convert to appropriate type
        if result.lower() == 'true':
            return True
        elif result.lower() == 'false':
            return False
        elif result.isdigit():
            return int(result)
        elif self._is_float(result):
            return float(result)
        else:
            return result

    @staticmethod
    def _is_float(value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _validate_config(self):
        """Validate required configuration fields"""
        required_fields = [
            'minio.endpoint',
            'minio.access_key',
            'minio.secret_key',
        ]

        for field in required_fields:
            if not self._get_nested(self._config, field):
                raise ValueError(f"Required configuration field missing: {field}")

    def _get_nested(self, config: Dict, path: str) -> Any:
        """Get nested config value using dot notation"""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation"""
        if self._config is None:
            self.load()
        return self._get_nested(self._config, path) or default


# Singleton instance
_config_loader = None


def get_config() -> Dict[str, Any]:
    """Get loaded configuration (singleton)"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
        _config_loader.load()
    return _config_loader._config


def get_config_value(path: str, default: Any = None) -> Any:
    """Get specific config value"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
        _config_loader.load()
    return _config_loader.get(path, default)
