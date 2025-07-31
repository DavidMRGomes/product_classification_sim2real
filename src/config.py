"""
Configuration management utilities for product classification training.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


class ConfigLoader:
    """Utility class for loading and merging configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str, base_config: str = "base_config") -> Dict[str, Any]:
        """
        Load a configuration file, inheriting from base config if specified.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            base_config: Name of the base config to inherit from
            
        Returns:
            Merged configuration dictionary
        """
        # Load base config first
        base_config_path = self.config_dir / f"{base_config}.yaml"
        if base_config_path.exists() and config_name != base_config:
            with open(base_config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Load specific config and merge
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                specific_config = yaml.safe_load(f)
                config = self._deep_merge(config, specific_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to file."""
        config_path = self.config_dir / f"{filename}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


@dataclass
class TrainingConfig:
    """Structured configuration for training parameters."""
    
    # Model parameters
    backbone: str
    num_classes: Optional[int]
    pretrained: bool
    dropout_rate: float
    use_batch_norm: bool
    use_l2_norm: bool
    
    # Training parameters
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    patience: int
    
    # Data parameters
    data_dir: str
    train_dir: str
    val_dir: str
    num_workers: int
    
    # Augmentation parameters
    augmentation_type: str
    progressive_enabled: bool
    
    # Domain adaptation parameters
    domain_adaptation_enabled: bool
    alignment_method: str
    
    # Device parameters
    device_type: str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from configuration dictionary."""
        return cls(
            # Model
            backbone=config_dict['model']['backbone'],
            num_classes=config_dict['model'].get('num_classes'),
            pretrained=config_dict['model']['pretrained'],
            dropout_rate=config_dict['model']['dropout_rate'],
            use_batch_norm=config_dict['model']['use_batch_norm'],
            use_l2_norm=config_dict['model']['use_l2_norm'],
            
            # Training
            epochs=config_dict['training']['epochs'],
            batch_size=config_dict['training']['batch_size'],
            learning_rate=config_dict['training']['learning_rate'],
            weight_decay=config_dict['training']['weight_decay'],
            optimizer=config_dict['training']['optimizer'],
            scheduler=config_dict['training']['scheduler'],
            patience=config_dict['training']['patience'],
            
            # Data
            data_dir=config_dict['data']['data_dir'],
            train_dir=config_dict['data']['train_dir'],
            val_dir=config_dict['data']['val_dir'],
            num_workers=config_dict['data']['num_workers'],
            
            # Augmentation
            augmentation_type=config_dict['augmentation']['type'],
            progressive_enabled=config_dict['augmentation']['progressive']['enabled'],
            
            # Domain adaptation
            domain_adaptation_enabled=config_dict['domain_adaptation']['enabled'],
            alignment_method=config_dict['domain_adaptation'].get('alignment_method', 'none'),
            
            # Device
            device_type=config_dict['device']['type'],
        )


def get_device(config: Dict[str, Any]) -> str:
    """Get the appropriate device based on configuration."""
    import torch
    
    device_config = config['device']
    
    if device_config['type'] == 'auto':
        if torch.cuda.is_available():
            return f"cuda:{device_config.get('gpu_id', 0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return device_config['type']


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# Example usage functions
def load_training_config(config_name: str = "base_config") -> Dict[str, Any]:
    """Convenience function to load a training configuration."""
    loader = ConfigLoader()
    return loader.load_config(config_name)


def create_experiment_config(base_config: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new experiment configuration with overrides."""
    loader = ConfigLoader()
    config = loader.load_config(base_config)
    
    # Apply overrides
    config = loader._deep_merge(config, overrides)
    
    return config


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_training_config("base_config")
        print("Base configuration loaded successfully!")
        
        # Test structured config
        training_config = TrainingConfig.from_dict(config)
        print("Using config:", training_config)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
