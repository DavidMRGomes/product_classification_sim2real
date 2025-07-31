# Configuration System for Product Classification

This configuration system provides flexible, YAML-based configuration management for training product classification models with different backbones, augmentation strategies, and domain adaptation techniques.

## Configuration Files

### Available Configurations

1. **`base_config.yaml`** - Complete base configuration with all options
2. **`fast_experiment.yaml`** - Lightweight config for quick experiments
3. **`production.yaml`** - Production-ready configuration with best practices

### Configuration Structure

```yaml
# Model Configuration
model:
  backbone: "vgg16"  # resnet18, resnet50, vgg16, efficientnet_b0, mobilenet_v3
  pretrained: true
  use_batch_norm: true
  use_l2_norm: true
  dropout_rate: 0.5

# Training Configuration
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3
  optimizer: "adam"  # adam, sgd, adamw
  scheduler: "step"  # step, cosine, none

# Augmentation Configuration
augmentation:
  type: "domain_randomization"  # basic, domain_randomization, progressive
  
# Domain Adaptation
domain_adaptation:
  enabled: true
  alignment_method: "coral"  # coral, domain_shift, center_scale
```

## Usage

### 1. Using the Training Script

```bash
# Basic usage with default config
python train_with_config.py --config base_config

# Fast experiment
python train_with_config.py --config fast_experiment

# Production training
python train_with_config.py --config production

# Custom overrides
python train_with_config.py --config base_config --epochs 20 --backbone resnet18
```

### 2. Using in Python Code

```python
from src.train import train_model_with_config

# Train with base configuration
model, accuracy = train_model_with_config("base_config")

# Train with custom overrides
custom_config = {
    'training': {'epochs': 10, 'batch_size': 64},
    'model': {'backbone': 'vgg16'}
}
model, accuracy = train_model_with_config("base_config", custom_config)
```

### 3. Creating Custom Configurations

```python
from src.config import ConfigLoader

loader = ConfigLoader()

# Create new config based on base_config
custom_overrides = {
    'model': {'backbone': 'efficientnet_b0'},
    'training': {'epochs': 100, 'learning_rate': 5e-4},
    'augmentation': {'type': 'progressive'}
}

config = loader.create_experiment_config("base_config", custom_overrides)
loader.save_config(config, "my_experiment")
```

## Key Features

### 1. **Multiple Backbones**
- ResNet18/50: Fast and reliable
- VGG16: Best performer from analysis
- EfficientNet-B0: Efficient and accurate
- MobileNet-V3: Lightweight for mobile

### 2. **Domain Randomization**
Based on our analysis, includes 6 synthetic domains:
- **Clean**: Original synthetic images
- **Bright**: Enhanced brightness/contrast
- **Dim**: Reduced brightness
- **Desaturated**: Reduced saturation
- **Noisy**: Added camera noise
- **Blurred**: Motion/focus blur

### 3. **Domain Adaptation**
Implements CORAL alignment from our successful analysis:
- Aligns feature covariance between synthetic and real domains
- Improves synthetic-to-real generalization

### 4. **Progressive Training**
- Starts with light augmentations (20% strength)
- Gradually increases to heavy augmentations (80% strength)
- Helps model learn robust features progressively

## Configuration Examples

### Fast Experiment (5 minutes)
```yaml
model:
  backbone: "resnet18"
training:
  epochs: 10
  batch_size: 64
augmentation:
  type: "basic"
domain_adaptation:
  enabled: false
```

### Production Ready (Best Results)
```yaml
model:
  backbone: "vgg16"  # Best from our analysis
training:
  epochs: 100
  optimizer: "adamw"
  scheduler: "cosine"
augmentation:
  type: "domain_randomization"
domain_adaptation:
  enabled: true
  alignment_method: "coral"  # Best from our analysis
```

### Custom Experiment
```yaml
model:
  backbone: "efficientnet_b0"
training:
  epochs: 50
  batch_size: 32
augmentation:
  type: "progressive"
  progressive:
    start_strength: 0.1
    end_strength: 0.9
```

## Best Practices

1. **Start with `fast_experiment.yaml`** for initial testing
2. **Use `production.yaml`** for final models
3. **VGG16 + CORAL** combination performed best in our analysis
4. **Domain randomization** is crucial for synthetic-to-real transfer
5. **Progressive training** helps with convergence

## Extending the System

### Adding New Backbones
1. Update `get_feature_dim()` in `src/model.py`
2. Add backbone case in `DomainRobustModel._get_backbone()`
3. Update configuration documentation

### Adding New Augmentation Types
1. Create new transform class in `src/transforms.py`
2. Update `get_config_transforms()` function
3. Add configuration options in YAML files

### Adding New Domain Adaptation Methods
1. Implement method in `src/transforms.py` or `src/model.py`
2. Update training loop in `src/train.py`
3. Add configuration options

## Troubleshooting

### Common Issues

1. **Config file not found**: Ensure `.yaml` files are in `configs/` directory
2. **CUDA out of memory**: Reduce `batch_size` in config
3. **Poor performance**: Try VGG16 backbone with CORAL alignment
4. **Slow training**: Use `fast_experiment.yaml` for testing

### Performance Tips

1. **Use larger batch sizes** if GPU memory allows
2. **Enable `pin_memory`** for faster data loading
3. **Increase `num_workers`** based on CPU cores
4. **Use mixed precision training** for speed (can be added to config)
