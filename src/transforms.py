import torchvision.transforms as T
import torch
import random
from typing import Dict, Any, List
import ast



def get_transforms(train=True, type='config', config=None):
    """
    Get transforms based on configuration.
    
    Args:
        train: Whether these are training transforms
        type: Transform type (must be 'config')
        config: Configuration dictionary
    """
    if type != 'config' or config is None:
        raise ValueError("Only configuration-based transforms are supported. Use type='config' with a valid config.")
    
    return get_config_transforms(train, config)


def get_config_transforms(train: bool, config: Dict[str, Any]):
    """Get transforms based on configuration."""
    if not train:
        # Validation transforms are always simple
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    aug_config = config['augmentation']
    aug_type = aug_config['type']
    
    if aug_type == 'standard':
        return get_standard_transforms()
    elif aug_type == 'simple':
        return get_simple_transforms(aug_config)
    elif aug_type == 'randaugment':
        return T.RandAugment(num_ops=3)
    elif aug_type == 'domain_randomization':
        return DomainRandomizationTransform(aug_config)
    elif aug_type == 'progressive':
        return ProgressiveTransform(aug_config)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def get_standard_transforms():
    """Create standard augmentation transforms."""
    transforms = [T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    
    return T.Compose(transforms)

def get_simple_transforms(aug_config: Dict[str, Any]):
    """Create simple augmentation transforms."""
    simple_aug_config = aug_config['simple_aug']
    
    transforms = [T.ToTensor()]
    
    # Random rotation
    if simple_aug_config.get('rotation', 0) > 0:
        transforms.append(T.RandomRotation(simple_aug_config['rotation']))

    # Random crop resized
    if simple_aug_config.get('random_crop', False):
        crop_size = ast.literal_eval(simple_aug_config.get('crop_size', (224, 224)))
        transforms.append(T.RandomResizedCrop(crop_size, scale=(0.6, 1.0)))

    # Random Perspective
    if simple_aug_config.get('random_perspective', False):
        transforms.append(T.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0))
    
    if 'color_jitter' in simple_aug_config:
        cj = simple_aug_config['color_jitter']
        transforms.append(T.RandomApply([
            T.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            )
        ], p=0.5))
    
    # Add normalization
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)


class DomainRandomizationTransform:
    # NOT IMPLEMENTED
    """Domain randomization transform that randomly selects from multiple domains."""
    
    def __init__(self, aug_config: Dict[str, Any]):
        self.domains = {}
        self.weights = []
        
        # Build domain transforms
        for domain_name, domain_config in aug_config['domains'].items():
            if not domain_config.get('enabled', True):
                continue
                
            transforms = [T.Resize((224, 224)), T.ToTensor()]
            
            # Add domain-specific augmentations
            if domain_name == 'bright':
                brightness = domain_config.get('brightness', [1.0, 1.3])
                contrast = domain_config.get('contrast', [0.9, 1.1])
                transforms.append(T.ColorJitter(brightness=brightness, contrast=contrast))
                
            elif domain_name == 'dim':
                brightness = domain_config.get('brightness', [0.6, 0.8])
                contrast = domain_config.get('contrast', [1.0, 1.2])
                transforms.append(T.ColorJitter(brightness=brightness, contrast=contrast))
                
            elif domain_name == 'desaturated':
                saturation = domain_config.get('saturation', [0.5, 0.7])
                transforms.append(T.ColorJitter(saturation=saturation))
                
            elif domain_name == 'noisy':
                noise_std = domain_config.get('noise_std', 0.02)
                transforms.append(T.Lambda(lambda x: x + torch.randn_like(x) * noise_std))
                
            elif domain_name == 'blurred':
                blur_sigma = domain_config.get('blur_sigma', [0.5, 1.5])
                transforms.append(T.GaussianBlur(kernel_size=3, sigma=blur_sigma))
            
            # Add normalization
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
            self.domains[domain_name] = T.Compose(transforms)
            self.weights.append(domain_config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        self.domain_names = list(self.domains.keys())
    
    def __call__(self, img):
        # Randomly select a domain based on weights
        domain_name = random.choices(self.domain_names, weights=self.weights)[0]
        return self.domains[domain_name](img)


class ProgressiveTransform:
    # NOT IMPLEMENTED
    """Progressive augmentation that increases strength over time."""
    
    def __init__(self, aug_config: Dict[str, Any]):
        self.aug_config = aug_config
        self.progressive_config = aug_config['progressive']
        self.current_strength = self.progressive_config['start_strength']
        self.base_transform = DomainRandomizationTransform(aug_config)
    
    def set_epoch(self, epoch: int, max_epochs: int):
        """Update augmentation strength based on current epoch."""
        progress = epoch / max_epochs
        start_strength = self.progressive_config['start_strength']
        end_strength = self.progressive_config['end_strength']
        self.current_strength = start_strength + (end_strength - start_strength) * progress
    
    def __call__(self, img):
        # Apply base transform with current strength
        # For simplicity, we'll use the base domain randomization
        # In a full implementation, you'd scale the augmentation parameters
        return self.base_transform(img)


# Utility functions for CORAL alignment (from our analysis)
def apply_coral_alignment(source_features, target_features):
    """Apply CORAL alignment to features."""
    import numpy as np
    
    # Center the features
    source_centered = source_features - np.mean(source_features, axis=0)
    target_centered = target_features - np.mean(target_features, axis=0)
    
    # Compute covariance matrices
    cov_source = np.cov(source_centered.T)
    cov_target = np.cov(target_centered.T)
    
    # Compute whitening and coloring transforms
    eigenvals_s, eigenvecs_s = np.linalg.eigh(cov_source)
    eigenvals_t, eigenvecs_t = np.linalg.eigh(cov_target)
    
    # Add regularization
    eigenvals_s += 1e-6
    eigenvals_t += 1e-6
    
    # Whitening transform for source
    whitening = eigenvecs_s @ np.diag(eigenvals_s**(-0.5)) @ eigenvecs_s.T
    # Coloring transform to target
    coloring = eigenvecs_t @ np.diag(eigenvals_t**(0.5)) @ eigenvecs_t.T
    
    # Apply transformation
    whitened_source = source_centered @ whitening.T
    aligned_source = whitened_source @ coloring.T + np.mean(target_features, axis=0)
    
    return aligned_source