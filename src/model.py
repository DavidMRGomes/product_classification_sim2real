import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional
from src.focal_loss.focal_loss import FocalLoss



class DomainRobustModel(nn.Module):
    """Domain-robust model with configurable architecture components."""
    
    def __init__(self, backbone_name: str, num_classes: int, feature_dim: int,
                 use_batch_norm: bool = True, use_l2_norm: bool = True, 
                 dropout_rate: float = 0.5, pretrained: bool = True):
        super().__init__()
        
        self.backbone = self._get_backbone(backbone_name, pretrained)
        self.feature_dim = feature_dim
        self.use_l2_norm = use_l2_norm
        
        # Domain-robust components
        if use_batch_norm:
            self.feature_norm = nn.BatchNorm1d(feature_dim)
        else:
            self.feature_norm = nn.Identity()
            
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def _get_backbone(self, backbone_name: str, pretrained: bool):
        """Get the specified backbone model."""
        if backbone_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            return torch.nn.Sequential(*list(model.children())[:-1])
            
        elif backbone_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            return torch.nn.Sequential(*list(model.children())[:-1])
            
        elif backbone_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            return model.features
            
        elif backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            return torch.nn.Sequential(*list(model.children())[:-1])
            
        elif backbone_name == 'mobilenet_v3':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            return torch.nn.Sequential(*list(model.children())[:-1])
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def forward(self, x, return_features: bool = False):
        # Extract features
        features = self.backbone(x)
        
        # Handle different output shapes
        if features.dim() > 2:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze()
        
        # Ensure 2D tensor for batch processing
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Apply domain-robust processing
        features = self.feature_norm(features)
        
        if self.use_l2_norm:
            features = F.normalize(features, p=2, dim=1)
        
        features_for_classification = self.dropout(features)
        logits = self.classifier(features_for_classification)
        
        if return_features:
            return logits, features
        return logits


def get_feature_dim(backbone_name: str) -> int:
    """Get the feature dimension for a given backbone."""
    feature_dims = {
        'resnet18': 512,
        'resnet50': 2048,
        'vgg16': 512,
        'efficientnet_b0': 1280,
        'mobilenet_v3': 960
    }
    
    if backbone_name not in feature_dims:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    return feature_dims[backbone_name]


def get_model(num_classes: int, config: Dict[str, Any]):
    """
    Get model based on configuration.
    
    Args:
        num_classes: Number of output classes
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    if config is None:
        raise ValueError("Configuration is required. Legacy model creation is no longer supported.")
    
    # Use configuration
    model_config = config['model']
    backbone_name = model_config['backbone']
    feature_dim = get_feature_dim(backbone_name)
    
    return DomainRobustModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        feature_dim=feature_dim,
        use_batch_norm=model_config.get('use_batch_norm', True),
        use_l2_norm=model_config.get('use_l2_norm', True),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        pretrained=model_config.get('pretrained', True)
    )


def get_optimizer(model: nn.Module, config: Dict[str, Any]):
    """Get optimizer based on configuration."""
    training_config = config['training']
    optimizer_name = training_config['optimizer']
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config.get('weight_decay', 0))
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, config: Dict[str, Any]):
    """Get learning rate scheduler based on configuration."""
    training_config = config['training']
    scheduler_name = training_config.get('scheduler', 'none')
    
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'step':
        step_size = training_config.get('step_size', 20)
        gamma = training_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        T_max = training_config['epochs']
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    

def get_loss_method(config: Dict[str, Any]):
    """Get loss method based on configuration."""
    training_config = config['training']
    loss_method = training_config.get('loss_method', 'none')

    if loss_method == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_method == 'focal_loss':
        num_classes = 9
        gamma = training_config.get('gamma', 2)
        alpha = training_config.get('alpha', [1] * num_classes)
        return FocalLoss(gamma=gamma, alpha=alpha, task_type='multi-class', num_classes=num_classes)