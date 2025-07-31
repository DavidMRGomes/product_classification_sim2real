import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets import ProductDataset
from src.transforms import get_transforms
from src.model import get_model, get_optimizer, get_scheduler, get_loss_method
from src.config import ConfigLoader, get_device, set_random_seeds
import os
from pathlib import Path
from typing import Optional, Dict, Any
import time


def train_model_with_config(config_name: str = "base_config", 
                           custom_config: Optional[Dict[str, Any]] = None):
    """
    Train model using configuration file.
    
    Args:
        config_name: Name of the configuration file to use
        custom_config: Optional custom configuration overrides
    """
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_name)
    
    # Apply custom overrides if provided
    if custom_config:
        config = config_loader._deep_merge(config, custom_config)
    
    # Set random seeds for reproducibility
    set_random_seeds(config.get('seed', 42))
    
    # Get device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Print configuration summary
    print_config_summary(config)
    
    # Setup data paths
    data_config = config['data']
    data_dir = data_config['data_dir']
    train_dir = os.path.join(data_dir, data_config['train_dir'])
    val_dir = os.path.join(data_dir, data_config['val_dir'])
    
    # Create datasets with configuration-based transforms
    train_dataset = ProductDataset(
        train_dir, 
        transform=get_transforms(train=True, type='config', config=config), 
        use_mask=True
    )
    val_dataset = ProductDataset(
        val_dir, 
        transform=get_transforms(train=False, type='config', config=config), 
        use_mask=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True)
    )
    
    # Update config with inferred number of classes
    config['model']['num_classes'] = len(train_dataset.class_to_idx)
    
    # Create model, optimizer, and scheduler
    model = get_model(len(train_dataset.class_to_idx), config).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_loss_method(config)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['training'].get('patience', 10)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['logging'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    print(f"Model: {config['model']['backbone']}")
    print(f"Augmentation: {config['augmentation']['type']}")
    print(f"Domain Adaptation: {'Enabled' if config['domain_adaptation']['enabled'] else 'Disabled'}")
    print("-" * 60)
    
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # Update progressive augmentation if enabled
        if hasattr(train_dataset.transform, 'set_epoch'):
            train_dataset.transform.set_epoch(epoch, config['training']['epochs'])
        
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if config['domain_adaptation'].get('enabled', False):
                outputs, features = model(imgs, return_features=True)
                
                # Add domain adaptation losses here if needed
                # For now, we'll use the basic classification loss
                loss = criterion(outputs, labels)
                
                # Add feature diversity regularization
                if config['regularization'].get('feature_diversity_loss', False):
                    diversity_weight = config['regularization'].get('feature_diversity_weight', 0.01)
                    diversity_loss = -torch.mean(torch.std(features, dim=0))
                    loss += diversity_weight * diversity_loss
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            
            # Log progress
            if batch_idx % config['logging'].get('log_interval', 10) == 0:
                print(f'Epoch {epoch+1}/{config["training"]["epochs"]} '
                      f'[{batch_idx * len(imgs)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        # Calculate training metrics
        train_acc = train_correct / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_acc, val_loss = validate_with_config(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Train Acc={train_acc*100:.2f}% "
              f"Val Loss={val_loss:.4f} Val Acc={val_acc*100:.2f}% Time={epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            if config['logging'].get('save_model', True):
                checkpoint_path = checkpoint_dir / f"best_model_{config_name}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': config
                }, checkpoint_path)
                print(f"New best model saved: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed! Best validation accuracy: {best_val_acc*100:.2f}%")
    return model, best_val_acc


def validate_with_config(model, val_loader, criterion, device):
    """Validation function that works with configuration-based models."""
    model.eval()
    val_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    
    val_acc = correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader)
    return val_acc, val_loss


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    print("Configuration Summary:")
    print(f"  Model: {config['model']['backbone']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  Augmentation: {config['augmentation']['type']}")
    print(f"  Domain Adaptation: {config['domain_adaptation']['enabled']}")
    print("-" * 60)


if __name__ == "__main__":
    # Example usage with different configurations
    
    # Fast experiment
    print("Running fast experiment...")
    model, acc = train_model_with_config("fast_experiment")
    
    # Production training (uncomment to run)
    # print("Running production training...")
    # model, acc = train_model_with_config("production")
    
    # Custom configuration
    custom_overrides = {
        'training': {
            'epochs': 5,
            'batch_size': 16
        },
        'model': {
            'backbone': 'resnet18'
        }
    }
    
    print("Running with custom overrides...")
    model, acc = train_model_with_config("base_config", custom_overrides)