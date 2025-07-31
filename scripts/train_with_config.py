#!/usr/bin/env python3
"""
Training script for product classification with configuration support.

Usage examples:
    python train_with_config.py --config base_config
    python train_with_config.py --config production --epochs 50
    python train_with_config.py --config fast_experiment --backbone resnet18
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import train_model_with_config


def main():
    parser = argparse.ArgumentParser(description='Train product classification model with configuration')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default='base_config',
                        help='Configuration file name (without .yaml extension)')
    
    # Override arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--backbone', type=str, help='Model backbone')
    parser.add_argument('--optimizer', type=str, help='Optimizer type')
    parser.add_argument('--augmentation', type=str, help='Augmentation type')
    
    args = parser.parse_args()
    
    # Build custom configuration overrides
    custom_config = {}
    
    if args.epochs is not None:
        custom_config.setdefault('training', {})['epochs'] = args.epochs
    
    if args.batch_size is not None:
        custom_config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.lr is not None:
        custom_config.setdefault('training', {})['learning_rate'] = args.lr
    
    if args.backbone is not None:
        custom_config.setdefault('model', {})['backbone'] = args.backbone
    
    if args.optimizer is not None:
        custom_config.setdefault('training', {})['optimizer'] = args.optimizer
    
    if args.augmentation is not None:
        custom_config.setdefault('augmentation', {})['type'] = args.augmentation
    
    # Run training
    print(f"Starting training with config: {args.config}")
    if custom_config:
        print(f"Custom overrides: {custom_config}")
    
    try:
        model, best_acc = train_model_with_config(
            config_name=args.config,
            custom_config=custom_config if custom_config else None
        )
        print(f"Training completed successfully! Best accuracy: {best_acc*100:.2f}%")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
