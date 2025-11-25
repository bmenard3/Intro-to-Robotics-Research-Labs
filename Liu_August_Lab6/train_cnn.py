#!/usr/bin/env python3
"""
CNN Training Script for Lab 6 Sign Classification
Designed for small dataset (~250 images + augmentation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import csv
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SignDataset(Dataset):
    """Custom dataset for sign images"""
    
    def __init__(self, data_path, transform=None, img_size=64, use_npy=False):
        """
        Args:
            data_path: Path to directory containing images and labels.txt OR .npy files
            transform: Optional transformations
            img_size: Size to resize images to (default: 64x64)
            use_npy: If True, load from images.npy and labels.npy
        """
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        self.use_npy = use_npy
        
        if use_npy:
            # Load from .npy files
            images_file = os.path.join(data_path, 'images.npy')
            labels_file = os.path.join(data_path, 'labels.npy')
            
            if not os.path.exists(images_file) or not os.path.exists(labels_file):
                raise ValueError(f"Could not find images.npy or labels.npy in {data_path}")
            
            self.images = np.load(images_file)  # Already normalized and sized
            self.labels = np.load(labels_file)
            
            print(f"Loaded from .npy: {len(self.labels)} images from {data_path}")
            print(f"  Images shape: {self.images.shape}, dtype: {self.images.dtype}")
            print(f"  Labels shape: {self.labels.shape}, dtype: {self.labels.dtype}")
        else:
            # Load from PNG files with labels.txt
            labels_file = os.path.join(data_path, 'labels.txt')
            self.data = []
            
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Format: "000, 1" (with space after comma)
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) == 2:
                            self.data.append((parts[0], int(parts[1])))
            
            print(f"Loaded {len(self.data)} images from {data_path}")
    
    def __len__(self):
        if self.use_npy:
            return len(self.labels)
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.use_npy:
            # Data is already preprocessed
            image = self.images[idx]  # Shape: (64, 64, 3), already normalized
            label = self.labels[idx]
            
            # Convert to PyTorch format (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            
            if self.transform:
                image = self.transform(image)
            
            return torch.FloatTensor(image), int(label)
        else:
            # Load from PNG
            img_name, label = self.data[idx]
            img_path = os.path.join(self.data_path, f"{img_name}.png")
            
            # Read and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            # Resize
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to PyTorch format (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            
            # Apply transforms if any
            if self.transform:
                image = self.transform(image)
            
            return torch.FloatTensor(image), label


class SmallCNN(nn.Module):
    """
    Small CNN architecture optimized for limited data
    ~150k parameters - small enough to avoid overfitting
    """
    
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(SmallCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class TinyCNN(nn.Module):
    """
    Even smaller CNN for very limited data
    ~50k parameters
    """
    
    def __init__(self, num_classes=6, dropout_rate=0.4):
        super(TinyCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Conv block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(6), yticklabels=range(6))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(train_path, test_path, model_type='small', 
                num_epochs=50, batch_size=32, learning_rate=0.001,
                img_size=64, save_dir='./models', use_npy=False):
    """
    Main training function
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        model_type: 'tiny' or 'small'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        img_size: Image size (default 64x64)
        save_dir: Directory to save models
        use_npy: If True, load from .npy files instead of PNG
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    train_dataset = SignDataset(train_path, img_size=img_size, use_npy=use_npy)
    test_dataset = SignDataset(test_path, img_size=img_size, use_npy=use_npy)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    # Create model
    print("\n" + "="*60)
    print(f"Creating {model_type.upper()} CNN model...")
    if model_type == 'tiny':
        model = TinyCNN(num_classes=6, dropout_rate=0.4)
    else:
        model = SmallCNN(num_classes=6, dropout_rate=0.5)
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, test_loader, 
                                                            criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, f'best_model_{model_type}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': model_type,
                'img_size': img_size
            }, best_model_path)
            print(f"✓ Saved best model (acc: {val_acc:.2f}%)")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test
    _, final_acc, final_preds, final_labels = validate(model, test_loader, 
                                                       criterion, device)
    
    print(f"Final test accuracy: {final_acc:.2f}%")
    
    # Plot confusion matrix
    cm_path = os.path.join(save_dir, f'confusion_matrix_{model_type}.png')
    plot_confusion_matrix(final_labels, final_preds, cm_path)
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Plot training history
    history_path = os.path.join(save_dir, f'training_history_{model_type}.png')
    plot_training_history(train_losses, train_accs, val_losses, val_accs, history_path)
    print(f"✓ Training history saved to {history_path}")
    
    # Print classification report
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    class_names = ['Empty', 'Left', 'Right', 'Do Not Enter', 'Stop', 'Goal']
    print(classification_report(final_labels, final_preds, 
                               target_names=class_names))
    
    return model, best_val_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN for sign classification")
    parser.add_argument("--train_path", type=str, required=True,
                       help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--model_type", type=str, default="small",
                       choices=['tiny', 'small'],
                       help="Model architecture (tiny or small)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--img_size", type=int, default=64,
                       help="Image size (will be resized to img_size x img_size)")
    parser.add_argument("--save_dir", type=str, default="./models",
                       help="Directory to save models")
    parser.add_argument("--use_npy", action="store_true",
                       help="Load data from .npy files instead of PNG (faster)")
    
    args = parser.parse_args()
    
    train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=args.img_size,
        save_dir=args.save_dir,
        use_npy=args.use_npy
    )
