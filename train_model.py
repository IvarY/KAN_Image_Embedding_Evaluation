import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt





def extract_embeddings(model, data_loader, device):
    """Extract embeddings and labels from a model and dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            emb = model(inputs, return_embeddings=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_model(model, model_name,
                train_loader, val_loader, test_loader=None, train_no_aug_loader=None,
                device=None,
                criterion=None, optimizer=None,
                num_epochs=50, batch_size=128, learning_rate=0.001, patience=10,
                random_state=42,
                test_model=False,
                print_epoch_logs=True,
                create_embeddings=True):
    """Main training function"""
    
    results_dir = Path('./results')
    models_dir = results_dir / 'models' / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, train_no_aug = get_data_loaders(batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    

    model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'epoch_time': epoch_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        if print_epoch_logs:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Time: {epoch_time:.2f}s")
        
        # Save intermediate model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = models_dir / f'_ep{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            if print_epoch_logs:
                print(f"Saved checkpoint: {checkpoint_path}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = models_dir / 'model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_architecture': 'SimpleCNN',
                'num_classes': 10
            }, best_model_path)
            if print_epoch_logs:
                print(f"New best model saved: {best_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience: {patience})")
            break
    
    print("-" * 60)
    print("Training completed!")
    
    # Save training history to DataFrame and CSV
    df_history = pd.DataFrame(training_history)
    history_path = models_dir / f'{model_name}_training_history.csv'
    df_history.to_csv(history_path, index=False)
    mean_time = df_history['epoch_time'].mean()
    print(f"Average epoch time: {mean_time:.2f}s")
    print(f"Training history saved to: {history_path}")
    
    if test_model and test_loader is not None:
        print("\nEvaluating on test set...")
        best_checkpoint = torch.load(models_dir / 'model.pt')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
        
        # plot_training_curves(df_history, results_dir / f'{model_name}_training_curves.png')

    print("\nExtracting and saving embeddings from best model...")
    best_checkpoint = torch.load(models_dir / 'model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.to(device)
    
    if create_embeddings and all(loader is not None for loader in [train_no_aug_loader, val_loader, test_loader]):
        emb_splits = {
            'train': train_no_aug_loader,
            'val': val_loader,
            'test': test_loader
        }
        for split, loader in emb_splits.items():
            if loader is not None:
                emb, labels = extract_embeddings(model, loader, device)
                torch.save({'embeddings': emb, 'labels': labels}, models_dir / f"{split}_embeddings.pt")
                print(f"Saved {split} embeddings to {models_dir / f'{split}_embeddings.pt'}")

    return df_history
    

def validate_model(model, val_loader, criterion, device):
    """Validate the model and return average loss and accuracy"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    return avg_val_loss, val_accuracy


class TransformDataset(torch.utils.data.Dataset):
    """Wrapper for a dataset to apply a transform on-the-fly"""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.base_dataset)
    

def get_data_loaders(batch_size=128, random_state=42, dataset_path="./data", num_workers=2):
    """Create data loaders for CIFAR-10 with augmentations"""
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    

    base_train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=None
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=val_test_transform
    )

    
    train_size = int(0.8 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        base_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    
    train_dataset = TransformDataset(train_subset, train_transform)
    train_no_aug_dataset = TransformDataset(train_subset, val_test_transform)
    val_dataset = TransformDataset(val_subset, val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_no_aug_loader = DataLoader(train_no_aug_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_no_aug_loader





























# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# import time

# def train_model(batch_size = 64):
#     transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
#     ])

#     train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")