import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

class MNISTModel(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        layers = []
        in_channels = 1  # MNIST is grayscale
        
        # Add convolutional layers with 3x3 kernels
        curr_size = 28
        for k in kernels:
            layers.extend([
                nn.Conv2d(in_channels, k, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = k
            curr_size = curr_size // 2
        
        # Calculate final feature map size
        final_size = curr_size * curr_size * kernels[-1]
        
        # Add classifier layer
        layers.extend([
            nn.Flatten(),
            nn.Linear(final_size, 10)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Print model architecture summary
        print(f"\nModel Architecture:")
        print(f"Input: 28x28x1")
        curr_size = 28
        for i, k in enumerate(kernels):
            print(f"Conv{i+1}: {curr_size}x{curr_size}x{k} (3x3 kernel)")
            curr_size = curr_size // 2
            print(f"MaxPool{i+1}: {curr_size}x{curr_size}x{k}")
        print(f"Flatten: {final_size}")
        print(f"Output: 10\n")
    
    def forward(self, x):
        return self.network(x)

def create_dataloaders(batch_size):
    """Create PyTorch DataLoaders for MNIST"""
    path = Path('data/mnist')
    path.mkdir(parents=True, exist_ok=True)
    
    # Get MNIST data using torchvision
    train_dataset = torchvision.datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    
    # Create train/valid split
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    
    # Create DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size*2)
    
    return train_dl, valid_dl

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, valid_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(valid_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def train_model(config):
    """Train a model with given configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract config parameters
    kernels = config['kernels']
    batch_size = config['batch_size']
    epochs = config['epochs']
    optimizer_name = config['optimizer']
    
    # Create DataLoaders
    train_loader, valid_loader = create_dataloaders(batch_size)
    
    # Create model
    model = MNISTModel(kernels).to(device)
    
    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training history
    history = []
    
    # Train the model
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc / 100,  # Convert to decimal
            'val_loss': val_loss,
            'val_accuracy': val_acc / 100
        })
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
    
    # Save model
    model_path = save_model(model, kernels)
    
    return {
        'history': history,
        'model_path': model_path,
        'model': model
    }

def save_model(model, kernels):
    """Save the trained model"""
    model_name = f"mnist_model_{'_'.join(map(str, kernels))}.pt"
    torch.save(model.state_dict(), model_name)
    return model_name

def compare_models(model1_config, model2_config):
    """Train and compare two models"""
    try:
        print("\n=== Model 1 Configuration ===")
        print(f"Kernels: {model1_config['kernels']}")
        print(f"Batch Size: {model1_config['batch_size']}")
        print(f"Epochs: {model1_config['epochs']}")
        print(f"Optimizer: {model1_config['optimizer']}")
        
        print("\nTraining Model 1...")
        model1_results = train_model(model1_config)
        
        print("\n=== Model 2 Configuration ===")
        print(f"Kernels: {model2_config['kernels']}")
        print(f"Batch Size: {model2_config['batch_size']}")
        print(f"Epochs: {model2_config['epochs']}")
        print(f"Optimizer: {model2_config['optimizer']}")
        
        print("\nTraining Model 2...")
        model2_results = train_model(model2_config)
        
        # Calculate comparison metrics
        m1_final = model1_results['history'][-1]
        m2_final = model2_results['history'][-1]
        
        comparison = {
            'accuracy_diff': m1_final['accuracy'] - m2_final['accuracy'],
            'loss_diff': m1_final['loss'] - m2_final['loss'],
            'model1_params': sum(p.numel() for p in model1_results['model'].parameters()),
            'model2_params': sum(p.numel() for p in model2_results['model'].parameters())
        }
        
        results = {
            'model1': {
                'config': model1_config,
                'history': model1_results['history'],
                'model_path': model1_results['model_path']
            },
            'model2': {
                'config': model2_config,
                'history': model2_results['history'],
                'model_path': model2_results['model_path']
            },
            'comparison': comparison
        }
        
        # Save results with comparison metrics
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == '__main__':
    # Example configurations
    model1_config = {
        'kernels': [16, 32, 64],
        'batch_size': 32,
        'epochs': 10,
        'optimizer': 'adam'
    }
    
    model2_config = {
        'kernels': [8, 8, 8],
        'batch_size': 32,
        'epochs': 10,
        'optimizer': 'sgd'
    }
    
    # Train and compare models
    results = compare_models(model1_config, model2_config) 