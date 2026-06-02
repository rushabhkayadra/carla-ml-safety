import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Bind core libraries to local runtime visibility mapping context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models import CarlaBinaryClassifier
from src.dataset import CarlaSafetyDataset

def evaluate_performance_metrics(model, dataloader, task_key, device):
    """
    Computes confusion matrix statistics over targeted validation/test datasets.
    Calculates Accuracy, Precision, Recall, and F1 configurations cleanly.
    """
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            targets = labels[task_key].to(device).unsqueeze(1)
            
            logits = model(images)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()
            
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score

def execute_training_pipeline(task_key, train_csv, img_dir, test_csv, epochs=10, batch_size=32, lr=1e-4):
    print(f"\n[PIPELINE START] Beginning Optimization Loop for Task: {task_key.upper()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting Processing Subsystem Asset: {device}")
    
    # Instantiate entire structured data container from shared classes
    full_dataset = CarlaSafetyDataset(metadata_csv=train_csv, img_dir=img_dir)
    test_dataset = CarlaSafetyDataset(metadata_csv=test_csv, img_dir=img_dir)
    
    # Carve out standard 80/20 train/validation splits for tracking convergence boundaries
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize separate classifier model container 
    model = CarlaBinaryClassifier(backbone_name='resnet18', pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            targets = labels[task_key].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Calculate localized validation tracking metrics
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                targets = labels[task_key].to(device).unsqueeze(1)
                logits = model(images)
                loss = criterion(logits, targets)
                epoch_val_loss += loss.item() * images.size(0)
        epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_val_loss:.4f}")
        
    # Plot tracking results to verify convergence characteristics
    plt.figure()
    plt.plot(range(1, epochs+1), history['train_loss'], label='Training Error')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Validation Error')
    plt.title(f'Convergence Profile Matrix: {task_key.upper()} Classifier')
    plt.xlabel('Epoch Counts')
    plt.ylabel('Loss Value Scale')
    plt.legend()
    plt.grid(True)
    plot_path = f"{task_key}_loss_convergence.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Loss curve graphics plotted safely to: {plot_path}")
    
    # Export model weights to central checkpoints directory for downstream dependencies
    os.makedirs("../../checkpoints", exist_ok=True)
    checkpoint_target = f"../../checkpoints/{task_key}_classifier.pt"
    torch.save(model.state_dict(), checkpoint_target)
    print(f"  -> Model weight matrix saved to artifact destination: {checkpoint_target}")
    
    # Final out-of-sample execution run over validation data metrics
    acc, prec, rec, f1 = evaluate_performance_metrics(model, test_loader, task_key, device)
    print(f"  -> Test Performance: Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
    
    return acc, prec, rec, f1

if __name__ == "__main__":
    # Check if we are running in a Google Colab environment
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        # Absolute paths for Google Colab environment
        train_csv = "/content/carla-ml-safety/data/train_metadata.csv"
        test_csv = "/content/carla-ml-safety/data/test_metadata.csv"
        img_dir = "/content/carla-ml-safety/data/images"
    else:
        # Standard relative paths for your local laptop environment
        train_csv = "../../data/train_metadata.csv"
        test_csv = "../../data/test_metadata.csv"
        img_dir = "../../data/images"
    
    tasks = ['pedestrian', 'vehicle', 'traffic_light']
    summary_results = {}
    
    for task in tasks:
        metrics = execute_training_pipeline(task, train_csv, img_dir, test_csv, epochs=8, batch_size=32)
        summary_results[task] = metrics
        
    print("\n========================= COMPLETE SYSTEM EVALUATION SUMMARY =========================")
    for task, metrics in summary_results.items():
        print(f"Task: {task.upper():<15} | Acc: {metrics[0]:.4f} | Prec: {metrics[1]:.4f} | Rec: {metrics[2]:.4f} | F1: {metrics[3]:.4f}")
    print("=======================================================================================")