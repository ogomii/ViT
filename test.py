import torch
import torchvision
import torchvision.transforms as transforms
import json
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from common.utils import TrainingConfig
from models.ViT import ViT, ViT_config, device
from models.DeformableViT import DeformableViT, DeformableViT_config

# Load test data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/datasets/', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False, num_workers=2)

# Load model configuration
train_name = "20260308_114553"
model_path = f'/home/ogomi/AI/ViT/weights/{train_name}.pth'
config_path = model_path.replace('.pth', '.json')

with open(config_path, 'r') as f:
    config_dict = json.load(f)
# Determine which config type based on presence of window_size
if 'window_size' in config_dict:
    config = DeformableViT_config.from_dict(config_dict)
    model = DeformableViT(config, n_classes=10)
    model_name = train_name + "DViT"
else:
    config = ViT_config.from_dict(config_dict)
    model = ViT(config, n_classes=10)
    model_name = train_name + "ViT"


model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluate on test dataset
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# For confusion matrix
all_predictions = []
all_labels = []

start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store predictions and labels for confusion matrix
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

end_time = time.time()
total_runtime = end_time - start_time

# Print metrics
print("=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Total Accuracy: {100 * correct / total:.2f}%")
print(f"Correct Predictions: {correct}/{total}")
print(f"\nInference Runtime: {total_runtime:.2f} seconds")
print(f"Average Time per Batch: {total_runtime / len(test_loader):.4f} seconds")
print("\nPer-Class Accuracy:")
for i in range(10):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  {classes[i]:10s}: {accuracy:6.2f}% ({class_correct[i]}/{class_total[i]})")
print("=" * 50)

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Visualize confusion matrix with improved styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Raw counts
im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
fig.colorbar(im1, ax=ax1, label='Count')

ax1.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label')
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax1.set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold', pad=20)
ax1.grid(False)

# Add text annotations for raw counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = ax1.text(j, i, str(cm[i, j]),
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# Plot 2: Normalized (percentages per true class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

im2 = ax2.imshow(cm_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
fig.colorbar(im2, ax=ax2, label='Percentage')

ax2.set(xticks=np.arange(cm_normalized.shape[1]),
        yticks=np.arange(cm_normalized.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label')
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax2.set_title('Confusion Matrix - Normalized (%)', fontsize=14, fontweight='bold', pad=20)
ax2.grid(False)

# Add text annotations for percentages
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        text = ax2.text(j, i, f'{cm_normalized[i, j]:.1%}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# Overall title
fig.suptitle(f'Confusion Matrix - {model_name} (Accuracy: {100 * correct / total:.2f}%)',
             fontsize=16, fontweight='bold', y=1.02)

fig.tight_layout()
confusion_matrix_path = f'/home/ogomi/AI/ViT/weights/confusion_matrix_{model_name}.png'
plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved to: {confusion_matrix_path}")