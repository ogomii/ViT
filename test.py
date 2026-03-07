import torch
import torchvision
import torchvision.transforms as transforms
import json
from common import TrainingConfig
from ViT import ViT, ViT_config, device
from DeformableViT import DeformableViT, DeformableViT_config

# Load test data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/datasets/', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False, num_workers=2)

# Load model configuration
model_path = '/home/ogomi/AI/ViT/weights/20260307_145439.pth'
config_path = model_path.replace('.pth', '.json')

with open(config_path, 'r') as f:
    config_dict = json.load(f)

# Determine which config type based on presence of window_size
if 'window_size' in config_dict:
    config = DeformableViT_config.from_dict(config_dict)
    model = DeformableViT(config, n_classes=10)
else:
    config = ViT_config.from_dict(config_dict)
    model = ViT(config, n_classes=10)

model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluate on test dataset
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print metrics
print("=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Total Accuracy: {100 * correct / total:.2f}%")
print(f"Correct Predictions: {correct}/{total}")
print("\nPer-Class Accuracy:")
for i in range(10):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  {classes[i]:10s}: {accuracy:6.2f}% ({class_correct[i]}/{class_total[i]})")
print("=" * 50)