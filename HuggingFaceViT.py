from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
import torchvision
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 5
batch_size = 32
learning_rate = 1e-4

url = '.'
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
dataset = torchvision.datasets.OxfordIIITPet(root='../datasets', download=False)
classes = dataset.classes

val_ratio = 0.2  # 20% for validation
n_total = len(dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

def preprocess_fn(batch):
    images = [img for img, label in batch]
    labels = [label for img, label in batch]
    processed = processor(images=images, return_tensors='pt')
    processed['labels'] = torch.tensor(labels)
    return processed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocess_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocess_fn)

class ViTFineTuned(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classification_head = torch.nn.Linear(self.pretrained.config.hidden_size, len(classes))

    def forward(self, pixel_values):
        hidden_states = self.pretrained(pixel_values=pixel_values).last_hidden_state # B x 197 x d_model
        # output embbeded state + 196 patches computed
        logits = self.classification_head(hidden_states[:, 0, :])
        return logits
model = ViTFineTuned().to(device)

# Zero-shot classification test
correct = 0
total = len(val_dataset)

model.eval()
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    images = batch['pixel_values']
    labels = batch['labels']
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)
    correct += (predictions == labels).sum().item()
accuracy = correct / total
print(f"Zero-shot classification accuracy: {accuracy:.2f}")


print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_loss_avg_arr = []
# Fine tune for dataset
for epoch in range(n_epochs):
    print(f"Running epoch: {epoch}")
    epoch_loss = []
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        images = batch['pixel_values']
        labels = batch['labels']
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        epoch_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    epoch_loss_avg = sum(epoch_loss) / len(epoch_loss) 
    epoch_loss_avg_arr.append(epoch_loss_avg)

    if epoch % 5 == 0:
        val_epoch_loss = []
        model.eval()
        # validation test
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            images = batch['pixel_values']
            labels = batch['labels']
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            val_epoch_loss.append(loss.item())
        print(f"Train: {epoch_loss_avg:.4f}, Val: {sum(val_epoch_loss) / len(val_epoch_loss):.4f}")


# Accuracy test after fine-tuning
correct = 0
total = len(train_dataset)
model.eval()
for batch in train_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    images = batch['pixel_values']
    labels = batch['labels']
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)
    correct += (predictions == labels).sum().item()
accuracy = correct / total
print(f"Train classification accuracy: {accuracy:.2f}")

correct = 0
total = len(val_dataset)
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    images = batch['pixel_values']
    labels = batch['labels']
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)
    correct += (predictions == labels).sum().item()
accuracy = correct / total
print(f"Val classification accuracy: {accuracy:.2f}")

# Train classification accuracy: 1.00
# Val classification accuracy: 0.94