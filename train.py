import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # loss tracking
from common import TrainingConfig
from ViT import ViT, ViT_config
from DeformableViT import DeformableViT, DeformableViT_config
from train_loop import train_loop

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vit_config = ViT_config(
    _channels=3,
    _height=32,
    _width=32,
    _n_patches=8,
    _d_model=1024,
    _n_heads=16,
    _n_layers=12,
    _dropout_rate=0.2
)

deformable_vit_config = DeformableViT_config(
    _channels = 3,
    _height = 32, 
    _width = 32,
    _n_patches = 4,
    _window_size = 2, # how many patches in a window dim
    _d_model = 1024,
    _n_heads = 8,
    _n_layers = 12,
    _dropout_rate = 0.2
)
save_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

training_config = TrainingConfig(
                    n_epochs=30,
                    batch_size=80, 
                    learning_rate=1e-5,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    patience=3,
                    early_stop=True,
                    weight_decay=1e-4,
                    model_save_path=f'weights/{save_date}.pth'
                    )

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/datasets/', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=training_config.batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/datasets/', train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(testset, batch_size=training_config.batch_size,
                                         shuffle=False, num_workers=2)
classes = trainset.classes

print("Creating model...")
m = DeformableViT(deformable_vit_config, len(classes))
# m = ViT(vit_config, len(classes))
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(
                    model.parameters(), 
                    lr=training_config.learning_rate,
                    weight_decay=training_config.weight_decay
                    )
writer = SummaryWriter()

print("Starting training loop...")
train_loop(model, training_config, train_loader, val_loader, optim, device, writer, deformable_vit_config)
# train_loop(model, training_config, train_loader, val_loader, optim, device, writer, vit_config)
