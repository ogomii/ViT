import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # loss tracking
from ViT import ViT, ViT_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vit_config = ViT_config(
    _channels=3,
    _height=32,
    _width=32,
    _n_patches=4,
    _d_model=1024,
    _n_heads=16,
    _n_layers=24,
    _dropout_rate=0.2
)

n_epochs = 5
batch_size = 80 # multiples of 16 only due to dataset size
learning_rate = 0.0005

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/ViT/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/ViT/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = trainset.classes


def LogGrads(model, epoch):
    with torch.no_grad():
        count = 0
        head_count = 0
        ff_count = 0
        for p in model.parameters():
            layerName: str = ""
            if len(p.shape) == 1: # bias is not ineresting in looking how gradiant passes
                continue
            if p.shape[0] == 64:
                head_count += 1
                if head_count < 46:
                    continue # if not the last head in layer, skip it
                else:
                    layerName = "LastHeadLinear" + str(-(46-head_count))
                    if head_count == 48: # last head in layer we can log to see head gradient norm
                        head_count = 0
            elif p.shape[0] == 1024 and count > 2:
                ff_count += 1
                if ff_count == 1:
                    layerName = "projection"
                else:
                    layerName = "ff"
                if ff_count == 3:
                    ff_count = 0
            if count == 0:
                layerName = "embed"    
            if count == 1:
                layerName = "pos_embed"
            count += 1
            layerName = str(count) + layerName
            writer.add_scalar('Grad_norm/train/layer'+layerName, p.grad.norm() / p.numel(), epoch)
            writer.add_histogram("Grad_dist/train/layer"+layerName, p.grad, global_step=epoch)


m = ViT(vit_config, len(classes))
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

with torch.no_grad():
    model.eval()
    test_loss = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        B, C, H, W = inputs.shape
        logits = m(inputs)
        loss = F.cross_entropy(logits, labels)
        test_loss.append(loss.item())
    train_loss_avg = sum(test_loss) / len(test_loss)  
    print(f'Initialized network train_loss: {train_loss_avg}')
    model.train()

print(f'------Commencing training:------')
epoch_loss_avg_arr = []
model.train()
for epoch in range(n_epochs):  # loop over the dataset multiple times
    epoch_loss = []
    print(f"Running epoch: {epoch}")

    # train on training data
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        logits = m(inputs)

        loss = F.cross_entropy(logits, labels)
        epoch_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    LogGrads(model, epoch)
    epoch_loss_avg = sum(epoch_loss) / len(epoch_loss) 
    epoch_loss_avg_arr.append(epoch_loss_avg)
    writer.add_scalar('Loss/train', epoch_loss_avg, epoch)

    # check performance on eval data
    if (epoch % 4 == 0) or (epoch == n_epochs-1):
        model.eval()
        test_loss = []
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            B, C, H, W = inputs.shape
            logits = m(inputs)
            test_loss.append(F.cross_entropy(logits, labels).item())
        test_loss_avg = sum(test_loss) / len(test_loss)          
        writer.add_scalar('Loss/test', test_loss_avg, epoch)
        print(f'epoch: {epoch} train_loss: {epoch_loss_avg}, val_loss: {test_loss_avg}')
        model.train()
    
    # early stopping
    if epoch > 5:
        if min(epoch_loss_avg_arr[-5:-3]) < (epoch_loss_avg + 0.01):
            print(f"Early stopping on epoch {epoch} due to current loss_avg: {epoch_loss_avg} compared to last 4: {epoch_loss_avg_arr[-5:-1]}")
            break
    
writer.close()

def check_accuracy(model, dataloader, acc_type='test'):
    model.eval()
    with torch.no_grad():
        # check accuracy on data
        correct = 0
        total = 0
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            B, C, H, W = images.shape
            logits = m(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on {acc_type} images: {100 * correct // total} %')
    model.train()

check_accuracy(model, trainloader, 'train')
check_accuracy(model, testloader, 'test')