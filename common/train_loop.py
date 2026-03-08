import torch
import torch.nn.functional as F
import json
import os
from common.utils import TrainingConfig

def save_model(model, path, config=None):
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)
    
    # Save config if provided
    if config is not None:
        config_path = os.path.splitext(path)[0] + '.json'
        print(f"Saving config to {config_path}...")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)


def train_loop(model, training_config: TrainingConfig, train_loader, val_loader, optim, device, writer, model_config=None):
    print(training_config)
    print(f"optim: {optim}")
    print(f"Device: {device}")
    print(f"Model:")
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)

    def check_accuracy(model, dataloader, acc_type='val'):
        model.eval()
        with torch.no_grad():
            # check accuracy on data
            total_loss = 0
            for data in dataloader:
                input, target = data[0].to(device), data[1].to(device)
                logits = model(input)
                loss = training_config.loss_fn(logits, target)
                total_loss += loss.item()
            print(f'Avg loss of the network on {acc_type} data: {total_loss / len(dataloader)}')
        model.train()
    print(f"Baseline loss:")
    check_accuracy(model, train_loader, 'train')
    check_accuracy(model, val_loader, 'val')

    print(f'------Commencing training:------')
    epoch_loss_avg_arr = []
    val_loss_avg_arr = []
    model.train()
    for epoch in range(training_config.n_epochs):  # loop over the dataset multiple times
        epoch_loss = []
        print(f"Running epoch: {epoch}")

        # train on training data
        for i, data in enumerate(train_loader, 0):
            inputs, target = data[0].to(device), data[1].to(device)
            
            logits = model(inputs)

            loss = training_config.loss_fn(logits, target)
            epoch_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        try:
            epoch_loss_avg = sum(epoch_loss) / len(epoch_loss) 
        except:
            epoch_loss_avg = 0
        epoch_loss_avg_arr.append(epoch_loss_avg)
        writer.add_scalar('Loss/train', epoch_loss_avg, epoch)

        # check performance on eval data
        model.eval()
        val_loss = []
        for i, data in enumerate(val_loader, 0):
            inputs, target = data[0].to(device), data[1].to(device)
            logits = model(inputs)
            val_loss.append(training_config.loss_fn(logits, target).item())
        try:
            val_loss_avg = sum(val_loss) / len(val_loss)          
        except:
            val_loss_avg = 0
        val_loss_avg_arr.append(val_loss_avg)
        writer.add_scalar('Loss/val', val_loss_avg, epoch)
        print(f'epoch: {epoch} train_loss: {epoch_loss_avg}, val_loss: {val_loss_avg}')
        model.train()
        
        # early stopping
        if training_config.early_stop and epoch > training_config.patience:
            if all(val_loss_avg >= prev for prev in val_loss_avg_arr[-training_config.patience:]):
                print(f"Early stopping at epoch {epoch}")
                break
    writer.close()

    print(f"Final loss:")
    check_accuracy(model, train_loader, 'train')
    check_accuracy(model, val_loader, 'val')
    save_model(model, training_config.model_save_path, model_config)