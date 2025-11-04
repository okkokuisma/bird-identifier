import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(path, model, optimizer, scheduler, epoch, rng_states):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'rng_states': rng_states, 
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            running_loss += loss.item()
    loss = running_loss / num_batches
    return loss

def train_one_epoch(model, optimizer, loss_fn, loader, device):
    running_loss = 0.0
    num_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    loss = running_loss / num_batches
    return loss

def train_loop(start_epoch, total_epochs, model, optimizer, loss_fn, scheduler, loader, device, ckpt_path):
    best_val_loss = 0.0
    model.train()
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = train_one_epoch(model, optimizer, loss_fn, loader, device)
        print(f"Epoch {epoch:03d} / {total_epochs:03d}  loss: {epoch_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
        
        val_loss = validate(model, loader, loss_fn, device)

        if val_loss < best_val_loss:
            rng_states = {
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state(),
                'torch_state': torch.get_rng_state(),
            }
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, rng_states)
            print(f"Saved checkpoint to {ckpt_path} (epoch {epoch})")
