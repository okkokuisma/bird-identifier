# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
import os
from train import set_seed, train_loop, load_checkpoint
from data_preprocess import augment_song

# %%
class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(CNNClassifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)  # [batch, 32, H/2, W/2]
        x = self.conv_block2(x)  # [batch, 64, H/4, W/4]
        x = self.conv_block3(x)  # [batch, 128, H/8, W/8]
        x = self.conv_block4(x)  # [batch, 256, H/16, W/16]
        x = self.global_pool(x)  # [batch, 128, 1, 1]
        logits = self.classifier(x)  # [batch, num_classes]
        return logits

def preprocess(sample):
    spectrogram = torch.tensor(sample[0], dtype=torch.float32)
    spectrogram = augment_song(spectrogram[None])
    label  = torch.tensor(sample[1]['species'], dtype=torch.long)
    return spectrogram, label

def pad_collate(batch):
    specs = []
    labels = []
    for spectrogram, species in batch:
        L = spectrogram.shape[-1]
        if L < 280:
            pad = 280 - L
            spectrogram = F.pad(spectrogram, (0,pad))
        specs.append(spectrogram)
        labels.append(species)
    specs = torch.stack([s for s in specs])  # [B,1,F,T]
    labels = torch.tensor(labels, dtype=torch.long)
    return specs, labels
# %%
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.device('cuda')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    set_seed(42)
    
    start_epoch = 0
    batch_size = 64
    lr = 0.002

    urls = '/Volumes/COCO-DATA/train_shards/shard-{000000..000229}.tar'
    dataset = (
        wds.WebDataset(urls, shardshuffle=1000)
        .decode()
        .rename(spectrogram='npy', species='json')
        .to_tuple('spectrogram', 'species') 
        .map(preprocess)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate, num_workers=4)
    
    model = CNNClassifier(num_classes=231).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_last.pth')

    if os.path.exists(ckpt_path):
        checkpoint = load_checkpoint(ckpt_path, model, optimizer, scheduler, map_location=device)
        loaded_epoch = checkpoint['epoch']
        start_epoch = loaded_epoch + 1

    train_loop(start_epoch, 250, model, optimizer, criterion, scheduler, dataloader, device, ckpt_path, 1)

