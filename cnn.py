# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader, random_split
import numpy as np
from occurrence_data import load_filtered_occurrences
from os import path
from data_download import get_preprocessed_song_ids


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

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)  # [batch, 32, H/2, W/2]
        x = self.conv_block2(x)  # [batch, 64, H/4, W/4]
        x = self.conv_block3(x)  # [batch, 128, H/8, W/8]
        x = self.global_pool(x)  # [batch, 128, 1, 1]
        logits = self.classifier(x)  # [batch, num_classes]
        return logits

# # %%
# class SpectrogramSegmentDataset(Dataset):

#     def __init__(self, occurrences, root_dir, transform=None):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.occurrences = occurrences
#         self.root_dir = root_dir
#         self.transform = transform
#         self.ids = occurrences.iloc[:,0].tolist()
#         self.labels = occurrences.iloc[:,1].astype(int).tolist()

#     def __len__(self):
#         return len(self.occurrences)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         id = self.ids[idx]
#         spectrogram_path = path.join(self.root_dir, str(id) + '.npy')
#         spectrogram = torch.tensor(np.load(spectrogram_path), dtype=torch.float32)

#         species = self.labels[idx]
#         sample = {'spectrogram': spectrogram, 'species': species}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


# # %%
# class Pad():

    # def __init__(self, padded_length):
    #     assert isinstance(padded_length, int)
    #     self.padded_length = padded_length

    # def __call__(self, sample):
    #     spectrogram, species = sample['spectrogram'], sample['species']
    #     if spectrogram.shape[1] < self.padded_length:
    #         pad_t = self.padded_length - spectrogram.shape[1]
    #         p = (0, pad_t)
    #         spectrogram = F.pad(spectrogram, p, "constant", 0)
    #     spectrogram = spectrogram[None, :, :]
    #     return {'spectrogram': spectrogram, 'species': species}

def preprocess(sample):
    spectrogram = torch.tensor(sample[0], dtype=torch.float32)
    label  = torch.tensor(sample[1]['species'], dtype=torch.long)
    return spectrogram, label

def pad_collate(batch):
    specs = []
    labels = []
    for spect, species in batch:
        L = spect.shape[-1]
        if L < 280:
            pad = 280 - L
            spect = F.pad(spect, (0,pad))
        specs.append(spect)
        labels.append(species)
    specs = torch.stack([s[None] for s in specs])  # [B,1,F,T]
    labels = torch.tensor(labels, dtype=torch.long)
    return specs, labels

# %%
if __name__ == '__main__':
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device('cpu')
    else:
        device = torch.device("mps")
    
    # occ = load_filtered_occurrences()
    # fs = get_preprocessed_song_ids()
    # occ = occ[occ['gbifID'].isin(fs)]
    # occ['species'] = occ['species'].cat.codes
    # dataset = SpectrogramSegmentDataset(occurrences=occ, root_dir='/Volumes/COCO-DATA/songs_npy/', transform=Pad(280), device=device)
    urls = "/Volumes/COCO-DATA/shards/shard-{000000..000240}.tar"
    dataset = (
        wds.WebDataset(urls, shardshuffle=1000)
        .decode()
        .rename(spectrogram='npy', species='json')
        .to_tuple('spectrogram', 'species') 
        .map(preprocess)
    )


    # train, val, test = random_split(dataset, [0.7, 0.15, 0.15])

    num_epochs = 50
    batch_size = 128
    lr = 0.005

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate, num_workers=10)
    model = CNNClassifier(num_classes=231).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    printing_interval = num_epochs // 10
    losses_ = []

    model.train()
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        n = 0
        for b in dataloader:
            X = b[0].to(device)
            y = b[1].to(device)
            y_hat = model(X)
            train_loss = criterion(y_hat, y)
            n += 1
            print(n)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += train_loss.item()

        avg_loss = running_loss / len(dataloader)
        losses_.append(avg_loss)

        # if(epoch % printing_interval == 1):
        #     print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

