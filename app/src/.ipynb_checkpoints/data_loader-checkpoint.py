import os
import torchaudio
from torch.utils.data import Dataset, DataLoader

class BirdCLEFDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".ogg")]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.filepaths[idx])
        if self.transform:
            audio = self.transform(audio)
        return audio, sr

def build_dataloaders(data_dir, batch_size=32):
    train_dataset = BirdCLEFDataset(os.path.join(data_dir, "train"))
    valid_dataset = BirdCLEFDataset(os.path.join(data_dir, "valid"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader