# src/cv_module/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class PlantVillageDataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, split="train", transforms=None):
        if csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            # fallback: build df by glob
            raise ValueError("Please provide csv_file")
        self.transforms = transforms or T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor()
        ])
        self.label2idx = {l:i for i,l in enumerate(sorted(self.df['label'].unique()))}
        self.df['label_idx'] = self.df['label'].map(self.label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        img = self.transforms(img)
        label = int(row.label_idx)
        return img, label
