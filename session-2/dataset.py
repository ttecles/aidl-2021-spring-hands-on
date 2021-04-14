import os
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        self.img_label = pd.read_csv(labels_path)
        self.images_path = Path(images_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        row = self.img_label.loc[idx]
        filename = f"input_{row.suite_id}_{row.sample_id}_{row.code}.jpg"
        image = Image.open(self.images_path / filename)
        if self.transform:
            image = self.transform(image)
        return {"label": row.code, "image": image}
