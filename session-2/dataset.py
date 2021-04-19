import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        basepath = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(basepath, labels_path)
        images_path = os.path.join(basepath, images_path)
        self.img_labels = pd.read_csv(labels_path)
        self.img_dir = images_path
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.loc[idx]
        file_name = f"input_{row.suite_id}_{row.sample_id}_{row.code}.jpg"
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        sample = (image, row.code)
        return sample
