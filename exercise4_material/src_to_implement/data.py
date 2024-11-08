from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        data.loc[:,'Image'] = data['filename'].apply(lambda x: imread(Path(x)))
        self.data = data
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self._transform(gray2rgb(self.data['Image'].iloc[idx])), self.data[['crack','inactive']].iloc[idx].values