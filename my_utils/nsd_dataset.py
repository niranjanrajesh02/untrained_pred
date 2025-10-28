from PIL import Image
import os
import torch
from torchvision import transforms
# import vgg16 transforms
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


transform = models.VGG16_Weights.IMAGENET1K_V1.transforms()
# transform = transforms.Compose([
#     transforms.Lambda(lambda img: img.convert('RGB')),  
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(
#     #     mean=[0.485, 0.456, 0.406],
#     #     std=[0.229, 0.224, 0.225]
#     # ),
# ])

class NSD_Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".bmp") and not f.startswith("MFO"))])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_NSD_dataset(folder):
    dataset = NSD_Dataset(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)

    return dataloader

def get_tripleN_fr(folder):
    triple_n = pd.read_csv(folder) #tripleN unit FRs
    start_ind = int(np.where(["stim_1" == s for s in list(triple_n.columns)])[0][0])
    triple_n_FR = triple_n.iloc[:,start_ind:start_ind+1000].to_numpy()
    triple_n_FR = np.transpose(triple_n_FR)  
    return triple_n_FR

