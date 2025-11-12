from PIL import Image
import os
import torch
from torchvision import transforms
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

class COCO_Subset(torch.utils.data.Dataset):
    def __init__(self, folder, ids, transform=None):
        """
        Args:
            folder (str): Path to COCO train2014 images folder
            ids (list[int]): List of integer image IDs to include
            transform (callable, optional): Optional transform to apply to each image
        """
        self.folder = folder
        self.ids = ids
        self.transform = transform

        # Build list of file paths from IDs
        self.paths = [
            os.path.join(folder, f"COCO_train2014_{id_:012d}.jpg")
            for id_ in ids
        ]

        # Optionally, you could filter out missing files:
        self.paths = [p for p in self.paths if os.path.exists(p)]
        if len(self.paths) < len(ids):
            print(f"Warning: {len(ids) - len(self.paths)} images not found in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_NSD_dataset(folder):
    dataset = NSD_Dataset(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    return dataloader

def get_tripleN_fr(folder, category_selective=False):
    triple_n = pd.read_csv(folder) #tripleN unit FRs
    triple_n = triple_n.sort_values(by="Reliability", ascending=False) #sort by reliability
    start_ind = int(np.where(["stim_1" == s for s in list(triple_n.columns)])[0][0])
    triple_n_FR = triple_n.iloc[:,start_ind:start_ind+1000].to_numpy()
    triple_n_FR = np.transpose(triple_n_FR)  

    if category_selective:
        cats = ['O', 'B', 'F']
        cat_units_FR = {}

        for cat in cats:
            cat_units_indices = np.where(triple_n['Category'].values == cat)[0]
            cat_FR = triple_n_FR[:, cat_units_indices]
            cat_units_FR[cat] = cat_FR
        return cat_units_FR

    else:
        return triple_n_FR

def get_tripleN_reliability(folder):
    triple_n = pd.read_csv(folder) #tripleN unit FRs
    triple_n = triple_n.sort_values(by="Reliability", ascending=False)
    neuron_reliability = list(triple_n['Reliability'])
    return neuron_reliability

def get_COCO_dataloader(folder):
    coco_neural = np.load("/home/nirajesh/untrained_pred/neural_data/NSD_monkey/macaque_NSD_IT.npy", allow_pickle=True).item()
    coco_ids = list(coco_neural['coco_IDs'])
    dataset = COCO_Subset(folder, ids, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
