from PIL import Image
import os
import torch
from torchvision import transforms


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # ),
])

class NSD_Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".bmp") and not f.startswith("MFO"))]
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
    return dataset