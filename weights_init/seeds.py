import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import random
import numpy as np
import os

def init_weights(m, method="he"):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if method == "constant":
            init.zeros_(m.weight)
        elif method == "uniform":
            init.uniform_(m.weight, a=0, b=1)
        elif method == "normal":
            init.normal_(m.weight, mean=0.0, std=1)
        elif method == "xavier_u":
            init.xavier_uniform_(m.weight)
        elif method == "xavier_n":
            init.xavier_normal_(m.weight)
        elif method == "kaiming_u":
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif method == "kaiming_n":
            init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif method == "orthogonal":
            init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unknown init method: {method}")
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Main loop ----
init_regimes = ["constant", "uniform", "normal", "xavier_u", "xavier_n", "kaiming_u", "kaiming_n", "orthogonal", "trained"]
num_seeds = 10
save_dir = '../saved_models/untrained/weights/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for method in init_regimes:

    for seed in range(num_seeds):
        set_seed(seed)
        print(f"Initialization method: {method}, Seed: {torch.initial_seed()}")
        if method == "trained":
            # Load pretrained model
            model = models.vgg16(weights='IMAGENET1K_V1')
            
        else:
            # Initialize untrained model
            model = models.vgg16(weights=None)
            # Apply initialization
            model.apply(lambda m: init_weights(m, method))

        model_save_path = '../saved_models/untrained/weights/vgg16_' + method + f'_seed{seed}.pth'
        torch.save(model.state_dict(), model_save_path)
        
        if (method == "constant") or (method == "trained"):
            # only need one model for constant initialization
            break
