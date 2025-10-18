import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import random
import numpy as np

# ---- Initialization functions ----
def init_weights(m, method="he"):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if method == "he":
            init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif method == "xavier":
            init.xavier_normal_(m.weight)
        elif method == "glorot":  # synonym for xavier
            init.xavier_uniform_(m.weight)
        elif method == "orthogonal":
            init.orthogonal_(m.weight)
        elif method == "normal":
            init.normal_(m.weight, mean=0.0, std=0.02)
        elif method == "uniform":
            init.uniform_(m.weight, a=-0.05, b=0.05)
        else:
            raise ValueError(f"Unknown init method: {method}")
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

# ---- Reproducibility helper ----
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Main loop ----
init_methods = ["he", "xavier", "glorot", "orthogonal", "normal", "uniform"]
num_seeds = 10
models_dict = {}

for method in init_methods:
    models_dict[method] = []
    for seed in range(num_seeds):
        set_seed(seed)

        # Load a new untrained VGG16 (not pretrained)
        model = models.vgg16(weights=None)

        # Apply initialization
        model.apply(lambda m: init_weights(m, method))

        models_dict[method].append(model)

        print(f"Initialized VGG16 with {method} (seed {seed})")
        break
    break

print("All models initialized successfully!")
