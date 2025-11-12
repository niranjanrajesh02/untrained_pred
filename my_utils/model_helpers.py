import torch
from torch import nn
import numpy as np


def get_layer_activations(model, layer, image_data, device_id=0):
  # iterate through conv and linear layers
    activations = []
    def hook_fn(module, input, output):
          activations.append(output.detach().cpu())

    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
      for images in image_data:
        images = images.to(f"cuda:{device_id}")
        _ = model(images)
        
    handle.remove()
    acts = torch.cat(activations, dim=0)
    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)
    max_val = torch.max(torch.abs(acts))

    target_dim = 4096 # fixing ddue to fc dimensions in VGG16
    
    if len(acts.shape) > 2:
      # For convolutional layers, apply adaptive average pooling to reduce spatial dimensions and match target_dim
      # when flattened
      n_channels = acts.shape[1]
      pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
      apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
      acts = apool(acts)
      acts = acts.flatten(1)

    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)
    max_val = torch.max(torch.abs(acts))
    if max_val > 1e6 and max_val != 0:
        # Normalize to keep within [-max_range, max_range] for numerical stability
        scale = 1e6 / max_val
        acts = acts * scale
    
    return acts.numpy()


# Wrapper to include input layer
class InputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_layer = nn.Identity()
        self.model = model
    def forward(self, x):
        x = self.input_layer(x)
        return self.model(x)
