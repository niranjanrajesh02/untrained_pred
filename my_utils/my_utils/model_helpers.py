import torch
import random
import numpy as np
from torch import nn
import torch.nn.init as init
from torchvision import models


### --- Model Initialization Helpers --- ###
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m, method="kaiming_u"):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if method == "uniform":
            init.uniform_(m.weight, a=-1, b=1)
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

# Wrapper to include input layer for extracting pixels as activations
class ModelWithInputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_layer = nn.Identity()
        self.model = model
    def forward(self, x):
        x = self.input_layer(x)
        return self.model(x)

def init_vgg(seed=0, regime='kaiming_n'):
    # Initialize VGG model with given seed and regime
    set_seed(seed)
    if regime == "trained":
        vgg = models.vgg16(weights="IMAGENET1K_V1")
    else:
      vgg = models.vgg16(weights=None)
      vgg.apply(lambda m: init_weights(m, method=regime))

    # add input wrapper
    vgg = ModelWithInputLayer(vgg)
    return vgg
    

def init_model(model_name, seed=0, regime='kaiming_n'):
    assert model_name in [
        'vgg16',
        'resnet18',
        'resnet50',
        'convnext_b',
        'vit_b_16',
        'dino_resnet50',
        'mocov3_resnet50',
        'simclr_resnet50'
    ]

    model = None
    set_seed(seed)
    if model_name == 'vgg16':
        if regime == "trained":
            model = models.vgg16(weights="IMAGENET1K_V1")
        else:
            model = models.vgg16(weights=None)
            model.apply(lambda m: init_weights(m, method=regime))
    elif model_name == 'resnet18':
        if regime == "trained":
            model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            model = models.resnet18(weights=None)
            model.apply(lambda m: init_weights(m, method=regime))
    elif model_name == 'resnet50':
        if regime == "trained":
            model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            model = models.resnet50(weights=None)
            model.apply(lambda m: init_weights(m, method=regime))
    elif model_name == 'convnext_b':
        if regime == "trained":
            model = models.convnext_base(weights="IMAGENET1K_V1")
        else:
            model = models.convnext_base(weights=None)
            model.apply(lambda m: init_weights(m, method=regime))
    elif model_name == 'vit_b_16':
        if regime == "trained":
            model = models.vit_b_16(weights="IMAGENET1K_V1")
        else:
            model = models.vit_b_16(weights=None)
            model.apply(lambda m: init_weights(m, method=regime))

    ## SSL models - have to be trained

    elif model_name == 'dino_resnet50':
        if regime == "trained":
            model = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_resnet50'
            )
    


    elif model_name == 'mocov3_resnet50':
        ckpt = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
                map_location="cpu"
        )
        state_dict = ckpt["state_dict"]

        # remove momentum / projector / prefix
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module.base_encoder."):
                new_state[k.replace("module.base_encoder.", "")] = v

        model.load_state_dict(new_state, strict=False)

    
    # add input wrapper
    model = ModelWithInputLayer(model)
    return model    

def count_nonzero_weights(model):
    total_weights = 0
    nonzero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            nonzero_weights += torch.sum(param != 0).item()
    return nonzero_weights, total_weights


def get_layer_names(model):
    layer_names = []
    for name, layer in model.named_modules():
        # skip container modules (those that have submodules)
        if len(list(layer.children())) == 0:
            # skip Relu and dropout
            if not isinstance(layer, (torch.nn.ReLU, torch.nn.Dropout)):
                layer_names.append(name)
    return layer_names[:-1]  # exclude the final classifier layer

def get_nice_layer_names(model, layer_names):
    nice_layer_names = []
    named_modules = dict(model.named_modules())
    counters = {}

    for layer in layer_names:
        if layer == 'input_layer':
            nice_layer_names.append('Input')
            continue
        
        module = named_modules.get(layer, None)
        if module is None:
            nice_layer_names.append(layer)
            continue
        
        layer_type = module.__class__.__name__
        counters[layer_type] = counters.get(layer_type, 0) + 1
        nice_layer_names.append(f"{layer_type.replace('2d','')}{counters[layer_type]}")

    return nice_layer_names

### --- Activation Extraction Helpers --- ###

# Extract activations from a specified layer module given image data
def get_layer_activations(model, layer_name, image_data, device_id=0):
    layer = dict(model.named_modules())[layer_name]
    activations = []
    def hook_fn(module, input, output):
          activations.append(output.detach().cpu())
    handle = layer.register_forward_hook(hook_fn)


    model.to(f"cuda:{device_id}")
    with torch.no_grad():
      for images in image_data:
        images = images.to(f"cuda:{device_id}")
        _ = model(images)      
    handle.remove()
  
    acts = torch.cat(activations, dim=0)
    
    # For convolutional layers, apply adaptive average pooling to reduce 
    # spatial dimensions and approx match target_dim when flattened
    target_dim = 4096 # fixing due to fc dimensions in VGG16

    if len(acts.shape) > 2:
      n_channels = acts.shape[1]
      pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
      apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
      acts = apool(acts)
      acts = acts.flatten(1)
    error_dim = abs(acts.shape[1] - target_dim)
    if error_dim > 1000:
        print(f"Warning: extracted acts dim {acts.shape[1]} differs from target dim {target_dim} by {error_dim} \n Layer: {layer_name}, original shape: {shape_info}")

    # bound acts and remove nans
    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)

    max_val = torch.max(torch.abs(acts))
    # Normalize to keep within [-max_val, max_val]
    if max_val > 1e6 and max_val != 0:
        scale = 1e6 / max_val
        acts = acts * scale
    
    return acts.numpy()

