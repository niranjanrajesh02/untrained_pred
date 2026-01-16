import torch
import random
import numpy as np
from torch import nn
from torchvision import models



### --- Sparsification Helpers --- ###


def sparsify_conv_weights(module, sparsity_k):
    out_c, in_c, h, w = module.weight.shape
    total_num_weights = out_c * in_c * h * w
    W = module.weight 

    with torch.no_grad():
        # keep k% of total weights across all channels
        W_flat = W.view(-1)
        num_weights_to_keep = round(sparsity_k * W_flat.numel())
        indices = torch.randperm(W_flat.numel())[:num_weights_to_keep]
        mask = torch.zeros_like(W_flat)
        mask[indices] = 1
        W_sparse = (W_flat * mask).view_as(W)
        module.weight.copy_(W_sparse)
    
    return W_sparse
    
def sparsify_linear_weights(module, sparsity_k, MHA_flag=False):
    W = module.weight if not MHA_flag else module.in_proj_weight
    with torch.no_grad():
        W_flat = W.view(-1)
        num_weights_to_keep = round(sparsity_k * W_flat.numel())
        indices = torch.randperm(W_flat.numel())[:num_weights_to_keep]
        mask = torch.zeros_like(W_flat)
        mask[indices] = 1
        W_sparse = (W_flat * mask).view_as(W)
        module.weight.copy_(W_sparse) if not MHA_flag else module.in_proj_weight.copy_(W_sparse)
    return W_sparse

def verify_sparsity(W_s, sparsity_k, name):
            actual_nonzero = torch.sum(W_s != 0).item()
            actual_sparsity = actual_nonzero / W_s.numel() # fraction of weights kept
            assert abs(actual_sparsity - sparsity_k) < 0.05, f"Sparsity check failed for layer {name}: expected {sparsity_k}, got {actual_sparsity}"



def sparsify_weights(model, arch, sparsity_k=0.5, target_layer_range="all"):
    """
    Sparsify the weights of the given model by zeroing out a fraction of the weights.
        For conv layers, sparsification is achieved randomly on weights across all channels.
        For linear layers, sparsification is achieved randomly on the weight matrix.
    Args:
        model: The neural network model to sparsify.
        arch: Architecture name
        layer_names: List of layer names in the model to process
        sparsity_k: Fraction of weights to keep (between 0 and 1).
        target_layer_range: Which layers to sparsify ("all", "early", "late")
    Returns:
        The sparsified model.

    """

    assert 0.0 <= sparsity_k <= 1.0, "sparsity_k must be between 0 and 1."
    assert arch in ["vgg16", "resnet18", "vit_b_16", "convnext_b"], "arch must be one of 'vgg16', 'resnet18', 'vit_b_16', 'convnext_b'."
    assert target_layer_range in ["all", "early", "middle", "late"], "target_layer_range must be one of 'all', 'early', 'middle', 'late'."
    layer_names = []

    # sample 20% of layers for "early", "mid" or "late" layers
    arch_numlayers_map = {
        'vgg16': 3,
        'resnet18': 4,
        'vit_b_16': 10,
        'convnext_b': 22,
    }
    num_layers_to_sparsify = arch_numlayers_map[arch]

    # forward pass to identify layers with weights
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            layer_names.append(name)

    # print layer_names here for debugging

    layer_names = layer_names[:-1]  # exclude the final classification layer
    num_layers = len(layer_names)

    if sparsity_k == 1.0:
        print("Sparsity k=1.0, no sparsification applied.")
        return model
    
    # define which layers to sparsify
    if target_layer_range == "all":
        layers_to_sparsify = layer_names  # all layers
        print(f'Layers to sparsify: all (n={num_layers})')

    elif target_layer_range == "early":
        layers_to_sparsify = layer_names[:num_layers_to_sparsify]  # first 20% layers
        print(f'Layers to sparsify: {layers_to_sparsify} (n={len(layers_to_sparsify)})')

    elif target_layer_range == "middle":
        mid_index = num_layers // 2
        sample_half = num_layers_to_sparsify // 2
        layers_to_sparsify = layer_names[mid_index - sample_half: mid_index + sample_half]  # middle 20% layers
        print(f'Layers to sparsify: {layers_to_sparsify} (n={len(layers_to_sparsify)})')

    elif target_layer_range == "late":
        layers_to_sparsify = layer_names[num_layers-num_layers_to_sparsify:]  # last 20% layers
        print(f'Layers to sparsify: {layers_to_sparsify} (n={len(layers_to_sparsify)})')
    
    modules = dict(model.named_modules())
    
    for name in layers_to_sparsify:
        module = modules[name]
        W_s = None

        #**CONV**
        if isinstance(module, nn.Conv2d):
            W_s = sparsify_conv_weights(module, sparsity_k)

        #**LINEAR**
        elif isinstance(module, nn.Linear):
            W_s = sparsify_linear_weights(module, sparsity_k)
        ##**MULTIHEAD ATTENTION** (Attn_in layers in ViT, Attn_out layers handled as Linear)
        elif isinstance(module, nn.MultiheadAttention):
            W_s = sparsify_linear_weights(module, sparsity_k, MHA_flag=True)
            
        # Verify sparsity to be within tolerance (5%)
        if W_s is not None:
            verify_sparsity(W_s, sparsity_k, name)

    return model


def sparsify_vit_weights(model, sparsity_k=0.5, layers="all", layer_type="all"):

    assert 0.0 <= sparsity_k <= 1.0, "sparsity_k must be between 0 and 1."
    assert layer_type in ["all", "attn_in", "attn_out", "mlp"], "layer_type must be one of 'all', 'attn_in', 'attn_out', 'mlp'."

    layer_names = []
    mlp_flag = layer_type in ["all", "mlp"]
    attn_in_flag = layer_type in ["all", "attn_in"]
    attn_out_flag = layer_type in ["all", "attn_out"]

    # for now only "all" layers supported
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)):
            # mlp layers
            if mlp_flag and 'mlp.0' in name:
                layer_names.append(name)
            
            # attn_out layers
            if attn_out_flag and 'out_proj' in name:
                layer_names.append(name) 
        
        # attn_in layers
        if attn_in_flag and isinstance(module, nn.MultiheadAttention):
            layer_names.append(name) # needs special attention later to get in_proj
    
    num_layers = len(layer_names)
    layers_to_sparsify = layer_names  # all layers TODO: add early/late options
    
    modules = dict(model.named_modules())
    for name in layers_to_sparsify:
        module = modules[name]
        W = None
        if isinstance(module, (nn.Linear)):
            # mlp layers and attn_out layers
            W = module.weight
            
        # attn_in layers
        elif isinstance(module, nn.MultiheadAttention):
            W = module.in_proj_weight
            
        if W is not None:
            with torch.no_grad():
                W_flat = W.view(-1)
                num_weights_to_keep = round(sparsity_k * W_flat.numel())
                indices = torch.randperm(W_flat.numel())[:num_weights_to_keep]
                mask = torch.zeros_like(W_flat)
                mask[indices] = 1
                W_sparse = (W_flat * mask).view_as(W)
                
                if isinstance(module, (nn.Linear)):
                    module.weight.copy_(W_sparse)
                elif isinstance(module, nn.MultiheadAttention):
                    module.in_proj_weight.copy_(W_sparse)

            # Verify sparsity to be within tolerance (5%)
            verify_sparsity(W, sparsity_k, name)
    return model


def sparsify_vgg_weights(model, sparsity_k=0.5, layers="all", layer_type="all"):

    assert 0.0 <= sparsity_k <= 1.0, "sparsity_k must be between 0 and 1."
    assert layer_type in ["all", "conv", "linear"], "layer_type must be one of 'all', 'conv', 'linear'."

    layer_names = []
    conv_flag = layer_type in ["all", "conv"]
    linear_flag = layer_type in ["all", "linear"]

    # for now only "all" layers supported
    for name, module in model.named_modules():
        # conv layers
        if conv_flag and isinstance(module, nn.Conv2d):
            layer_names.append(name)
        
        # linear layers
        if linear_flag and isinstance(module, nn.Linear):
            layer_names.append(name)

    layer_names = layer_names[:-1]  # exclude the final classification layer
    num_layers = len(layer_names)

    layers_to_sparsify = layer_names  # all layers TODO: add early/late options
    
    modules = dict(model.named_modules())
    for name in layers_to_sparsify:
        module = modules[name]
        W_s = None

        #**CONV**
        if isinstance(module, nn.Conv2d):
            W_s = sparsify_conv_weights(module, sparsity_k)

        #**LINEAR**
        elif isinstance(module, nn.Linear):
            W_s = sparsify_linear_weights(module, sparsity_k)
        
        if W_s is not None:
            verify_sparsity(W_s, sparsity_k, name)
    
    return model



        


def sparsify_units(model, sparsity_k=0.5, target_layer_range="all"):
    assert 0.0 <= sparsity_k <= 1.0, "sparsity_k must be between 0 and 1."
    
    layer_names = []
    num_layers_to_sparsify = 3  # for "early", "mid" or "late" layers

    # forward pass to identify layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
    layer_names = layer_names[:-1]  # exclude the final classification layer
    num_layers = len(layer_names)

    if sparsity_k == 1.0:
        print("Sparsity k=1.0, no sparsification applied.")
        return model
    
    # define which layers to sparsify
    if target_layer_range == "all":
        layers_to_sparsify = layer_names  # all layers
        print(f'Layers to sparsify: all')
        
    elif target_layer_range == "early":
        layers_to_sparsify = layer_names[:num_layers_to_sparsify]  # first few layers
        print(f'Layers to sparsify: {layers_to_sparsify}')
    
    elif target_layer_range == "middle":
        mid_index = num_layers // 2
        sample_half = num_layers_to_sparsify // 2
        layers_to_sparsify = layer_names[mid_index - sample_half: mid_index + sample_half]  # middle few layers
        print(f'Layers to sparsify: {layers_to_sparsify}')

    elif target_layer_range == "late":
        layers_to_sparsify = layer_names[num_layers-num_layers_to_sparsify:]  # last few layers
        print(f'Layers to sparsify: {layers_to_sparsify}')
    modules = dict(model.named_modules())


    for name in layers_to_sparsify:
        module = modules[name]
        #**CONV** - only keep k% output channels
        if isinstance(module, nn.Conv2d):
            out_c, in_c, h, w = module.weight.shape
            num_channels_to_keep = round(sparsity_k * out_c)
            indices = torch.randperm(out_c)[:num_channels_to_keep]
            mask = torch.zeros(out_c, 1, 1, 1)
            mask[indices] = 1
            W = module.weight 
            with torch.no_grad():
                W_sparse = W * mask
                module.weight.copy_(W_sparse)

        #**LINEAR** - only keep k% output units
        elif isinstance(module, nn.Linear):
            out_u, in_u = module.weight.shape
            num_units_to_keep = round(sparsity_k * out_u)
            indices = torch.randperm(out_u)[:num_units_to_keep]
            mask = torch.zeros(out_u, 1)
            mask[indices] = 1
            W = module.weight 
            with torch.no_grad():
                W_sparse = W * mask
                module.weight.copy_(W_sparse)

        # Verify sparsity to be within tolerance (5%)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
          W = module.weight
          actual_nonzero = torch.sum(W != 0).item()
          actual_sparsity = actual_nonzero / W.numel() # fraction of weights kept
          
          assert abs(actual_sparsity - sparsity_k) < 0.02, f"Sparsity check failed for layer {name}: expected {sparsity_k}, got {actual_sparsity}"
    return model