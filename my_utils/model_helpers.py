import torch


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

    if len(acts.shape) > 2:
      # avg each filter
      acts = torch.nanmean(acts, dim=(-2, -1))
    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)


    max_val = torch.max(torch.abs(acts))
    
    if max_val > 1e6 and max_val != 0:
        # Normalize to keep within [-max_range, max_range]
        scale = 1e6 / max_val
        acts = acts * scale
    
    return acts.numpy()