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
    if len(acts.shape) > 2:
      # avg each filter
      acts = torch.mean(acts, dim=(-2, -1))

    return acts.numpy()