import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models
from pyaml_env import parse_config, BaseConfig
from my_utils.alignment_metrics import compute_linear_predictivity, compute_RSA
from my_utils.model_helpers import init_model, get_layer_names, get_layer_activations
from my_utils.sparsify import sparsify_weights, sparsify_units, sparsify_vit_weights
from my_utils.imagenet_probe import eval_model_linear_probe, get_imagenet_dataloaders, eval_model_val


def expt_main():
  config = BaseConfig(parse_config('./config.yaml'))
  model_name = 'vit_b_16'
  # get layer names from untrained model
  model = init_model(model_name, 0, "uniform")  
  layer_names = get_layer_names(model)
  del model

  sparse_k_levels = [0.1,0.25,0.5,0.75,1] # 0% to 100% sparsity
  sparse_type = 'random_weights'

  layer_types = ['attn_in', 'attn_out', 'mlp', 'all']
  for layer_type in layer_types:
   
    results_path = f"./weights_sparsity/expt_results/{sparse_type}/{model_name}/all_layers/performance/"
    if not os.path.exists(results_path):
      os.makedirs(results_path)

    _, val_dl = get_imagenet_dataloaders(config.imagenet_train, config.imagenet_val)
    print("Loaded ImageNet dataloaders.")

    imagenet_probe_performance = pd.DataFrame(index=range(len(sparse_k_levels)), columns=['Top-1', 'Top-5']) #rows: sparsity levels, cols: top1 and top5 val accuracy
    for ki, k in enumerate(sparse_k_levels):
      model = init_model(model_name, 0, 'trained')
      model_sparse = sparsify_vit_weights(model, sparsity_k=k, layer_type=layer_type)
      print(f'Evaluating Imagenet Accuracy for {model_name} sparsity level: {k}, sparse type: {sparse_type}, layers: {layer_type}')

      val_acc = eval_model_val(model_sparse, val_dl, device_id=config.device_id)
      print(f'Full model validation accuracy at sparsity {k}: Top-1: {val_acc[0]}, Top-5: {val_acc[1]}')

      imagenet_probe_performance.iloc[ki, 0] = val_acc[0]  # Top-1 accuracy
      imagenet_probe_performance.iloc[ki, 1] = val_acc[1]  # Top-5 accuracy
      # index as sparsity level
      imagenet_probe_performance.index.name = 'sparsity_k'
      imagenet_probe_performance.index = sparse_k_levels

      # save intermediate results
      imagenet_probe_performance.to_csv(f'{results_path}/{layer_type}_layers_sparsity_performance.csv')
      

  
if __name__ == "__main__":
  expt_main()