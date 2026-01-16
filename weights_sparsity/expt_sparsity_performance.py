import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models
from pyaml_env import parse_config, BaseConfig
from my_utils.alignment_metrics import compute_linear_predictivity, compute_RSA
from my_utils.sparsify import sparsify_weights, sparsify_units
from my_utils.imagenet_probe import eval_model_linear_probe, get_imagenet_dataloaders, eval_model_val
from my_utils.model_helpers import init_model, get_layer_names, get_layer_activations, count_nonzero_weights, set_seed


def expt_main():
  config = BaseConfig(parse_config('./config.yaml'))

  # get layer names from untrained model
  arch_names = ["vgg16", "resnet18", "convnext_b", "vit_b_16"]
  sparse_k_levels = [0.1,0.25,0.5,0.75,1] # 0% to 100% sparsity
  seeds = [1,2,3,4,5]
  sparse_type = config.sparse_type 
  target_layer_options = ['all', 'early', 'middle', 'late']

  # arch = config.arch
  for arch in arch_names:
    # target_layers = ['all']
    model_weight_counter = pd.DataFrame(index=range(len(target_layer_options)), columns=sparse_k_levels) # rows: target region, #cols: frac of pruned weights

    for ti, target_layers in enumerate(target_layer_options):

      results_path = f"./weights_sparsity/expt_results/{sparse_type}/{arch}/{target_layers}_layers/performance/"
      if not os.path.exists(results_path):
        os.makedirs(results_path)

      _, val_dl = get_imagenet_dataloaders(config.imagenet_train, config.imagenet_val)
      print("Loaded ImageNet dataloaders.")
    
      imagenet_probe_performance = pd.DataFrame(index=range(len(sparse_k_levels)), 
                                            columns=['Seed1', 'Seed2', 'Seed3', 'Seed4', 'Seed5']) #rows: sparsity levels, cols: seeds

      for ki, k in enumerate(sparse_k_levels):
        for si, s in enumerate(seeds):
          set_seed(s)
          model = init_model(arch,  regime='trained')

          if sparse_type == 'random_weights':
            model_sparse = sparsify_weights(model, arch, sparsity_k=k, target_layer_range=target_layers)
          elif sparse_type == 'random_units':
            model_sparse = sparsify_units(model, sparsity_k=k, target_layer_range=target_layers)
          del model

          print(f'Evaluating Imagenet Accuracy for {arch} sparsity level: {k}, sparse type: {sparse_type}, layers: {target_layers}')

          val_acc = eval_model_val(model_sparse, val_dl, device_id=config.device_id)
          print(f'Full model validation accuracy at sparsity {k}: Top-1: {val_acc[0]}, Top-5: {val_acc[1]}')

          
          imagenet_probe_performance.iloc[ki, si] = val_acc[1]  # Top-5 accuracy
        
          # index as sparsity level
          imagenet_probe_performance.index.name = 'sparsity_k'
          imagenet_probe_performance.index = sparse_k_levels

          # save intermediate results
          imagenet_probe_performance.to_csv(f'{results_path}/seeds_model_sparsity_performance.csv')


        # save weight count tracker for each level
        num_nonzero, num_total = count_nonzero_weights(model_sparse)
        model_weight_counter.iloc[ti, ki] = num_nonzero / num_total  # fraction of nonzero weights
        
      model_weight_counter.index.name = 'target_range'
      model_weight_counter.index = target_layer_options
      model_weight_counter.to_csv(f'./weights_sparsity/expt_results/{sparse_type}/{arch}/model_weight_counter.csv')



  
if __name__ == "__main__":
  expt_main()