import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models
from pyaml_env import parse_config, BaseConfig
from my_utils.alignment_metrics import compute_linear_predictivity, compute_RSA
from my_utils.neural_data import get_tripleN_fr, get_NSD_dataset_imgs
from my_utils.model_helpers import init_vgg, get_layer_names, get_layer_activations
from my_utils.sparsify import sparsify_weights


def expt_main():
  config = BaseConfig(parse_config('./config.yaml'))
  neural_data = get_tripleN_fr(config.triplen_path)
  images_dl = get_NSD_dataset_imgs(config.nsd_path)

  vgg = init_vgg(0, "uniform")
  layer_names = get_layer_names(vgg)
  del vgg

  vgg_inits = ["trained", "kaiming_n"]
  sparse_k_levels = [0.1, 0.25, 0.5, 0.75, 1.0] # 0% to 100% sparsity


  assert config.sparsify_layers in ['early', 'middle', 'late', 'all'], "sparsify_layers must be one of ['early', 'middle', 'late', 'all']"

  
  results_path = f"./weights_sparsity/expt_results/random_weights/vgg16/{config.sparsify_layers}_layers/"


  seeds_to_test = [0]  # For simplicity, using a single seed here
  for init in vgg_inits:
    for si, seed in enumerate(seeds_to_test):
      
      for k in sparse_k_levels:
        # check if results already exist
        if os.path.exists(f'{results_path}/predictivity/{init}_sparsity_{k:.2f}.csv'):
          print(f'Results already exist for init: {init}, sparsity level: {k}, skipping...')
          continue

        predictivity_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))
        neuron_corr_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))
        rsa_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))
        
        vgg = init_vgg(seed, init)
        vgg_sparse = sparsify_weights(vgg, sparsity_k=k, layers=config.sparsify_layers, conv_global=config.sparsify_conv_global)
        
        print(f'Running init: {init}, sparsity level: {k}, seed: {seed}, layers: {config.sparsify_layers}, conv_global: {config.sparsify_conv_global}')
        
        
        for li, layer_name in enumerate(tqdm(layer_names)):
          # extract activations
          acts = get_layer_activations(vgg_sparse, layer_name, images_dl, device_id=config.device_id)
      
          # compute predictivity
          pred_data = compute_linear_predictivity(acts, neural_data, k_folds=config.k_folds, normalize_reliability=config.normalize)
          
          predictivity_results.iloc[si, li] = pred_data['linear_predictivity']
          neuron_corr_results.iloc[si, li] = pred_data['neuron_corrs']

          rsa_corr = compute_RSA(acts, neural_data, dist_metric='correlation', corr_metric='pearson')
          rsa_results.iloc[si, li] = rsa_corr[0]
         

        # save results for each sparsity level 
        predictivity_results.to_csv(f'{results_path}/predictivity/{init}_sparsity_{k:.2f}.csv')
        neuron_corr_results.to_pickle(f'{results_path}/neuron_corrs/{init}_sparsity_{k:.2f}.pkl')
        rsa_results.to_csv(f'{results_path}/rsa/{init}_sparsity_{k:.2f}.csv')
        
    
if __name__ == "__main__":
    expt_main()