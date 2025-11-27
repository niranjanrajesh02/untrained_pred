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





def expt_main():
  config = BaseConfig(parse_config('./config.yaml'))

  regimes_to_test = ["uniform", "normal", "xavier_u", "xavier_n", "kaiming_u", "kaiming_n", "orthogonal", "trained"]
  seeds_to_test = range(0,10)

  neural_data = get_tripleN_fr(config.triplen_path)
  images_dl = get_NSD_dataset_imgs(config.nsd_path)
  vgg = init_vgg(0, "uniform")
  layer_names = get_layer_names(vgg)
  del vgg

  results_path = "./weights_init/expt_results/"

  for regime in regimes_to_test:
    predictivity_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))
    neuron_corr_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))
    rsa_results = pd.DataFrame(index=range(len(seeds_to_test)), columns=range(len(layer_names)))


    for si,seed in enumerate(seeds_to_test):
      print(f'Running regime: {regime}, seed: {seed}')
      vgg = init_vgg(seed, regime)

      for li, layer_name in enumerate(tqdm(layer_names)):
        # extract activations
        acts = get_layer_activations(vgg, layer_name, images_dl, device_id=config.device_id)
    
        # compute predictivity
        pred_data = compute_linear_predictivity(acts, neural_data, k_folds=config.k_folds, normalize_reliability=config.normalize)
        
        predictivity_results.iloc[si, li] = pred_data['linear_predictivity']
        neuron_corr_results.iloc[si, li] = pred_data['neuron_corrs']

        rsa_corr = compute_RSA(acts, neural_data, dist_metric='correlation', corr_metric='pearson')
        rsa_results.iloc[si, li] = rsa_corr[0]
        

      # save each seed's results
      predictivity_results.to_csv(f'{results_path}/neural_predictivity/{regime}.csv')
      neuron_corr_results.to_pickle(f'{results_path}/neuron_corrs/{regime}.pkl')
      rsa_results.to_csv(f'{results_path}/rsa/{regime}.csv')
        
  return

if __name__ == "__main__":
    expt_main()