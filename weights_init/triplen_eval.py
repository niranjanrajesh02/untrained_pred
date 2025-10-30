import pandas as pd
from my_utils.nsd_dataset import get_NSD_dataset, get_tripleN_fr
from my_utils.model_helpers import get_layer_activations
from my_utils.alignment_metrics import compute_linear_predictivity, compute_RSA
import torch
from torchvision import models
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


device_id = 2


def layerwise_alignment(model, image_data, neural_data):
  k_folds = 5 # folds for cross-validation
  results = {}
  model = model.to(f"cuda:{device_id}")
  for name, layer in model.named_modules():
    # layers: final 2 conv blocks and 2 fc layers
    if name in ['features.23', 'features.30', 'classifier.0', 'classifier.3']:
      results[name] = {}
      
      # forward: predicting neural data from model activations
      X = get_layer_activations(model, layer, image_data, device_id) #[1000x4096]
      Y = neural_data #[1000x17000]
      res = compute_linear_predictivity(X, Y, k_folds=k_folds)
      results[name]['forward'] = res
      

      # backward: predicting model activations from neural data
      Y = X #[1000x4096]
      X = neural_data #[1000x17000]
      res = compute_linear_predictivity(X, Y, k_folds=k_folds)
      results[name]['backward']= res
      
      
      # RSA
      rsa_pearson_corr = float(compute_RSA(X, Y, dist_metric='correlation', corr_metric='pearson')[0])
      results[name]['RSA_pearson'] = rsa_pearson_corr
      
      
  return results

def main():
  dl = get_NSD_dataset("/home/nirajesh/untrained_pred/neural_data/tripleN/images") #tripleN images
  unit_frs = get_tripleN_fr("/home/nirajesh/untrained_pred/neural_data/tripleN/filtered_units_by_roi_all.csv") #tripleN unit FRs

  model_load_path = '/home/nirajesh/untrained_pred/saved_models/untrained/weights'
  model_names = os.listdir(model_load_path)
  model_names = [mn for mn in model_names if mn.endswith('.pth')]
  model_names = [mn[:-4] for mn in model_names]

  results_list = []
  results_path = '/home/nirajesh/untrained_pred/weights_init/results'
  os.makedirs(results_path, exist_ok=True)
  results_file = os.path.join(results_path, 'weights_lin_pred.csv')


  if os.path.exists(results_file):
      results_df = pd.read_csv(results_file)
      completed_models = set(results_df['model_name'].unique())
      print(f"Loaded {len(completed_models)} completed models.")
  else:
      results_df = pd.DataFrame()
      completed_models = set()
  
  

  for model_name in tqdm(model_names, desc="Evaluating models", unit="model"):
    # Skip already processed models
    if model_name in completed_models:
        print(f"Skipping {model_name} (already processed).")
        continue

    try:
      print(f"Processing {model_name}...")
      model = models.vgg16(weights=None)
      model.load_state_dict(torch.load(os.path.join(model_load_path, model_name + '.pth')))
      results = layerwise_alignment(model, dl, unit_frs)
      
      
      for layer_name, res in results.items():
        layer_name = layer_name.replace('_forward', '').replace('_backward', '')

        results_list.append({
            'model_name': model_name,
            'layer_name': layer_name,
            'forward_lin_pred': res['forward']['linear_predictivity'],
            'forward_best_alpha': res['forward']['best_alpha'],
            'backward_lin_pred': res['backward']['linear_predictivity'],
            'backward_best_alpha': res['backward']['best_alpha'],
            'RSA_pearson': res['RSA_pearson']})

      new_df = pd.DataFrame(results_list)
      results_df = pd.concat([results_df, new_df], ignore_index=True)
      results_df.to_csv(results_file, index=False)
      results_list = []  # reset accumulator
      
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue  # skip to next model
    
  print("Results saved.")



main()