import pandas as pd
from my_utils.nsd_dataset import get_NSD_dataset, get_tripleN_fr
from my_utils.model_helpers import get_layer_activations, InputWrapper
from my_utils.alignment_metrics import compute_linear_predictivity, compute_RSA
import torch
from torchvision import models
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
import pickle

def layerwise_brain_alignment(model, image_data, neural_data, all_layers=False, subsample_neurons=False):
  k_folds = 5 # folds for cross-validation
  results = {}
  if all_layers:
    layer_names = []
    model = InputWrapper(model)
    for name, layer in model.named_modules():
      # skip container modules (those that have submodules)
      if len(list(layer.children())) == 0:
          # all layers except Relu and dropout
          if not isinstance(layer, (torch.nn.ReLU, torch.nn.Dropout)):
              layer_names.append(name)

    # skip final classification layer
    layer_names = layer_names[:-1]

  else:
    layer_names = ['features.23', 'features.30', 'classifier.0', 'classifier.3']
  
  model = model.to(f"cuda:{device_id}")
  for name, layer in tqdm(model.named_modules(), desc="Evaluating layers", unit="layer"):
    # layers: final 2 conv blocks and 2 fc layers
    if name in layer_names:
      results[name] = {}
      
      # forward: predicting neural data from model activations
      X = get_layer_activations(model, layer, image_data, device_id) #[1000x4096]
      Y = neural_data #[1000x17000]
      
      if subsample_neurons:
          Y = Y[:, :X.shape[1]]  # subsample neurons to match dimensionality (top neurons by reliability)


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


def layerwise_predict_neurons(model, dl, unit_frs):
  model = model.to(f"cuda:{device_id}")
  results = {}

  # for each model layer, predict each neuron FRs
  for name, layer in model.named_modules():
    # layers: final 2 conv blocks and 2 fc layers
    if name in ['features.23', 'features.30', 'classifier.0', 'classifier.3']:
      X = get_layer_activations(model, layer, dl, device_id) #[1000x4096]
      Y = unit_frs #[1000x17000]
      res = compute_linear_predictivity(X, Y, k_folds=5)
      results[name] = res['neuron_corrs']  
      
  return results


# evaluate all seeds for brain alignment in 4 last layers (linear predictivity + RSA)
def model_brain_alignment(model_names, dl, unit_frs, all_layers=False, subsample_neurons=False):
  results_list = []
  global brain_eval_results_file
  if all_layers:
      brain_eval_results_file = os.path.join(results_path, 'weights_brain_eval_all_layers.csv')
  
  if subsample_neurons:
      brain_eval_results_file = os.path.join(results_path, 'weights_brain_eval_subsampled.csv')

  print(f"Results will be saved to {brain_eval_results_file}")

  if os.path.exists(brain_eval_results_file):
      results_df = pd.read_csv(brain_eval_results_file)
      completed_models = set(results_df['model_name'].unique())
      print(f"Loaded {len(completed_models)} completed models.")
  else:
      results_df = pd.DataFrame()
      completed_models = set()
  
  for model_name in model_names:
    # Skip already processed models
    if model_name in completed_models:
        print(f"Skipping {model_name} (already processed).")
        continue

    
    print(f"Processing {model_name}...")
    model = models.vgg16(weights=None)
    model.load_state_dict(torch.load(os.path.join(model_load_path, model_name + '.pth')))
    
    results = layerwise_brain_alignment(model, dl, unit_frs, all_layers=all_layers, subsample_neurons=subsample_neurons)
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
    results_df.to_csv(brain_eval_results_file, index=False)
    results_list = []  # reset accumulator
    
  
  print("Results saved.")




# evaluate all seeds for neuron prediction from units in 4 last layers
def model_neuron_prediction(model_names, dl, unit_frs):
  results_list = []

  if os.path.exists(brain_neuronpred_results_file):
      results_df = pd.read_pickle(brain_neuronpred_results_file)
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
      
      res = layerwise_predict_neurons(model, dl, unit_frs)

      for layer_name, neuron_corrs in res.items():
        results_list.append({
            'model_name': model_name,
            'layer_name': layer_name,
            'neuron_corrs': neuron_corrs})
      new_df = pd.DataFrame(results_list)
      results_df = pd.concat([results_df, new_df], ignore_index=True)
      results_df.to_pickle(brain_neuronpred_results_file)
      results_list = []  # reset accumulator
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue  # skip to next model
    
    
  print("Results saved.")






def main():
  dl = get_NSD_dataset("/home/nirajesh/untrained_pred/neural_data/tripleN/images") #tripleN images
  unit_frs = get_tripleN_fr("/home/nirajesh/untrained_pred/neural_data/tripleN/filtered_units_by_roi_all.csv") #tripleN unit FRs
  model_names = os.listdir(model_load_path)
  model_names = [mn for mn in model_names if mn.endswith('.pth')]
  model_names = [mn[:-4] for mn in model_names]
  model_names.sort()

  model_brain_alignment(model_names, dl, unit_frs, all_layers=False, subsample_neurons=False)
  # model_neuron_prediction(model_names, dl, unit_frs)
  # model_names = ['vgg16_trained_seed0', 'vgg16_kaiming_n_seed0']
  # model_brain_alignment(model_names, dl, unit_frs, all_layers=True)
  

  
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--device_id', type=int, default=2, help='GPU device ID to use')
  args = parser.parse_args()
  device_id = args.device_id
  model_load_path = '/home/nirajesh/untrained_pred/saved_models/untrained/weights'
  results_path = '/home/nirajesh/untrained_pred/weights_init/results'
  os.makedirs(results_path, exist_ok=True)
  brain_eval_results_file = os.path.join(results_path, 'weights_brain_eval.csv')
  brain_neuronpred_results_file = os.path.join(results_path, 'weights_neuron_pred_eval.pkl')

  main()

