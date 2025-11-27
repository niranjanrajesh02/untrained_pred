import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr  
from scipy.stats import kendalltau
from my_utils.neural_data import get_tripleN_reliability
from pyaml_env import parse_config, BaseConfig





def normalize(X):
    mean = np.nanmean(X, 0) 
    stddev = np.nanstd(X, 0) + 1e-8
    X_zm = X - mean    
    X_zt = X_zm / stddev
    X_zt[np.isnan(X_zt)] = 0
    return X_zt


def partial_correlation(x, y, z):
    """
    Compute the partial correlation between x and y, controlling for z.
    Parameters:
    x : array-like, shape (n_samples,)
        First variable.
    y : array-like, shape (n_samples,)
        Second variable.
    z : array-like, shape (n_samples,)
        Control variable.
    Returns:
    float
        Partial correlation coefficient between x and y controlling for z.
    """
    # Regress x on z and get residuals
    reg_x = LinearRegression().fit(z.reshape(-1, 1), x)
    x_residuals = x - reg_x.predict(z.reshape(-1, 1))
    
    # Regress y on z and get residuals
    reg_y = LinearRegression().fit(z.reshape(-1, 1), y)
    y_residuals = y - reg_y.predict(z.reshape(-1, 1))
    
    # Compute Pearson correlation between the residuals
    p_corr, _ = pearsonr(x_residuals, y_residuals)
    
    return p_corr

def compute_linear_predictivity(X, Y, k_folds=5, normalize_reliability=False):
  assert X.shape[0] == Y.shape[0], "Mismatch in number of samples"
  n_samples = X.shape[0]
  results = {}
  

  # nested cross validation for ridge regression
  # outer k-fold cross-validation for generalized test performance
  alpha_space = np.logspace(-8, 8, 17)
  kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
  across_folds_corrs = [] # avg corrs across neurons for each fold (n_folds,1)
  across_folds_alphas = []
  neuron_corrs_all = [] # neuron-wise corrs for each fold (n_folds, n_neurons)
  
  for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      Y_train, Y_test = Y[train_index], Y[test_index]

      predictor = RidgeCV(alphas=alpha_space)
      predictor.fit(normalize(X_train), Y_train)
      Y_pred = predictor.predict(normalize(X_test))
      best_alpha = predictor.alpha_
      across_neuron_corrs =  []
      Y_predc = Y_pred - np.nanmean(Y_pred, axis=0, keepdims=True)
      Y_testc = Y_test - np.nanmean(Y_test, axis=0, keepdims=True)
      Y_predc = Y_predc.astype(np.float64) # shape: (n_samples_test, n_neurons)
      Y_testc = Y_testc.astype(np.float64) 
      
      num = np.sum(Y_predc * Y_testc, axis=0)
      den = np.sqrt(np.sum(Y_predc**2, axis=0) * np.sum(Y_testc**2, axis=0)) + 1e-8
      across_neuron_corrs = num / den # shape: (n_neurons,)

      if normalize_reliability:
          # divide each neuron's corr by its reliability
          config = BaseConfig(parse_config('./config.yaml'))
          neuron_reliability = get_tripleN_reliability(config.triplen_path)
          across_neuron_corrs = across_neuron_corrs / neuron_reliability
      
      fold_linear_predictivity = np.nanmean(across_neuron_corrs)
      across_folds_corrs.append(fold_linear_predictivity)
      across_folds_alphas.append(best_alpha)
      neuron_corrs_all.append(across_neuron_corrs)

  linear_predictivity = np.nanmean(across_folds_corrs)
  neuron_corrs_all = np.array(neuron_corrs_all)  # shape: (k_folds, n_neurons)
  mean_neuron_corrs = np.nanmean(neuron_corrs_all, axis=0)  # shape: (n_neurons,)

  results = {
      'linear_predictivity': linear_predictivity,
      'best_alpha': np.nanmean(across_folds_alphas),
      'neuron_corrs': mean_neuron_corrs
  }

  return results


def compute_RSA(X,Y, dist_metric='correlation', corr_metric='pearson'):

    if dist_metric=='correlation':
      X_dist=1-np.corrcoef(X)
      Y_dist=1-np.corrcoef(Y)
    elif dist_metric=='euclidean':
      X_dist=squareform(pdist(X, metric='euclidean'))
      Y_dist=squareform(pdist(Y, metric='euclidean'))

    X_dist_flat = X_dist[np.triu_indices(X_dist.shape[0], k=1)]
    Y_dist_flat = Y_dist[np.triu_indices(Y_dist.shape[0], k=1)]
    
    valid_indices = ~np.isnan(X_dist_flat) & ~np.isnan(Y_dist_flat)
    X_dist_flat = X_dist_flat[valid_indices]
    Y_dist_flat = Y_dist_flat[valid_indices]

    if len(X_dist_flat) < 2:
        rsa_corr= [np.nan]
    else:
      if corr_metric=='pearson':
        rsa_corr = [pearsonr(X_dist_flat, Y_dist_flat)[0]]
      elif corr_metric=='spearman':
        rsa_corr = [spearmanr(X_dist_flat, Y_dist_flat)[0]]
      elif corr_metric=='kendall':
        rsa_corr = [kendalltau(X_dist_flat, Y_dist_flat)[0]]

    
    return rsa_corr