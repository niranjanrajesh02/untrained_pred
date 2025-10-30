import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr  
from scipy.stats import kendalltau


def normalize(X):
    mean = np.nanmean(X, 0) 
    stddev = np.nanstd(X, 0) + 1e-8
    X_zm = X - mean    
    X_zt = X_zm / stddev
    X_zt[np.isnan(X_zt)] = 0
    return X_zt


def compute_linear_predictivity(X, Y, k_folds=5):
  assert X.shape[0] == Y.shape[0], "Mismatch in number of samples"
  n_samples = X.shape[0]
  results = {}

  # nested cross validation for ridge regression
  # outer k-fold cross-validation for generalized test performance
  alpha_space = np.logspace(-8, 8, 17)
  kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
  across_folds_corrs = []
  across_folds_alphas = []
  
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
      Y_predc = Y_predc.astype(np.float64)
      Y_testc = Y_testc.astype(np.float64)


      corr_matrix = (Y_predc.T @ Y_testc) / (
          np.sqrt(np.sum(Y_predc**2, axis=0, keepdims=True)).T
          * np.sqrt(np.sum(Y_testc**2, axis=0, keepdims=True))
          + 1e-8
      )
      across_neuron_corrs = corr_matrix.diagonal()
      fold_linear_predictivity = np.nanmean(across_neuron_corrs)
      across_folds_corrs.append(fold_linear_predictivity)
      across_folds_alphas.append(best_alpha)
      
  linear_predictivity = np.nanmean(across_folds_corrs)
  results = {
      'linear_predictivity': linear_predictivity,
      'best_alpha': np.nanmean(across_folds_alphas)
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