import numpy as np
import pandas as pd
import random
import copy
import warnings
from tqdm import tqdm
from scipy.special import expit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
import matplotlib.pyplot as plt


# R imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

# Activate automatic conversion
pandas2ri.activate()
numpy2ri.activate()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

nprobust = importr('nprobust')

# Local imports
from data import generate_synthetic_data, generate_synthetic_data_v2
from models import get_ps_model, get_or_model, get_w_model

from joblib import Parallel, delayed
from scipy.stats import norm


def calc_mse_bias(estimated_ates, true_ate):
    mse = np.mean((estimated_ates - true_ate) ** 2)
    bias = np.mean(estimated_ates - true_ate)
    return mse, bias


def ate_mr_estimator(hat_w1, hat_w0, df):
    return np.mean(hat_w1 * df['y']) - np.mean(hat_w0 * df['y'])

def ate_amr_estimator(hat_tilde_w, hat_propen, hat_mu1, hat_mu0, df):
    tilde_y = df['y'] - (1 - hat_propen)*hat_mu1 - hat_propen*hat_mu0
    return np.mean(hat_tilde_w * tilde_y)

def ate_aipw_estimator(hat_mu1, hat_mu0, hat_propen, df):
    """
    Doubly robust: 
    ATE = E[ A*(Y - mu1)/p(X) - (1-A)*(Y - mu0)/(1-p(X)) + mu1 - mu0 ]
    """
    return np.mean(
        df['A']*(df['y'] - hat_mu1)/hat_propen 
        - (1 - df['A'])*(df['y'] - hat_mu0)/(1 - hat_propen) 
        + (hat_mu1 - hat_mu0)
    )

def ate_ipw_estimator(hat_propen, df):
    """
    IPW estimator: E[ A * Y / p(X) - (1-A) * Y / (1 - p(X)) ]
    """
    return np.mean(
        df['A']*df['y']/hat_propen 
        - (1 - df['A'])*df['y']/(1 - hat_propen)
    )

# -------------------------------------------------------------------
# ------------------------- Std Estimations -------------------------
# -------------------------------------------------------------------
def std_ate_mr_estimator(hat_w1, hat_w0, hat_propen, df):
    hat_theta = np.mean((hat_w1 - hat_w0) * df['y'])
    inf = df['A']*df['y']/hat_propen - (1 - df['A'])*df['y']/(1 - hat_propen)

    var_ = 1/(df.shape[0]*(df.shape[0]-1)) * np.sum((inf-hat_theta)**2)
    return np.sqrt(var_)

def std_ate_amr_estimator(hat_tilde_w, hat_propen, hat_mu1, hat_mu0, df):
    # tilde_y = df['y'] - (1 - hat_propen)*hat_mu1 - hat_propen*hat_mu0
    # hat_theta = np.mean(hat_tilde_w*tilde_y)
    # hat_h = df['A']/hat_propen - (1 - df['A'])/(1 - hat_propen)
    # var_tilde_w = np.mean((hat_h - hat_tilde_w)**2)

    # inf_smoothed = hat_tilde_w*tilde_y - hat_theta
    # b = np.sqrt(var_tilde_w)*abs(tilde_y)
    # var_ = 1/(df.shape[0]**2) * np.sum(inf_smoothed**2 + b**2)
    hat_theta =ate_amr_estimator(hat_tilde_w, hat_propen, hat_mu1, hat_mu0, df)
    var_ = 1/(df.shape[0]*(df.shape[0]-1)) * np.sum((df['A']*(df['y'] - hat_mu1)/hat_propen - (1 - df['A'])*(df['y'] - hat_mu0)/(1 - hat_propen) + (hat_mu1 - hat_mu0) - hat_theta)**2)
    return np.sqrt(var_)

def std_ate_aipw_estimator(hat_mu1, hat_mu0, hat_propen, df):
    val = df['A']*(df['y'] - hat_mu1)/hat_propen - (1 - df['A'])*(df['y'] - hat_mu0)/(1 - hat_propen) + (hat_mu1 - hat_mu0)
    return np.sqrt(np.var(val, ddof=1)/df.shape[0])

def std_ate_ipw_estimator(hat_propen, df):
    val = df['A']*df['y']/hat_propen - (1 - df['A'])*df['y']/(1 - hat_propen)
    return np.sqrt(np.var(val, ddof=1)/df.shape[0])

# -------------------------------------------------------------------
# -------------- True w/Tilde for Debugging ------------------------
# -------------------------------------------------------------------

def true_w(y, df_mc):
    """
    Calculate the true W for debugging purposes.
    
    Parameters:
    - y (float): Outcome value.
    - df_mc (pd.DataFrame): DataFrame containing 'propen', 'mu0', 'mu1'.
    
    Returns:
    - float: True W value.
    """
    p_y_A1_x = df_mc.apply(lambda row: norm.pdf(y, loc=row['mu1'], scale=1), axis=1)
    p_y_A0_x = df_mc.apply(lambda row: norm.pdf(y, loc=row['mu0'], scale=1), axis=1)
    numerator = np.mean(p_y_A1_x - p_y_A0_x)
    denominator = np.mean(df_mc['propen'] * p_y_A1_x + (1 - df_mc['propen']) * p_y_A0_x)
    return 0 if denominator == 0 else (numerator / denominator)

def calculate_true_w(df, df_mc):
    """
    Vectorized calculation of true W for the entire DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing outcomes 'y'.
    - df_mc (pd.DataFrame): DataFrame containing 'propen', 'mu0', 'mu1'.
    
    Returns:
    - np.array: Array of true W values.
    """
    return np.array([true_w(y, df_mc) for y in df['y']])




# Nuisance parameters estimation
def fit_predict_w(w_mdl, y_train, tilde_y_train, y_valid, tilde_y_valid, rho1, rho0, rho, target='rho1'):
    """
    Fit the W model and predict on validation data.

    Parameters:
    - w_mdl: Model instance for W-fitting.
    - y_train (np.array): Outcome variable for training (used for 'rho1' and 'rho0').
    - tilde_y_train (np.array): Tilted outcome for training (used for 'tilde_rho').
    - target (str): Target variable ('rho1', 'rho0', 'tilde_rho').
    - feature (str): Feature to use for prediction ('y', 'tilde_y').

    Returns:
    - np.array: Predicted values for W on validation data.
    """
    w_mdl_copy = copy.deepcopy(w_mdl)
    if target == 'rho1':
        pred_target = rho1
        feature_train = y_train.reshape(-1,1)
        feature_pred = y_valid.reshape(-1,1)
    elif target == 'rho0':
        pred_target = rho0
        feature_train = y_train.reshape(-1,1)
        feature_pred = y_valid.reshape(-1,1)
    elif target == 'tilde_rho':
        pred_target = rho
        feature_train = tilde_y_train.reshape(-1,1)
        feature_pred = tilde_y_valid.reshape(-1,1)
    
    # Fit the model and predict
    if isinstance(w_mdl_copy, RandomizedSearchCV):
        w_mdl_copy.fit(feature_train, pred_target)
        preds = w_mdl_copy.predict(feature_pred)
    elif hasattr(w_mdl, 'predict'):
        w_mdl_copy.fit(feature_train, pred_target)
        preds = w_mdl_copy.predict(feature_pred)
    else:
        # For models like nprobust which might return tuple
        preds, _ = w_mdl_copy(feature_train.flatten(), pred_target, feature_train.flatten())
    
    return preds

def single_split_nuisance_parameter_estimator(
    df1, df2, p, ps_model_type="lr", or_model_type="lr", w_model_type="kernel_ridge_regression",
    hyperparam_search=True, random_seed=42
):
    """
    Perform single-split nuisance parameter estimation.

    Parameters:
    - df1 (pd.DataFrame): Training split containing covariates, treatment 'A', and outcome 'y'.
    - df2 (pd.DataFrame): Validation split to predict nuisance parameters.
    - p (int): Number of covariates (excluding instrumental variables).
    - ps_model_type (str): Model type for propensity score ('lr', 'lasso', 'none').
    - or_model_type (str): Model type for outcome regression ('lr', 'lasso', 'none').
    - w_model_type (str): Model type for W-fitting ('kernel_ridge_regression', 'rf', 'mlp').
    - hyperparam_search (bool): Whether to perform hyperparameter tuning.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - dict: Dictionary containing estimated nuisance parameters for df2:
        'hat_propen', 'hat_mu0', 'hat_mu1', 'hat_w1', 'hat_w0', 'hat_tilde_w'
    """
    # Define feature columns
    X_cols = [f"X{i}" for i in range(1, p + 1)]
    
    # 1. Fit Propensity Score Model on df1
    ps_model = get_ps_model(ps_model_type, hyperparam_search, random_seed)
    if ps_model is not None:
        ps_model.fit(df1[X_cols].values, df1['A'].values)
        hat_propen_df2 = ps_model.predict_proba(df2[X_cols].values)[:, 1]
    else:
        # Use true propensity scores if available
        hat_propen_df2 = df2['propen'].values
    
    # 2. Fit Outcome Regression Models for A=0 and A=1 on df1
    or_model_0 = get_or_model(or_model_type, hyperparam_search, random_seed)
    or_model_1 = get_or_model(or_model_type, hyperparam_search, random_seed)
    
    if or_model_0 is not None:
        mask0 = (df1['A'] == 0)
        or_model_0.fit(df1.loc[mask0, X_cols].values, df1.loc[mask0, 'y'].values)
        hat_mu0_df2 = or_model_0.predict(df2[X_cols].values)
    else:
        hat_mu0_df2 = df2['mu0'].values
    
    if or_model_1 is not None:
        mask1 = (df1['A'] == 1)
        or_model_1.fit(df1.loc[mask1, X_cols].values, df1.loc[mask1, 'y'].values)
        hat_mu1_df2 = or_model_1.predict(df2[X_cols].values)
    else:
        hat_mu1_df2 = df2['mu1'].values
    
    # 3. Prepare Data for W-fitting
    if ps_model is not None:
        p_train = ps_model.predict_proba(df1[X_cols].values)[:, 1]
    else:
        p_train = df1['propen'].values
    
    if or_model_0 is not None:
        mu0_train = or_model_0.predict(df1[X_cols].values)
    else:
        mu0_train = df1['mu0'].values
    
    if or_model_1 is not None:
        mu1_train = or_model_1.predict(df1[X_cols].values)
    else:
        mu1_train = df1['mu1'].values
    
    # Calculate nuisance parameters in training data
    rho1 = df1['A'] / p_train
    rho0 = (1 - df1['A']) / (1 - p_train)
    tilde_rho = rho1 - rho0
    mu_combined = (1 - p_train) * mu1_train + p_train * mu0_train
    tilde_y = df1['y'] - mu_combined

    tilde_y_df2 =  df2['y'] - ((1 - hat_propen_df2) * hat_mu1_df2 + hat_propen_df2 * hat_mu0_df2)
    
    # 4. Fit W Models on df1 and predict on df2
    w_model = get_w_model(w_model_type, hyperparam_search, random_seed)
    
    # Fit and predict for rho1, rho0, rho
    # hat_w1_df2 = fit_predict_w(w_model, df1['y'].values, tilde_y.values, df1[X_cols].values, df2[X_cols].values, target='rho1', feature='y')
    hat_w1_df2 = fit_predict_w(w_mdl = w_model, y_train=df1['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df2['y'].values, tilde_y_valid = tilde_y_df2.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='rho1')

    hat_w0_df2 = fit_predict_w(w_mdl = w_model, y_train=df1['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df2['y'].values, tilde_y_valid = tilde_y_df2.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='rho0')

    hat_tilde_w_df2 = fit_predict_w(w_mdl = w_model, y_train=df1['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df2['y'].values, tilde_y_valid = tilde_y_df2.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='tilde_rho')

    return {
        'hat_propen': hat_propen_df2,
        'hat_mu0': hat_mu0_df2,
        'hat_mu1': hat_mu1_df2,
        'hat_w1': hat_w1_df2,
        'hat_w0': hat_w0_df2,
        'hat_tilde_w': hat_tilde_w_df2
    }

def kfold_cross_fit_nuisance_parameter_estimator(
    df_full, p, ps_model_type="lr", or_model_type="lr", w_model_type="kernel_ridge_regression",
    hyperparam_search=True, random_seed=42, n_folds=5
):
    """
    Perform K-Fold cross-fitting to estimate nuisance parameters.

    Parameters:
    - df_full (pd.DataFrame): The entire dataset containing covariates, treatment 'A', and outcome 'y'.
    - p (int): Number of covariates (excluding instrumental variables).
    - ps_model_type (str): Model type for propensity score ('lr', 'lasso', 'none').
    - or_model_type (str): Model type for outcome regression ('lr', 'lasso', 'none').
    - w_model_type (str): Model type for W-fitting ('kernel_ridge_regression', 'rf', 'mlp').
    - hyperparam_search (bool): Whether to perform hyperparameter tuning.
    - random_seed (int): Random seed for reproducibility.
    - n_folds (int): Number of K-Folds.

    Returns:
    - pd.DataFrame: The original DataFrame augmented with estimated nuisance parameters:
        'hat_propen', 'hat_mu0', 'hat_mu1', 'hat_w1', 'hat_w0', 'hat_tilde_w'
    """
    # Define feature columns
    X_cols = [f"X{i}" for i in range(1, p + 1)]
    
    # Initialize arrays to store predictions
    hat_propen = np.empty(len(df_full))
    hat_mu0 = np.empty(len(df_full))
    hat_mu1 = np.empty(len(df_full))
    hat_w1 = np.empty(len(df_full))
    hat_w0 = np.empty(len(df_full))
    hat_tilde_w = np.empty(len(df_full))
    
    # Initialize K-Fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Shuffle the data
    # df_shuffled = df_full.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_shuffled= copy.deepcopy(df_full)
    indices = df_shuffled.index.to_numpy()
    
    for train_idx, valid_idx in kf.split(df_shuffled):
        df_train = df_shuffled.iloc[train_idx]
        df_valid = df_shuffled.iloc[valid_idx]
        
        X_train = df_train[X_cols].values
        A_train = df_train['A'].values
        Y_train = df_train['y'].values
        X_valid = df_valid[X_cols].values
        
        # 1. Fit Propensity Score Model
        ps_model = get_ps_model(ps_model_type, hyperparam_search, random_seed)
        if ps_model is not None:
            ps_model.fit(X_train, A_train)
            p_valid = ps_model.predict_proba(X_valid)[:, 1]
        else:
            # Use true propensity scores if available
            p_valid = df_valid['propen'].values
        
        hat_propen[valid_idx] = p_valid
        
        # 2. Fit Outcome Regression Models for A=0 and A=1
        or_model_0 = get_or_model(or_model_type, hyperparam_search, random_seed)
        or_model_1 = get_or_model(or_model_type, hyperparam_search, random_seed)
        
        if or_model_0 is not None:
            mask0 = (A_train == 0)
            or_model_0.fit(X_train[mask0], Y_train[mask0])
            mu0_valid = or_model_0.predict(X_valid)
        else:
            mu0_valid = df_valid['mu0'].values
        
        if or_model_1 is not None:
            mask1 = (A_train == 1)
            or_model_1.fit(X_train[mask1], Y_train[mask1])
            mu1_valid = or_model_1.predict(X_valid)
        else:
            mu1_valid = df_valid['mu1'].values
        
        hat_mu0[valid_idx] = mu0_valid
        hat_mu1[valid_idx] = mu1_valid
        
        # 3. Prepare Data for W-fitting
        if ps_model is not None:
            p_train = ps_model.predict_proba(X_train)[:, 1]
        else:
            p_train = df_train['propen'].values
        
        if or_model_0 is not None:
            mu0_train = or_model_0.predict(X_train)
        else:
            mu0_train = df_train['mu0'].values
        
        if or_model_1 is not None:
            mu1_train = or_model_1.predict(X_train)
        else:
            mu1_train = df_train['mu1'].values
        
        # Calculate nuisance parameters in training data
        rho1 = df_train['A'] / p_train
        rho0 = (1 - df_train['A']) / (1 - p_train)
        tilde_rho = rho1 - rho0
        mu_combined = (1 - p_train) * mu1_train + p_train * mu0_train
        tilde_y = df_train['y'] - mu_combined
        
        tilde_y_valid =  df_valid['y'] - ((1 - p_valid) * mu1_valid + p_valid * mu0_valid)
    
        # 4. Fit W Models
        w_model = get_w_model(w_model_type, hyperparam_search, random_seed)
        
        # Fit and predict for rho1
        hat_w1_valid = fit_predict_w(w_mdl = w_model, y_train=df_train['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df_valid['y'].values, tilde_y_valid = tilde_y_valid.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='rho1')
        
        # Fit and predict for rho0
        hat_w0_valid = fit_predict_w(w_mdl = w_model, y_train=df_train['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df_valid['y'].values, tilde_y_valid = tilde_y_valid.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='rho0')

        # Fit and predict for tilde_rho
        hat_tilde_w_valid = fit_predict_w(w_mdl = w_model, y_train=df_train['y'].values, tilde_y_train=tilde_y.values,
                                y_valid=df_valid['y'].values, tilde_y_valid = tilde_y_valid.values, rho1=rho1, rho0=rho0, rho=tilde_rho, target='tilde_rho')

        hat_w1[valid_idx] = hat_w1_valid
        hat_w0[valid_idx] = hat_w0_valid
        hat_tilde_w[valid_idx] = hat_tilde_w_valid
    
    # Assign predictions back to the original DataFrame
    df_full['hat_propen'] = hat_propen
    df_full['hat_mu0'] = hat_mu0
    df_full['hat_mu1'] = hat_mu1
    df_full['hat_w1'] = hat_w1
    df_full['hat_w0'] = hat_w0
    df_full['hat_tilde_w'] = hat_tilde_w
    
    #return df_full
    return {
        'hat_propen': df_full['hat_propen'].values,
        'hat_mu0': df_full['hat_mu0'].values,
        'hat_mu1': df_full['hat_mu1'].values,
        'hat_w1':df_full['hat_w1'].values,
        'hat_w0': df_full['hat_w0'].values,
        'hat_tilde_w': df_full['hat_tilde_w'].values
    }

def nuisance_parameter_estimator(
    df, p, ps_model_type="lr", or_model_type="lr", w_model_type="kernel_ridge_regression",
    hyperparam_search=True, random_seed=42, cross_fit=True, n_folds=5, df1=None, df2=None
):
    """
    Master function that estimates nuisance parameters either via single-split or K-fold cross-fitting.
    
    Parameters:
    - df (pd.DataFrame): The entire dataset.
    - p (int): Number of covariates.
    - ps_model_type (str): Model type for propensity score ('lr', 'lasso', 'none').
    - or_model_type (str): Model type for outcome regression ('lr', 'lasso', 'none').
    - w_model_type (str): Model type for W-fitting.
    - hyperparam_search (bool): Whether to perform hyperparameter tuning.
    - random_seed (int): Random seed for reproducibility.
    - cross_fit (bool): Whether to perform K-fold cross-fitting.
    - n_folds (int): Number of K-Folds if cross_fit=True.
    
    Returns:
    - dict: Dictionary containing estimated nuisance parameters.
    """
    if cross_fit:
        # Perform K-Fold cross-fitting on the entire dataset
        nuis_params = kfold_cross_fit_nuisance_parameter_estimator(
            df_full=df,
            p=p,
            ps_model_type=ps_model_type,
            or_model_type=or_model_type,
            w_model_type=w_model_type,
            hyperparam_search=hyperparam_search,
            random_seed=random_seed,
            n_folds=n_folds
        )

    else:
        # Single-split approach
        # Estimate nuisance parameters on df1 and predict on df2
        nuis_params = single_split_nuisance_parameter_estimator(
            df1=df1,
            df2=df2,
            p=p,
            ps_model_type=ps_model_type,
            or_model_type=or_model_type,
            w_model_type=w_model_type,
            hyperparam_search=hyperparam_search,
            random_seed=random_seed
        )
        
    return nuis_params
