import numpy as np
import pandas as pd
from scipy.special import expit, logit
import scipy.stats as stats
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import io
import contextlib

import torch
import os
import pickle
from sklearn.model_selection import train_test_split
import urllib
import zipfile

# BERT embedding generation for text data
import math
import random
import os.path as osp
from collections import defaultdict

import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

def sim_data_causl_non_linear(n=10000, 
    nI=3, 
    nX=1, 
    nO=1, 
    nS=1, 
    ate=2, 
    beta_cov=0, 
    strength_instr=3, 
    strength_conf=1, 
    strength_outcome=1
):
    """
    Generate synthetic data using the R script's 'data.causl' function.

    Parameters
    ----------
    n : int
        Number of samples.
    nI : int
        Number of instruments.
    nX : int
        Number of other covariates (X).
    nO : int
        Number of outcome variables?
    nS : int
        Other parameter for the underlying DGP (as per R script).
    ate : float
        Average treatment effect parameter.
    beta_cov : float
        Coefficient for covariates.
    strength_instr : float
        Strength of instrument(s).
    strength_conf : float
        Strength of confounding.
    strength_outcome : float
        Strength of outcome model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [X1, X2, ..., A, y, propen, mu0, mu1].
    """

    @contextlib.contextmanager
    def suppress_r_output():
        r_output = io.StringIO()
        with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
            yield

    pandas2ri.activate()

    # Source the R script to load 'data.causl' function
    with suppress_r_output():
        robjects.r['source'](r'../R/data.r')
        generate_data = robjects.globalenv['data.causl.non_linear']

        # Generate data in R
        r_dataframe = generate_data(
            n=n, 
            nI=nI, 
            nX=nX, 
            nO=nO, 
            nS=nS, 
            ate=ate, 
            beta_cov=beta_cov, 
            strength_instr=strength_instr, 
            strength_conf=strength_conf, 
            strength_outcome=strength_outcome
        )
        

    # Convert R dataframe to Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)

    return df

def sim_data_causl_non_linear_propen(n=10000, 
    nI=3, 
    nX=1, 
    nO=1, 
    nS=1, 
    ate=2, 
    beta_cov=0, 
    strength_instr=3, 
    strength_conf=1, 
    strength_outcome=1
):
    """
    Generate synthetic data using the R script's 'data.causl' function.

    Parameters
    ----------
    n : int
        Number of samples.
    nI : int
        Number of instruments.
    nX : int
        Number of other covariates (X).
    nO : int
        Number of outcome variables?
    nS : int
        Other parameter for the underlying DGP (as per R script).
    ate : float
        Average treatment effect parameter.
    beta_cov : float
        Coefficient for covariates.
    strength_instr : float
        Strength of instrument(s).
    strength_conf : float
        Strength of confounding.
    strength_outcome : float
        Strength of outcome model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [X1, X2, ..., A, y, propen, mu0, mu1].
    """

    @contextlib.contextmanager
    def suppress_r_output():
        r_output = io.StringIO()
        with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
            yield

    pandas2ri.activate()

    # Source the R script to load 'data.causl' function
    with suppress_r_output():
        robjects.r['source'](r'../R/data.r')
        generate_data = robjects.globalenv['data.causl.non_linear_propen']

        # Generate data in R
        r_dataframe = generate_data(
            n=n, 
            nI=nI, 
            nX=nX, 
            nO=nO, 
            nS=nS, 
            ate=ate, 
            beta_cov=beta_cov, 
            strength_instr=strength_instr, 
            strength_conf=strength_conf, 
            strength_outcome=strength_outcome
        )
        

    # Convert R dataframe to Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)

    return df


def generate_data_causl(
    n=10000, 
    nI=3, 
    nX=1, 
    nO=1, 
    nS=1, 
    ate=2, 
    beta_cov=0, 
    strength_instr=3, 
    strength_conf=1, 
    strength_outcome=1
):
    """
    Generate synthetic data using the R script's 'data.causl' function.

    Parameters
    ----------
    n : int
        Number of samples.
    nI : int
        Number of instruments.
    nX : int
        Number of other covariates (X).
    nO : int
        Number of outcome variables?
    nS : int
        Other parameter for the underlying DGP (as per R script).
    ate : float
        Average treatment effect parameter.
    beta_cov : float
        Coefficient for covariates.
    strength_instr : float
        Strength of instrument(s).
    strength_conf : float
        Strength of confounding.
    strength_outcome : float
        Strength of outcome model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [X1, X2, ..., A, y, propen, mu0, mu1].
    """

    @contextlib.contextmanager
    def suppress_r_output():
        r_output = io.StringIO()
        with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
            yield

    pandas2ri.activate()

    # Source the R script to load 'data.causl' function
    with suppress_r_output():
        robjects.r['source'](r'../R/data.r')
        generate_data = robjects.globalenv['data.causl']

        # Generate data in R
        r_dataframe = generate_data(
            n=n, 
            nI=nI, 
            nX=nX, 
            nO=nO, 
            nS=nS, 
            ate=ate, 
            beta_cov=beta_cov, 
            strength_instr=strength_instr, 
            strength_conf=strength_conf, 
            strength_outcome=strength_outcome
        )

    # Convert R dataframe to Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)

    return df

def generate_synthetic_data(
    n, 
    p=2, 
    p_instr=1, 
    gamma=1, 
    beta=0, 
    propen_model='lr', 
    strength_instrument=5, 
    heteroscedasticity=False, 
    mu1_type='homo_separable', 
):
    """
    Generate synthetic data for causal inference experiments.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of total covariates/features.
    p_instr : int
        Number of instrument-like covariates used in the propensity score model.
    gamma : float
        Treatment effect strength (used in the outcome model).
    beta : float
        (Unused variable in this function, but retained for compatibility.)
    propen_model : str
        Propensity score model type: 'lr', 'lr_poly', or 'probit'.
    strength_instrument : float
        Coefficient used for instrumental features in the propensity model.
    heteroscedasticity : bool
        If True, heteroscedastic noise is introduced in Y0 and Y1.
    mu1_type : str
        Type of the treated outcome mean function: 'non_linear', 'hete_linear', or 'homo_separable'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [X1, X2, ..., A, y, propen, mu0, mu1].
    """
    # Generate features X
    X = np.random.normal(0, 1, (n, p))


    # Define the instruments for propensity score estimation
    X1 = X[:, 0:p_instr]

    # Compute propensity scores
    if propen_model == 'lr':
        propen = expit(np.sum(strength_instrument * X1, axis=1) + 0.01 * X[:, p_instr])
    elif propen_model == 'lr_poly':
        propen = expit(np.sum(strength_instrument * (X1 ** 2), axis=1) + 0.01 * X[:, p_instr])
    elif propen_model == 'probit':
        propen = stats.norm.cdf(np.sum(strength_instrument * (X1 ** 2), axis=1) + 0.01 * X[:, p_instr])
    else:
        raise ValueError(f"Unknown propen_model: {propen_model}")

    # Treatment assignment
    A = np.random.binomial(1, propen, n)

    # Generate potential outcomes
    X2 = X[:, p_instr]  # For simplicity, the outcome depends on X2
    mu0 = 10 * np.sin(np.pi * X2) + 20 * (X2) ** 2 + (X2 * np.cos(np.pi * X2 + X2)) ** 2

    if mu1_type == 'non_linear':
        mu1 = mu0 + (X2 * np.cos(np.pi * X2)) ** 2 * gamma
    elif mu1_type == 'hete_linear':
        mu1 = mu0 + X2 * gamma
    elif mu1_type == 'homo_separable':
        mu1 = mu0 + gamma
    else:
        raise ValueError(f"Unknown mu1_type: {mu1_type}")

    # Add noise
    if heteroscedasticity:
        sigma = 1 + 3 * (X2 * (1 - X2))
        Y0 = mu0 + np.random.normal(0, sigma, n)
        Y1 = mu1 + np.random.normal(0, sigma, n)
    else:
        Y0 = mu0 + np.random.normal(0, 1, n)
        Y1 = mu1 + np.random.normal(0, 1, n)

    # Observed outcome
    Y = A * Y1 + (1 - A) * Y0

    # Create a DataFrame
    data = np.column_stack((X, A, Y, propen, mu0, mu1))
    colnames = [f'X{i+1}' for i in range(p)] + ['A', 'y', 'propen', 'mu0', 'mu1']
    df = pd.DataFrame(data, columns=colnames)

    return df

def generate_synthetic_data_v2(
    n, 
    p_instr=1, 
    p_confound=1, 
    p_spurious=1, 
    p_outcome=1, 
    gamma_homo=1, 
    gamma_hetero=1,
    beta_instr=1, 
    beta_confound=1, 
    beta_outcome=1, 
    propen_model='lr', 
    heteroscedasticity=False, 
    outcome_type='nonlinear',
    misspecified_propen=False, 
    heterogeneous_treatment=True
):
    """
    Generate synthetic data with instruments, confounders, spurious variables,
    and separate outcome variables.

    Parameters
    ----------
    n : int
        Number of samples.
    p_instr : int
        Number of instrumental variables.
    p_confound : int
        Number of confounders.
    p_spurious : int
        Number of spurious (noise) variables.
    p_outcome : int
        Number of outcome-specific variables.
    gamma_homo : float
        Homogeneous treatment effect magnitude.
    gamma_hetero : float
        Additional magnitude for heterogeneous treatment effect.
    beta_instr : float
        Coefficient for instrument in propensity score model.
    beta_confound : float
        Coefficient for confounders in propensity score model and outcome.
    beta_outcome : float
        Coefficient for outcome-only covariates in outcome.
    propen_model : str
        Propensity score model, e.g. 'lr', 'lr_poly', or 'probit'.
    heteroscedasticity : bool
        If True, adds variable noise to Y0 and Y1.
    outcome_type : str
        'nonlinear' or 'linear' outcome.
    misspecified_propen : bool
        If True, uses additional interaction/higher-order terms in propensity model.
    heterogeneous_treatment : bool
        If True, uses a heterogeneous treatment effect (mu1_hetero).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [X1...Xp, A, y, propen, mu0, mu1].
    """
    # Total number of covariates
    p = p_instr + p_confound + p_outcome + p_spurious

    # Generate features from standard normal
    X = np.random.normal(0, 1, (n, p))

    # Partition X into instrument, confounders, outcome, and spurious groups
    X_instr = X[:, :p_instr]
    X_confound = X[:, p_instr:p_instr + p_confound]
    X_outcome = X[:, p_instr + p_confound:p_instr + p_confound + p_outcome]
    X_spurious = X[:, p_instr + p_confound + p_outcome:]  # Not used directly

    # Propensity score
    if not misspecified_propen:
        # Correct model
        if propen_model == 'lr':
            propen = expit(
                np.sum(beta_instr * X_instr, axis=1) + 
                beta_confound * np.sum(X_confound, axis=1)
            )
        elif propen_model == 'lr_poly':
            propen = expit(
                np.sum(beta_instr * (X_instr ** 2), axis=1) + 
                beta_confound * np.sum(X_confound, axis=1)
            )
        elif propen_model == 'probit':
            propen = stats.norm.cdf(
                np.sum(beta_instr * (X_instr ** 2), axis=1) + 
                beta_confound * np.sum(X_confound, axis=1)
            )
        else:
            raise ValueError(f"Unknown propen_model: {propen_model}")
    else:
        # Misspecified model
        interaction_term = np.sum(X_instr * X_confound, axis=1)
        higher_order_term = np.sum(X_instr ** 2, axis=1)
        propen = expit(
            np.sum(beta_instr * X_instr, axis=1) + 
            beta_confound * np.sum(X_confound, axis=1) +
            0.05 * interaction_term + 
            0.05 * higher_order_term
        )

    # Treatment assignment
    A = np.random.binomial(1, propen, n)

    # Potential outcome base (mu0)
    if outcome_type == 'nonlinear':
        # Nonlinear function of X_outcome + confounders
        sum_outcome = np.sum(beta_outcome * X_outcome, axis=1)
        sum_confound = np.sum(beta_confound * X_confound, axis=1)
        mu0 = (
            10 * np.sin(np.pi * sum_outcome) + 
            20 * (sum_outcome) ** 2 + 
            (sum_confound * np.cos(np.pi * sum_outcome)) ** 2
        )
    elif outcome_type == 'linear':
        sum_outcome = np.sum(beta_outcome * X_outcome, axis=1)
        sum_confound = np.sum(beta_confound * X_confound, axis=1)
        mu0 = sum_outcome + sum_confound
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")

    # Potential outcomes (mu1)
    # Heterogeneous vs. homogeneous
    sum_outcome2 = np.sum(beta_outcome * X_outcome, axis=1)
    mu1_hetero = mu0 + sum_outcome2 * gamma_hetero
    mu1_homo = mu0 + gamma_homo

    # Add noise / heteroscedasticity
    if heteroscedasticity:
        sigma = 1 + 3 * (np.sum(X_outcome, axis=1) * (1 - np.sum(X_outcome, axis=1)))
        Y0 = mu0 + np.random.normal(0, sigma, n)
        Y1_hetero = mu1_hetero + np.random.normal(0, sigma, n)
        Y1_homo = mu1_homo + np.random.normal(0, sigma, n)
    else:
        Y0 = mu0 + np.random.normal(0, 1, n)
        Y1_hetero = mu1_hetero + np.random.normal(0, 1, n)
        Y1_homo = mu1_homo + np.random.normal(0, 1, n)

    if heterogeneous_treatment:
        Y = A * Y1_hetero + (1 - A) * Y0
        mu1 = mu1_hetero
    else:
        Y = A * Y1_homo + (1 - A) * Y0
        mu1 = mu1_homo

    # Combine data into DataFrame
    data = np.column_stack((X, A, Y, propen, mu0, mu1))
    colnames = [f'X{i+1}' for i in range(p)] + ['A', 'y', 'propen', 'mu0', 'mu1']
    df = pd.DataFrame(data, columns=colnames)

    return df

class News:
    """
    Data handler for the News dataset. Returns train, validation, or test subsets
    as specified. Also includes a method get_df() to produce a DataFrame with 
    columns matching the typical causal notation [X1,... Xp, A, y, propen, mu0, mu1].
    """

    def __init__(
        self, 
        exp_num, 
        dataset='train', 
        tensor=True, 
        device="cpu", 
        train_size=0.6, 
        val_size=0.2,
        data_folder=None, 
        scale=True, 
        seed=0
    ):
        if data_folder is None:
            data_folder = '../data'

        # Ensure data is created
        if not os.path.isdir(os.path.join(data_folder, 'News/numpy_dicts/')):
            self._create_data(data_folder)

        # Load data
        with open(
            os.path.join(data_folder, f'News/numpy_dicts/data_as_dicts_with_numpy_seed_{exp_num}'),
            'rb'
        ) as file:
            data = pickle.load(file)

        # Add the 'cate_true' = mu1 - mu0 for convenience
        data['cate_true'] = data['mu1'] - data['mu0']

        x = data['x']
        n_samples = x.shape[0]


        # Create reproducible indices
        rng = np.random.default_rng(seed=seed)
        original_indices = rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)

        itr = original_indices[:n_train]                # train set
        iva = original_indices[n_train:n_train + n_val] # val set
        idxtrain = original_indices[:n_train + n_val]   # train + val set
        ite = original_indices[n_train + n_val:]        # test set

        # Choose the subset based on 'dataset'
        if dataset == 'train':
            subset_indices = itr
        elif dataset == 'val':
            subset_indices = iva
        elif dataset == "train_val":
            subset_indices = idxtrain
        else:
            subset_indices = ite

        self.original_indices = subset_indices

        # Convert and store each field as tensor if required
        for key, value in data.items():
            value = value[subset_indices]
            if tensor:
                value = torch.tensor(value, dtype=torch.float32, device=device)
            setattr(self, key, value)
        
        self.data = data

    @staticmethod
    def _create_data(data_folder):
        """
        Download, unzip, and convert the News dataset from sparse CSV to dense arrays,
        then pickle them for future reuse.
        """
        print('News : no data, creating it')
        print('Downloading zipped csvs')
        zip_path = os.path.join(data_folder, 'News/csv.zip')
        urllib.request.urlretrieve(
            'http://www.fredjo.com/files/NEWS_csv.zip', 
            zip_path
        )

        print('Unzipping csvs with sparse data')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_folder, 'News'))

        print('Densifying the sparse data')
        os.mkdir(os.path.join(data_folder, 'News/numpy_dicts/'))

        for f_index in range(1, 50 + 1):
            # Load the sparse matrix
            csv_x = os.path.join(data_folder, 
                                 f'News/csv/topic_doc_mean_n5000_k3477_seed_{f_index}.csv.x')
            mat = pd.read_csv(csv_x)
            n_rows, n_cols = int(mat.columns[0]), int(mat.columns[1])
            x = np.zeros((n_rows, n_cols), dtype=int)

            for i, j, val in zip(mat.iloc[:, 0], mat.iloc[:, 1], mat.iloc[:, 2]):
                x[i - 1, j - 1] = val

            # Load metadata with treatment/outcomes
            csv_y = os.path.join(
                data_folder, 
                f'News/csv/topic_doc_mean_n5000_k3477_seed_{f_index}.csv.y'
            )
            meta = pd.read_csv(
                csv_y, 
                names=['t', 'y', 'y_cf', 'mu0', 'mu1']
            )

            data = {
                'x': x,
                't': np.array(meta['t']).reshape((-1, 1)),
                'y': np.array(meta['y']).reshape((-1, 1)),
                'y_cf': np.array(meta['y_cf']).reshape((-1, 1)),
                'mu0': np.array(meta['mu0']).reshape((-1, 1)),
                'mu1': np.array(meta['mu1']).reshape((-1, 1))
            }

            out_path = os.path.join(
                data_folder, 
                f'News/numpy_dicts/data_as_dicts_with_numpy_seed_{f_index}'
            )
            with open(out_path, 'wb') as file:
                pickle.dump(data, file)

        print('Done!')

    def __getitem__(self, index, attrs=None):
        """
        Retrieve tuple of requested attributes for a single index, e.g., 
        (x, y, t, mu0, mu1, cate_true) by default.
        """
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        return tuple(getattr(self, attr)[index] for attr in attrs)

    def __len__(self):
        return len(self.original_indices)
    
    def get_cov_shape(self):
        return self.data['x'].shape

    def get_df(self):
        """
        Return a pandas DataFrame containing columns [X1, X2, ..., Xp, A, y, propen, mu0, mu1]
        consistent with the synthetic data frames.

        - 'x' is high-dimensional, so each dimension is split into separate columns: X1, X2, ..., Xp.
        - 't' corresponds to 'A' in the synthetic data.
        - 'propen' is not directly available in the News data, so we insert a placeholder (NaN).
        """
        # Move data to CPU numpy if needed
        def to_numpy(arr):
            return arr.detach().cpu().numpy() if torch.is_tensor(arr) else arr

        x_np = to_numpy(self.x)   # shape (n, p)
        t_np = to_numpy(self.t).flatten()
        y_np = to_numpy(self.y).flatten()
        mu0_np = to_numpy(self.mu0).flatten()
        mu1_np = to_numpy(self.mu1).flatten()

        # Create placeholder for 'propen'
        propen_column = np.full_like(t_np, np.nan, dtype=float)

        # Create individual columns for each dimension of x
        num_features = x_np.shape[1]  # p
        x_col_names = [f'X{i+1}' for i in range(num_features)]
        x_df = pd.DataFrame(x_np, columns=x_col_names)

        # Create a DataFrame for A, y, propen, mu0, mu1
        other_df = pd.DataFrame({
            'A': t_np,
            'y': y_np,
            'propen': propen_column,
            'mu0': mu0_np,
            'mu1': mu1_np
        })

        # Concatenate the two DataFrames column-wise
        df = pd.concat([x_df, other_df], axis=1)
        return df


def pre_process_text(df):                    
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")
    model = DistilBertModel.from_pretrained("bert-base-uncased")
    # Function to get the [CLS] token embedding
    def get_cls_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Get the embedding for the [CLS] token (first token in the sequence)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().numpy()
        return cls_embedding

    # Add the [CLS] embeddings to the dataframe
    df["x"] = df["text"].apply(get_cls_embedding)
    return df


