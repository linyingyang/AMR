import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

# R's nprobust
nprobust = importr('nprobust')


class TorchMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], lr=1e-3, max_epochs=100, batch_size=64, random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        layers = []
        current_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))  # Add dropout for regularization
            current_dim = hdim
        layers.append(nn.Linear(current_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.optimizer = None
        self.criterion = None  # Defined in `fit()`

    def forward(self, x):
        return self.network(x)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        # Class balancing
        pos_weight = torch.tensor([len(y) / (2 * sum(y))], dtype=torch.float32)  # Balance positive samples
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

        self.train()
        for epoch in range(self.max_epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()

    def predict_proba(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.forward(X_tensor).flatten()
            probs = torch.sigmoid(logits).clamp(min=1e-4, max=1 - 1e-4)
        probs_2col = torch.stack([1 - probs, probs], dim=1).numpy()
        return probs_2col



class TorchMLPRegressor(nn.Module):
    """
    Simple feed-forward neural network for regression (outcome model).
    We'll wrap in a scikit-like API for .fit(), .predict().
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], lr=1e-3, max_epochs=100, batch_size=64, random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        layers = []
        current_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            current_dim = hdim
        # final layer: dimension 1 for regression
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.optimizer = None
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        for epoch in range(self.max_epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.forward(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.forward(X_tensor).flatten()
        return preds.numpy()




class TorchClassificationWrapper:
    """
    A scikit-like wrapper around the TorchMLPClassifier 
    so that we can do e.g. .fit(X, y), .predict_proba(X).
    We'll figure out input_dim from X at .fit() time.
    """
    def __init__(self, hidden_dims=[64, 32], lr=1e-3, max_epochs=100, batch_size=64, random_seed=42):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        input_dim = X.shape[1]
        self.model_ = TorchMLPClassifier(
            input_dim=input_dim, 
            hidden_dims=self.hidden_dims, 
            lr=self.lr, 
            max_epochs=self.max_epochs,
            batch_size=self.batch_size, 
            random_seed=self.random_seed
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet!")
        return self.model_.predict_proba(X)


class TorchRegressionWrapper:
    """
    A scikit-like wrapper around the TorchMLPRegressor 
    so that we can do e.g. .fit(X, y), .predict(X).
    We'll figure out input_dim from X at .fit() time.
    """
    def __init__(self, hidden_dims=[64, 32], lr=1e-3, max_epochs=100, batch_size=64, random_seed=42):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        input_dim = X.shape[1]
        self.model_ = TorchMLPRegressor(
            input_dim=input_dim, 
            hidden_dims=self.hidden_dims, 
            lr=self.lr, 
            max_epochs=self.max_epochs,
            batch_size=self.batch_size, 
            random_seed=self.random_seed
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet!")
        return self.model_.predict(X)

        
def get_ps_model(ps_model_type="lr", hyperparam_search=True, random_seed=42):
    """
    Get the propensity score model based on the specified type.
    
    Parameters:
    - ps_model_type (str): Type of model ('lr', 'lasso', 'torch_nn', 'none').
    - hyperparam_search (bool): Whether to perform hyperparameter tuning.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - sklearn estimator or None
    """
    if ps_model_type == 'lr':
        if hyperparam_search:
            param_dist = {'C': np.logspace(-4, 4, 20)}
            model = RandomizedSearchCV(
                LogisticRegression(max_iter=1000, random_state=random_seed),  # Plain logistic regression
                param_distributions=param_dist,
                n_iter=10,  # Specify the number of random samples to draw
                cv=5,
                scoring='neg_log_loss',
                n_jobs=-1,
                random_state=random_seed
            )
        else:
            model = LogisticRegression(random_state=random_seed, max_iter=1000)  # Plain logistic regression
        return model

    elif ps_model_type == 'lasso':
        if hyperparam_search:
            param_dist = {'C': np.logspace(-4, 4, 20)}  # C for inverse regularization strength
            model = RandomizedSearchCV(
                LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=random_seed),
                param_distributions=param_dist,
                n_iter=10,
                cv=5,
                scoring='neg_log_loss',
                n_jobs=-1,
                random_state=random_seed
            )
        else:
            model = LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0,  # Default regularization strength
                random_state=random_seed,
                max_iter=1000
            )
        return model
    elif ps_model_type == "torch_nn":

        return TorchClassificationWrapper(hidden_dims=[64, 32], random_seed=random_seed)

    elif ps_model_type == "original":
        return None  # skip fitting, use 'propen' column directly
    # Feel free to add more PS models 
    raise ValueError(f"Unknown ps_model: {ps_model_type}")


def get_or_model(or_model_type, hyperparam_search=True, random_seed=42):
    """
    Get the outcome regression model based on the specified type.
    
    Parameters:
    - or_model_type (str): Type of model ('lr', 'lasso', 'rf', 'torch_nn', 'none').
    - hyperparam_search (bool): Whether to perform hyperparameter tuning.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - sklearn estimator or None
    """
    if or_model_type == "lr":
        return LinearRegression()

    # Example snippet
    elif or_model_type == 'lasso':
        if hyperparam_search:
            # Define the parameter distributions to sample from
            # You can adjust alpha range, step, or add more parameters to tune
            param_distribution = {
                'alpha': np.logspace(-3, 2, 100),  # 0.001 to 100, log-spaced
            }

            # Randomized Search setup
            model = RandomizedSearchCV(
                estimator=Lasso(random_state=random_seed),
                param_distributions=param_distribution,
                n_iter=20,              # how many random samples to draw
                cv=5,                   # 5-fold cross-validation
                scoring='neg_mean_squared_error',  # or other metric
                random_state=random_seed,
                n_jobs=-1               # use all available cores
            )
        else:
            model = Lasso(alpha=1.0, random_state=random_seed)

        return model

    elif or_model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_seed)
    elif or_model_type == "rf_binary":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=random_seed)
    elif or_model_type == "torch_nn":
        return TorchRegressionWrapper(hidden_dims=[64, 32], random_seed=random_seed)
    elif or_model_type == "torch_nn_binary":
        return TorchClassificationWrapper(hidden_dims=[64, 32], random_seed=random_seed)
    elif or_model_type == "original":
        return None
    raise ValueError(f"Unknown or_model: {or_model_type}")


def get_w_model(w_model_name, random_seed=None, hyperparam_search=True, alpha=None, gamma=None):
    """
    Return a model object (for W-fitting) given w_model_name.
    Extended to allow more flexible kernel ridge param searching.
    """
    if w_model_name == "lr":
        return LinearRegression()
    elif w_model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_seed)
    elif w_model_name == "mlp":
        # scikit MLP
        return MLPRegressor(
            hidden_layer_sizes=(512, 256, 32), 
            max_iter=500, 
            alpha=0.01, 
            random_state=random_seed
        )
    elif w_model_name == "kernel_ridge_regression":
        # Make param search flexible here:
        if hyperparam_search:
            param_dist = {
                "alpha": [0.01, 0.1, 0.2, 0.5],
                "gamma": np.logspace(-3, 2, 10)
            }
            n_iter_ = 50  # user can adjust or pass in
            return RandomizedSearchCV(
                KernelRidge(kernel="rbf"), 
                param_distributions=param_dist, 
                n_iter=n_iter_, 
                cv=5, 
                scoring='neg_mean_squared_error',
                random_state=random_seed, 
                n_jobs=-1
            )
        else:
            # If user wants direct alpha/gamma usage
            return KernelRidge(alpha=alpha, gamma=gamma, kernel="rbf")

    elif w_model_name == 'huber':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', HuberRegressor())
        ])
        if hyperparam_search:
            param_dist = {
                'model__epsilon': np.linspace(1.2, 3, 20),
                'model__alpha': np.linspace(0.001, 0.01, 20)
            }
            return RandomizedSearchCV(
                pipeline, 
                param_distributions=param_dist, 
                n_iter=50, 
                cv=5, 
                scoring='neg_mean_squared_error', 
                random_state=random_seed, 
                n_jobs=-1
            )
        else:
            return pipeline

    elif w_model_name == 'nprobust':
        # Return a callable that does local poly regress in R
        def nprobust_model(X_train, y_train, X_eval):
            X_train_r = ro.FloatVector(X_train)
            y_train_r = ro.FloatVector(y_train)
            X_eval_r = ro.FloatVector(X_eval)
            lpr_result = nprobust.lprobust(
                x=X_train_r, 
                y=y_train_r, 
                eval=X_eval_r, 
                kernel='gau', 
                p=1
            )
            estimates = np.array(lpr_result.rx2('Estimate'))[:, 5]
            return estimates, lpr_result
        return nprobust_model

    # If you have custom NW or LocalLinear classes, import them and wrap here
    raise ValueError(f"Unknown w_model: {w_model_name}")