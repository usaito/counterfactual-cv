import numpy as np
from sklearn.metrics import mean_squared_error as mse


def mu_risk(y: np.ndarray, t: np.ndarray, mu0_pred: np.array, mu1_pred: np.array) -> float:
    """Factual Validation (Mu-risk) with normalized mean-squared-error."""
    nmse_control = nmse(y_true=y[t == 0], y_pred=mu0_pred[t == 0])
    nmse_treat = nmse(y_true=y[t == 1], y_pred=mu1_pred[t == 1])
    return np.mean(1 - t) * nmse_control + np.mean(t) * nmse_treat


def plug_in(mu0: np.array, mu1: np.array, ite_pred: np.array) -> float:
    """Plug-in estimator."""
    return mse(mu1 - mu0, ite_pred)


def ipw_mse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, ite_pred: np.ndarray) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    pseudo_ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
    return mse(pseudo_ite, ite_pred)


def cfcv_mse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, ite_pred: np.ndarray) -> float:
    """Mean-squared-error with Counterfactual Cross Validation."""
    pseudo_ite = (t * (y - mu1) / ps) - ((1 - t) * (y - mu0) / (1 - ps)) + (mu1 - mu0)
    return mse(pseudo_ite, ite_pred)


def tau_risk(y: np.ndarray, t: np.ndarray, ps: np.ndarray, mu: np.ndarray, ite_pred: np.ndarray) -> float:
    """Tau-risk."""
    return mse(y - mu, (t - ps) * ite_pred)


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)
