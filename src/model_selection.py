import os
import pickle
import time
import json
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR

from metrics import cfcv_mse, ipw_mse, plug_in, tau_risk, nmse
from model import CFR, WeightedCFR
from visualizer import base_models, meta_models, metric_names

# load pre-tuned (not yet trained) models.
with open('../models/base_model_dict.pickle', mode='rb') as fp:
    base_model_dict = pickle.load(fp)

with open('../models/meta_model_dict.pickle', mode='rb') as fp:
    meta_model_dict = pickle.load(fp)

NUM_META_MODELS = len(base_model_dict) * len(meta_model_dict)

# load best params for GBR and CFR
best_params_gbr = json.loads(json.load(open(f'../models/best_params_gbr.json', 'r')))
best_params_cfr = json.loads(json.load(open(f'../models/best_params_cfr.json', 'r')))


class MetaModel:

    def __init__(self, base_model: str, meta_model: str) -> None:
        """Initialize Class."""
        assert base_model in ('dt', 'rf', 'gbr', 'ridge', 'svm')
        assert meta_model in ('X', 'S', 'T', 'DA', 'DR')
        _base_model = base_model_dict[base_model]
        _meta_model = meta_model_dict[meta_model]
        if meta_model in ('X', 'T', 'DR'):
            self.model = _meta_model(_base_model, _base_model)
        elif meta_model in ('S'):
            self.model = _meta_model(_base_model)
        elif meta_model in ('DA'):
            self.model = _meta_model(_base_model, _base_model, _base_model)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.array) -> None:
        """Fit meta-model."""
        self.model.fit(Y, T, X)

    def predict(self, X: np.array) -> np.ndarray:
        """Make prediction."""
        return self.model.effect(X)


def _make_preds(Xtr: np.ndarray, Xval: np.ndarray, Xte: np.ndarray,
                Ttr: np.ndarray, Ytr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make CATE predictions on validation and test data."""
    models = product(base_models, meta_models)
    preds_val = np.zeros((Xval.shape[0], NUM_META_MODELS))
    preds_te = np.zeros((Xte.shape[0], NUM_META_MODELS))

    for i, (b, m) in enumerate(models):
        model = MetaModel(base_model=b, meta_model=m)
        model.fit(Xtr, Ttr, Ytr)
        preds_val[:, i] = model.predict(Xval)
        preds_te[:, i] = model.predict(Xte)

    return preds_val, preds_te


def _make_val_preds(Xval: np.ndarray, Tval: np.ndarray, Yval: np.ndarray,
                    alpha_list: Optional[List[float]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Estimate prameters of validation data."""
    tf.set_random_seed(12345)
    ops.reset_default_graph()
    params = np.zeros((Xval.shape[0], 6))
    obs_outcome_model_ = GBR(
        n_estimators=100,
        learning_rate=best_params_gbr['eta'],
        max_depth=best_params_gbr['max_depth'],
        min_samples_leaf=best_params_gbr['min_leaf'],
        random_state=12345)
    propensity_model = LogisticRegression(random_state=12345)
    potential_outcome_model_ = CFR(
        hidden_layer_size=best_params_cfr['hidden_layer_size'],
        num_layers=best_params_cfr['num_layers'],
        batch_size=2 ** 8,
        learning_rate=best_params_cfr['eta'],
        dropout=0.2,
        imbalance_loss_weight=best_params_cfr['alpha'])
    cfcv_model_ = WeightedCFR(
        hidden_layer_size=best_params_cfr['hidden_layer_size'],
        num_layers=best_params_cfr['num_layers'],
        batch_size=2 ** 8,
        learning_rate=best_params_cfr['eta'],
        dropout=0.2,
        imbalance_loss_weight=best_params_cfr['alpha'])
    # estimate observed outcome.
    obs_outcome_model_.fit(Xval, Yval)
    params[:, 0] = obs_outcome_model_.predict(Xval)
    # estimate potential outcomes for plug-in metric.
    potential_outcome_model_.train(x=Xval, t=Tval, y=Yval)
    mu0_preds, mu1_preds = potential_outcome_model_.predict(Xval)
    params[:, 1] = mu0_preds
    params[:, 2] = mu1_preds
    potential_outcome_model_.sess.close()
    # estimate propensity score.
    propensity_model.fit(Xval, Tval)
    params[:, -1] = propensity_model.predict_proba(Xval)[:, 1]
    # estimate potential outcomes for Counterfactual Cross Validation.
    cfcv_model_.train(x=Xval, t=Tval, y=Yval, e=params[:, -1])
    mu0_preds, mu1_preds = cfcv_model_.predict(Xval)
    cfcv_model_.sess.close()
    params[:, 3] = mu0_preds
    params[:, 4] = mu1_preds

    params_alpha = None
    if alpha_list is not None:
        params_alpha = np.zeros((Xval.shape[0], 2 * len(alpha_list)))
        for i, alpha in enumerate(alpha_list):
            tf.set_random_seed(12345)
            ops.reset_default_graph()
            cfcv_model_ = WeightedCFR(
                hidden_layer_size=best_params_cfr['hidden_layer_size'],
                num_layers=best_params_cfr['num_layers'],
                batch_size=2 ** 8,
                learning_rate=best_params_cfr['eta'],
                dropout=0.2,
                imbalance_loss_weight=alpha)
            cfcv_model_.train(x=Xval, t=Tval, y=Yval, e=params[:, -1])
            mu0_preds, mu1_preds = cfcv_model_.predict(Xval)
            cfcv_model_.sess.close()
            params_alpha[:, 2 * i] = mu0_preds
            params_alpha[:, 2 * i + 1] = mu1_preds

    return params, params_alpha


def run_preds(data: str = 'ihdp_B', alpha_list: Optional[List[float]] = None) -> None:
    """Run predictions on all scenarios."""
    # mkdirs
    os.makedirs(f'../logs/{data}/', exist_ok=True)
    # load data.
    start = time.time()
    Xtr = np.load(f'../data/{data}/Xtr.npy')
    Ttr = np.load(f'../data/{data}/Ttr.npy')
    Ytr = np.load(f'../data/{data}/Ytr.npy')
    Xval = np.load(f'../data/{data}/Xval.npy')
    Tval = np.load(f'../data/{data}/Tval.npy')
    Yval = np.load(f'../data/{data}/Yval.npy')
    Xte = np.load(f'../data/{data}/Xte.npy')
    ITEte = np.load(f'../data/{data}/ITEte.npy')
    # fit model.
    preds_list_val = []
    preds_list_te = []
    params_list = []
    ground_truths = np.zeros((Xtr.shape[0], NUM_META_MODELS))
    params_alpha_list = []
    for i in np.arange(Xtr.shape[0]):
        preds_val, preds_te = _make_preds(Xtr=Xtr[i], Xval=Xval[i], Xte=Xte[i], Ttr=Ttr[i], Ytr=Ytr[i])
        preds_list_val.append(preds_val)
        preds_list_te.append(preds_te)
        for j in np.arange(NUM_META_MODELS):
            ground_truths[i, j] = np.sqrt(nmse(ITEte[i], preds_te[:, j]))
        params, params_alpha = _make_val_preds(Xval=Xval[i], Tval=Tval[i], Yval=Yval[i], alpha_list=alpha_list)
        params_list.append(params)
        params_alpha_list.append(params_alpha)

        print(f'ITERS#{i + 1}: {np.round(time.time() -  start)}s')
    # save predictions, parameters, and ground truths.
    np.save(arr=np.concatenate(preds_list_val).reshape((Xval.shape[0], Xval.shape[1], NUM_META_MODELS)),
            file=f'../logs/{data}/predictions_val.npy')
    np.save(arr=np.concatenate(preds_list_te).reshape((Xte.shape[0], Xte.shape[1], NUM_META_MODELS)),
            file=f'../logs/{data}/predictions_te.npy')
    np.save(arr=np.concatenate(params_list).reshape((Xval.shape[0], Xval.shape[1], 6)),
            file=f'../logs/{data}/parameters.npy')
    if alpha_list is not None:
        np.save(arr=np.concatenate(params_alpha_list).reshape((Xval.shape[0], Xval.shape[1], 2 * len(alpha_list))),
                file=f'../logs/{data}/parameters_alpha.npy')
    np.save(arr=ground_truths, file=f'../logs/{data}/ground_truths.npy')

    print('\n', '=' * 25, '\n')
    print(f'Finished model selection: {np.round(time.time() -  start)}s')
    print('\n', '=' * 25, '\n')


NUM_METRICS = len(metric_names)


def estimate_metrics_on_val(data: str = 'ihdp_B') -> None:
    """Estimate true performance using validation sets."""
    params = np.load(f'../logs/{data}/parameters.npy')
    preds = np.load(f'../logs/{data}/predictions_val.npy')
    Tval = np.load(f'../data/{data}/Tval.npy')
    Yval = np.load(f'../data/{data}/Yval.npy')
    metrics_list = []
    for i in np.arange(Tval.shape[0]):
        mu, mu0, mu1, mu0_cf, mu1_cf, ps =\
            params[i, :, 0], params[i, :, 1], params[i, :, 2], params[i, :, 3], params[i, :, 4], params[i, :, 5]
        # calc metrics
        metrics = np.zeros((NUM_METRICS, NUM_META_MODELS))
        for j in np.arange(NUM_META_MODELS):
            _preds = preds[i, :, j]
            metrics[0, j] = ipw_mse(y=Yval[i], t=Tval[i], ps=ps, ite_pred=_preds)
            metrics[1, j] = tau_risk(y=Yval[i], t=Tval[i], ps=ps, mu=mu, ite_pred=_preds)
            metrics[2, j] = plug_in(mu0=mu0, mu1=mu1, ite_pred=_preds)
            metrics[3, j] = cfcv_mse(y=Yval[i], t=Tval[i], mu0=mu0_cf, mu1=mu1_cf, ps=ps, ite_pred=_preds)
        metrics_list.append(metrics)
    np.save(arr=np.concatenate(metrics_list).reshape((Tval.shape[0], NUM_METRICS, NUM_META_MODELS)),
            file=f'../logs/{data}/metrics.npy')


def estimate_alpha_metrics_on_val(data: str = 'ihdp_B', alpha_list: Optional[List[float]] = None) -> None:
    """Estimate true performance using validation sets."""
    params = np.load(f'../logs/{data}/parameters.npy')
    params_alpha = np.load(f'../logs/{data}/parameters_alpha.npy')
    preds = np.load(f'../logs/{data}/predictions_val.npy')
    Tval = np.load(f'../data/{data}/Tval.npy')
    Yval = np.load(f'../data/{data}/Yval.npy')
    metrics_list = []
    for i in np.arange(Tval.shape[0]):
        metrics = np.zeros((len(alpha_list), NUM_META_MODELS))
        for j in np.arange(NUM_META_MODELS):
            _preds = preds[i, :, j]
            for k in np.arange(len(alpha_list)):
                mu0_cf, mu1_cf, ps = params_alpha[i, :, 2 * k], params_alpha[i, :, 2 * k + 1], params[i, :, -1]
                metrics[k, j] = cfcv_mse(y=Yval[i], t=Tval[i], mu0=mu0_cf, mu1=mu1_cf, ps=ps, ite_pred=_preds)
        metrics_list.append(metrics)
    np.save(arr=np.concatenate(metrics_list).reshape((Tval.shape[0], len(alpha_list), NUM_META_MODELS)),
            file=f'../logs/{data}/metrics_alpha.npy')
