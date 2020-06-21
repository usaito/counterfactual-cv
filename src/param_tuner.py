"""Hyperparameter tuning of causal inference model using Optuna."""
import os
import time
from typing import Dict

import numpy as np
import optuna
from econml.metalearners import DomainAdaptationLearner as DAL
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingRegressor as GBR

from metrics import cfcv_mse, ipw_mse, plug_in, tau_risk, nmse

metric_dict = {'ipw': ipw_mse, 'tau': tau_risk, 'cfcv': cfcv_mse, 'plug-in': plug_in}
metric_names = ['ipw', 'tau', 'cfcv', 'plug-in']
sampler = TPESampler(seed=12345)


class Objective:
    """Objective class for optuna."""

    def __init__(self, metric_name: str, data: str = 'ihdp_B', trial_id: int = 0) -> None:
        """Initialize Class."""
        self.metric_name = metric_name
        self.trial_id = trial_id
        self.Xtr = np.load(f'../data/{data}/Xtr.npy')
        self.Ttr = np.load(f'../data/{data}/Ttr.npy')
        self.Ytr = np.load(f'../data/{data}/Ytr.npy')
        self.Xval = np.load(f'../data/{data}/Xval.npy')
        self.Tval = np.load(f'../data/{data}/Tval.npy')
        self.Yval = np.load(f'../data/{data}/Yval.npy')
        self.params = np.load(f'../logs/{data}/parameters.npy')

    def __call__(self, trial: optuna.Trial) -> float:
        """Calculate an objective value by using the extra arguments."""
        _preds = self._make_cate_predictions(trial, self.trial_id)
        _params_metric = self._create_dict_for_metric()[self.metric_name]
        return metric_dict[self.metric_name](ite_pred=_preds, **_params_metric)

    def _create_dict_for_metric(self) -> Dict[str, Dict]:
        """Create input for metrics."""
        i = self.trial_id
        mu, mu0, mu1, mu0_cf, mu1_cf, ps =\
            self.params[i, :, 0], self.params[i, :, 1], self.params[i, :, 2],\
            self.params[i, :, 3], self.params[i, :, 4], self.params[i, :, 5]
        return {'ipw': {'y': self.Yval[i], 't': self.Tval[i], 'ps': ps},
                'tau': {'y': self.Yval[i], 't': self.Tval[i], 'ps': ps, 'mu': mu},
                'plug-in': {'mu0': mu0, 'mu1': mu1},
                'cfcv': {'y': self.Yval[i], 't': self.Tval[i], 'mu0': mu0_cf, 'mu1': mu1_cf, 'ps': ps}}

    def _make_cate_predictions(self, trial: optuna.Trial, i: int) -> np.ndarray:
        """Make predictions of CATE by a sampled set of hyperparameters."""
        # hyparparameters
        # for control model
        eta_con = trial.suggest_loguniform('eta_control', 1e-5, 1e-1)
        min_leaf_con = trial.suggest_int('min_samples_leaf_control', 1, 20)
        max_depth_con = trial.suggest_int('max_depth_control', 1, 20)
        subsample_con = trial.suggest_uniform('sub_sample_control', 0.1, 1.0)
        control_params = {'n_estimators': 100, 'learning_rate': eta_con, 'min_samples_leaf': min_leaf_con,
                          'max_depth': max_depth_con, 'subsample': subsample_con, 'random_state': 12345}
        # for treated model
        eta_trt = trial.suggest_loguniform('eta_treat', 1e-5, 1e-1)
        min_leaf_trt = trial.suggest_int('min_samples_leaf_treat', 1, 20)
        max_depth_trt = trial.suggest_int('max_depth_treat', 1, 20)
        subsample_trt = trial.suggest_uniform('sub_sample_treat', 0.1, 1.0)
        treated_params = {'n_estimators': 100, 'learning_rate': eta_trt, 'min_samples_leaf': min_leaf_trt,
                          'max_depth': max_depth_trt, 'subsample': subsample_trt, 'random_state': 12345}
        # for overall model
        eta_ova = trial.suggest_loguniform('eta_overall', 1e-5, 1e-1)
        min_leaf_ova = trial.suggest_int('min_samples_leaf_overall', 1, 20)
        max_depth_ova = trial.suggest_int('max_depth_overall', 1, 20)
        subsample_ova = trial.suggest_uniform('sub_sample_overall', 0.1, 1.0)
        overall_params = {'n_estimators': 100, 'learning_rate': eta_ova, 'min_samples_leaf': min_leaf_ova,
                          'max_depth': max_depth_ova, 'subsample': subsample_ova, 'random_state': 12345}
        # define DAL model
        meta_learner = DAL(controls_model=GBR(**control_params),
                           treated_model=GBR(**treated_params),
                           overall_model=GBR(**overall_params))
        meta_learner.fit(X=self.Xtr[i], T=self.Ttr[i], Y=self.Ytr[i])
        return meta_learner.effect(X=self.Xval[i])


def main_tuner(data: str = 'ihdp_B', iters: int = 30, n_trials: int = 30, verbose: bool = True) -> None:
    """Run experiments on hyperparameter tuning."""
    os.makedirs(f'../results/{data}/', exist_ok=True)
    # load data
    Xtr = np.load(f'../data/{data}/Xtr.npy')
    Ttr = np.load(f'../data/{data}/Ttr.npy')
    Ytr = np.load(f'../data/{data}/Ytr.npy')
    Xte = np.load(f'../data/{data}/Xte.npy')
    ITEte = np.load(f'../data/{data}/ITEte.npy')
    # results
    results = np.zeros((iters, len(metric_names)))
    trials_frames = [[] for i in np.arange(len(metric_names))]

    start = time.time()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    for j, met in enumerate(metric_names):
        for i in np.arange(iters):
            # optimize hyperparameters.
            study = optuna.create_study(sampler=sampler)
            study.optimize(Objective(data=data, metric_name=met, trial_id=i), n_trials=n_trials)
            trials_frames[j].append(study.trials_dataframe())
            params = study.best_params
            # define the best model.
            control_model = GBR(
                n_estimators=100, learning_rate=params['eta_control'],
                min_samples_leaf=params['min_samples_leaf_control'], max_depth=params['max_depth_control'],
                subsample=params['sub_sample_control'], random_state=12345)
            treated_model = GBR(
                n_estimators=100, learning_rate=params['eta_treat'],
                min_samples_leaf=params['min_samples_leaf_treat'], max_depth=params['max_depth_treat'],
                subsample=params['sub_sample_treat'], random_state=12345)
            overall_model = GBR(
                n_estimators=100, learning_rate=params['eta_overall'],
                min_samples_leaf=params['min_samples_leaf_overall'], max_depth=params['max_depth_overall'],
                subsample=params['sub_sample_overall'], random_state=12345)
            meta_leaner = DAL(controls_model=control_model, treated_model=treated_model, overall_model=overall_model)
            # fit and predict the best model.
            meta_leaner.fit(X=Xtr[i], T=Ttr[i], Y=Ytr[i])
            preds_te = meta_leaner.effect(Xte[i])
            results[i, j] = np.sqrt(nmse(ITEte[i], preds_te))

        if verbose:
            print('\n', '=' * 25, '\n')
            print(f'Finished {met}: {np.round(time.time() - start)}s')
            print('\n', '=' * 25, '\n')

    np.save(arr=results, file=f'../results/{data}/optuna.npy')
