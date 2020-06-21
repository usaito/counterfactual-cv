import argparse
import os
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf

from model_selection import estimate_alpha_metrics_on_val, estimate_metrics_on_val, run_preds
from data_generator import generate_data
from param_tuner import main_tuner
from visualizer import Visualizer, metric_names

parser = argparse.ArgumentParser()
parser.add_argument('--iters', default=100, type=int)
parser.add_argument('--n_trials', default=100, type=int)
parser.add_argument('--alpha_list', default=None, nargs='*')


def save_results(data: str = 'ihdp_B') -> None:
    """Save final results."""
    os.makedirs(f'../results/{data}/', exist_ok=True)
    # load true and estimated performance.
    ground_truths = np.load(f'../logs/{data}/ground_truths.npy')
    estimated_metrics = np.load(f'../logs/{data}/metrics.npy')

    # spearman rank correlation between metric values and true performance.
    corr = np.array(
        [[spearmanr(ground_truths[j, :], estimated_metrics[j, i, :])[0]
          for i in np.arange(len(metric_names))]
            for j in np.arange(ground_truths.shape[0])])
    np.save(arr=corr, file=f'../results/{data}/corr.npy')
    pd.DataFrame(corr, columns=metric_names).to_csv(f'../results/{data}/corr.csv')
    # regret in model selection.
    regret = np.array(
        [[(ground_truths[j, np.argmin(estimated_metrics[j, i, :])] / ground_truths[j].min()) - 1
          for i in np.arange(len(metric_names))]
            for j in np.arange(ground_truths.shape[0])])
    np.save(arr=regret, file=f'../results/{data}/regret.npy')
    pd.DataFrame(regret, columns=metric_names).to_csv(f'../results/{data}/regret.csv')


def save_results_alpha(data: str = 'ihdp_B', alpha_list: Optional[List[float]] = None) -> None:
    """Save final results."""
    os.makedirs(f'../results/{data}/', exist_ok=True)
    # load true and estimated performance.
    ground_truths = np.load(f'../logs/{data}/ground_truths.npy')
    estimated_metrics = np.load(f'../logs/{data}/metrics_alpha.npy')

    # spearman rank correlation between metric values and true performance.
    corr_alpha = np.array(
        [[spearmanr(ground_truths[j, :], estimated_metrics[j, i, :])[0]
          for i in np.arange(len(alpha_list))]
            for j in np.arange(ground_truths.shape[0])])
    np.save(arr=corr_alpha, file=f'../results/{data}/corr_alpha.npy')
    pd.DataFrame(corr_alpha, columns=alpha_list).to_csv(f'../results/{data}/corr_alpha.csv')
    # regret in model selection.
    regret_alpha = np.array(
        [[(ground_truths[j, np.argmin(estimated_metrics[j, i, :])] / ground_truths[j].min()) - 1
          for i in np.arange(len(alpha_list))]
            for j in np.arange(ground_truths.shape[0])])
    np.save(arr=regret_alpha, file=f'../results/{data}/regret_alpha.npy')
    pd.DataFrame(regret_alpha, columns=alpha_list).to_csv(f'../results/{data}/regret_alpha.csv')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    iters = args.iters
    n_trials = args.n_trials
    alpha_list = args.alpha_list

    # run simulations
    # generate and preprocess semi-synthetic datasets
    generate_data(iters=iters)

    # prediction by meta-learners, parameter estimation by metrics.
    run_preds(alpha_list=alpha_list)
    estimate_metrics_on_val()  # model evaluation by metrics.
    # calculate and save the evaluation performacen of metrics.
    save_results()

    if alpha_list:
        estimate_alpha_metrics_on_val(alpha_list=alpha_list)
        save_results_alpha(alpha_list=alpha_list)

    main_tuner(iters=iters, n_trials=n_trials)  # optuna experiments

    visualizer = Visualizer(iters=iters, alpha_list=alpha_list)
    visualizer.plot_prediction_mse()

    if alpha_list is not None:
        visualizer.plot_rank_alpha()
        visualizer.plot_regret_alpha()
