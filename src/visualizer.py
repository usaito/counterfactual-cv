import os
from itertools import product
from typing import List, Optional

import numpy as np
from plotly.graph_objs import Box, Figure, Layout, Scatter
from plotly.offline import plot

base_models = ['dt', 'rf', 'gbr', 'ridge', 'svm']
meta_models = ['X', 'S', 'T', 'DA', 'DR']
model_names = [b + '-' + a for (a, b) in product(base_models, meta_models)]
metric_names = ['IPW', 'τ-risk', 'Plug-in', 'CF-CV']

colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

fill_colors = ['rgba(31, 119, 180, 0.2)', 'rgba(255, 127, 14, 0.2)',
               'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)',
               'rgba(148, 103, 189, 0.2)', 'rgba(140, 86, 75, 0.2)',
               'rgba(227, 119, 194, 0.2)', 'rgba(127, 127, 127, 0.2)',
               'rgba(188, 189, 34, 0.2)', 'rgba(23, 190, 207, 0.2)']


class Visualizer:
    """Class for result visualization."""

    def __init__(self, data: str = 'ihdp_B', iters: int = 30, alpha_list: Optional[List[float]] = None) -> None:
        """Initialize Class."""
        os.makedirs(f'../plots/{data}/', exist_ok=True)
        self.iters = iters
        self.data = data
        self.alpha_list = alpha_list
        # load true and estimated performance.
        self.ground_truth_mse = np.load(f'../logs/{data}/ground_truths.npy')
        self.corr = np.load(f'../results/{data}/corr.npy')
        self.corr_alpha = np.load(f'../results/{data}/corr_alpha.npy')
        self.regret = np.load(f'../results/{data}/regret.npy')
        self.regret_alpha = np.load(f'../results/{data}/regret_alpha.npy')
        self.optuna_results = np.load(f'../results/{data}/optuna.npy')

    def plot_prediction_mse(self) -> None:
        """Plot ground truth relative root mean_squared_error (RMSE) of 25 meta-learners."""
        box_list = [Box(y=self.ground_truth_mse[:, i], name=name, boxpoints='all', showlegend=True)
                    for i, name in enumerate(model_names)]

        plot(Figure(data=box_list, layout=layout_prediction_mse), auto_open=False,
             filename=f'../plots/{self.data}/ground_truth_mse.html')

    def plot_rank_alpha(self) -> None:
        """Plot spearman rank correlation of CFCV with varying values of alpha."""
        rank_corrs = np.mean(self.corr_alpha, 0)
        upper = rank_corrs + np.std(self.corr_alpha, 0) / np.sqrt(self.iters)
        lower = rank_corrs - np.std(self.corr_alpha, 0) / np.sqrt(self.iters)

        scatter_list = [Scatter(x=self.alpha_list, y=rank_corrs, name='CF-CV',
                                mode='markers+lines', marker=dict(size=25), line=dict(color=colors[0], width=7))]

        scatter_list.append(Scatter(
            x=np.r_[self.alpha_list[::-1], self.alpha_list],
            y=np.r_[upper[::-1], lower], showlegend=False, fill='tozerox', fillcolor=fill_colors[0],
            mode='lines', line=dict(color='rgba(255,255,255,0)')))

        scatter_list.append(Scatter(
            x=self.alpha_list, y=np.mean(
                self.corr[:, 2]) * np.ones_like(self.alpha_list),
            name='Plug-in', mode='markers+lines', marker=dict(size=25),
            line=dict(color=colors[1], width=7)))

        plot(Figure(data=scatter_list, layout=layout_rank_alpha), auto_open=False,
             filename=f'../plots/{self.data}/rank_corr_alpha.html')

    def plot_regret_alpha(self) -> None:
        """Plot Regret in model selection of CFCV with varying values of alpha."""
        regret = np.mean(self.regret_alpha, 0)
        upper = regret + np.std(self.regret_alpha, 0) / np.sqrt(self.iters)
        lower = regret - np.std(self.regret_alpha, 0) / np.sqrt(self.iters)

        scatter_list = [Scatter(x=self.alpha_list, y=regret, name='CF-CV',
                                mode='markers+lines', marker=dict(size=25), line=dict(color=colors[0], width=7))]

        scatter_list.append(Scatter(
            x=np.r_[self.alpha_list[::-1], self.alpha_list],
            y=np.r_[upper[::-1], lower], showlegend=False, fill='tozerox', fillcolor=fill_colors[0],
            mode='lines', line=dict(color='rgba(255,255,255,0)')))

        scatter_list.append(Scatter(
            x=self.alpha_list,
            y=np.mean(self.regret[:, 2], 0) * np.ones_like(self.alpha_list),
            name='Plug-in', mode='markers+lines', marker=dict(size=25),
            line=dict(color=colors[1], width=7)))

        plot(Figure(data=scatter_list, layout=layout_regret_alpha), auto_open=False,
             filename=f'../plots/{self.data}/regret_alpha.html')


layout_prediction_mse = Layout(
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)', width=1500, height=750,
    xaxis=dict(showgrid=False, titlefont=dict(size=30), tickfont=dict(size=15), gridcolor='rgb(255,255,255)'),
    yaxis=dict(title='Ground Truth Normalized RMSE', range=[0, 1.0], titlefont=dict(
        size=30), tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    legend=dict(x=1., y=1., font=dict(size=15)),
    margin=dict(l=80, t=50, b=60))

layout_rank_alpha = Layout(
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)',
    width=1200, height=800,
    xaxis=dict(title='Varying values of alpha (log scaled)', type='log', titlefont=dict(size=30),
               showgrid=False, tickfont=dict(size=22), gridcolor='rgb(255,255,255)'),
    yaxis=dict(title='Spearman Rank Correlation', titlefont=dict(size=35),
               tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(240,240,240)', x=0.01, y=1.01,
                orientation='h', xanchor='left', yanchor='bottom', font=dict(size=32)),
    margin=dict(l=110, t=50, b=80))

layout_regret_alpha = Layout(
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)',
    width=1200, height=800,
    xaxis=dict(title='Varying values of alpha (log scaled)', type='log', titlefont=dict(size=30),
               showgrid=False, tickfont=dict(size=22), gridcolor='rgb(255,255,255)'),
    yaxis=dict(title='Regret in Model Selection',
               titlefont=dict(size=35), tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(240,240,240)', x=0.01, y=1.01,
                orientation='h', xanchor='left', yanchor='bottom', font=dict(size=32)),
    margin=dict(l=110, t=50, b=80))
