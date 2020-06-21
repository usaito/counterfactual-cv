import os

import numpy as np

from econml.data.dgps import ihdp_surface_B as ihdp_B
from sklearn.model_selection import train_test_split


def generate_data(data: str = 'ihdp_B', iters: int = 50) -> None:
    """Generate ihdp datasets used in semi-synthetic experiments."""
    names = ['Xtr', 'Xval', 'Xte', 'Ttr', 'Tval', 'Ytr', 'Yval', 'ITEte']
    os.makedirs(f'../data/{data}/', exist_ok=True)
    data_lists = [[] for i in np.arange(len(names))]
    for i in np.arange(iters):
        Y, T, X, ITE = ihdp_B(random_state=i)
        # train/test split
        X, Xte, T, _, Y, _, _, ITEte = train_test_split(X, T, Y, ITE, test_size=0.3, random_state=i)
        # train/val split
        Xtr, Xval, Ttr, Tval, Ytr, Yval = train_test_split(X, T, Y, test_size=0.5, random_state=i)
        _data = (Xtr, Xval, Xte, Ttr, Tval, Ytr, Yval, ITEte)
        for j, _ in enumerate(_data):
            data_lists[j].append(_)
    for j in np.arange(len(_data)):
        if j < 3:
            np.save(arr=np.concatenate(data_lists[j]).reshape((iters, _data[j].shape[0], _data[j].shape[1])),
                    file=f'../data/{data}/{names[j]}.npy')
        else:
            np.save(arr=np.c_[data_lists[j]], file=f'../data/{data}/{names[j]}.npy')
