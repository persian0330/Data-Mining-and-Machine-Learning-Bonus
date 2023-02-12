"""
Created on June 22, 2021
@author: J. Czech
Machine Learning Group, TU Darmstadt
"""
import os.path
import numpy as np


def get_x_down_sampled(filepath: str) -> np.ndarray:
    """Returns a down sampled version of the data given by filepath."""

    fname = 'X_sample.npy'
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            return np.load(f)

    X_data = np.loadtxt(open(filepath, 'r'), delimiter=",", skiprows=0)
    X_sample = X_data[:, ::100]
    np.save(fname, X_sample)
    return X_sample
