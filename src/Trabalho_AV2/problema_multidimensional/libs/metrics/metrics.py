import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2(y_teste, y_pred):
    SS_res = np.sum((y_teste - y_pred) ** 2)
    SS_tot = np.sum((y_teste - np.mean(y_teste)) ** 2)
    return 1 - (SS_res / SS_tot)