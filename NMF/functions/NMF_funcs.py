import os
import numpy as np
import sys
import sklearn
from scipy import stats
from sklearn.decomposition import NMF
import pandas as pd
import random

def get_nnmf(X, rank, init='nndsvda',it=2000):
    """Non-negative matrix factorization, remove zero rows before computation."""
    W = np.zeros((X.shape[0], rank))
    zero_rows = np.where(X.mean(axis=1) == 0)[0]
    nonzero_rows = np.where(X.mean(axis=1) > 0)[0]
    X0 = np.delete(X, zero_rows, 0)

    model = NMF(n_components=rank, init=init, max_iter=it)
    W0 = model.fit_transform(X0)
    H = model.components_
    W[nonzero_rows, :] = W0

    return W, H

def get_nnmf_forced(m, rank, H0, W0, it=500):
    """Non-negative matrix factorization with initial guess, remove zero rows before computation."""
    W = np.zeros((m.shape[0], rank))
    zero_rows = np.where(m.mean(axis=1) == 0)[0]
    nonzero_rows = np.where(m.mean(axis=1) > 0)[0]
    X0 = np.delete(m, zero_rows, 0)
    W0 = np.delete(W0, zero_rows, 0)
    model = NMF(n_components=rank, init='custom', max_iter=it)
    W0 = model.fit_transform(X0, H=H0, W=W0)
    H = model.components_
    W[nonzero_rows, :] = W0
    return W, H

def get_W_corr(Wa, Wb):
    """Construct n by k matrix of Pearson product-moment correlation coefficients for every combination of two columns in A and B"""
    return np.corrcoef(Wa.T, Wb.T)

def max_corr(corr):
    """Get the mean of the absolute maximum correlation coefficient for each row."""
    return np.abs(corr).max(axis=0).mean()

def amariMaxError(correlation):
    """
    Computes what Wu et al. (2016) described as a 'amari-type error'
    based on average distance between factorization solutions.
    """
    maxCol = correlation.abs().max(axis=0)
    colTemp = (1 - maxCol).mean()
    maxRow = correlation.abs().max(axis=1)
    rowTemp = (1 - maxRow).mean()
    return (rowTemp + colTemp) / 2

def stabNMF(M_input, num_it=100, k0=2, k1=10, init='nndsvda',it=2000):
    d = M_input.shape[0]  # number of features
    stability = np.zeros((k1 - k0 + 1,))
    instability = np.zeros((k1 - k0 + 1,))

    for k_num, k in enumerate(range(k0, k1 + 1)):  # for each rank
        print(str(k_num)+'/'+str(k1 - k0 + 1), end="\r")
        # for each rank value
        W_all = np.zeros((num_it, d, k))
        for n in range(num_it):
            W, H = get_nnmf(M_input, k)
            W_all[n, :, :] = W

        for i in range(num_it):
            for j in range(i, num_it):
                x = W_all[i]
                y = W_all[j]
                CORR = get_W_corr(x, y)
                simMat_ij = max_corr(CORR) if i != j else 0  # amariMaxError(CORR) if i == j else max_corr(CORR)
                distMat_ij = amariMaxError(CORR)
                stability[k_num] += simMat_ij / (num_it * (num_it - 1) / 2)
                instability[k_num] += distMat_ij / (num_it * (num_it - 1))

    return stability, instability
