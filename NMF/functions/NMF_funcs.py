import os
import numpy as np
import sys
from scipy import stats
from sklearn.decomposition import NMF
import pandas as pd
import random
# import staNMF as st
# from staNMF.nmf_models import spams_nmf
from scipy.stats import entropy


# This function will compute the entropy for each row of the input matrix
def get_entropy(L):
    E = np.array([entropy(row) for row in L])
    A = entropy(E)
    B = np.mean(E)
    return E, A, B

def hier_staNMF(X, k_range, max_clusters, clusters=None, idx=None, parent_entropy=None, cluster_label='',
                threshold=0.1, level=0, stab_it = 20):
    if idx is None:
        idx = np.arange(X.shape[0])

    if clusters is None:
        clusters = []

    # Stop if the number of data points is less than the smallest possible number of clusters or if maximum number of clusters has been reached
    if min(X.shape) < min(k_range) or len(clusters) >= max_clusters:
        clusters.append((cluster_label, X, idx))
        return clusters

    # run stability NMF for different ranks
    _, instability = stabNMF(X, num_it=stab_it, k0=min(k_range), k1=max(k_range), init='nndsvda', it=2000)
    best_k = k_range[np.argmin(instability)]

    # Run NMF with best k
    model = NMF(n_components=best_k, init='random', random_state=0)
    W = model.fit_transform(X)

    best_f_values = []
    new_clusters = []  # New list to keep track of the new clusters
    for i in range(best_k):
        # todo: find better way for cluster assignment
        component_idx = np.argmax(W, axis=1) == i
        best_clusters = X[component_idx]
        new_label = cluster_label + 'X' + str(level + 1) + str(i + 1)  # Append the child label to the parent label

        _, A_cluster, B_cluster = get_entropy(best_clusters)
        f_value = A_cluster - B_cluster

        if parent_entropy is None or f_value < parent_entropy:
            best_f_values.append(f_value)
            new_clusters.append((new_label, best_clusters, idx[component_idx]))

    # Only go to next level of recursion if the conditions based on f and g values are met
    for new_cluster in new_clusters:
        new_label, best_clusters, new_idx = new_cluster

        clusters = hier_staNMF(best_clusters, k_range, max_clusters, clusters, new_idx, f_value, new_label,
                               threshold, level + 1, stab_it)

        if len(best_f_values) == 0:  # New condition to handle empty sub_f_values
            clusters.append(new_cluster)
        else:
            g_value = sum(abs(np.array(best_f_values) - np.array(best_f_values)))
            if g_value > threshold:
                clusters.append(new_cluster)

    return clusters

def recursive_stanmf(X, k_range, max_clusters, clusters=None, idx=None, parent_entropy=None, cluster_label='',
                     threshold=0.1, level=0):
    if idx is None:
        idx = np.arange(X.shape[0])

    if clusters is None:
        clusters = []

    # Stop if the number of data points is less than the smallest possible number of clusters or if maximum number of clusters has been reached
    if X.shape[0] < min(k_range) or len(clusters) >= max_clusters:
        clusters.append((cluster_label, X, idx))
        return clusters

    # Apply stability NMF and compute instability
    folderID = "your_folder_"
    model = st.staNMF(X, folderID=folderID, K1=min(k_range), K2=max(k_range), replicates=20, seed=123)
    model.NMF_finished = True
    model.runNMF(spams_nmf(bootstrap=False))
    model.instability("spams_nmf")
    best_k = k_range[np.argmin(model.get_instability())]

    # Run NMF with best k
    model = NMF(n_components=best_k, init='random', random_state=0)
    W = model.fit_transform(X)

    best_f_values = []
    new_clusters = []  # New list to keep track of the new clusters
    for i in range(best_k):
        # todo: find better way for cluster assignment
        component_idx = np.argmax(W, axis=1) == i
        best_clusters = X[component_idx]
        new_label = cluster_label + 'X' + str(level + 1) + str(i + 1)  # Append the child label to the parent label

        _, A_cluster, B_cluster = get_entropy(best_clusters)
        f_value = A_cluster - B_cluster

        if parent_entropy is None or f_value < parent_entropy:
            best_f_values.append(f_value)
            new_clusters.append((new_label, best_clusters, idx[component_idx]))

    # Only go to next level of recursion if the conditions based on f and g values are met
    for new_cluster in new_clusters:
        new_label, best_clusters, new_idx = new_cluster

        clusters = recursive_stanmf(best_clusters, k_range, max_clusters, clusters, new_idx, f_value, new_label,
                                    threshold, level + 1)

        if len(best_f_values) == 0:  # New condition to handle empty sub_f_values
            clusters.append(new_cluster)
        else:
            g_value = sum(abs(np.array(best_f_values) - np.array(best_f_values)))
            if g_value > threshold:
                clusters.append(new_cluster)

    return clusters


def get_nnmf(X, rank, init='nndsvda', it=2000):
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
    maxCol = np.abs(correlation).max(axis=0)
    colTemp = (1 - maxCol).mean()
    maxRow = np.abs(correlation).max(axis=1)
    rowTemp = (1 - maxRow).mean()
    return (rowTemp + colTemp) / 2


def stabNMF(M_input, num_it=100, k0=2, k1=10, init='nndsvda', it=2000):
    d = M_input.shape[0]  # number of features
    stability = np.zeros((k1 - k0 + 1,))
    instability = np.zeros((k1 - k0 + 1,))

    for k_num, k in enumerate(range(k0, k1 + 1)):  # for each rank
        print(str(k_num) + '/' + str(k1 - k0 + 1), end="\r")
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
