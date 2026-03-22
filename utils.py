"""
utils.py — INCENT utility functions
=====================================
Unchanged from original except minor clarity improvements.
All FGW / conditional gradient logic is preserved exactly.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import ot

from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances


# ═════════════════════════════════════════════════════════════════════════════
# Sparse / dense helpers
# ═════════════════════════════════════════════════════════════════════════════

to_dense_array    = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.obsm[rep]


# ═════════════════════════════════════════════════════════════════════════════
# Jensen-Shannon divergence helpers
# ═════════════════════════════════════════════════════════════════════════════

def kl_divergence_corresponding_backend(X, Y):
    """
    Pairwise KL divergence (matching rows) between X and Y.
    Returns a (n,) array where entry i = KL(X[i] || Y[i]).
    """
    assert X.shape == Y.shape
    nx = ot.backend.get_backend(X, Y)
    X  = X / nx.sum(X, axis=1, keepdims=True)
    Y  = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i', X, log_X)
    X_log_Y = nx.einsum('ij,ij->i', X, log_Y)
    return nx.to_numpy(X_log_X - X_log_Y)


def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    JSD between one row X (shape 1×K) and every row of Y (shape m×K).
    Returns a (m,) array.
    """
    assert X.shape[1] == Y.shape[1] and X.shape[0] == 1
    nx  = ot.backend.get_backend(X, Y)
    X   = nx.concatenate([X] * Y.shape[0], axis=0)
    X   = X / nx.sum(X, axis=1, keepdims=True)
    Y   = Y / nx.sum(Y, axis=1, keepdims=True)
    M   = (X + Y) / 2.0
    kl1 = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl2 = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    return nx.sqrt((kl1 + kl2) / 2.0).squeeze()


def jensenshannon_divergence_backend(X, Y):
    """
    Full (n × m) JSD matrix between all rows of X and all rows of Y.
    """
    assert X.shape[1] == Y.shape[1]
    print("Computing JSD cost matrix …")
    nx  = ot.backend.get_backend(X, Y)
    X   = X / nx.sum(X, axis=1, keepdims=True)
    Y   = Y / nx.sum(Y, axis=1, keepdims=True)
    n, m = X.shape[0], Y.shape[0]
    D   = nx.zeros((n, m))
    for i in tqdm(range(n)):
        D[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)
    print("JSD matrix done.")
    if torch.cuda.is_available():
        try:
            return D.numpy()
        except Exception:
            return D
    return D


# ═════════════════════════════════════════════════════════════════════════════
# Mean-squared distance
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_msd(A, B):
    """Pairwise mean-squared distance: (m, n) array."""
    A = np.asarray(A)
    B = np.asarray(B)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (m, n, d)
    return np.mean(diff ** 2, axis=2)                  # (m, n)


def get_neighborhood_distribution(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    # print ("radius", radius)

    unique_cell_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((curr_slice.shape[0], len(unique_cell_types)), dtype=float)

    # print("time taken for cell type", time_cell_type_end-time_cell_type_start)

    source_coords = curr_slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)

    for i in tqdm(range(curr_slice.shape[0])):
        # find the indices of the cells within the radius

        target_indices = np.where(distances[i] <= radius)[0]
        # print("i", i)
        # print(target_indices)

        for ind in target_indices:
            cell_type_str_j = str(curr_slice.obs['cell_type_annot'][ind])
            cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1

    return np.array(cells_within_radius)

def cosine_dist_calculator(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = None, use_gpu = False, nx = ot.backend.NumpyBackend(), beta = 0.8, overwrite = False):
    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    
    one_hot_cell_type_sliceA = pd.get_dummies(sliceA.obs['cell_type_annot'])
    # print ("one_hot_cell_type_sliceA type: ", type(one_hot_cell_type_sliceA))
    one_hot_cell_type_sliceA = one_hot_cell_type_sliceA.to_numpy()

    one_hot_cell_type_sliceB = pd.get_dummies(sliceB.obs['cell_type_annot'])
    one_hot_cell_type_sliceB = one_hot_cell_type_sliceB.to_numpy()

    if isinstance(nx,ot.backend.TorchBackend):
        s_A = s_A.cpu().detach().numpy()
        s_B = s_B.cpu().detach().numpy()

    # Concatenate along a specified axis (0 for rows, 1 for columns)
    s_A = np.concatenate((s_A, beta * one_hot_cell_type_sliceA), axis=1)
    s_B = np.concatenate((s_B, beta * one_hot_cell_type_sliceB), axis=1)

    s_A = torch.from_numpy(s_A)
    s_B = torch.from_numpy(s_B)

    if torch.cuda.is_available():
        print("CUDA is available on your system.")
        s_A = s_A.to('cuda')
        s_B = s_B.to('cuda')

    else:
        print("CUDA is not available on your system.")

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
    else:
        print("Calculating cosine dist of gene expression for slice A and slice B")

        # calculate cosine distance manually
        # cosine_dist_gene_expr = 1 - (s_A @ s_B.T) / s_A.norm(dim=1)[:, None] / s_B.norm(dim=1)[None, :]
        # cosine_dist_gene_expr = cosine_dist_gene_expr.cpu().detach().numpy()

        # use sklearn's cosine_distances
        if torch.cuda.is_available():
            s_A = s_A.cpu().detach().numpy()
            s_B = s_B.cpu().detach().numpy()
        cosine_dist_gene_expr = cosine_distances(s_A, s_B)

        print("Saving cosine dist of gene expression for slice A and slice B")
        np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr

