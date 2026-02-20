import os

import ot
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

from .utils import to_dense_array, extract_data_matrix


def kl_divergence_corresponding_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return nx.to_numpy(D)


def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    Returns pairwise Jensenshannon distance (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nx = ot.backend.get_backend(X,Y)        # np or torch depending upon gpu availability
    X = nx.concatenate([X] * Y.shape[0], axis=0) # broadcast X
    X = X/nx.sum(X,axis=1, keepdims=True)   # normalize
    Y = Y/nx.sum(Y,axis=1, keepdims=True)   # normalize
    M = (X + Y) / 2.0
    kl_X_M = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl_Y_M = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist


def jensenshannon_divergence_backend(X, Y):
    """
    This function is added ny Nuwaisir
    
    Returns pairwise JS divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)        
    
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)
        
    print("Finished calculating cost matrix")
    # print(nx.unique(nx.isnan(js_dist)))

    if torch.cuda.is_available():
        try:
            return js_dist.numpy()
        except:
            return js_dist
    else:
        return js_dist


def pairwise_msd(A, B):
    """
    Returns pairwise mean squared distance (over all pairs of samples) of two matrices A and B.
    
    Args:
        A: np array with dim (m_samples by d_features)
        B: np array with dim (n_samples by d_features)
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # A: (m, d), B: (n, d)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    msd = np.mean(diff ** 2, axis=2)  # shape: (m, n)
    return msd


def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep=None, use_gpu=False, nx=ot.backend.NumpyBackend(), beta=0.8, overwrite=False):
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
