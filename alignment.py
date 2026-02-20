import os
import time
import datetime
import torch
import numpy as np
import ot
from typing import Optional, Tuple, Union
from numpy.typing import NDArray
from anndata import AnnData

from .metrics import cosine_distance, neighborhood_distribution, jensenshannon_divergence_backend, pairwise_msd
from .ot_solvers import (
    fused_gromov_wasserstein_incent_heuristic_cg,
    fused_gromov_wasserstein_incent_exact_bcd,
    fused_gromov_wasserstein_incent_entropic
)

def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    tau: float = 0.1,
    epsilon: float = 0.01,
    solver_type: str = 'exact_bcd',
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None, 
    norm: bool = False, 
    numItermax: int = 6000, 
    backend = ot.backend.NumpyBackend(), 
    use_gpu: bool = False, 
    return_obj: bool = False,
    verbose: bool = False, 
    gpu_verbose: bool = True, 
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite = False,
    neighborhood_dissimilarity: str='jsd',
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET
    Adapted for Unbalanced OT.

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  weight for spatial distance
        gamma: weight for gene expression distance (JSD)
        beta: weight for cell type one hot encoding
        radius: radius for cellular neighborhood
        tau: marginal relaxation penalty for unbalanced OT. Higher tau = closer to balanced.
        epsilon: entropic regularization parameter (only used if solver_type='entropic').
        solver_type: 'exact_bcd' (default), 'entropic', or 'heuristic_cg'.

        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
   
    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:
        
        - Objective function output of cost 
    """

    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")

    logFile.write(f"pairwise_align_INCENT_UNBALANCED\n")
    currDateTime = datetime.datetime.now()

    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
   

    logFile.write(f"alpha: {alpha}\n")
    logFile.write(f"beta: {beta}\n")
    logFile.write(f"gamma: {gamma}\n")
    logFile.write(f"radius: {radius}\n")
    logFile.write(f"tau (unbalanced penalty): {tau}\n")


    
    # Determine if gpu or cpu is being used
    if use_gpu:
        if torch.cuda.is_available():
            nx = ot.backend.TorchBackend()
            if gpu_verbose:
                print("gpu is available, using gpu.")
        else:
            use_gpu = False
            nx = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("gpu is not available, resorting to torch cpu.")
    else:
        nx = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.") 
    
    # Calculate spatial distances
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = sliceB.obsm['spatial'].copy()
    coordinatesB = nx.from_numpy(coordinatesB)
    
    if isinstance(nx,ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA,coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB,coordinatesB, metric='euclidean')

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()


    # Calculate gene expression dissimilarity
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = use_rep, use_gpu = use_gpu, nx = nx, beta = beta, overwrite=overwrite)

    M1 = nx.from_numpy(cosine_dist_gene_expr)


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius)
        neighborhood_distribution_sliceA += 0.01 # for avoiding zero division error


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius)
        neighborhood_distribution_sliceB += 0.01 # for avoiding zero division error


    if ('numpy' in str(type(neighborhood_distribution_sliceA))) and use_gpu:
        neighborhood_distribution_sliceA = torch.from_numpy(neighborhood_distribution_sliceA)
    if ('numpy' in str(type(neighborhood_distribution_sliceB))) and use_gpu:
        neighborhood_distribution_sliceB = torch.from_numpy(neighborhood_distribution_sliceB)

    if use_gpu:
        neighborhood_distribution_sliceA = neighborhood_distribution_sliceA.cuda()
        neighborhood_distribution_sliceB = neighborhood_distribution_sliceB.cuda()

    if neighborhood_dissimilarity == 'jsd':
        if os.path.exists(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy") and not overwrite:
            print("Loading precomputed JSD of neighborhood distribution for slice A and slice B")
            js_dist_neighborhood = np.load(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy")
            
        else:
            print("Calculating JSD of neighborhood distribution for slice A and slice B")

            js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

            if isinstance(js_dist_neighborhood, torch.Tensor):
                js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()
  
        M2 = nx.from_numpy(js_dist_neighborhood)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor) or isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndA = neighborhood_distribution_sliceA
            ndB = neighborhood_distribution_sliceB
            if not isinstance(ndA, torch.Tensor):
                ndA = torch.from_numpy(np.asarray(ndA))
            if not isinstance(ndB, torch.Tensor):
                ndB = torch.from_numpy(np.asarray(ndB))
            if use_gpu:
                ndA = ndA.cuda()
                ndB = ndB.cuda()
            numerator = ndA @ ndB.T
            denom = ndA.norm(dim=1)[:, None] * ndB.norm(dim=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            cosine_dist_neighborhood = cosine_dist_neighborhood.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
            ndB = np.asarray(neighborhood_distribution_sliceB)
            numerator = ndA @ ndB.T
            denom = np.linalg.norm(ndA, axis=1)[:, None] * np.linalg.norm(ndB, axis=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
        M2 = nx.from_numpy(cosine_dist_neighborhood)

    elif neighborhood_dissimilarity == 'msd':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor):
            ndA = neighborhood_distribution_sliceA.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
        if isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndB = neighborhood_distribution_sliceB.detach().cpu().numpy()
        else:
            ndB = np.asarray(neighborhood_distribution_sliceB)

        msd_neighborhood = pairwise_msd(ndA, ndB)
        M2 = nx.from_numpy(msd_neighborhood)

    else:
        raise ValueError(
            "Invalid neighborhood_dissimilarity. Expected one of {'jsd','cosine','msd'}; "
            f"got {neighborhood_dissimilarity!r}."
        )

    
    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        M1 = M1.cuda()
        M2 = M2.cuda()
    
    # init distributions
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()
    
    if norm:
        D_A /= nx.min(D_A[D_A>0])
        D_B /= nx.min(D_B[D_B>0])
    
    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
        if isinstance(nx,ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init = G_init.cuda()


    G = np.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

    if neighborhood_dissimilarity == 'jsd':
        initial_obj_neighbor = np.sum(js_dist_neighborhood*G)
    if neighborhood_dissimilarity == 'msd':
        initial_obj_neighbor = np.sum(msd_neighborhood*G)
    elif neighborhood_dissimilarity == 'cosine':
        initial_obj_neighbor = np.sum(cosine_dist_neighborhood*G)

    initial_obj_gene = np.sum(cosine_dist_gene_expr*G)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Initial objective neighbor (jsd): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'msd':
        logFile.write(f"Initial objective neighbor (mean sq distance): {initial_obj_neighbor}\n")

    logFile.write(f"Initial objective (cosine_dist): {initial_obj_gene}\n")
    

    # D_A: pairwise dist matrix of sliceA spots coords
    # a: initial distribution(uniform) of sliceA spots
    if solver_type == 'heuristic_cg':
        pi, logw = fused_gromov_wasserstein_incent_heuristic_cg(M1, M2, D_A, D_B, a, b, tau=tau, G_init=G_init, loss_fun='square_loss', alpha=alpha, gamma=gamma, log=True, numItermax=numItermax, verbose=verbose, use_gpu=use_gpu)
    elif solver_type == 'entropic':
        pi, logw = fused_gromov_wasserstein_incent_entropic(M1, M2, D_A, D_B, a, b, tau=tau, epsilon=epsilon, G_init=G_init, alpha=alpha, gamma=gamma, log=True, numItermax=numItermax, use_gpu=use_gpu)
    elif solver_type == 'exact_bcd':
        pi, logw = fused_gromov_wasserstein_incent_exact_bcd(M1, M2, D_A, D_B, a, b, tau=tau, G_init=G_init, alpha=alpha, gamma=gamma, log=True, numItermax=numItermax, use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. Choose from 'heuristic_cg', 'entropic', 'exact_bcd'.")
        
    pi = nx.to_numpy(pi)

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        jsd_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * js_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(msd_neighborhood*pi)

    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        cos_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * cosine_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(cos_error)


    final_obj_gene = np.sum(cosine_dist_gene_expr * pi)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")

    logFile.write(f"Final objective gene expr(cosine_dist): {final_obj_gene}\n")
    

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.write(f"---------------------------------------------\n\n\n")

    logFile.close()

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi


def age_progression_score(sliceA, sliceB, data1, data2, filePath):
    '''
    Compute age progression score for each slice and return the score in the obs column of the slice.
    Updated for Unbalanced OT: Normalizes by the actual transported mass per cell instead of 1/N.
    '''

    cosine_dist_gene_expr = np.load(f"{filePath}/cosine_dist_gene_expr_{data1}_{data2}.npy")
    pi_mat = np.load(f"{filePath}/pi_matrix_{data1}_{data2}.npy")

    age_progression_score_mat = pi_mat * cosine_dist_gene_expr

    # Unbalanced normalization: divide by the actual mass transported by each cell
    mass_A = np.sum(pi_mat, axis=1, dtype=np.float64)
    mass_B = np.sum(pi_mat, axis=0, dtype=np.float64)

    # Avoid division by zero for cells that transported no mass
    mass_A[mass_A == 0] = 1.0
    mass_B[mass_B == 0] = 1.0

    sliceA.obs['age_progression_score'] = np.sum(age_progression_score_mat, axis=1, dtype=np.float64) / mass_A * 100
    sliceB.obs['age_progression_score'] = np.sum(age_progression_score_mat, axis=0, dtype=np.float64) / mass_B * 100

    return sliceA, sliceB
