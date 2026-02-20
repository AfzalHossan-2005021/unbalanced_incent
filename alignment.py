import os
import ot
import time
import torch
import datetime
import numpy as np

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances

from .transport import fused_gromov_wasserstein_incent
from .utils import to_dense_array, extract_data_matrix
from .distances import jensenshannon_divergence_backend, pairwise_msd, cosine_distance
from .neighborhood import neighborhood_distribution


def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
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
    neighborhood_dissimilarity: str = 'jsd',
    rho1: float = 1.0,
    rho2: float = 1.0,
    balanced_fallback_threshold: float = 1e6,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Unbalanced version of pairwise_align. Marginal constraints are softly enforced
    via KL penalties (rho1 for source, rho2 for target). All cost matrices and the
    GW structure term are unchanged from the balanced version.

    Parameters
    ----------
    rho1 : float, optional
        KL penalty weight for the source marginal (sliceA). Default 1.0.
        Set to >= balanced_fallback_threshold to recover the balanced solution.
    rho2 : float, optional
        KL penalty weight for the target marginal (sliceB). Default 1.0.
        Set to >= balanced_fallback_threshold to recover the balanced solution.
    balanced_fallback_threshold : float, optional
        When both rho1 and rho2 >= this value, exact EMD is used instead. Default 1e6.

    All other parameters are identical to modular_incent.alignment.pairwise_align.
    """

    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")

    logFile.write(f"pairwise_align_INCENT\n")
    currDateTime = datetime.datetime.now()

    # logFile.write(f"{currDateTime.date()}, {currDateTime.strftime("%I:%M %p")} BDT, {currDateTime.strftime("%A")} \n")

    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
   

    logFile.write(f"alpha: {alpha}\n")
    logFile.write(f"beta: {beta}\n")
    logFile.write(f"gamma: {gamma}\n")
    logFile.write(f"radius: {radius}\n")
    logFile.write(f"rho1 (source KL penalty): {rho1}\n")
    logFile.write(f"rho2 (target KL penalty): {rho2}\n")

    
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

    # print the shape of D_A and D_B
    # print("D_A.shape: ", D_A.shape)
    # print("D_B.shape: ", D_B.shape)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()


    # Calculate gene expression dissimilarity
    # filePath = '/content/drive/MyDrive/Thesis_data_anup/local_data'
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = use_rep, use_gpu = use_gpu, nx = nx, beta = beta, overwrite=overwrite)

    M1 = nx.from_numpy(cosine_dist_gene_expr)


    # jensenshannon_divergence_backend actually returns jensen shannon distance
    # neighborhood_distribution_slice_1, neighborhood_distribution_slice_1 will be pre computed

    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius)


        neighborhood_distribution_sliceA += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice A")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy", neighborhood_distribution_sliceA)


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius)


        neighborhood_distribution_sliceB += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice B")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy", neighborhood_distribution_sliceB)


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

            # print("Saving precomputed JSD of neighborhood distribution for slice A and slice B")
            # np.save(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy", js_dist_neighborhood)
  
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
        # M = M.cuda()

        M1 = M1.cuda()
        M2 = M2.cuda()
    
    # init distributions
    if a_distribution is None:
        # uniform distribution, a = array([1/n, 1/n, ...])
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
    else:
        types_A = np.asarray(sliceA.obs['cell_type_annot']).astype(str)
        types_B = np.asarray(sliceB.obs['cell_type_annot']).astype(str)

        match = (types_A[:, None] == types_B[None, :]).astype(np.float64)
        
        G_ct = 1.0 + match
        G_ct = G_ct / G_ct.sum()
        G_init = nx.from_numpy(G_ct)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init = G_init.cuda()


    # added by Anup Bhowmik

    G = np.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

    if neighborhood_dissimilarity == 'jsd':
        initial_obj_neighbor = np.sum(js_dist_neighborhood*G)
    if neighborhood_dissimilarity == 'msd':
        initial_obj_neighbor = np.sum(msd_neighborhood*G)
    elif neighborhood_dissimilarity == 'cosine':
        initial_obj_neighbor = np.sum(cosine_dist_neighborhood*G)

    initial_obj_gene = np.sum(cosine_dist_gene_expr*G)

    if neighborhood_dissimilarity == 'jsd':
        # print(f"Initial objective neighbor (jsd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (jsd): {initial_obj_neighbor}\n")

    elif neighborhood_dissimilarity == 'cosine':
        # print(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor_cos}")
        logFile.write(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'msd':
        # print(f"Initial objective neighbor (msd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (mean sq distance): {initial_obj_neighbor}\n")

    # print(f"Initial objective gene expr (cosine_dist): {initial_obj_gene}")
    logFile.write(f"Initial objective (cosine_dist): {initial_obj_gene}\n")
    

    # D_A: pairwise dist matrix of sliceA spots coords
    # a: initial distribution(uniform) of sliceA spots
    pi, logw = fused_gromov_wasserstein_incent(M1, M2, D_A, D_B, a, b, G_init=G_init, loss_fun='square_loss',
                                               alpha=alpha, gamma=gamma, log=True, numItermax=numItermax,
                                               verbose=verbose, use_gpu=use_gpu,
                                               rho1=rho1, rho2=rho2,
                                               balanced_fallback_threshold=balanced_fallback_threshold)
    pi = nx.to_numpy(pi)
    # obj = nx.to_numpy(logw['fgw_dist'])

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        jsd_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * js_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(msd_neighborhood*pi)

    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        cos_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * cosine_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(cos_error)


    final_obj_gene = np.sum(cosine_dist_gene_expr * pi)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")

    logFile.write(f"Final objective gene expr(cosine_dist): {final_obj_gene}\n")
    # print(f"Final objective (cosine_dist): {final_obj_gene}\n")

    # Log marginal violations (how far the unbalanced plan deviates from strict marginals)
    a_np = nx.to_numpy(a)
    b_np = nx.to_numpy(b)
    source_violation = np.sum(np.abs(pi.sum(axis=1) - a_np))
    target_violation = np.sum(np.abs(pi.sum(axis=0) - b_np))
    logFile.write(f"Source marginal violation (L1): {source_violation}\n")
    logFile.write(f"Target marginal violation (L1): {target_violation}\n")

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    # print(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.write(f"---------------------------------------------\n\n\n")

    logFile.close()

    # new code ends

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi
