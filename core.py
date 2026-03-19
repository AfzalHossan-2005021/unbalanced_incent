"""
core.py — INCENT  (balanced FGW  +  unbalanced FUGW variant)
=============================================================
Two public functions share one private preprocessing helper:

  pairwise_align()            — original INCENT FGW (balanced)
  pairwise_align_unbalanced() — FUGW via ot.gromov.fused_unbalanced_gromov_wasserstein

Shared fix in both: D_A and D_B are normalised by the SAME scale
(max of D_B) so that GW correctly embeds A as a spatial subregion of B.
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import ot

from typing import Optional, Tuple, Union
from numpy.typing import NDArray
from anndata import AnnData

from .utils import (
    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood distribution
# ─────────────────────────────────────────────────────────────────────────────

def neighborhood_distribution(curr_slice: AnnData, radius: float) -> np.ndarray:
    """
    Normalised cell-type neighbourhood distribution for every cell.

    Parameters
    ----------
    curr_slice : AnnData — .obsm['spatial'], .obs['cell_type_annot'] required
    radius     : float  — Euclidean radius of the local neighbourhood

    Returns
    -------
    dist : (n_cells, n_cell_types) float64, rows sum to 1
    """
    from tqdm import tqdm
    from sklearn.neighbors import BallTree

    cell_types     = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct      = np.unique(cell_types)
    ct2idx         = {c: i for i, c in enumerate(unique_ct)}
    coords         = curr_slice.obsm['spatial']
    n, K           = curr_slice.shape[0], len(unique_ct)

    tree           = BallTree(coords)
    neighbor_lists = tree.query_radius(coords, r=radius)

    dist = np.zeros((n, K), dtype=np.float64)
    for i in tqdm(range(n), desc="Neighbourhood distribution"):
        for idx in neighbor_lists[i]:
            dist[i, ct2idx[cell_types[idx]]] += 1.0

    row_sums = dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance on gene expression
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name,
                    filePath, use_rep=None, use_gpu=False,
                    nx=ot.backend.NumpyBackend(), beta=0.8, overwrite=False):
    """Pairwise cosine distance on gene expression. Results cached to filePath."""
    A_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep)))
    B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"

    if os.path.exists(fileName) and not overwrite:
        print("Loading cached cosine distance matrix")
        mat = np.load(fileName)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            return torch.from_numpy(mat).cuda()
        return mat

    print("Computing cosine distance matrix")
    if isinstance(s_A, torch.Tensor) and isinstance(s_B, torch.Tensor):
        norm_A = s_A / s_A.norm(dim=1, keepdim=True)
        norm_B = s_B / s_B.norm(dim=1, keepdim=True)
        mat    = 1.0 - torch.mm(norm_A, norm_B.T)
        np.save(fileName, mat.cpu().detach().numpy())
        return mat
    else:
        from sklearn.metrics.pairwise import cosine_distances
        mat = cosine_distances(
            to_dense_array(s_A) if not isinstance(s_A, np.ndarray) else s_A,
            to_dense_array(s_B) if not isinstance(s_B, np.ndarray) else s_B,
        )
        np.save(fileName, mat)
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# Helper: bring any matrix to numpy float64
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Private preprocessing helper — shared by both align functions
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(
        sliceA, sliceB,
        alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite,
        neighborhood_dissimilarity,
        logFile,
):
    """
    All preprocessing shared by pairwise_align and pairwise_align_unbalanced.
    Returns a dict with every artefact both callers need.
    """
    # ── GPU / backend ─────────────────────────────────────────────────────────
    if use_gpu:
        if torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            if gpu_verbose:
                print("GPU available — using CUDA.")
        else:
            use_gpu = False
            backend = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("GPU requested but not available — using CPU.")
    else:
        backend = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using CPU backend.")
    nx = backend

    # ── Input validation ───────────────────────────────────────────────────────
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Empty AnnData: {s}")

    # ── Shared genes ───────────────────────────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]

    # ── Shared cell types ──────────────────────────────────────────────────────
    shared_ct = (pd.Index(sliceA.obs['cell_type_annot'])
                 .unique()
                 .intersection(pd.Index(sliceB.obs['cell_type_annot']).unique()))
    if len(shared_ct) == 0:
        raise ValueError("No shared cell types between slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_ct)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_ct)]

    logFile.write(f"n_A={sliceA.shape[0]}  n_B={sliceB.shape[0]}\n")
    logFile.write(f"shared_genes={len(shared_genes)}  shared_ct={len(shared_ct)}\n\n")

    # ── Spatial distance matrices ──────────────────────────────────────────────
    coordsA = nx.from_numpy(sliceA.obsm['spatial'].copy())
    coordsB = nx.from_numpy(sliceB.obsm['spatial'].copy())
    if isinstance(nx, ot.backend.TorchBackend):
        coordsA = coordsA.float()
        coordsB = coordsB.float()

    D_A = ot.dist(coordsA, coordsA, metric='euclidean')
    D_B = ot.dist(coordsB, coordsB, metric='euclidean')

    # ── ★ Shared-scale normalisation (the critical fix) ★ ─────────────────────
    #
    # Both matrices divided by max(D_B).
    #   D_B → [0, 1.0]
    #   D_A → [0, diameter_A / diameter_B]  < 1 for partial slices
    #
    # GW embeds A as a spatial *subregion* of B.
    # Old independent normalisation made both span [0,1] → mixing.
    #
    scale = nx.max(D_B)
    if float(scale) < 1e-12:
        raise ValueError("D_B is all zeros — check spatial coordinates.")

    D_A = D_A / scale
    D_B = D_B / scale

    logFile.write(f"Shared-scale normalisation: scale={float(scale):.4f}\n")
    logFile.write(f"D_A max={float(nx.max(D_A)):.6f}   "
                  f"D_B max={float(nx.max(D_B)):.6f}\n\n")

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # ── Gene-expression cost ───────────────────────────────────────────────────
    cosine_dist_gene_expr = cosine_distance(
        sliceA, sliceB, sliceA_name, sliceB_name, filePath,
        use_rep=use_rep, use_gpu=use_gpu, nx=nx, beta=beta, overwrite=overwrite)

    # ── Cell-type mismatch penalty ─────────────────────────────────────────────
    lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (lab_A[:, None] != lab_B[None, :]).astype(np.float64)

    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_ct = torch.from_numpy(M_celltype).to(cosine_dist_gene_expr.device)
        M1   = (1.0 - beta) * cosine_dist_gene_expr + beta * M_ct
    else:
        M1 = nx.from_numpy(
            (1.0 - beta) * cosine_dist_gene_expr + beta * M_celltype)

    logFile.write(f"M_celltype shape={M_celltype.shape}  beta={beta}\n")

    # ── Neighbourhood distributions ────────────────────────────────────────────
    nd_cache_A = f"{filePath}/nd_{sliceA_name}.npy"
    nd_cache_B = f"{filePath}/nd_{sliceB_name}.npy"

    if os.path.exists(nd_cache_A) and not overwrite:
        print("Loading cached neighbourhood distribution A")
        nd_A = np.load(nd_cache_A)
    else:
        print("Computing neighbourhood distribution A")
        nd_A = neighborhood_distribution(sliceA, radius=radius)
        np.save(nd_cache_A, nd_A)

    if os.path.exists(nd_cache_B) and not overwrite:
        print("Loading cached neighbourhood distribution B")
        nd_B = np.load(nd_cache_B)
    else:
        print("Computing neighbourhood distribution B")
        nd_B = neighborhood_distribution(sliceB, radius=radius)
        np.save(nd_cache_B, nd_B)

    nd_A += 0.01
    nd_B += 0.01

    if use_gpu:
        if isinstance(nd_A, np.ndarray):
            nd_A = torch.from_numpy(nd_A).cuda()
        if isinstance(nd_B, np.ndarray):
            nd_B = torch.from_numpy(nd_B).cuda()

    # ── Neighbourhood dissimilarity M2 ────────────────────────────────────────
    if neighborhood_dissimilarity == 'jsd':
        jsd_cache = f"{filePath}/jsd_{sliceA_name}_{sliceB_name}.npy"
        if os.path.exists(jsd_cache) and not overwrite:
            print("Loading cached JSD matrix")
            js_dist = np.load(jsd_cache)
            M2 = (torch.from_numpy(js_dist).cuda()
                  if use_gpu and isinstance(nx, ot.backend.TorchBackend)
                  else nx.from_numpy(js_dist))
        else:
            print("Computing JSD matrix")
            js_dist = jensenshannon_divergence_backend(nd_A, nd_B)
            if isinstance(js_dist, torch.Tensor):
                np.save(jsd_cache, js_dist.cpu().numpy())
                M2 = js_dist
            else:
                np.save(jsd_cache, js_dist)
                M2 = nx.from_numpy(js_dist)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(nd_A, torch.Tensor):
            na  = nd_A.cuda() if use_gpu else nd_A
            nb  = nd_B.cuda() if use_gpu else nd_B
            num = na @ nb.T
            den = na.norm(dim=1)[:, None] * nb.norm(dim=1)[None, :]
            M2  = 1.0 - num / (den + 1e-12)
        else:
            na  = np.asarray(nd_A)
            nb  = np.asarray(nd_B)
            num = na @ nb.T
            den = (np.linalg.norm(na, axis=1)[:, None]
                   * np.linalg.norm(nb, axis=1)[None, :])
            M2  = nx.from_numpy(1.0 - num / (den + 1e-12))

    elif neighborhood_dissimilarity == 'msd':
        na = nd_A.cpu().numpy() if isinstance(nd_A, torch.Tensor) else np.asarray(nd_A)
        nb = nd_B.cpu().numpy() if isinstance(nd_B, torch.Tensor) else np.asarray(nd_B)
        M2 = nx.from_numpy(pairwise_msd(na, nb))

    else:
        raise ValueError(
            f"neighborhood_dissimilarity must be 'jsd', 'cosine', or 'msd'. "
            f"Got: {neighborhood_dissimilarity!r}")

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        if not isinstance(M1, torch.Tensor):
            M1 = torch.from_numpy(np.asarray(M1)).cuda()
        if not isinstance(M2, torch.Tensor):
            M2 = torch.from_numpy(np.asarray(M2)).cuda()
        M1, M2 = M1.cuda(), M2.cuda()

    # ── Marginals ──────────────────────────────────────────────────────────────
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        a = a.cuda()
        b = b.cuda()

    # ── Initial transport plan ─────────────────────────────────────────────────
    if G_init is not None:
        G_init_t = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init_t = G_init_t.float()
            if use_gpu:
                G_init_t = G_init_t.cuda()
    else:
        G_init_t = None

    return dict(
        nx=nx, use_gpu=use_gpu,
        sliceA=sliceA, sliceB=sliceB,
        D_A=D_A, D_B=D_B,
        M1=M1, M2=M2,
        cosine_dist_gene_expr=cosine_dist_gene_expr,
        a=a, b=b,
        G_init_t=G_init_t,
        nd_dissim=neighborhood_dissimilarity,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 1 — balanced FGW  (original INCENT, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    use_rep:   Optional[str]   = None,
    G_init                     = None,
    a_distribution             = None,
    b_distribution             = None,
    norm:      bool            = False,
    numItermax: int            = 6000,
    backend                    = ot.backend.NumpyBackend(),
    use_gpu:   bool            = False,
    return_obj: bool           = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    **kwargs,
) -> Union[NDArray[np.floating],
           Tuple[NDArray[np.floating], float, float, float, float]]:
    """
    Balanced Fused Gromov-Wasserstein alignment (original INCENT).

    Parameters
    ----------
    alpha  : weight of the GW spatial term  (0 = biology only, 1 = space only)
    beta   : weight of cell-type mismatch inside M1
    gamma  : weight of neighbourhood dissimilarity M2
    radius : neighbourhood radius (same units as spatial coordinates)

    Key fix vs original INCENT
    --------------------------
    D_A and D_B are both normalised by max(D_B), preserving the true size
    relationship so GW embeds A as a spatial subregion of B.
    """
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align — INCENT balanced FGW\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n\n")

    p = _preprocess(
        sliceA, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )
    nx     = p['nx']
    M1     = p['M1']
    M2     = p['M2']
    D_A    = p['D_A']
    D_B    = p['D_B']
    a      = p['a']
    b      = p['b']
    sliceA = p['sliceA']
    sliceB = p['sliceB']

    # Initial objective logging
    G0_np = np.ones((sliceA.shape[0], sliceB.shape[0])) / (
        sliceA.shape[0] * sliceB.shape[0])

    init_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        init_nb = float(np.sum(_to_np(M2) * G0_np))
        logFile.write(f"Initial obj neighbour (jsd): {init_nb:.6f}\n")
    init_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * G0_np))
    logFile.write(f"Initial obj gene (cosine):    {init_gene:.6f}\n\n")

    # ── Solve balanced FGW ────────────────────────────────────────────────────
    pi, logw = fused_gromov_wasserstein_incent(
        M1, M2, D_A, D_B, a, b,
        G_init=p['G_init_t'],
        loss_fun='square_loss',
        alpha=alpha,
        gamma=gamma,
        log=True,
        numItermax=numItermax,
        verbose=verbose,
        use_gpu=p['use_gpu'],
    )
    pi = nx.to_numpy(pi)

    # Final objective logging
    final_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        max_idx  = np.argmax(pi, axis=1)
        jsd_np   = _to_np(M2)
        final_nb = float(sum(pi[i, max_idx[i]] * jsd_np[i, max_idx[i]]
                             for i in range(len(max_idx))))
        logFile.write(f"Final obj neighbour (jsd): {final_nb:.6f}\n")

    final_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * pi))
    logFile.write(f"Final obj gene (cosine):   {final_gene:.6f}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    if p['use_gpu'] and isinstance(nx, ot.backend.TorchBackend):
        torch.cuda.empty_cache()

    if return_obj:
        return pi, init_nb, init_gene, final_nb, final_gene
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 2 — unbalanced FUGW  (new)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align_unbalanced(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    # ── new FUGW parameters ───────────────────────────────────────────────────
    reg_marginals:     float = 1.0,
    epsilon:           float = 0.0,
    divergence:        str   = 'kl',
    unbalanced_solver: str   = 'mm',
    max_iter:          int   = 100,
    tol:               float = 1e-7,
    max_iter_ot:       int   = 500,
    tol_ot:            float = 1e-7,
    # ── identical to pairwise_align ───────────────────────────────────────────
    use_rep:   Optional[str]   = None,
    G_init                     = None,
    a_distribution             = None,
    b_distribution             = None,
    norm:      bool            = False,
    numItermax: int            = 6000,   # kept for API compat (unused by FUGW)
    backend                    = ot.backend.NumpyBackend(),
    use_gpu:   bool            = False,
    return_obj: bool           = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    **kwargs,
) -> Union[NDArray[np.floating],
           Tuple[NDArray[np.floating], float, float]]:
    """
    Unbalanced Fused Gromov-Wasserstein alignment.

    Uses ``ot.gromov.fused_unbalanced_gromov_wasserstein`` as the solver.
    Everything before the solver call is identical to ``pairwise_align``:
    same shared-scale normalisation, same M1 / M2 construction.

    The unbalanced marginal relaxation allows cells with no good counterpart
    in the other slice to remain (partially) unmatched, which naturally
    handles unknown partial overlap without specifying the overlap fraction.

    Parameters shared with pairwise_align
    ----------------------------------------
    alpha  : GW spatial weight [0, 1].
             Converted internally to FUGW's linear-term weight:
               alpha_fugw = (1 - alpha) / alpha
             so the GW/biology ratio is the same as in the balanced version.
             alpha=0.5 → alpha_fugw=1.0 (equal weighting).
    beta   : cell-type mismatch weight inside M1
    gamma  : neighbourhood dissimilarity weight
    radius : neighbourhood radius (same units as spatial coords)

    New FUGW-specific parameters
    --------------------------------
    reg_marginals : float, default 1.0
        KL (or L2) penalty on marginal violations.
        Smaller  → more cells can be "destroyed" → stronger partial-overlap.
        Larger   → approaches the balanced solution.
        Typical range:  0.1 (strongly unbalanced) … 10.0 (nearly balanced).
        Start with 1.0 and lower if the plan mass is close to 1.0
        (meaning the solver is behaving like balanced OT).

    epsilon : float, default 0.0
        Entropic regularisation (Sinkhorn smoothing).
        0.0 uses the MM solver (exact, recommended for small problems).
        > 0 uses Sinkhorn (faster for large problems). Try 0.01–0.1.

    divergence : 'kl' | 'l2', default 'kl'
        Divergence for marginal relaxation and entropic term.

    unbalanced_solver : 'mm' | 'lbfgsb' | 'sinkhorn' | 'sinkhorn_log'
        Inner OT solver.
        'mm'  works for any divergence and epsilon=0 (default, recommended).
        'sinkhorn' requires epsilon > 0 and divergence='kl'.

    max_iter : int, default 100
        BCD outer iterations.

    tol : float, default 1e-7
        BCD convergence tolerance.

    max_iter_ot, tol_ot : inner solver budget per BCD step.

    Returns
    -------
    pi  : (n_A, n_B) float64  — FUGW sample coupling (alignment plan).
          Rows no longer sum to 1/n_A for unmatched cells (mass is "destroyed").
          pi.sum() < 1 indicates partial overlap was detected.

    If return_obj=True: (pi, linear_cost, fugw_cost)
    """
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_ub_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log_ub.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_unbalanced — INCENT FUGW\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    logFile.write(f"reg_marginals={reg_marginals}  epsilon={epsilon}  "
                  f"divergence={divergence}  solver={unbalanced_solver}\n\n")

    # ── All preprocessing identical to pairwise_align ─────────────────────────
    p = _preprocess(
        sliceA, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )

    sliceA = p['sliceA']
    sliceB = p['sliceB']
    M1     = p['M1']
    M2     = p['M2']
    D_A    = p['D_A']
    D_B    = p['D_B']
    a      = p['a']
    b      = p['b']

    # ── Convert to numpy float64 for FUGW ─────────────────────────────────────
    # ot.gromov.fused_unbalanced_gromov_wasserstein accepts any POT-backend
    # array; numpy float64 is always safe and avoids dtype surprises.
    D_A_np = _to_np(D_A)
    D_B_np = _to_np(D_B)
    a_np   = _to_np(a)
    b_np   = _to_np(b)

    # ── Build FUGW linear cost  M_bio = M1 + gamma * M2 ──────────────────────
    #
    # Balanced INCENT objective:
    #   (1-alpha) * [M1 + gamma*M2]  +  alpha * GW(D_A, D_B, pi)
    #
    # FUGW objective (POT convention):
    #   GW(D_A, D_B, pi)  +  alpha_fugw * <M_bio, pi>  +  unbalanced terms
    #
    # Matching the GW/biology ratio gives:
    #   alpha_fugw = (1 - alpha) / alpha
    #
    # Examples:
    #   alpha=0.5  →  alpha_fugw=1.0  (equal weight)
    #   alpha=0.3  →  alpha_fugw=2.33 (biology dominates)
    #   alpha=0.7  →  alpha_fugw=0.43 (space dominates)
    #
    M_bio_np = _to_np(M1) + gamma * _to_np(M2)   # (n_A, n_B) float64

    if alpha < 1e-6:
        alpha_fugw = 1e6        # effectively biology only
    elif alpha > 1.0 - 1e-6:
        alpha_fugw = 0.0        # effectively space only
    else:
        alpha_fugw = (1.0 - alpha) / alpha

    logFile.write(f"alpha → alpha_fugw: {alpha} → {alpha_fugw:.6f}\n")
    logFile.write(f"M_bio range: [{M_bio_np.min():.4f}, {M_bio_np.max():.4f}]\n\n")

    # ── Initial plan ──────────────────────────────────────────────────────────
    init_pi_np = _to_np(p['G_init_t']) if p['G_init_t'] is not None else None

    # ── Solve FUGW ────────────────────────────────────────────────────────────
    #
    # Returns (pi_samp, pi_feat, log_dict) when log=True.
    #   pi_samp  — the sample coupling  ← this is our alignment plan
    #   pi_feat  — second coupling (identical to pi_samp for pure GW; ignore)
    #   log_dict — cost breakdown
    #
    pi_samp, _pi_feat, log_dict = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A_np,
        Cy=D_B_np,
        wx=a_np,
        wy=b_np,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha_fugw,
        M=M_bio_np,
        init_pi=init_pi_np,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=verbose,
    )

    pi = np.asarray(pi_samp, dtype=np.float64)

    # ── Log ───────────────────────────────────────────────────────────────────
    linear_cost = float(log_dict.get('linear_cost', 0.0))
    fugw_cost   = float(log_dict.get('fugw_cost',   0.0))
    pi_mass     = float(pi.sum())

    logFile.write(f"FUGW linear cost: {linear_cost:.6f}\n")
    logFile.write(f"FUGW total cost:  {fugw_cost:.6f}\n")
    logFile.write(f"pi mass:          {pi_mass:.6f}  "
                  f"(< 1.0 = partial overlap detected)\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    if p['use_gpu'] and isinstance(p['nx'], ot.backend.TorchBackend):
        torch.cuda.empty_cache()

    print(f"[FUGW] pi_mass={pi_mass:.4f}  "
          f"linear_cost={linear_cost:.4f}  fugw_cost={fugw_cost:.4f}")

    if return_obj:
        return pi, linear_cost, fugw_cost
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 3 — rigid alignment  (no stretching)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align_rigid(
    sliceA:    AnnData,
    sliceB:    AnnData,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    # ── OT bootstrap parameters ───────────────────────────────────────────────
    alpha_bootstrap:      float = 0.5,
    reg_marginals:        float = 1.0,
    epsilon:              float = 0.0,
    unbalanced_solver:    str   = 'mm',
    max_iter_bootstrap:   int   = 100,
    # ── rigid matching parameters ─────────────────────────────────────────────
    search_radius_k:      float = 4.0,
    soft_temp:            float = 0.0,
    top_frac_svd:         float = 0.15,
    em_iters:             int   = 3,
    # ── standard parameters ───────────────────────────────────────────────────
    use_rep:   Optional[str]   = None,
    a_distribution             = None,
    b_distribution             = None,
    use_gpu:   bool            = False,
    return_obj: bool           = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = False,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    **kwargs,
):
    """
    Rigid alignment — no stretching, no deformation.

    The physical structure of both slices is fully preserved.
    Slice A is mapped into B's coordinate frame by a rigid transform
    (rotation + translation only), then each A cell is matched to its
    most biologically similar B cell within a small spatial radius.

    Algorithm
    ---------
    Step 1  Bootstrap OT
        Run ``pairwise_align_unbalanced`` with a moderate alpha to obtain
        a soft transport plan pi.  This gives approximate correspondences
        without needing any prior knowledge of the coordinate frames.

    Step 2  Infer rigid transform T  (weighted SVD, no deformation)
        Take the top ``top_frac_svd`` highest-weight pairs from pi.
        Fit rotation R and translation t via weighted SVD.
        This is a rigid transform — it cannot stretch or shear.

    Step 3  Proximity-constrained biological matching
        Apply T: coords_A_reg = coords_A @ R.T + t
        For each A cell i, find all B cells j within ``search_radius``
        (auto-estimated from B's cell density × search_radius_k).
        Within that spatial neighborhood, solve a tiny local OT using
        only the biological cost  (1-beta)*cosine + beta*celltype + gamma*JSD.
        Cells with no B neighbor within the radius are unmatched.

    Step 4  EM refinement
        Re-estimate T from the current matching.  Re-run Step 3.
        Repeat ``em_iters`` times.  Typically converges in 2–3 iterations.

    Parameters
    ----------
    beta            : cell-type mismatch weight  [0, 1]
    gamma           : neighbourhood JSD weight   [0, 1]
    radius          : neighbourhood radius for niche distribution
    alpha_bootstrap : GW weight used ONLY in the bootstrap OT step.
                      Does not affect the final rigid matching.
    reg_marginals   : unbalanced marginal penalty for the bootstrap step.
    search_radius_k : final matching search radius = k × median NN dist in B.
                      4.0 (default) captures ~5–15 candidate B cells per A cell.
    soft_temp       : matching softness inside each spatial neighborhood.
                      0.0 = hard argmin (best single match per cell).
                      0.5 = soft assignment proportional to bio similarity.
    top_frac_svd    : fraction of highest-weight pairs used to fit T.
    em_iters        : EM refinement iterations.

    Returns
    -------
    pi              : (n_A, n_B) float32  transport plan
                      Each matched A cell has exactly one nonzero entry per row
                      (hard assignment) or a soft distribution (soft_temp > 0).
                      Unmatched cells have an all-zero row.
    coords_A_reg    : (n_A, 2) float64  A's coordinates in B's physical frame.
                      Always returned as second element when return_obj=True,
                      otherwise available via infer_transform(pi, sliceA, sliceB).
    """
    from sklearn.neighbors import BallTree
    from scipy.linalg import svd as la_svd
    from scipy.spatial.distance import jensenshannon

    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_rigid_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log_rigid.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_rigid — no stretching\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"beta={beta}  gamma={gamma}  radius={radius}\n")
    logFile.write(f"alpha_bootstrap={alpha_bootstrap}  "
                  f"reg_marginals={reg_marginals}\n\n")

    # ── 0. Preprocessing — shared with other align functions ──────────────────
    p = _preprocess(
        sliceA, sliceB,
        alpha_bootstrap, beta, gamma, radius, filePath,
        use_rep, None, a_distribution, b_distribution,
        200, ot.backend.NumpyBackend(), use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )

    sliceA  = p['sliceA']
    sliceB  = p['sliceB']
    n_A     = sliceA.shape[0]
    n_B     = sliceB.shape[0]
    M1      = p['M1']
    M2      = p['M2']
    D_A     = p['D_A']
    D_B     = p['D_B']
    a       = p['a']
    b       = p['b']

    # Raw physical coordinates (not normalised — we need these for rigid fit)
    cA_raw = sliceA.obsm['spatial'].astype(np.float64)
    cB_raw = sliceB.obsm['spatial'].astype(np.float64)

    # ── 1. Bootstrap OT ───────────────────────────────────────────────────────
    print("[Rigid] Step 1: bootstrap OT …")

    M_bio_np  = _to_np(M1) + gamma * _to_np(M2)
    D_A_np    = _to_np(D_A)
    D_B_np    = _to_np(D_B)
    a_np      = _to_np(a)
    b_np      = _to_np(b)

    if alpha_bootstrap < 1e-6:
        alpha_fugw = 1e6
    elif alpha_bootstrap > 1.0 - 1e-6:
        alpha_fugw = 0.0
    else:
        alpha_fugw = (1.0 - alpha_bootstrap) / alpha_bootstrap

    pi_boot, _pif, log_boot = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A_np, Cy=D_B_np,
        wx=a_np,   wy=b_np,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence='kl',
        unbalanced_solver=unbalanced_solver,
        alpha=alpha_fugw,
        M=M_bio_np,
        max_iter=max_iter_bootstrap,
        tol=1e-7,
        max_iter_ot=300,
        tol_ot=1e-7,
        log=True,
        verbose=verbose,
    )
    pi_boot = np.asarray(pi_boot, dtype=np.float64)
    logFile.write(f"Bootstrap pi mass: {pi_boot.sum():.4f}\n")
    print(f"[Rigid] Bootstrap pi mass: {pi_boot.sum():.4f}")

    # ── 2. Infer rigid transform T from bootstrap plan ────────────────────────
    #
    # Uses the top top_frac_svd highest-weight correspondences.
    # Weighted SVD gives the rotation R and translation t
    # that best maps cA to cB in a least-squares sense.
    # This is a RIGID transform — no scale, no shear, no stretch.
    #
    def fit_rigid(pi, cA, cB, top_frac):
        flat    = pi.flatten()
        n_top   = max(30, int(top_frac * (pi > 0).sum()))
        n_top   = min(n_top, len(flat))
        top_idx = np.argsort(flat)[-n_top:]
        rows    = top_idx // pi.shape[1]
        cols    = top_idx %  pi.shape[1]
        w       = flat[top_idx].astype(np.float64)
        w      /= w.sum() + 1e-12

        pA      = cA[rows]
        pB      = cB[cols]
        cA_bar  = (w[:, None] * pA).sum(0)
        cB_bar  = (w[:, None] * pB).sum(0)

        H       = (pA - cA_bar).T @ np.diag(w) @ (pB - cB_bar)
        U, _, Vt = la_svd(H)
        R       = Vt.T @ U.T
        if np.linalg.det(R) < 0:        # fix improper rotation
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cB_bar - R @ cA_bar
        return R, t

    R, t = fit_rigid(pi_boot, cA_raw, cB_raw, top_frac_svd)
    coords_A_reg = cA_raw @ R.T + t     # rigid transform — no stretching

    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    logFile.write(f"\nRigid transform: angle={angle:.2f}°  "
                  f"t=({t[0]:.2f}, {t[1]:.2f})\n")
    print(f"[Rigid] T: angle={angle:.1f}°  t=({t[0]:.1f}, {t[1]:.1f})")

    # ── 3. Precompute biological cost matrices (numpy, cell-level) ────────────
    cos_np  = _to_np(p['cosine_dist_gene_expr'])   # (n_A, n_B)
    ct_A    = np.asarray(sliceA.obs['cell_type_annot'].values)
    ct_B    = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_ct_np = (ct_A[:, None] != ct_B[None, :]).astype(np.float64)

    # Load neighbourhood distributions from cache (already computed + saved above)
    nd_cache_A = f"{filePath}/nd_{sliceA_name}.npy"
    nd_cache_B = f"{filePath}/nd_{sliceB_name}.npy"
    nd_A = np.load(nd_cache_A).astype(np.float64) + 0.01
    nd_B = np.load(nd_cache_B).astype(np.float64) + 0.01

    # ── 4. Auto search radius ─────────────────────────────────────────────────
    tree_B_tmp = BallTree(cB_raw)
    d_nn, _    = tree_B_tmp.query(cB_raw, k=2)
    search_radius = float(np.median(d_nn[:, 1]) * search_radius_k)
    logFile.write(f"Search radius: {search_radius:.2f} "
                  f"({search_radius_k}× median NN dist in B)\n")
    print(f"[Rigid] Search radius: {search_radius:.1f}")

    # ── 5. Proximity-constrained biological matching + EM ─────────────────────
    def local_match(coords_A_reg, cB_raw, search_radius):
        """
        For each A cell find B candidates within search_radius.
        Assign using bio cost only. No GW, no stretching.
        """
        tree_B  = BallTree(cB_raw)
        pi_out  = np.zeros((n_A, n_B), dtype=np.float32)
        n_unmatched = 0

        for i in range(n_A):
            cands = tree_B.query_radius([coords_A_reg[i]], r=search_radius)[0]
            if len(cands) == 0:
                n_unmatched += 1
                continue

            # Local biological cost: gene cosine + cell-type mismatch + JSD niche
            cost = ((1.0 - beta) * cos_np[i, cands]
                    + beta       * M_ct_np[i, cands]
                    + gamma      * np.array([
                        float(jensenshannon(nd_A[i], nd_B[j]))
                        for j in cands]))

            if soft_temp < 1e-6:
                # Hard assignment — no stretching, single best match per cell
                pi_out[i, cands[np.argmin(cost)]] = 1.0 / n_A
            else:
                # Soft — proportional to biological similarity within radius
                span  = cost.max() - cost.min() + 1e-12
                tau   = soft_temp * span + 1e-12
                logw  = -cost / tau
                logw -= logw.max()
                w     = np.exp(logw)
                w    /= w.sum()
                for m, j in enumerate(cands):
                    pi_out[i, j] = float(w[m]) / n_A

        return pi_out, n_unmatched

    pi, n_unmatched = local_match(coords_A_reg, cB_raw, search_radius)
    logFile.write(f"Unmatched A cells: {n_unmatched}/{n_A}\n")
    print(f"[Rigid] Step 3: matched={n_A - n_unmatched}/{n_A}  "
          f"unmatched={n_unmatched}")

    # ── 6. EM refinement ──────────────────────────────────────────────────────
    print(f"[Rigid] Step 4: EM refinement ({em_iters} iters) …")
    for em in range(em_iters):
        R_new, t_new      = fit_rigid(pi, cA_raw, cB_raw, top_frac_svd)
        coords_A_reg_new  = cA_raw @ R_new.T + t_new
        delta_t           = float(np.linalg.norm(t_new - t))
        R, t, coords_A_reg = R_new, t_new, coords_A_reg_new
        logFile.write(f"EM {em+1}: |Δt|={delta_t:.4f}\n")
        print(f"         EM iter {em+1}: |Δt|={delta_t:.3f}")
        pi, n_unmatched = local_match(coords_A_reg, cB_raw, search_radius)
        if delta_t < 0.5:
            break

    logFile.write(f"Final unmatched: {n_unmatched}/{n_A}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    print(f"[Rigid] Done in {time.time()-start:.1f}s  "
          f"(matched {n_A - n_unmatched}/{n_A})")

    if return_obj:
        return pi, coords_A_reg
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 4 — FDesc-RANSAC (feature-descriptor RANSAC alignment)
#
# Solves the bidirectional partial overlap problem with no GW, no stretching.
# Works when neither slice fully contains the other.
# ═════════════════════════════════════════════════════════════════════════════

def _fit_rigid_from_pairs(p_src, p_dst, weights=None):
    """
    Fit rotation R and translation t such that p_dst ≈ R @ p_src.T + t.
    Uses weighted SVD. Enforces det(R) = +1 (proper rotation, no reflection).

    Parameters
    ----------
    p_src, p_dst : (N, 2) float64
    weights      : (N,) float64 or None (uniform)

    Returns R (2,2), t (2,)
    """
    from scipy.linalg import svd as la_svd
    N = len(p_src)
    if weights is None:
        weights = np.ones(N, dtype=np.float64) / N
    else:
        weights = weights / (weights.sum() + 1e-12)

    cA = (weights[:, None] * p_src).sum(0)
    cB = (weights[:, None] * p_dst).sum(0)
    H  = (p_src - cA).T @ np.diag(weights) @ (p_dst - cB)
    U, _, Vt = la_svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def _fit_rigid_2pt(p1_src, p2_src, p1_dst, p2_dst):
    """
    Fit rigid transform from exactly 2 point correspondences.
    Returns TWO solutions: (R_proper, t_proper) and (R_reflect, t_reflect).
    One has det(R)=+1, the other det(R)=-1.
    """
    dp = p2_src - p1_src
    dq = p2_dst - p1_dst

    dp_len = np.linalg.norm(dp) + 1e-12
    dq_len = np.linalg.norm(dq) + 1e-12

    # Proper rotation (det=+1)
    cos_t = np.dot(dp, dq) / (dp_len * dq_len)
    sin_t = (dp[0]*dq[1] - dp[1]*dq[0]) / (dp_len * dq_len)
    cos_t = np.clip(cos_t, -1, 1)
    R_proper = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    t_proper = p1_dst - R_proper @ p1_src

    # Reflection (det=-1): flip y axis then rotate
    R_reflect = np.array([[cos_t, sin_t], [sin_t, -cos_t]])
    t_reflect = p1_dst - R_reflect @ p1_src

    return (R_proper, t_proper), (R_reflect, t_reflect)


def _build_putative_correspondences(nd_S, nd_L,
                                     k=10,
                                     n_anchor=2000,
                                     jsd_percentile=60,
                                     seed=42):
    """
    Build putative (i, j) feature correspondences by k-NN in JSD space.

    For each anchor cell i in S, find its k nearest neighbours in L by JSD
    on neighbourhood distributions. Filter to pairs below the jsd_percentile
    threshold to remove obviously bad matches.

    Parameters
    ----------
    nd_S, nd_L   : (n_S, K) and (n_L, K) neighbourhood distributions
    k            : number of neighbours per anchor cell
    n_anchor     : max number of S cells used as query anchors
                   (subsampled if n_S > n_anchor, prioritising low min-JSD cells)
    jsd_percentile: filter threshold — only keep pairs in this percentile of JSD

    Returns
    -------
    pairs   : list of (i, j) index tuples
    jsds    : list of corresponding JSD values
    """
    from scipy.spatial.distance import jensenshannon

    n_S = nd_S.shape[0]
    n_L = nd_L.shape[0]
    rng = np.random.default_rng(seed)

    # ── Choose anchor cells: subsample if large ────────────────────────────────
    if n_S > n_anchor:
        # Prefer cells whose best JSD match in L is small (likely overlapping)
        # Sample min-JSD by checking a random subset of L
        n_probe = min(500, n_L)
        probe_idx = rng.choice(n_L, n_probe, replace=False)
        nd_L_probe = nd_L[probe_idx]

        min_jsd = np.full(n_S, np.inf)
        for j_probe in range(n_probe):
            jsd_row = np.array([
                jensenshannon(nd_S[i] + 1e-9, nd_L_probe[j_probe] + 1e-9)
                for i in range(n_S)])
            min_jsd = np.minimum(min_jsd, jsd_row)

        # Take the n_anchor cells with lowest min-JSD
        anchor_idx = np.argsort(min_jsd)[:n_anchor]
    else:
        anchor_idx = np.arange(n_S)

    # ── k-NN for each anchor ───────────────────────────────────────────────────
    # Build full JSD matrix for anchor cells vs all L cells
    print(f"  Computing JSD for {len(anchor_idx)} anchor cells × {n_L} L cells …")
    pairs, jsds = [], []
    for i in anchor_idx:
        row_jsd = np.array([
            jensenshannon(nd_S[i] + 1e-9, nd_L[j] + 1e-9)
            for j in range(n_L)])
        nn_idx = np.argsort(row_jsd)[:k]
        for j in nn_idx:
            pairs.append((int(i), int(j)))
            jsds.append(float(row_jsd[j]))

    # ── Filter by JSD threshold ────────────────────────────────────────────────
    if len(jsds) == 0:
        return [], []
    thresh = float(np.percentile(jsds, jsd_percentile))
    filtered = [(p, d) for p, d in zip(pairs, jsds) if d <= thresh]
    if not filtered:
        filtered = list(zip(pairs, jsds))
    pairs_f, jsds_f = zip(*filtered)
    print(f"  Putative pairs: {len(pairs_f)}  (JSD threshold={thresh:.4f})")
    return list(pairs_f), list(jsds_f)


def _ransac_rigid(pairs, jsds, coord_S, coord_L,
                   n_iter=2000,
                   spatial_thresh=None,
                   seed=42):
    """
    RANSAC to find the rigid transform T aligning coord_S into coord_L.

    Parameters
    ----------
    pairs        : list of (i, j) putative correspondence indices
    jsds         : corresponding JSD values (used to weight final refit)
    coord_S      : (n_S, 2) physical coords of S
    coord_L      : (n_L, 2) physical coords of L'
    n_iter       : RANSAC iterations
    spatial_thresh: inlier threshold in physical units (auto if None)

    Returns
    -------
    R, t         : best rigid transform
    inlier_pairs : list of (i,j) inlier correspondences
    inlier_frac  : float
    """
    if len(pairs) < 4:
        raise RuntimeError("Too few putative correspondences for RANSAC. "
                           "Lower k or jsd_percentile threshold.")

    pairs_arr = np.array(pairs, dtype=int)       # (N_pairs, 2)
    jsds_arr  = np.array(jsds,  dtype=np.float64)

    # Auto spatial threshold: 3× median NN distance estimated from the pairs
    if spatial_thresh is None:
        coord_S_pairs = coord_S[pairs_arr[:, 0]]
        coord_L_pairs = coord_L[pairs_arr[:, 1]]
        # Rough estimate of cell spacing from pairwise partner distances
        spacing = np.median(np.linalg.norm(coord_S_pairs[:200] - coord_L_pairs[:200], axis=1))
        spatial_thresh = max(spacing * 5,
                             np.median(np.linalg.norm(np.diff(coord_S[:200], axis=0), axis=1)) * 3)
    print(f"  RANSAC spatial threshold: {spatial_thresh:.2f}")

    rng = np.random.default_rng(seed)
    N   = len(pairs)
    best_n_inliers = -1
    best_R, best_t = np.eye(2), np.zeros(2)
    best_inliers   = []

    # Pre-extract coords for speed
    pS = coord_S[pairs_arr[:, 0]]   # (N, 2)
    pL = coord_L[pairs_arr[:, 1]]   # (N, 2)

    for _ in range(n_iter):
        # Minimal sample: 2 pairs
        idx = rng.choice(N, size=2, replace=False)
        (R_pr, t_pr), (R_rf, t_rf) = _fit_rigid_2pt(
            pS[idx[0]], pS[idx[1]],
            pL[idx[0]], pL[idx[1]])

        for R_cand, t_cand in [(R_pr, t_pr), (R_rf, t_rf)]:
            pS_T    = pS @ R_cand.T + t_cand     # (N, 2)
            resid   = np.linalg.norm(pS_T - pL, axis=1)
            inliers = np.where(resid < spatial_thresh)[0]
            if len(inliers) > best_n_inliers:
                best_n_inliers = len(inliers)
                best_R, best_t = R_cand, t_cand
                best_inliers   = inliers.tolist()

    # Final refit from all inliers, weighted by 1/JSD
    if len(best_inliers) >= 4:
        inl    = np.array(best_inliers)
        w      = 1.0 / (jsds_arr[inl] + 1e-6)
        best_R, best_t = _fit_rigid_from_pairs(pS[inl], pL[inl], weights=w)

    inlier_pairs = [pairs[k] for k in best_inliers]
    inlier_frac  = len(best_inliers) / max(len(pairs), 1)
    angle        = float(np.degrees(np.arctan2(best_R[1, 0], best_R[0, 0])))

    print(f"  RANSAC: {len(best_inliers)}/{N} inliers ({100*inlier_frac:.0f}%)  "
          f"angle={angle:.1f}°  t=({best_t[0]:.1f},{best_t[1]:.1f})")

    return best_R, best_t, inlier_pairs, inlier_frac


def pairwise_align_fdesc_ransac(
    sliceA:    AnnData,
    sliceB:    AnnData,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    # ── feature matching parameters ───────────────────────────────────────────
    k_nn:             int   = 10,
    n_anchor:         int   = 2000,
    jsd_percentile:   float = 60.0,
    n_ransac:         int   = 2000,
    spatial_thresh:   float = None,
    # ── matching parameters ───────────────────────────────────────────────────
    search_radius_k:  float = 4.0,
    soft_temp:        float = 0.0,
    em_iters:         int   = 3,
    top_frac_svd:     float = 0.20,
    # ── standard parameters ───────────────────────────────────────────────────
    use_rep:   Optional[str]   = None,
    a_distribution             = None,
    b_distribution             = None,
    use_gpu:   bool            = False,
    return_extra: bool         = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = False,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    **kwargs,
):
    """
    FDesc-RANSAC alignment — solves the bidirectional partial overlap problem.

    Neither slice needs to fully contain the other. Cells unique to sliceA
    (outside sliceB's coverage) and cells unique to sliceB (outside sliceA's
    coverage) are both correctly identified and left unmatched.

    No GW term. No stretching. The physical structure of both slices is
    preserved exactly.

    Algorithm
    ---------
    1. Build putative correspondences using k-NN in JSD neighbourhood space.
       Cells at the same anatomical location have similar neighbourhood
       distributions regardless of which slice they are in.

    2. RANSAC: find the rigid transform T (rotation + translation) that
       maximises the number of spatially-consistent putative pairs.
       The minimal sample is 2 pairs → works even with 50% outliers (bilateral
       symmetry) because the correct T has many more inliers than any wrong T.

    3. Apply T: coord_A_reg = coord_A @ R.T + t  (rigid, no stretching)

    4. Bidirectional overlap detection:
       - sliceA cell with no sliceB neighbour within radius → unique to A
       - sliceB cell with no sliceA neighbour within radius → unique to B

    5. Local biological matching within radius (overlapping cells only).

    6. EM refinement: re-estimate T from matching, repeat steps 4–5.

    Parameters
    ----------
    beta            : cell-type mismatch weight [0,1]
    gamma           : neighbourhood JSD weight [0,1]
    radius          : neighbourhood radius for niche distributions
    k_nn            : neighbours per anchor in JSD space (default 10)
    n_anchor        : max anchor cells from A for putative matches (default 2000)
    jsd_percentile  : filter threshold for putative pairs (default 60)
    n_ransac        : RANSAC iterations (default 2000)
    spatial_thresh  : RANSAC inlier threshold in physical units (auto if None)
    search_radius_k : final matching radius = k × median NN dist in B (default 4)
    soft_temp       : 0 = hard single best match; >0 = soft assignment
    em_iters        : EM refinement iterations (default 3)
    top_frac_svd    : fraction of best pairs used for SVD refit (default 0.20)

    Returns
    -------
    pi              : (n_A, n_B) float32  transport plan
                      Unmatched cells have all-zero rows.
    If return_extra=True: (pi, coord_A_reg, R, t, overlap_mask_A, overlap_mask_B)
    """
    from sklearn.neighbors import BallTree
    from scipy.spatial.distance import jensenshannon

    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_fdesc_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log_fdesc.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_fdesc_ransac\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"beta={beta}  gamma={gamma}  radius={radius}\n")
    logFile.write(f"k_nn={k_nn}  n_anchor={n_anchor}  "
                  f"jsd_percentile={jsd_percentile}  n_ransac={n_ransac}\n\n")

    # ── 0. Preprocessing (shared genes, cell types, bio costs) ────────────────
    # We call _preprocess with alpha=0 since we do not use D_A/D_B here.
    # The preprocessing still builds M1, M2, nd caches — all needed downstream.
    p = _preprocess(
        sliceA, sliceB,
        0.0, beta, gamma, radius, filePath,
        use_rep, None, a_distribution, b_distribution,
        200, ot.backend.NumpyBackend(), use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )

    sliceA  = p['sliceA']
    sliceB  = p['sliceB']
    n_A     = sliceA.shape[0]
    n_B     = sliceB.shape[0]

    # Physical coordinates — never normalised, always in original units
    cA_raw  = sliceA.obsm['spatial'].astype(np.float64)
    cB_raw  = sliceB.obsm['spatial'].astype(np.float64)

    # Biological cost matrices (numpy)
    cos_np  = _to_np(p['cosine_dist_gene_expr'])
    ct_A    = np.asarray(sliceA.obs['cell_type_annot'].values)
    ct_B    = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_ct_np = (ct_A[:, None] != ct_B[None, :]).astype(np.float64)

    # Neighbourhood distributions from cache
    nd_cache_A = f"{filePath}/nd_{sliceA_name}.npy"
    nd_cache_B = f"{filePath}/nd_{sliceB_name}.npy"
    nd_A = np.load(nd_cache_A).astype(np.float64) + 0.01
    nd_B = np.load(nd_cache_B).astype(np.float64) + 0.01

    # ── 1. Build putative correspondences ─────────────────────────────────────
    print(f"[FDesc] Step 1: building putative correspondences …")
    pairs, jsds = _build_putative_correspondences(
        nd_A, nd_B, k=k_nn, n_anchor=n_anchor,
        jsd_percentile=jsd_percentile)

    if len(pairs) < 10:
        raise RuntimeError(
            f"Only {len(pairs)} putative correspondences found. "
            "Increase k_nn or jsd_percentile, or check that slices share cell types.")

    logFile.write(f"Putative pairs: {len(pairs)}\n")

    # ── 2. RANSAC ────────────────────────────────────────────────────────────
    print(f"[FDesc] Step 2: RANSAC ({n_ransac} iters) …")
    R, t, inlier_pairs, inlier_frac = _ransac_rigid(
        pairs, jsds, cA_raw, cB_raw,
        n_iter=n_ransac,
        spatial_thresh=spatial_thresh)

    logFile.write(f"RANSAC inliers: {len(inlier_pairs)}/{len(pairs)} "
                  f"({100*inlier_frac:.0f}%)\n")
    logFile.write(f"Rigid T: angle={float(np.degrees(np.arctan2(R[1,0], R[0,0]))):.2f}°  "
                  f"t=({t[0]:.2f},{t[1]:.2f})\n\n")

    if inlier_frac < 0.05:
        logFile.write("WARNING: very low inlier fraction. "
                      "Slices may have very little overlap.\n")
        print("[FDesc] WARNING: low RANSAC inliers — check slice overlap.")

    # ── 3. Apply rigid transform ───────────────────────────────────────────────
    coord_A_reg = cA_raw @ R.T + t     # rigid only — det(R)=+1, no stretch

    # ── 4. Auto search radius for matching ───────────────────────────────────
    tree_B_tmp   = BallTree(cB_raw)
    d_nn, _      = tree_B_tmp.query(cB_raw, k=2)
    search_radius = float(np.median(d_nn[:, 1]) * search_radius_k)
    logFile.write(f"Search radius: {search_radius:.2f}  "
                  f"({search_radius_k}× median NN dist in B)\n")
    print(f"[FDesc] Search radius: {search_radius:.1f}")

    # ── 5+6. Local matching + EM ──────────────────────────────────────────────
    def local_match_bidir(coord_A_reg, cB_raw, search_radius):
        """
        Bidirectional local biological matching.
        - A cells with no B neighbor → unique to A (unmatched, zero row in pi)
        - B cells with no A neighbor → unique to B (ignored in pi)
        Returns pi (n_A, n_B) and overlap masks for both slices.
        """
        tree_B  = BallTree(cB_raw)
        tree_A  = BallTree(coord_A_reg)

        pi_out         = np.zeros((n_A, n_B), dtype=np.float32)
        overlap_mask_A = np.zeros(n_A, dtype=bool)
        overlap_mask_B = np.zeros(n_B, dtype=bool)
        n_unmatched_A  = 0

        for i in range(n_A):
            cands = tree_B.query_radius([coord_A_reg[i]], r=search_radius)[0]
            if len(cands) == 0:
                n_unmatched_A += 1
                continue

            overlap_mask_A[i] = True
            for j in cands:
                overlap_mask_B[j] = True

            # Biological cost within spatial neighbourhood only
            cost = ((1.0 - beta) * cos_np[i, cands]
                    + beta        * M_ct_np[i, cands]
                    + gamma       * np.array([
                        float(jensenshannon(nd_A[i], nd_B[j]))
                        for j in cands]))

            if soft_temp < 1e-6:
                # Hard: best single biological match within radius
                pi_out[i, cands[np.argmin(cost)]] = 1.0 / n_A
            else:
                span  = cost.max() - cost.min() + 1e-12
                tau   = soft_temp * span + 1e-12
                logw  = -cost / tau
                logw -= logw.max()
                w     = np.exp(logw)
                w    /= w.sum()
                for m, j in enumerate(cands):
                    pi_out[i, j] = float(w[m]) / n_A

        return pi_out, overlap_mask_A, overlap_mask_B, n_unmatched_A

    print(f"[FDesc] Step 3+4: local matching + EM ({em_iters} iters) …")
    pi, om_A, om_B, n_unm = local_match_bidir(coord_A_reg, cB_raw, search_radius)
    logFile.write(f"Initial: A unmatched={n_unm}/{n_A}  "
                  f"B unique={int((~om_B).sum())}/{n_B}\n")
    print(f"[FDesc] Matched A: {n_A - n_unm}/{n_A}  "
          f"Unique to B: {int((~om_B).sum())}/{n_B}")

    for em in range(em_iters):
        # Re-estimate T from high-confidence matched pairs
        flat     = pi.flatten()
        n_top    = max(30, int(top_frac_svd * (pi > 0).sum()))
        top_idx  = np.argsort(flat)[-n_top:]
        rows_em  = top_idx // n_B
        cols_em  = top_idx %  n_B
        w_em     = flat[top_idx]
        if w_em.sum() < 1e-12:
            break
        w_em    /= w_em.sum()

        R_new, t_new   = _fit_rigid_from_pairs(
            cA_raw[rows_em], cB_raw[cols_em], weights=w_em)
        coord_A_reg_new = cA_raw @ R_new.T + t_new
        delta_t         = float(np.linalg.norm(t_new - t))

        R, t, coord_A_reg = R_new, t_new, coord_A_reg_new
        logFile.write(f"EM {em+1}: |Δt|={delta_t:.4f}  "
                      f"angle={float(np.degrees(np.arctan2(R[1,0], R[0,0]))):.2f}°\n")
        print(f"[FDesc] EM {em+1}: |Δt|={delta_t:.2f}")

        pi, om_A, om_B, n_unm = local_match_bidir(
            coord_A_reg, cB_raw, search_radius)
        if delta_t < 0.5:
            break

    logFile.write(f"\nFinal: A unmatched={n_unm}/{n_A}  "
                  f"B unique={int((~om_B).sum())}/{n_B}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    print(f"[FDesc] Done in {time.time()-start:.1f}s  |  "
          f"A matched={n_A-n_unm}/{n_A}  B unique={int((~om_B).sum())}/{n_B}")

    if return_extra:
        return pi, coord_A_reg, R, t, om_A, om_B
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# CAST descriptor — Coarse Architecture Spatial Topology
# Expression-invariant, fine-type-invariant spatial feature descriptor.
# Preserved across time points because it captures tissue geometry, not state.
# ═════════════════════════════════════════════════════════════════════════════

def _auto_coarse_types(cell_types, n_coarse=6, seed=42):
    """
    When no coarse type mapping is provided, automatically group fine-grained
    cell types into n_coarse major classes using k-means on cell type names
    encoded as one-hot frequency vectors.

    In practice the user should provide a domain-informed mapping
    (e.g. {'L2/3 ExN': 'ExN', 'L5 ExN': 'ExN', 'Astro': 'Glia', ...}).
    This function is a safe fallback when no mapping is given.

    Parameters
    ----------
    cell_types : (n,) array of fine-grained cell type strings
    n_coarse   : number of coarse groups

    Returns
    -------
    mapping : dict {fine_type -> coarse_label}
    """
    from sklearn.cluster import KMeans

    unique_fine = np.unique(cell_types)
    n_fine      = len(unique_fine)

    if n_fine <= n_coarse:
        # Already coarse enough — identity mapping
        return {ct: ct for ct in unique_fine}

    # Encode each fine type by its co-occurrence frequency with others
    # (which fine types appear in the same cells' neighbourhoods)
    # Simple fallback: cluster by string similarity via character n-gram
    # For biology: cluster by frequency of the first word (major class name)
    def first_word(s):
        return s.split()[0].split('_')[0].split('-')[0]

    prefix_map = {}
    for ct in unique_fine:
        p = first_word(ct)
        if p not in prefix_map:
            prefix_map[p] = []
        prefix_map[p].append(ct)

    if len(prefix_map) <= n_coarse:
        # Use prefix grouping directly
        mapping = {}
        for p, types in prefix_map.items():
            for ct in types:
                mapping[ct] = p
        return mapping

    # Last resort: k-means on one-hot fine-type matrix
    ft2idx = {ct: i for i, ct in enumerate(unique_fine)}
    X = np.eye(n_fine, dtype=np.float32)   # identity — each type is its own feature
    km = KMeans(n_clusters=n_coarse, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    mapping = {ct: f"coarse_{labels[ft2idx[ct]]}" for ct in unique_fine}
    return mapping


def _build_cast_descriptors(coords, cell_types_coarse,
                              radii=(0.1, 0.3, 0.6),
                              n_sectors=8,
                              normalize=True):
    """
    Build CAST (Coarse Architecture Spatial Topology) descriptors.

    For each cell, the descriptor captures three complementary aspects
    of the local tissue architecture — all using only coarse cell type
    identity, not expression:

    Component 1 — Multi-scale coarse neighbourhood distribution
        At each of len(radii) spatial scales, count the fraction of
        neighbouring cells of each coarse type.  Captures the cell-type
        composition of the local environment at multiple resolutions.
        Dimension: K_coarse × len(radii)

    Component 2 — Angular sector profile
        Divide the 2π neighbourhood into n_sectors equal angular sectors.
        For each sector, count the dominant coarse type.
        Captures the directional tissue geometry (e.g., cortical layer
        direction, vessel orientation).
        Dimension: K_coarse × n_sectors

    Component 3 — Local density gradient
        The ratio of cell density at small vs large radius.
        High ratio → cell is at a density peak.  Low ratio → cell is at
        a region boundary.
        Dimension: 1

    All components are z-scored within the slice, making the descriptor
    comparable across time points without requiring cross-slice normalization.

    Parameters
    ----------
    coords            : (n, 2) float64 physical coordinates
    cell_types_coarse : (n,)   str     coarse cell type labels
    radii             : tuple of floats  spatial scales (as fractions of
                        the slice's 95th-pctile NN distance × multipliers)
                        These are MULTIPLIERS of the median NN distance:
                        0.1×med, 0.3×med, 0.6×med etc.
    n_sectors         : number of angular sectors
    normalize         : if True, z-score within the slice

    Returns
    -------
    desc : (n, D) float32   CAST descriptor matrix
    """
    from sklearn.neighbors import BallTree, KDTree

    n              = len(coords)
    unique_coarse  = np.unique(cell_types_coarse)
    K              = len(unique_coarse)
    ct2idx         = {c: i for i, c in enumerate(unique_coarse)}
    ct_idx         = np.array([ct2idx[c] for c in cell_types_coarse], dtype=int)

    # Estimate physical scale from median NN distance
    tree = BallTree(coords)
    d_nn, _ = tree.query(coords, k=2)
    med_nn  = float(np.median(d_nn[:, 1])) + 1e-9
    phys_radii = [med_nn * r * 20 for r in radii]   # r=0.1 → 2×med_nn etc.

    # ── Component 1: multi-scale neighbourhood distribution ────────────────────
    comp1_dim = K * len(phys_radii)
    comp1     = np.zeros((n, comp1_dim), dtype=np.float32)

    for ri, r in enumerate(phys_radii):
        nbrs = tree.query_radius(coords, r=r)
        for i in range(n):
            counts = np.zeros(K)
            for j in nbrs[i]:
                counts[ct_idx[j]] += 1.0
            s = counts.sum()
            if s > 0:
                counts /= s
            comp1[i, ri*K : (ri+1)*K] = counts

    # ── Component 2: angular sector profile ────────────────────────────────────
    comp2_dim = K * n_sectors
    comp2     = np.zeros((n, comp2_dim), dtype=np.float32)
    r_max     = phys_radii[-1]   # use largest radius for angular profile
    nbrs_ang  = tree.query_radius(coords, r=r_max)
    sector_w  = 2.0 * np.pi / n_sectors

    for i in range(n):
        for j in nbrs_ang[i]:
            if j == i:
                continue
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            angle = np.arctan2(dy, dx) % (2 * np.pi)
            sec   = int(angle / sector_w) % n_sectors
            comp2[i, sec * K + ct_idx[j]] += 1.0

    # Normalize each sector independently
    for sec in range(n_sectors):
        block = comp2[:, sec*K : (sec+1)*K]
        s     = block.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        comp2[:, sec*K : (sec+1)*K] = block / s

    # ── Component 3: local density gradient ────────────────────────────────────
    count_small = np.array([len(tree.query_radius([coords[i]], r=phys_radii[0])[0])
                             for i in range(n)], dtype=np.float32)
    count_large = np.array([len(tree.query_radius([coords[i]], r=phys_radii[-1])[0])
                             for i in range(n)], dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        density_ratio = np.where(count_large > 0,
                                 count_small / count_large,
                                 0.0).reshape(-1, 1).astype(np.float32)

    # ── Concatenate ────────────────────────────────────────────────────────────
    desc = np.concatenate([comp1, comp2, density_ratio], axis=1)

    # ── Z-score within this slice ──────────────────────────────────────────────
    # Critical: normalise each dimension independently within the slice so
    # absolute expression-driven shifts between time points cancel out.
    # We do NOT jointly normalise across slices — that would require knowing
    # the alignment, creating the circular dependency we are trying to break.
    if normalize:
        mu  = desc.mean(axis=0, keepdims=True)
        std = desc.std( axis=0, keepdims=True) + 1e-8
        desc = (desc - mu) / std

    return desc.astype(np.float32)


def _build_putative_correspondences_cast(desc_A, desc_B,
                                          k=10,
                                          n_anchor=2000,
                                          dist_percentile=60,
                                          seed=42):
    """
    Build putative correspondences using Euclidean distance in CAST space.
    Same logic as _build_putative_correspondences but uses CAST descriptors
    instead of raw JSD on neighbourhood distributions.

    Parameters
    ----------
    desc_A, desc_B : (n_A, D) and (n_B, D) CAST descriptor matrices
    k              : neighbours per anchor
    n_anchor       : max anchors from A
    dist_percentile: filter threshold

    Returns
    -------
    pairs, dists   : same format as _build_putative_correspondences
    """
    from sklearn.neighbors import BallTree

    n_A, n_B = desc_A.shape[0], desc_B.shape[0]
    rng      = np.random.default_rng(seed)

    # ── Choose anchor cells ────────────────────────────────────────────────────
    if n_A > n_anchor:
        # Prefer cells whose best CAST match in B is small
        n_probe   = min(500, n_B)
        probe_idx = rng.choice(n_B, n_probe, replace=False)
        probe     = desc_B[probe_idx]

        # For each A cell, compute min distance to any probe B cell
        from sklearn.metrics.pairwise import euclidean_distances
        D_probe   = euclidean_distances(desc_A, probe)   # (n_A, n_probe)
        min_d     = D_probe.min(axis=1)
        anchor_idx = np.argsort(min_d)[:n_anchor]
    else:
        anchor_idx = np.arange(n_A)

    # ── k-NN in CAST space ─────────────────────────────────────────────────────
    print(f"  CAST k-NN: {len(anchor_idx)} anchors × {n_B} B cells …")
    tree_B = BallTree(desc_B)
    dists_knn, idx_knn = tree_B.query(desc_A[anchor_idx], k=k)

    pairs, dists = [], []
    for ai, i in enumerate(anchor_idx):
        for ki in range(k):
            j = int(idx_knn[ai, ki])
            d = float(dists_knn[ai, ki])
            pairs.append((int(i), j))
            dists.append(d)

    # ── Filter by distance threshold ───────────────────────────────────────────
    thresh    = float(np.percentile(dists, dist_percentile))
    filtered  = [(p, d) for p, d in zip(pairs, dists) if d <= thresh]
    if not filtered:
        filtered = list(zip(pairs, dists))
    pairs_f, dists_f = zip(*filtered)
    print(f"  Putative pairs: {len(pairs_f)}  (CAST-dist threshold={thresh:.4f})")
    return list(pairs_f), list(dists_f)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 5 — cross-condition alignment (different time points)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align_cross_condition(
    sliceA:    AnnData,
    sliceB:    AnnData,
    radius:    float,
    filePath:  str,
    # ── coarse type mapping ───────────────────────────────────────────────────
    coarse_type_map: dict = None,
    n_coarse:        int  = 6,
    # ── CAST descriptor parameters ────────────────────────────────────────────
    cast_radii:   tuple = (0.1, 0.3, 0.6),
    n_sectors:    int   = 8,
    # ── feature matching parameters ───────────────────────────────────────────
    k_nn:             int   = 15,
    n_anchor:         int   = 2000,
    dist_percentile:  float = 60.0,
    n_ransac:         int   = 2000,
    spatial_thresh:   float = None,
    # ── matching parameters ───────────────────────────────────────────────────
    search_radius_k:  float = 4.0,
    soft_temp:        float = 0.0,
    em_iters:         int   = 3,
    top_frac_svd:     float = 0.20,
    # ── matching cost weights (no expression term) ────────────────────────────
    beta_coarse:      float = 0.5,
    gamma_cast:       float = 0.5,
    # ── standard parameters ───────────────────────────────────────────────────
    use_gpu:   bool            = False,
    return_extra: bool         = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = False,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    **kwargs,
):
    """
    Cross-condition alignment for slices from different time points.

    Gene expression and fine-grained cell type annotations are NOT used
    for registration or matching because they change between conditions.
    Only the spatial architecture of major tissue compartments is used,
    which is preserved across time points.

    Key difference from pairwise_align_fdesc_ransac
    ------------------------------------------------
    - Neighbourhood JSD on fine cell types → CAST descriptor (coarse type
      spatial topology, angular geometry, multi-scale density)
    - Gene cosine distance → discarded entirely
    - Cell type mismatch on fine types → coarse type mismatch only
    - All descriptors z-scored within each slice independently

    Parameters
    ----------
    coarse_type_map : dict {fine_type_name -> coarse_class_name} or None
        Maps fine-grained cell types to major biological classes.
        Example for brain:
            {'L2/3 ExN': 'Excitatory', 'L5 ExN': 'Excitatory',
             'Astro': 'Glia', 'Oligo': 'Glia',
             'Endo': 'Vasculature', 'Peri': 'Vasculature',
             'PV': 'Inhibitory', 'SST': 'Inhibitory'}
        If None, auto-detected by grouping similar cell type names.
        User-provided mapping is strongly preferred for biological accuracy.

    n_coarse : int
        Number of coarse classes if auto-detecting (default 6).

    cast_radii : tuple of float
        Spatial scale multipliers for CAST descriptor.
        Each value × median NN distance gives the physical neighbourhood radius.
        (0.1, 0.3, 0.6) captures local, medium, and regional context.

    n_sectors : int
        Angular sectors for the CAST directional profile (default 8).

    beta_coarse : float
        Weight of coarse cell-type mismatch in local matching cost.
        0 = ignore cell type entirely, 1 = use only cell type.

    gamma_cast : float
        Weight of CAST descriptor Euclidean distance in local matching.
        0 = ignore spatial architecture, 1 = use only spatial architecture.

    All other parameters: same as pairwise_align_fdesc_ransac.

    Returns
    -------
    pi              : (n_A, n_B) float32  transport plan
    If return_extra : (pi, coord_A_reg, R, t, overlap_A, overlap_B,
                       coarse_A, coarse_B, desc_A, desc_B)
    """
    from sklearn.neighbors import BallTree
    import scipy.sparse as sp

    start    = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_cc_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log_cc.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_cross_condition\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"radius={radius}  n_coarse={n_coarse}\n")
    logFile.write(f"cast_radii={cast_radii}  n_sectors={n_sectors}\n\n")

    # ── 0. Shared genes + cell types (minimal preprocessing) ──────────────────
    # We do NOT use expression or niche here — only cell type labels and coords.
    # We still intersect genes so downstream callers can use pi with expression
    # data if they choose to. But the alignment itself ignores expression.
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Empty AnnData: {s}")

    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes):
        sliceA = sliceA[:, shared_genes]
        sliceB = sliceB[:, shared_genes]

    # Cell types — do NOT filter to shared: different time points may have
    # different cell type sets. We use the coarse mapping to bridge them.
    ct_A_fine = np.array(sliceA.obs['cell_type_annot'].astype(str))
    ct_B_fine = np.array(sliceB.obs['cell_type_annot'].astype(str))
    n_A       = sliceA.shape[0]
    n_B       = sliceB.shape[0]

    cA_raw = sliceA.obsm['spatial'].astype(np.float64)
    cB_raw = sliceB.obsm['spatial'].astype(np.float64)

    logFile.write(f"n_A={n_A}  n_B={n_B}  shared_genes={len(shared_genes)}\n")
    logFile.write(f"A fine types: {len(np.unique(ct_A_fine))}  "
                  f"B fine types: {len(np.unique(ct_B_fine))}\n\n")

    # ── 1. Coarse type mapping ─────────────────────────────────────────────────
    # Build a unified mapping covering both slices' cell types
    all_fine = np.unique(np.concatenate([ct_A_fine, ct_B_fine]))

    if coarse_type_map is None:
        print("[CrossCond] Auto-detecting coarse cell type groups …")
        coarse_type_map = _auto_coarse_types(all_fine, n_coarse=n_coarse)
        print(f"[CrossCond] Coarse map: "
              + ", ".join(f"{k}→{v}" for k, v in
                          sorted(coarse_type_map.items())[:8])
              + (" …" if len(coarse_type_map) > 8 else ""))
    else:
        # Fill in any fine types not covered by user map
        for ft in all_fine:
            if ft not in coarse_type_map:
                coarse_type_map[ft] = 'Other'

    ct_A_coarse = np.array([coarse_type_map.get(c, 'Other') for c in ct_A_fine])
    ct_B_coarse = np.array([coarse_type_map.get(c, 'Other') for c in ct_B_fine])

    unique_coarse = np.unique(np.concatenate([ct_A_coarse, ct_B_coarse]))
    logFile.write(f"Coarse types ({len(unique_coarse)}): "
                  f"{', '.join(unique_coarse)}\n\n")
    print(f"[CrossCond] Coarse types: {list(unique_coarse)}")

    # ── 2. Build CAST descriptors ──────────────────────────────────────────────
    print("[CrossCond] Building CAST descriptors for A …")
    desc_A = _build_cast_descriptors(cA_raw, ct_A_coarse,
                                      radii=cast_radii,
                                      n_sectors=n_sectors,
                                      normalize=True)
    print("[CrossCond] Building CAST descriptors for B …")
    desc_B = _build_cast_descriptors(cB_raw, ct_B_coarse,
                                      radii=cast_radii,
                                      n_sectors=n_sectors,
                                      normalize=True)

    D_desc = desc_A.shape[1]
    logFile.write(f"CAST descriptor dimension: {D_desc}\n\n")
    print(f"[CrossCond] CAST descriptor dim: {D_desc}")

    # ── 3. Build putative correspondences in CAST space ────────────────────────
    print("[CrossCond] Building putative correspondences …")
    pairs, dists = _build_putative_correspondences_cast(
        desc_A, desc_B,
        k=k_nn, n_anchor=n_anchor,
        dist_percentile=dist_percentile)

    if len(pairs) < 10:
        raise RuntimeError(
            f"Only {len(pairs)} putative correspondences found. "
            "Try increasing k_nn or dist_percentile, or provide a better "
            "coarse_type_map.")

    logFile.write(f"Putative pairs: {len(pairs)}\n")

    # ── 4. RANSAC ─────────────────────────────────────────────────────────────
    print(f"[CrossCond] RANSAC ({n_ransac} iters) …")
    R, t, inlier_pairs, inlier_frac = _ransac_rigid(
        pairs, dists, cA_raw, cB_raw,
        n_iter=n_ransac, spatial_thresh=spatial_thresh)

    logFile.write(f"RANSAC: {len(inlier_pairs)}/{len(pairs)} inliers  "
                  f"({100*inlier_frac:.0f}%)\n")
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    logFile.write(f"T: angle={angle:.2f}°  t=({t[0]:.2f},{t[1]:.2f})\n\n")

    if inlier_frac < 0.05:
        logFile.write("WARNING: very low RANSAC inlier fraction.\n")
        print("[CrossCond] WARNING: low inlier fraction — check overlap.")

    # ── 5. Apply rigid transform ───────────────────────────────────────────────
    coord_A_reg = cA_raw @ R.T + t   # rigid, det(R)=+1, no stretching

    # ── 6. Auto search radius ─────────────────────────────────────────────────
    tree_B_tmp   = BallTree(cB_raw)
    d_nn, _      = tree_B_tmp.query(cB_raw, k=2)
    search_radius = float(np.median(d_nn[:, 1]) * search_radius_k)
    logFile.write(f"Search radius: {search_radius:.2f}\n")
    print(f"[CrossCond] Search radius: {search_radius:.1f}")

    # ── 7. Precompute coarse type mismatch matrix ──────────────────────────────
    # Coarse type mismatch only — fine-type and expression differences
    # across conditions are NOT penalised here.
    M_coarse = (ct_A_coarse[:, None] != ct_B_coarse[None, :]).astype(np.float32)

    # ── 8. Local matching + EM ────────────────────────────────────────────────
    def local_match_cc(coord_A_reg, cB_raw, search_radius):
        """
        Bidirectional local matching using coarse type + CAST distance only.
        No gene expression, no fine-type penalty.
        """
        tree_B  = BallTree(cB_raw)
        tree_A  = BallTree(coord_A_reg)

        pi_out         = np.zeros((n_A, n_B), dtype=np.float32)
        overlap_mask_A = np.zeros(n_A, dtype=bool)
        overlap_mask_B = np.zeros(n_B, dtype=bool)
        n_unmatched_A  = 0

        for i in range(n_A):
            cands = tree_B.query_radius([coord_A_reg[i]], r=search_radius)[0]
            if len(cands) == 0:
                n_unmatched_A += 1
                continue

            overlap_mask_A[i] = True
            for j in cands:
                overlap_mask_B[j] = True

            # Cross-condition local cost: coarse type + CAST descriptor distance
            # Crucially: NO fine gene expression term
            cast_dists = np.linalg.norm(
                desc_A[i] - desc_B[cands], axis=1).astype(np.float64)
            cost = (beta_coarse  * M_coarse[i, cands].astype(np.float64)
                    + gamma_cast * cast_dists / (cast_dists.max() + 1e-8))

            if soft_temp < 1e-6:
                pi_out[i, cands[np.argmin(cost)]] = 1.0 / n_A
            else:
                span  = cost.max() - cost.min() + 1e-12
                tau   = soft_temp * span + 1e-12
                logw  = -cost / tau
                logw -= logw.max()
                w     = np.exp(logw)
                w    /= w.sum()
                for m, j in enumerate(cands):
                    pi_out[i, j] = float(w[m]) / n_A

        return pi_out, overlap_mask_A, overlap_mask_B, n_unmatched_A

    print(f"[CrossCond] Local matching + EM ({em_iters} iters) …")
    pi, om_A, om_B, n_unm = local_match_cc(coord_A_reg, cB_raw, search_radius)
    logFile.write(f"Initial: unmatched_A={n_unm}/{n_A}  "
                  f"unique_B={(~om_B).sum()}/{n_B}\n")
    print(f"[CrossCond] Matched A: {n_A-n_unm}/{n_A}  "
          f"Unique to B: {int((~om_B).sum())}/{n_B}")

    for em in range(em_iters):
        flat     = pi.flatten()
        n_top    = max(30, int(top_frac_svd * (pi > 0).sum()))
        top_idx  = np.argsort(flat)[-n_top:]
        rows_em  = top_idx // n_B
        cols_em  = top_idx %  n_B
        w_em     = flat[top_idx]
        if w_em.sum() < 1e-12:
            break
        w_em    /= w_em.sum()

        R_new, t_new    = _fit_rigid_from_pairs(
            cA_raw[rows_em], cB_raw[cols_em], weights=w_em)
        coord_A_reg_new = cA_raw @ R_new.T + t_new
        delta_t         = float(np.linalg.norm(t_new - t))
        R, t, coord_A_reg = R_new, t_new, coord_A_reg_new

        logFile.write(f"EM {em+1}: |Δt|={delta_t:.4f}\n")
        print(f"[CrossCond] EM {em+1}: |Δt|={delta_t:.2f}")

        pi, om_A, om_B, n_unm = local_match_cc(
            coord_A_reg, cB_raw, search_radius)
        if delta_t < 0.5:
            break

    logFile.write(f"\nFinal: unmatched_A={n_unm}/{n_A}  "
                  f"unique_B={(~om_B).sum()}/{n_B}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    print(f"[CrossCond] Done in {time.time()-start:.1f}s  |  "
          f"A matched={n_A-n_unm}/{n_A}  B unique={int((~om_B).sum())}/{n_B}")

    if return_extra:
        return (pi, coord_A_reg, R, t,
                om_A, om_B,
                ct_A_coarse, ct_B_coarse,
                desc_A, desc_B)
    return pi