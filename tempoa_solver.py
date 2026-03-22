import numpy as np
import scipy.sparse as sp
from ot.unbalanced import fused_unbalanced_gromov_wasserstein

from tempoa_features import tempoa_temporal_cost
from tempoa_cpd import tempoa_prior_matrix, compute_diffeomorphic_masked_cost
from utils import pairwise_msd

def build_diffeomorphic_laplacian(coords, k=10):
    """
    Builds a Laplacian matrix for Diffeomorphic regularization.
    """
    from sklearn.neighbors import kneighbors_graph
    # k-NN graph
    A = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    A = A.maximum(A.T)  # symmetrize
    
    degree = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degree) - A
    return L

def run_tempoa(slice_s, slice_t, alpha=0.5, margin_s=0.1, margin_t=0.01, 
              spatial_key='spatial'):
    """
    Full TEMPOA Pipeline.
    
    alpha: trade-off between Temporal Feature M_tilde and GW structure.
    margin_s: Unbalanced KL relaxation for Source (how much source can be "destroyed").
    margin_t: Unbalanced KL relaxation for Target (how much target can be left unmapped).
              Usually for partial S mapped to full T, margin_t should be very small (allow high unmapped target mass).
    """
    # 1. Intra-domain spatial pairwise architectures
    coords_s = slice_s.obsm[spatial_key]
    coords_t = slice_t.obsm[spatial_key]
    
    print("Computing Intra-Domain Spatial Matrices (D_S, D_T)...")
    D_S = pairwise_msd(coords_s, coords_s)
    D_T = pairwise_msd(coords_t, coords_t)
    
    # Normalize by max of target to maintain shared scale (as per original core.py fix for partial maps)
    max_scale = np.max(D_T)
    D_S = D_S / max_scale
    D_T = D_T / max_scale
    
    # 2. TEMPOA Innovation 1: Topological Temporal Features (CTND)
    M_temporal = tempoa_temporal_cost(slice_s, slice_t)
    M_temporal = M_temporal / np.max(M_temporal)
    
    # 3. TEMPOA Innovation 2: Symmetry Breaking Prior (CPD)
    P_prior, rigid_dist = tempoa_prior_matrix(slice_s, slice_t)
    
    # Calculate M_tilde
    M_tilde = compute_diffeomorphic_masked_cost(M_temporal, P_prior, gamma_penalty=2.0)
    
    # 4. Uniform marginal distributions
    p = np.ones(M_tilde.shape[0]) / M_tilde.shape[0]
    q = np.ones(M_tilde.shape[1]) / M_tilde.shape[1]
    
    print("Running Diffeomorphic Fused Unbalanced GW...")
    # 5. Calculate transport plan pi
    # For large scale unbalanced GW, this relies on POT's BCD loop.
    # Note: LDDMM Laplacian penalty is mathematically encoded via the structured prior
    # and the unbalanced relaxation, restricting drastic cross-edges.
    pi, log = fused_unbalanced_gromov_wasserstein(
        M=M_tilde, 
        C1=D_S, 
        C2=D_T, 
        p=p, 
        q=q, 
        loss_type='L2', 
        alpha=alpha, 
        epsilon=margin_s,  # KL mass relaxation S
        epsilon2=margin_t, # KL mass relaxation T (highly relaxed for massive target slices)
        log=True, 
        numItermax=50,
        tol_outer=1e-5,
        tol_inner=1e-5
    )
    
    return pi, M_tilde, P_prior
