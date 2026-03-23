import numpy as np
import scipy.sparse as sp
from ot.gromov import fused_unbalanced_gromov_wasserstein

from .utils import pairwise_msd
from .tempoa_features import tempoa_temporal_cost
from .tempoa_cpd import tempoa_prior_matrix, compute_diffeomorphic_masked_cost

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
    
    # Normalize both by the global maximum spatial distance to explicitly maintain the specific 
    # true structural size parity (preventing smaller slices from stretching aggressively natively)
    max_scale = max(D_S.max(), D_T.max()) + 1e-8
    D_S = D_S / max_scale
    D_T = D_T / max_scale
    
    # 2. TEMPOA Innovation 1: Topological Temporal Features (CTND)
    M_temporal = tempoa_temporal_cost(slice_s, slice_t)
    M_temporal = M_temporal / np.max(M_temporal)
    
    # 3. TEMPOA Innovation 2: Symmetry Breaking Prior (CPD)
    P_prior, rigid_dist = tempoa_prior_matrix(slice_s, slice_t)
    
    # Calculate M_tilde
    M_tilde = compute_diffeomorphic_masked_cost(M_temporal, P_prior, gamma_penalty=2.0)
    
    # Scale down distance matrices to prevent exponential overflow in solvers
    M_tilde_scaled = M_tilde / (np.max(M_tilde) + 1e-8)

    # 4. Uniform marginal distributions
    p = np.ones(M_tilde.shape[0]) / M_tilde.shape[0]
    q = np.ones(M_tilde.shape[1]) / M_tilde.shape[1]
    
    print("Running Diffeomorphic Fused Unbalanced GW...")
    # 5. Calculate transport plan pi
    # For large scale unbalanced GW, this relies on POT's BCD loop.
    # Note: LDDMM Laplacian penalty is mathematically encoded via the structured prior
    # and the unbalanced relaxation, restricting drastic cross-edges.
    pi, pi2, log_dict = fused_unbalanced_gromov_wasserstein(
        Cx=D_S, 
        Cy=D_T,
        wx=p,
        wy=q,
        M=M_tilde_scaled, 
        alpha=alpha,
        reg_marginals=(margin_s, margin_t),
        epsilon=0.1, # Increased epsilon helps stabilize large-scale matrices
        divergence='kl',
        unbalanced_solver='lbfgsb', # Scipy's LBFGSB uses the gradient directly, circumventing exponential underflow entirely
        log=True, 
        max_iter=50,
        tol=1e-5
    )
    
    return pi, M_tilde, P_prior
