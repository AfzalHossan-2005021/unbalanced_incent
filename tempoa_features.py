import numpy as np
from sklearn.neighbors import BallTree
from .utils import jensenshannon_divergence_backend

def compute_ctnd(adata, radius, spatial_key='spatial', ct_key='cell_type_annot'):
    """
    Cell-Type Neighborhood Descriptor (CTND).
    Computes a spatially-decayed local neighborhood profile for each cell to 
    circumvent absolute temporal gene expression shifts.
    """
    coords = adata.obsm[spatial_key]
    cell_types = np.array(adata.obs[ct_key].astype(str))
    unique_ct = np.unique(cell_types)
    ct2idx = {c: i for i, c in enumerate(unique_ct)}
    
    n_cells = adata.shape[0]
    K = len(unique_ct)
    
    tree = BallTree(coords)
    # Query neighbors within radius
    neighbor_lists, dist_lists = tree.query_radius(coords, r=radius, return_distance=True)
    
    ctnd = np.zeros((n_cells, K), dtype=np.float64)
    
    # Use a Gaussian decay for neighbors
    # w(d) = exp(-d^2 / (2 * (radius / 3)^2) )
    sigma = radius / 3.0
    
    for i in range(n_cells):
        neighbors = neighbor_lists[i]
        dists = dist_lists[i]
        
        weights = np.exp(- (dists**2) / (2 * sigma**2))
        
        for idx, w in zip(neighbors, weights):
            ct_idx = ct2idx[cell_types[idx]]
            ctnd[i, ct_idx] += w
            
    # Normalize to form a probability distribution per cell
    row_sums = ctnd.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ctnd = ctnd / row_sums
    
    return ctnd, unique_ct

def tempoa_temporal_cost(adata_s, adata_t, radius=50.0):
    """
    Calculates the base Cost Matrix M based on topological CTND rather than raw genes.
    """
    print("Extracting Structure: CTND Source...")
    curr_ctnd_s, cts_s = compute_ctnd(adata_s, radius)
    print("Extracting Structure: CTND Target...")
    curr_ctnd_t, cts_t = compute_ctnd(adata_t, radius)
    
    # Ensure same dimension across timepoints
    # We take the union of cell types
    all_cts = np.union1d(cts_s, cts_t)
    
    def align_ctnd(ctnd, cts, all_cts):
        aligned = np.zeros((ctnd.shape[0], len(all_cts)))
        for i, ct in enumerate(all_cts):
            if ct in cts:
                idx = np.where(cts == ct)[0][0]
                aligned[:, i] = ctnd[:, idx]
        return aligned
        
    ctnd_s_aligned = align_ctnd(curr_ctnd_s, cts_s, all_cts)
    ctnd_t_aligned = align_ctnd(curr_ctnd_t, cts_t, all_cts)
    
    # Calculate JSD as the cost matrix M
    print("Computing CTND JSD Matrix M...")
    M_temporal = jensenshannon_divergence_backend(ctnd_s_aligned, ctnd_t_aligned)
    return M_temporal

