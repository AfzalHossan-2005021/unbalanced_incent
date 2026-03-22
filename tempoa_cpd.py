import numpy as np
from scipy.spatial import KDTree

def icp_registration(source_pts, target_pts, max_iterations=50, tolerance=1e-5):
    """
    Iterative Closest Point (ICP) for rigid registration.
    Returns scaling, rotation, translation.
    """
    src = source_pts.copy()
    tgt = target_pts.copy()
    
    prev_error = float('inf')
    
    for i in range(max_iterations):
        tree = KDTree(tgt)
        distances, indices = tree.query(src)
        
        # Mean squared error
        mean_error = np.mean(distances**2)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
        # Compute rigid transformation
        # Centering
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt[indices], axis=0)
        
        src_centered = src - centroid_src
        tgt_centered = tgt[indices] - centroid_tgt
        
        # SVD for rotation
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        
        # Reflection fix
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
            
        t_vec = centroid_tgt - (R_mat @ centroid_src.T).T
        
        # Apply transformation
        src = (R_mat @ src.T).T + t_vec
        
    return src

def tempoa_prior_matrix(adata_s, adata_t, spatial_key='spatial', alpha=2.0):
    """
    Calculates the spatial prior probability matrix P_ij based on CPD/ICP rigid initialization.
    Breaks GW isometry/reflection invariance.
    """
    coords_s = adata_s.obsm[spatial_key]
    coords_t = adata_t.obsm[spatial_key]
    
    print("Running Rigid Pre-alignment (ICP) to break structural reflection...")
    # Map S to T
    aligned_s = icp_registration(coords_s, coords_t)
    
    # Calculate Euclidean distances between rigidly aligned Source and Target
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(aligned_s, coords_t, metric='euclidean')
    
    # Convert to probability prior: P_ij = exp(-alpha * dist / max_dist)
    # The further the mapped point, the close to 0 the probability.
    dist_matrix = dist_matrix / np.max(dist_matrix)
    P = np.exp(-alpha * dist_matrix)
    
    # Normalize prior
    P = P / np.max(P)
    
    return P, dist_matrix

def compute_diffeomorphic_masked_cost(M_temporal, P_prior, gamma_penalty=2.0):
    """
    Combines the Temporal Feature Cost with the Rigid Spatial Prior.
    M_tilde = M_temporal * (-log(P + eps))
    """
    epsilon = 1e-8
    mask = -np.log(P_prior + epsilon)
    # Normalize mask 
    mask = mask / np.max(mask)
    
    # Weight the temporal cost heavily by the spatial distance penalty
    # If P_prior is low (far away in rigid map), the cost multiplies immensely.
    M_tilde = M_temporal + (gamma_penalty * mask)
    
    return M_tilde
