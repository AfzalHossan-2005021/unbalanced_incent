import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_neighborhood_distribution, jensenshannon_divergence_backend, cosine_dist_calculator

# Assuming tempoa modules are in the same directory or accessible in PYTHONPATH
from .tempoa_solver import run_tempoa as execute_tempoa

def run_tempoa(sliceA, sliceB, data1, data2, dataPath, overwrite=False):
    """
    Helper function to run TEMPOA (Temporal & Morphological Partial Overlap Alignment)
    and format the output similarly to run_stalign for benchmarking.
    """
    filePath = f'{dataPath}/{data1}_{data2}'
    if not os.path.exists(filePath):
        os.makedirs(filePath, exist_ok=True)
        
    print("Running TEMPOA Alignment...")

    # --- 1. Metric Helper Functions (same as in STalign benchmarking setup) ---
    def get_js_dist_neighborhood(slice1, slice2, radius):
        neighborhood_dist_slice1 = get_neighborhood_distribution(slice1, radius) + 0.01
        neighborhood_dist_slice2 = get_neighborhood_distribution(slice2, radius) + 0.01
        js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_dist_slice1, neighborhood_dist_slice2)
        return np.asarray(js_dist_neighborhood)

    def get_cosine_dist_gene_expr(slice1, slice2):
        cosine_dist_gene_expr = cosine_dist_calculator(slice1, slice2, data1, data2, filePath)
        return cosine_dist_gene_expr

    def cellular_neighborhood_gene_expr_metric(pi_mat, radius, slice1, slice2):
        js_dist_neighborhood = get_js_dist_neighborhood(slice1, slice2, radius)
        obj_neighbor = np.sum(js_dist_neighborhood * pi_mat)
        cosine_dist_gene_expr = get_cosine_dist_gene_expr(slice1, slice2)
        obj_gene_cos = np.sum(cosine_dist_gene_expr * pi_mat)
        return obj_neighbor, obj_gene_cos

    radius = 100  # 100um logic from previous run_stalign

    # --- 2. Calculate Initial Objective Values ---
    print("Calculating Initial Metrics...")
    a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    G = np.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

    initial_obj_gene_cos = np.sum(get_cosine_dist_gene_expr(sliceA, sliceB) * G)
    initial_obj_neighbor = np.sum(get_js_dist_neighborhood(sliceA, sliceB, radius) * G)


    # --- 3. Execute Core TEMPOA Pipeline ---
    # Determine the unbalanced margins. If it's partial A to full B, give high relaxation to B.
    # alpha controls feature vs morphology trade-off.
    pi_mat, M_tilde, P_prior = execute_tempoa(
        sliceA, sliceB, 
        alpha=0.6, 
        margin_s=0.1,  # KL mass relaxation for source (some cells in cropped slice might not map safely)
        margin_t=0.01, # Higher relaxation value for Target (many target cells naturally won't map)
        spatial_key='spatial'
    )


    # --- 4. Create "Aligned" coordinates for Slice A based on mappings ---
    # To compute post-alignment metrics uniformly across benchmarking scripts,
    # we physically project slice A's coordinates to the Target space (Slice B).
    
    # 4a. Basic Barycentric Projection using Pi
    # S_aligned = (Pi / row_sums) @ T_coords
    print("Transforming Source Coordinates to Aligned Latent Space...")
    coords_t = sliceB.obsm['spatial']
    
    row_sums = pi_mat.sum(axis=1)[:, np.newaxis]
    # small eps to avoid div by zero
    row_sums[row_sums == 0] = 1e-10 
    
    pi_normalized_rows = pi_mat / row_sums
    mapped_coords_A = pi_normalized_rows @ coords_t
    
    # Alternatively, use the rigid ICP intermediate if mapping tore the tissue:
    # mapped_coords_A = icp_registration(sliceA.obsm['spatial'], sliceB.obsm['spatial'])

    sliceA_TEMPOA = sliceA.copy()
    sliceA_TEMPOA.obsm['spatial'] = mapped_coords_A
    new_slices = [sliceA_TEMPOA, sliceB]

    # --- 5. Visualize Alignment (Benchmarking style) ---
    plt.clf()
    plt.scatter(mapped_coords_A[:, 0], mapped_coords_A[:, 1], c='blue', s=2, alpha=0.6, label='Source TEMPOA-aligned')
    plt.scatter(coords_t[:, 0], coords_t[:, 1], c='lightgrey', s=2, alpha=0.3, label='Target (Full)')
    plt.legend(markerscale=5, loc='lower left')
    plt.axis("off")
    plt.title(f"TEMPOA Alignment: {data1} -> {data2}")
    if not os.path.exists(filePath): os.makedirs(filePath)
    plt.savefig(f"{filePath}/tempoa_alignment_plot.png")
    plt.show()
    plt.clf()


    # --- 6. Calculate Final Objective Values ---
    print("Calculating Final Metrics...")
    # NOTE: Since we updated spatial coordinates on sliceA_TEMPOA,
    # we need spatial metrics. Because TEMPOA optimizes a direct joint coupling matrix pi_mat,
    # we can evaluate metrics directly on pi_mat (no need to reconstruct RBF logic unless strict parity mandated).
    
    # Assuming the unified benchmarking pipeline judges the cost using the coupling matrix:
    final_obj_neighbor, final_obj_gene_cos = cellular_neighborhood_gene_expr_metric(pi_mat, radius, sliceA_TEMPOA, sliceB)

    return pi_mat, initial_obj_neighbor, initial_obj_gene_cos, final_obj_neighbor, final_obj_gene_cos, new_slices

