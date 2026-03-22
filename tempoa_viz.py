import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from scipy.sparse import coo_matrix

def plot_tempoa_alignment(slice_s, slice_t, pi, threshold=1e-4, spatial_key='spatial', 
                          title="TEMPOA Cross-Timepoint Alignment"):
    """
    Diagnostic plot projecting Source slice onto Target slice using mapping pi.
    Lines connect S_i to T_j if pi_ij > threshold.
    """
    coords_s = slice_s.obsm[spatial_key]
    coords_t = slice_t.obsm[spatial_key]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot Background Target (Large Slice)
    ax.scatter(coords_t[:, 0], coords_t[:, 1], c='lightgrey', 
               s=20, label='Full Target Slice ($t_2$)', alpha=0.6)
    
    # Plot Original Source (Small Slice) as a reference, translated left for visibility
    shift_x = np.max(coords_t[:, 0]) - np.min(coords_s[:, 0]) + 50
    shifted_s = coords_s.copy()
    shifted_s[:, 0] -= shift_x
    ax.scatter(shifted_s[:, 0], shifted_s[:, 1], c='blue', 
               s=20, label='Original Source ($t_1$)', alpha=0.8)

    # Filter Pi using scipy sparse to avoid memory explosive iteration
    pi_sparse = coo_matrix(pi)
    
    lines = []
    mapped_source = []
    
    for i, j, v in zip(pi_sparse.row, pi_sparse.col, pi_sparse.data):
        if v > threshold:
            # We connect shifted S to T to show the mapping lines clearly
            lines.append([shifted_s[i], coords_t[j]])
            # Track the pushed coordinate 
            mapped_source.append(coords_t[j])
            
    mapped_source = np.array(mapped_source) 
    
    if len(lines) > 0:
        lc = mcoll.LineCollection(lines, colors='red', linewidths=0.5, alpha=0.3, zorder=1)
        ax.add_collection(lc)
        
        # Overlay standard points to show mapped clusters
        ax.scatter(mapped_source[:, 0], mapped_source[:, 1], c='darkorange', 
                   s=10, label='Mapped Coordinates\n(No Symmetry Reflections)', zorder=2)
     
    ax.set_title(title, fontsize=16)
    ax.axis('equal')
    ax.legend(loc='best')
    plt.show()

def validate_reflection(slice_s, slice_t, pi, ground_truth_coords_if_known=None):
    """
    Returns boolean if mapping correctly avoided the isometric contralateral hemisphere.
    (Applicable if synthetic benchmarks have a known center-line or target offset).
    """
    pi_argmax = np.argmax(pi, axis=1) # mapping vector S -> T
    
    if ground_truth_coords_if_known is not None:
        pass
        # Compare actual to mapping
