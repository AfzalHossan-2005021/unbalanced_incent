import numpy as np


def age_progression_score(sliceA, sliceB, data1, data2, filePath):
    '''
    Compute age progression score for each slice and return the score in the obs column of the slice
    '''

    cosine_dist_gene_expr = np.load(f"{filePath}/cosine_dist_gene_expr_{data1}_{data2}.npy")
    pi_mat = np.load(f"{filePath}/pi_matrix_{data1}_{data2}.npy")

    age_progression_score_mat = pi_mat * cosine_dist_gene_expr

    sliceA.obs['age_progression_score'] = np.sum(age_progression_score_mat, axis=1, dtype=np.float64) / (1 / sliceA.n_obs) * 100
    sliceB.obs['age_progression_score'] = np.sum(age_progression_score_mat, axis=0, dtype=np.float64) / (1 / sliceB.n_obs) * 100

    return sliceA, sliceB
