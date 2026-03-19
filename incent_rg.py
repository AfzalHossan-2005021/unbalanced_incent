import numpy as np



############################################
# BARYCENTRIC PROJECTION
############################################

def barycentric_projection(pi,coordsB):

    mass=pi.sum(axis=1,keepdims=True)

    prob=pi/(mass+1e-12)

    mapped=prob@coordsB

    return mapped,mass.squeeze()



############################################
# ROBUST HUBER WEIGHTING
############################################

def huber_weights(errors):

    sigma=np.median(errors)+1e-8

    w=1/(1+(errors/sigma)**2)

    return w



############################################
# WEIGHTED PROCRUSTES
############################################

def weighted_procrustes(X,Y,w):

    w=w/w.sum()

    muX=(w[:,None]*X).sum(0)
    muY=(w[:,None]*Y).sum(0)

    Xc=X-muX
    Yc=Y-muY

    H=(w[:,None]*Xc).T@Yc

    U,S,Vt=np.linalg.svd(H)

    R=Vt.T@U.T

    if np.linalg.det(R)<0:

        Vt[-1]*=-1

        R=Vt.T@U.T

    t=muY-R@muX

    return R,t



############################################
# RANSAC INITIALIZATION
############################################

def ransac_pose(X,Y,trials=100):

    best_R=None
    best_t=None

    best_err=1e20

    N=len(X)

    for _ in range(trials):

        idx=np.random.choice(N,3)

        R,t=weighted_procrustes(

            X[idx],
            Y[idx],
            np.ones(3)

        )

        Xp=X@R.T+t

        err=((Xp-Y)**2).sum(1).mean()

        if err<best_err:

            best_err=err

            best_R=R
            best_t=t

    return best_R,best_t



############################################
# MAIN METHOD
############################################

def incent_rg(

        sliceA,
        sliceB,

        fugw_func,

        coords_key='spatial',

    n_iter=6,
    return_obj=False

):

    coordsA=sliceA.obsm[coords_key].copy()

    coordsB=sliceB.obsm[coords_key]

    R_total=np.eye(coordsA.shape[1])

    t_total=np.zeros(coordsA.shape[1])

    init_nb = None
    init_gene = None
    final_nb = None
    final_gene = None

    def _call_fugw():
        if return_obj:
            try:
                return fugw_func(sliceA, sliceB, return_obj=True)
            except TypeError:
                pass
        return fugw_func(sliceA, sliceB)

    def _unpack_fugw(result):
        if isinstance(result, tuple):
            if len(result) == 5:
                return result
            if len(result) == 1:
                return result[0], None, None, None, None
            return result[0], None, None, None, None
        return result, None, None, None, None

    print("Initial OT")

    pi, init_nb, init_gene, final_nb, final_gene = _unpack_fugw(_call_fugw())

    mapped,mass=barycentric_projection(pi,coordsB)

    print("RANSAC init")

    R,t=ransac_pose(coordsA,mapped)

    coordsA=coordsA@R.T+t

    for k in range(n_iter):

        print("Iteration",k)

        pi, _, _, _, _ = _unpack_fugw(_call_fugw())

        mapped,mass=barycentric_projection(pi,coordsB)

        errors=np.linalg.norm(coordsA-mapped,axis=1)

        robust=huber_weights(errors)

        mask=robust>np.percentile(robust,20)

        R,t=weighted_procrustes(

            coordsA[mask],
            mapped[mask],
            robust[mask]

        )

        coordsA=coordsA@R.T+t

        R_total=R@R_total

        t_total=R@t_total+t

    if return_obj:
        return coordsA, R_total, t_total, pi, init_nb, init_gene, final_nb, final_gene

    return coordsA,R_total,t_total,pi