import os
import ot
import torch
import inspect

import numpy as np

from ot.optim import line_search_armijo

from .linesearch import solve_gromov_linesearch
from .conditional_gradient import cg_incent


def fused_gromov_wasserstein_incent(M1, M2, C1, C2, p, q, gamma, G_init=None, loss_fun='square_loss',
                                    alpha=0.1, beta=0.8, armijo=False, log=False, numItermax=6000,
                                    tol_rel=1e-9, tol_abs=1e-9, use_gpu=False,
                                    rho1=1.0, rho2=1.0, balanced_fallback_threshold=1e6,
                                    **kwargs):
    """
    This method is written by Anup Bhowmik, CSE, BUET

    Unbalanced extension of fused_gromov_wasserstein_incent.
    Marginal constraints are softly enforced via KL penalties (rho1, rho2).
    All cost matrices, the GW structure term, and the line-search are unchanged.

    Parameters
    ----------
    rho1 : float, optional
        KL penalty weight for the source marginal. Default 1.0.
    rho2 : float, optional
        KL penalty weight for the target marginal. Default 1.0.
    balanced_fallback_threshold : float, optional
        Fall back to exact EMD when both rho1, rho2 >= this value. Default 1e6.

    All other parameters are identical to modular_incent.transport.fgw.
    """

    p, q = ot.utils.list_to_array(p, q)

    p0, q0, C10, C20, M10, M20 = p, q, C1, C2, M1, M2
    nx = ot.backend.get_backend(p0, q0, C10, C20, M10, M20)

    # constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1/nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()

    def f(G):
   
        # print("G.shape: ", G.shape)
        # print("C1.shape: ", C1.shape)
        # print("C2.shape: ", C2.shape)
        # print("G", G)
        # print("C1", C1)
        # print("C2", C2)
        return nx.sum((G @ G.T)  * C1) + nx.sum((G.T @ G)  * C2)

    def df(G):
        # Gradient of f(G)=<C1, GG^T> + <C2, G^T G> is 2*(C1G + GC2)
        return 2 * (nx.dot(C1, G) + nx.dot(G, C2))
    
    # armijo is default to False and loss_fun is default to square_loss
    if loss_fun == 'kl_loss':
        armijo = True  # there is no closed form line-search with KL

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)
    else:
        # we are using this line search
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M=0., reg=1., nx=nx, **kwargs)
    
    module_path = inspect.getfile(ot)

    # Get the directory containing the module
    module_directory = os.path.dirname(module_path)

    # print(f"Module path: {module_path}")
    # print(f"Module directory: {module_directory}")

    if log:
   
        res, log = cg_incent(p, q, (1 - alpha) * M1, (1 - alpha) * M2, alpha, f, df, gamma=gamma, G0=G0,
                             line_search=line_search, log=True, numItermax=numItermax,
                             stopThr=tol_rel, stopThr2=tol_abs,
                             rho1=rho1, rho2=rho2,
                             balanced_fallback_threshold=balanced_fallback_threshold,
                             **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        # 'u' and 'v' are dual variables only present in the EMD (balanced) log;
        # mm_unbalanced does not produce them, so we guard with .get()
        log['u'] = log.get('u', None)
        log['v'] = log.get('v', None)
        return res, log

    else:
        return cg_incent(p, q, (1 - alpha) * M1, (1 - alpha) * M2, alpha, f, df, gamma=gamma, G0=G0,
                         line_search=line_search, log=True, numItermax=numItermax,
                         stopThr=tol_rel, stopThr2=tol_abs,
                         rho1=rho1, rho2=rho2,
                         balanced_fallback_threshold=balanced_fallback_threshold,
                         **kwargs)
