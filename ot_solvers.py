import os
import inspect
import numpy as np
import ot
from ot.unbalanced import mm_unbalanced
from ot.optim import line_search_armijo
from ot.utils import list_to_array, get_backend

def solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M, reg,
                            alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the FW iterations
    """
    if nx is None:
        G, deltaG, C1, C2, M = ot.utils.list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = ot.backend.get_backend(G, deltaG, C1, C2)
        else:
            nx = ot.backend.get_backend(G, deltaG, C1, C2, M)

    # ensure all tensors share the same dtype when using torch backend
    if isinstance(nx, ot.backend.TorchBackend):
        # use dtype of G for all
        dt = G.dtype
        if isinstance(deltaG, torch.Tensor):
            deltaG = deltaG.to(dt)
        if isinstance(C1, torch.Tensor):
            C1 = C1.to(dt)
        if isinstance(C2, torch.Tensor):
            C2 = C2.to(dt)
        if isinstance(M, torch.Tensor):
            M = M.to(dt)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG))

    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G


def generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg1, reg2, lp_solver, line_search,
                                         gamma, G0=None, numItermax=6000, stopThr=1e-9,
                                         stopThr2=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.
    """

    # new code starts
    a, b, M1, M2, G0 = list_to_array(a, b, M1, M2, G0)
    if isinstance(M1, int) or isinstance(M1, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M1)

    if isinstance(M2, int) or isinstance(M2, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M2)

    # new code ends

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        # G0 is kept None by default
        
        G2 = nx.outer(a, b)
        # make G uniform distribution matrix of size (ns, nt)
        G1 = nx.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

        # todo: integrate the cell-type aware initialization


        G = G1
        # print the shape of G
        # print("G shape: ", G.shape)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    def cost(G):
        alpha = reg1
        
        # with niche aware
        return (1-alpha) * (nx.sum(M1 * G) + gamma * nx.sum(M2 * G)) + alpha * f(G)

        # without niche aware
        # return (1-alpha) * (nx.sum(M1 * G)) + alpha * f(G)

    

    cost_G = cost(G)
    if log:
        log['loss'].append(cost_G)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    while loop:

        it += 1
        old_cost_G = cost_G
        # problem linearization
        # gradient descent
        Mi = M1 + reg1 * df(G)

        if not (reg2 is None):
            Mi = Mi + reg2 * (1 + nx.log(G))
        # set M positive
        Mi = Mi + nx.min(Mi)

        # solve linear program
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        # line search
        deltaG = Gc - G

        alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G)
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log['loss'].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, relative_delta_cost_G, abs_delta_cost_G))

    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G


def cg_incent(a, b, M1, M2, reg, f, df, gamma, tau, G0=None, line_search=line_search_armijo,
       numItermax=6000, numItermaxEmd=100000, stopThr=1e-9, stopThr2=1e-9,
       verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem with conditional gradient
    """

    def lp_solver(a, b, M, **kwargs):
        # Use unbalanced MM solver instead of exact EMD
        # tau is the marginal relaxation penalty
        return mm_unbalanced(a, b, M, reg_m=tau, numItermax=numItermaxEmd, log=True)

    return generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg, None, lp_solver, line_search, G0=G0,
                                               gamma = gamma, numItermax=numItermax, stopThr=stopThr,
                                               stopThr2=stopThr2, verbose=verbose, log=log, **kwargs)


def fused_gromov_wasserstein_incent_heuristic_cg(M1, M2, C1, C2, p, q, gamma, tau=0.1, G_init = None, loss_fun='square_loss', alpha = 0.1, beta = 0.8, armijo=False, log=False,numItermax=6000, tol_rel=1e-9, tol_abs=1e-9, use_gpu = False, **kwargs):
    """
    This method is written by Anup Bhowmik, CSE, BUET
    Adapted for Unbalanced OT using Heuristic Conditional Gradient.
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
        return nx.sum((G @ G.T)  * C1) + nx.sum((G.T @ G)  * C2)

    def df(G):
        # Gradient of f(G)=<C1, GG^T> + <C2, G^T G> is 2*(C1G + GC2)
        # ensure cost matrices have the same dtype as G to avoid torch errors
        if isinstance(nx, ot.backend.TorchBackend):
            c1 = C1.to(G.dtype)
            c2 = C2.to(G.dtype)
        else:
            # numpy backend
            c1 = C1.astype(G.dtype)
            c2 = C2.astype(G.dtype)
        return 2 * (nx.dot(c1, G) + nx.dot(G, c2))
    
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

    if log:
   
        res, log = cg_incent(p, q, (1 - alpha) * M1, (1 - alpha) * M2, alpha, f, df, gamma = gamma, tau=tau, G0 = G0, line_search = line_search, log=True, numItermax=numItermax, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        return res, log

    else:
        return cg_incent(p, q, (1 - alpha) * M1, (1 - alpha) * M2, alpha, f, df, gamma = gamma, tau=tau, G0 = G0, line_search = line_search, log=True, numItermax=numItermax, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)


def fused_gromov_wasserstein_incent_exact_bcd(M1, M2, C1, C2, p, q, gamma, tau=0.1, G_init=None, alpha=0.1, numItermax=6000, tol_rel=1e-9, tol_abs=1e-9, use_gpu=False, log=False, **kwargs):
    """
    Exact Unbalanced FGW using Block Coordinate Descent (Majorization-Minimization).
    This avoids the line-search issues of the heuristic CG approach.
    """
    p, q, C1, C2, M1, M2 = ot.utils.list_to_array(p, q, C1, C2, M1, M2)
    nx = ot.backend.get_backend(p, q, C1, C2, M1, M2)
    
    M = (1 - alpha) * (M1 + gamma * M2)
    
    if G_init is None:
        G = p[:, None] * q[None, :]
    else:
        G = nx.copy(G_init)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            G = G.cuda()
            
    if log:
        log_dict = {'loss': []}
        
    def cost(G):
        return nx.sum(M * G) + alpha * (nx.sum((G @ G.T) * C1) + nx.sum((G.T @ G) * C2))
        
    cost_G = cost(G)
    if log:
        log_dict['loss'].append(cost_G)
        
    for it in range(numItermax):
        old_cost_G = cost_G
        
        # Linearize GW cost (INCENT specific gradient)
        # ensure cost matrices and G share dtype
        if isinstance(nx, ot.backend.TorchBackend):
            c1 = C1.to(G.dtype)
            c2 = C2.to(G.dtype)
        else:
            c1 = C1.astype(G.dtype)
            c2 = C2.astype(G.dtype)
        grad_GW = 2 * (nx.dot(c1, G) + nx.dot(G, c2))
        
        # Total linearized cost matrix
        C_G = M + alpha * grad_GW
        
        # Solve unbalanced linear OT
        G = ot.unbalanced.mm_unbalanced(p, q, C_G, reg_m=tau, numItermax=1000)
        
        cost_G = cost(G)
        if log:
            log_dict['loss'].append(cost_G)
            
        abs_delta_cost_G = abs(cost_G - old_cost_G)
        if abs_delta_cost_G < tol_abs or abs_delta_cost_G / abs(cost_G) < tol_rel:
            break
            
    if log:
        log_dict['fgw_dist'] = cost_G
        return G, log_dict
    return G


def fused_gromov_wasserstein_incent_entropic(M1, M2, C1, C2, p, q, gamma, tau=0.1, epsilon=0.01, G_init=None, alpha=0.1, numItermax=6000, tol_rel=1e-9, tol_abs=1e-9, use_gpu=False, log=False, **kwargs):
    """
    Entropic Unbalanced FGW using Sinkhorn projections.
    This is the most robust and standard way to solve UFGW, though it introduces slight blurring.
    """
    p, q, C1, C2, M1, M2 = ot.utils.list_to_array(p, q, C1, C2, M1, M2)
    nx = ot.backend.get_backend(p, q, C1, C2, M1, M2)
    
    M = (1 - alpha) * (M1 + gamma * M2)
    
    if G_init is None:
        G = p[:, None] * q[None, :]
    else:
        G = nx.copy(G_init)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            G = G.cuda()
            
    if log:
        log_dict = {'loss': []}
        
    def cost(G):
        return nx.sum(M * G) + alpha * (nx.sum((G @ G.T) * C1) + nx.sum((G.T @ G) * C2))
        
    cost_G = cost(G)
    if log:
        log_dict['loss'].append(cost_G)
        
    for it in range(numItermax):
        old_cost_G = cost_G
        
        # Linearize GW cost (INCENT specific gradient)
        # ensure cost matrices and G share dtype
        if isinstance(nx, ot.backend.TorchBackend):
            c1 = C1.to(G.dtype)
            c2 = C2.to(G.dtype)
        else:
            c1 = C1.astype(G.dtype)
            c2 = C2.astype(G.dtype)
        grad_GW = 2 * (nx.dot(c1, G) + nx.dot(G, c2))
        C_G = M + alpha * grad_GW
        
        # Solve entropic unbalanced linear OT
        G = ot.unbalanced.sinkhorn_unbalanced(p, q, C_G, reg=epsilon, reg_m=tau, numItermax=1000)
        
        cost_G = cost(G)
        if log:
            log_dict['loss'].append(cost_G)
            
        abs_delta_cost_G = abs(cost_G - old_cost_G)
        if abs_delta_cost_G < tol_abs or abs_delta_cost_G / abs(cost_G) < tol_rel:
            break
            
    if log:
        log_dict['fgw_dist'] = cost_G
        return G, log_dict
    return G
