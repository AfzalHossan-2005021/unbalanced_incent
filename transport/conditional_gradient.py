import ot
import numpy as np

from ot.lp import emd
from ot.optim import line_search_armijo
from ot.utils import list_to_array, get_backend

from .linesearch import solve_gromov_linesearch


def generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg1, reg2, lp_solver, line_search,
                                         gamma, G0=None, numItermax=6000, stopThr=1e-9,
                                         stopThr2=1e-9, verbose=False, log=False,
                                         rho1=None, rho2=None,
                                         **kwargs):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.

        The function solves the following optimization problem if set as a conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b} (optional constraint)

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

        The function solves the following optimization problem if set a generalized conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot f(\gamma) + \mathrm{reg_2}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples weights in the target domain

    a: initial distribution(uniform) of sliceA spots
    b: initial distribution(uniform) of sliceB spots

    M1: cosine dist of gene expression matrices of two slices
    M2: jensenshannon dist of niche of two slices
    f : function
        Regularization function taking a transportation matrix as argument
    df: function
        Gradient of the regularization function taking a transportation matrix as argument
    reg1 : float
        Regularization term >0
    reg2 : float,
        Entropic Regularization term >0. Ignored if set to None.
    lp_solver: function,
        linear program solver for direction finding of the (generalized) conditional gradient.
        If set to emd will solve the general regularized OT problem using cg.
        If set to lp_semi_relaxed_OT will solve the general regularized semi-relaxed OT problem using cg.
        If set to sinkhorn will solve the general regularized OT problem using generalized cg.
    line_search: function,
        Function to find the optimal step. Currently used instances are:
        line_search_armijo (generic solver). solve_gromov_linesearch for (F)GW problem.
        solve_semirelaxed_gromov_linesearch for sr(F)GW problem. gcg_linesearch for the Generalized cg.
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Added by Anup Bhowmik
    ------------------------
    gamma: float, optional
        weight of the second regularization term (default is 0.5)
    --------------------------


    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    .. _references_gcg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
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
        transport_cost = (1-alpha) * (nx.sum(M1 * G) + gamma * nx.sum(M2 * G)) + alpha * f(G)

        # without niche aware
        # transport_cost = (1-alpha) * (nx.sum(M1 * G)) + alpha * f(G)

        # KL marginal penalties (unbalanced terms)
        # generalized KL: KL(p||q) = sum_i [p_i*log(p_i/q_i) - p_i + q_i]
        if rho1 is not None and rho2 is not None:
            eps = 1e-16
            row_marginal = nx.sum(G, axis=1)
            col_marginal = nx.sum(G, axis=0)
            kl_source = nx.sum(row_marginal * (nx.log(row_marginal + eps) - nx.log(a + eps)) - row_marginal + a)
            kl_target = nx.sum(col_marginal * (nx.log(col_marginal + eps) - nx.log(b + eps)) - col_marginal + b)
            transport_cost = transport_cost + rho1 * kl_source + rho2 * kl_target

        return transport_cost

    

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


def cg_incent(a, b, M1, M2, reg, f, df, gamma, G0=None, line_search=line_search_armijo,
       numItermax=6000, numItermaxEmd=100000, stopThr=1e-9, stopThr2=1e-9,
       verbose=False, log=False,
       rho1=1.0, rho2=1.0, balanced_fallback_threshold=1e6,
       **kwargs):
    r"""
    Solve the general regularized OT problem with conditional gradient.

    Unbalanced version: the marginal constraints gamma @ 1 = a and gamma.T @ 1 = b
    are softly enforced via KL penalties with weights rho1 (source) and rho2 (target).
    Setting rho1 = rho2 >= balanced_fallback_threshold recovers the balanced (EMD) solution.

    Parameters
    ----------
    rho1 : float, optional
        KL penalty weight for the source marginal constraint. Default 1.0.
        Larger values enforce the source marginal more strictly.
    rho2 : float, optional
        KL penalty weight for the target marginal constraint. Default 1.0.
        Larger values enforce the target marginal more strictly.
    balanced_fallback_threshold : float, optional
        If both rho1 and rho2 >= this value, fall back to exact EMD (balanced).
        Default 1e6.

    All other parameters are identical to the balanced cg_incent in modular_incent.

    References
    ----------
    .. [1] Ferradans et al. (2014). Regularized discrete optimal transport.
    .. [U] Chapel et al. (2021). Unbalanced Optimal Transport through Non-negative
           Penalized Linear Regression. NeurIPS.
    """

    def lp_solver(a, b, M, **kwargs):
        # Balanced fallback: if both penalties are very large, use exact EMD
        if rho1 >= balanced_fallback_threshold and rho2 >= balanced_fallback_threshold:
            return emd(a, b, M, numItermaxEmd, log=True)
        # Unbalanced: KL marginal relaxation via multiplicative updates (no extra entropy)
        return ot.unbalanced.mm_unbalanced(a, b, M,
                                           reg_m=(rho1, rho2),
                                           div='kl',
                                           log=True)

    return generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg, None, lp_solver, line_search, G0=G0,
                                               gamma=gamma, numItermax=numItermax, stopThr=stopThr,
                                               stopThr2=stopThr2, verbose=verbose, log=log,
                                               rho1=rho1, rho2=rho2,
                                               **kwargs)
