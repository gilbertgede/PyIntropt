from numpy import matrix, array, eye, asarray, vstack
from .qp_solver import qp_dispatch
from pyintropt.functions import eps, col, big
from numpy.linalg import inv
from scipy.sparse import issparse


def _slack_finder(y, yl, yu, lambd, rho=None):
    """ Solves for 2.28/2.29 in [Betts2010] """
    y = asarray(y).squeeze()
    yl = asarray(yl).squeeze()
    yu = asarray(yu).squeeze()
    lambd = asarray(lambd).squeeze()
    out = y * 0
    if rho is not None:
        y_term = y - lambd / asarray(rho).squeeze()
    else:
        y_term = y
    return matrix((yl > y_term) * yl + (yu < y_term) * yu + ((yl < y_term) * (y_term < yu)) * y_term).reshape(-1, 1)


def sqp_solver(x0, f, c, f_x, c_x, H, x_l, x_u, c_l, c_u):
    """
    An implementation of the SQP algorithm shown be Betts in [Betts2010].

    Parameters
    ==========
    x0 : numpy array
    The inital point to start the optimization from.
    f : function
    The objective function.
    c : function
    The constraint function.
    f_x : function
    The function for the gradient of the objective function w.r.t. x.
    c_x : function
    The function for the gradient of the constraint function w.r.t. x.
    H : function
    The function for the Hessian of the problem Lagrangian.
    x_l : numpy array
    Lower simple bounds on x.
    x_u : numpy array
    Upper simple bounds on x.
    c_l : numpy array
    Lower simple bounds on c(x).
    c_u : numpy array
    Upper simple bounds on c(x).
    """

    # some constants
    psi_0 = eps

    x = matrix(x0.copy()).reshape(-1, 1)
    n = len(x)
    c_k = c(x)
    m = len(c_k)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)


    sparse = True
    if sparse:
        from scipy.sparse import eye, dia_matrix
        eye = eye(n + m).tocsc()
        diag = lambda x: dia_matrix((asarray(x).squeeze(), [0]), shape=(len(x), len(x)))
    else:
        from numpy import eye, diag
        eye = eye(n + m)
        diag = lambda x: diag(asarry(x).squeeze())


    active_set = set()
    k = 0

    """ Stuff for initialization """
    f_k = f(x)
    c_k = col(c(x))
    f_x_k = f_x(x)
    c_x_k = c_x(x)
    H_k = eye[:n, :n]


    out = qp_dispatch(x, f_x_k, H_k, c_x_k, c_l - c_k, c_u - c_k, x_l, x_u, active_set)

    p = out[0] - x
    active_set = out[1]
    qp_multipliers = out[2]
    active_list = list(active_set)
    qp_multipliers = eye[active_list].T * qp_multipliers


    # TODO check if these should be 0 for initial penalty parameter calculation
    nu_k = 0 * qp_multipliers[:n]       # simple bounds
    lambda_k = 0 * qp_multipliers[n:]   # constraint bounds
    # calculate penalty parameters
    t = _slack_finder(x, x_l, x_u, nu_k)
    s = _slack_finder(c_k, c_l, c_u, lambda_k)
    delta_t = p + x - t
    delta_s = c_x_k * p + c_k - s
    sigma = - 0.5 * p.T * H_k * p + nu_k.T * delta_t + lambda_k.T * delta_s - psi_0 * (x - t).T * (x - t) - psi_0 * (c_k - s).T * (c_k - s)
    a = matrix(vstack([asarray(x - t)**2, asarray(c_k - s)**2]))
    psi = a * inv(a.T * a) * sigma #TODO change to solve
    xi = psi[:n] + psi_0
    theta = psi[n:] + psi_0
    x += p
    nu_k = qp_multipliers[:n]       # simple bounds
    lambda_k = qp_multipliers[n:]   # constraint bounds
    # TODO add actual merit function value
    # TODO also check that merit function actually shows reduction (check sign
    # basically)
    """ End stuff for initialization """

    while True:
        print('Iteration of SQP: ' + str(k))
        f_k = f(x)
        c_k = col(c(x))
        f_x_k = f_x(x)
        c_x_k = c_x(x)
        H_l_k = H(x, lambda_k)

        #TODO improve check for termination - needs multiplier sign check
        # (sort of) using termination criteria from [Gill2005]
        if k > 0:
            eps_p = 1.e-6 # tau_p in paper
            eps_d = 1.e-6 # tau_d in paper
            tau_x = eps_p * (1 + x.abs().max())
            tau_l = eps_d * (1 + lambda_k.abs().max()) # tau_pi in paper
            # check constraints
            # check lambda signs
            term1 = (c_l <= c_k) and (c_k <= c_u) and (x_l <= x) and (x <= x_u)
            term3 = abs(f_x_k - c_x_k.T * lambda_k) <= tau_l
            if term1 and term3:
                break

        # TODO add (proper) hessian guard from [Betts2010]
        from scipy.sparse.linalg import eigsh
        i = 0.001
        H_k = H_l_k.copy()
        if not issparse(H_l_k):
            from scipy.sparse import csc_matrix
            H_k = csc_matrix(H_k)

        while (eigsh(H_k, k=1, which='SA', maxiter=100000,
               return_eigenvectors=False)[0] < 0):
            H_k = H_k + i * eye[:n, :n]
            i *= 10

        out = qp_dispatch(x, f_x_k, H_k, c_x_k, c_l - c_k, c_u - c_k, x_l, x_u, active_set)

        p = out[0]
        active_set = out[1]
        qp_multipliers = out[2]
        active_list = list(active_set)
        qp_multipliers = eye[active_list].T * qp_multipliers

        qp_nu = qp_multipliers[:n]       # simple bounds
        qp_lambda = qp_multipliers[n:]   # constraint bounds
        delta_nu = nu_k - qp_nu
        delta_lambda = lambda_k - qp_lambda

        # calculate penalty parameters
        t = _slack_finder(x, x_l, x_u, nu_k, xi)
        s = _slack_finder(c_k, c_l, c_u, lambda_k, theta)
        delta_t = p + x - t
        delta_s = c_x_k * p + c_k - s

        # Compute new penalty functions
        a = matrix(vstack([asarray(x - t)**2, asarray(c_k - s)**2]))
        sigma = - 0.5 * p.T * H_k * p + nu_k.T * delta_t + lambda_k.T * delta_s - 2 * delta_nu.T * (x - t) - 2 * delta_lambda.T * (c_k - s) - psi_0 * (x - t).T * (x - t) - psi_0 * (c_k - s).T * (c_k - s)
        psi = a * inv(a.T * a) * sigma #TODO change to solve
        xi = psi[:n] + psi_0
        theta = psi[n:] + psi_0
        xi_mat = diag(xi)
        theta_mat = diag(theta)
        # define merit function
        M = lambda x, n, l, t, s: f(x) - n.T * (x - t) - l.T * (c(x) - s) + 0.5 * (x - t).T * xi * (x - t) + 0.5 * (c(x) - s).T * theta * (c(x) - s)
        dMda_0 = lambda x, n, l, t, s: f_x(x) * p - l.T * (c_x(x) * p - delta_s) - delta_lambda.T * (c(x) - s) - n.T * (p - delta_t) - delta_nu.T * (x - t) + (c_x(x) * p - delta_s).T * theta_mat * (c(x) - s) + (p - delta_t).T * xi_mat * (x - t)
        # TODO define merit function derivative as a function of alpha, for interpolation line search

        # do line search
        alpha = 1
        kappa1 = 1e-5
        kappa2 = 0.9

        M0 = M(x, nu_k, lambda_k, t, s)
        dMda0 = dMda_0(x, nu_k, lambda_k, t, s)

        while True:
            if M(x + alpha * p, nu_k + alpha * delta_nu, lambda_k + alpha * delta_lambda, t + alpha * delta_t, s + alpha * delta_s) - M0 < kappa1 * alpha * dMda0:
                break
            alpha *= 0.5

        # take step
        x = x + alpha * p
        nu_k = nu_l + alpha * delta_nu
        lambda_k = lambda_k + alpha * delta_lambda
        k += 1

    return x
