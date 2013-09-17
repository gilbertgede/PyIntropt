from numpy import zeros, finfo, vstack, ones, matrix
from scipy.sparse import csr_matrix
eps = finfo(float).eps
big = 1 / eps**2


def factiz(K):
    """
    Helper function to behave the same way scipy.sparse.factorized does, but
    for dense matrices.
    """
    from scipy.linalg import lu_factors, lu_solve
    luf = lu_factors(K)
    return lambda x: lu_solve(luf, x)


def row(x):
    """
    Helper function to return x as a n x 1 numpy matrix.
    """
    return matrix(x).reshape(-1, 1)


def col(x):
    """
    Helper function to return x as a 1 x n numpy matrix.
    """
    return matrix(x).reshape(1, -1)


def vec_clamp(x, clamp_tol=10.):
    """
    Helper function to clamp a numpy object x to 0 if it is within a tolerance.
    """
    return x * (x < clamp_tol * eps)


def approx_jacobian(x0, f, order=2, _fdscale=1./3., sparse=True):
    """
    Uses finite differencing to compute approximate jacobian for provided
    function f about point x0. The code finds the 'optimal' step size, h, for
    the differencing, for the various orders supported.
    See:
    Course Notes: University of Washington, AMATH 301 Beginning Scientific
    Computing. J. N. Kutz, 2009

    """

    # TODO describe _fdscale, and examine why I put comment on M re: max f'''
    M = _fdscale # eq. to max(f'''(x0)), for 2nd order
    if order == 1:
        h = (2. * eps / M) ** (1. / 2.)
        row = lambda dx: ((f(x0 + dx) - f0) / h).T[0]
    elif order == 2:
        h = (3. * eps / M) ** (1. / 3.)
        row = lambda dx: ((f(x0 + dx) - f(x0 - dx)) / (2. * h)).T[0]
    elif order == 4:
        h = (45. * eps / 4. / M) ** (1. / 5.)
        row = lambda dx: ((-f(x0 + 2. * dx) + 8. * f(x0 + dx) - 8. * f(x0
                            - dx) + f(x0 - 2. * dx)) / (12. * h)).T[0]
    else:
        raise ValueError('Not a valid finite difference order')

    # Note: shape should be m x n, wher m is the length of f0
    #       if f0 returns a scalar, returns n x 1
    f0 = f(x0)
    rows = []
    if len(f0) == 0:
        col_vec = True
    else:
        col_vec = False

    dx = zeros((len(x0), 1))
    for i in range(len(x0)):
        dx[i] = h
        rows.append(clamp( row(dx).squeeze() ))
        dx[i] = 0

    if col_vec:
        matrix
        pass
    else:
        if sparse == True:
            jac = csr_matrix(rows)
        else:
            jac = array(rows)

    if jac.shape[0] == 1:
        return jac.T
    elif jac.shape[1] == 1:
        return jac
    elif jac.shape[0] == len(x0):
        return jac.T
    else:
        return jac



