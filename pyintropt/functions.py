import numpy as np
import scipy.sparse as sp
import time

from numpy import zeros, finfo, vstack, ones, matrix, empty, where, squeeze, prod, asarray, multiply
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, isspmatrix, isspmatrix, coo_matrix
from scipy.sparse.linalg import spsolve
from pyintropt.ss_spqr import QR

from numba import autojit

mbmat = lambda x: sp.bmat(x).tocsc()

eps = finfo(float).eps
big = 1 / eps**2


def qp_feas(x, c_x, x_l, x_u, c_l, c_u, tol=10.):
    """
    Helper function to test if the current point (in the current quadratic
    programming problem) is feasible.
    Assumes "original" form of problem, not the form with slack variables.
    """
    c = densify(c_x * x)
    #TODO add ability to know where the violation is, and actual tolerance?
    if (x_l <= x).all() and (x <= x_u).all() and (c_l <= c).all() and (c <= c_u).all():
        return True
    return False


def densify(x):
    """
    Helper function to convert to dense, only if needed.
    """
    if isspmatrix(x):
        return x.todense()
    return x


def get_independent_rows(A, tol=1e3*eps, R=None, perm=False):
    """ Code to get independent rows in A, using QR decomposition. """
    if R is None:
        A = A.T.tocsc()
        #Q, R, rank = QR(A)
        qr_out = QR(A, perm)
        R = qr_out[1]
    m, n = R.shape
    cols = []
    row_ind = 0
    for i in range(n):
        if abs(R[row_ind, i]) > tol:
            row_ind += 1
            cols += [i]
        if row_ind == m:
            break
    # un-permutate
    if perm:
        permutate = qr_out[3]
        return permutate[cols]
    else:
        return cols


def get_updated_independent_rows(AT_Q, AT_R, C, tol=1e3*eps):
    """ Code to get independent rows from an update to the QR decomposition. """
    # TODO this entire function is apparently very, very, slow...

    def _givens(i, j, A):
        mat = sp.eye(A.shape[0], format='dok')
        a = A[(i > j) * j + (i < j) * i, j]
        b = A[(i > j) * i + (i < j) * j, j]
        r = (a**2 + b**2)**0.5
        if r == 0:
            c = 1
            s = 0
        else:
            c = a / r
            s = b / r
        mat[i, i] = c
        mat[j, j] = c
        mat[i, j] = -(i > j) * s + (i < j) * s
        mat[j, i] = (i > j) * s - (i < j) * s
        return mat

    def _HH(v1, v2): # already know ||v1||==||v2||==1
        I = sp.eye(len(v1))
        u = v1 - v2
        u = u
        return I - 2 * u * u.T

    Q = AT_Q.copy()
    R = AT_R.copy()
    mc = C.shape[0]
    for i in range(mc):
        m, n = R.shape
        a = C[i, :]

    """
    GIVENS CODE - REALLY SLOW
    R2 = (C * AT_Q).T
    # Has parts of R for C, but needs to be rotated (triangularized)
    R_new = mbmat([[AT_R, R2]])

    tempR = R_new.copy()
    for i in range(AT_R.shape[1], R_new.shape[1]):
        for j in reversed(range(i + 1, R_new.shape[0])):
            g = _givens(j, i, tempR)
            tempR = g.dot(tempR)
            print(i,j)
    #TODO remove some of the non-zero entries, to save size...?
    return get_independent_rows(None, tol=tol, R=tempR)
    """


def factiz(K):
    """
    Helper function to behave the same way scipy.sparse.factorized does, but
    for dense matrices.
    """
    luf = lu_factor(K)
    return lambda x: matrix(lu_solve(luf, x))


def sp_solve(A, b):
    """
    Helper function to do sparse matrix / vector solving, which occasionally
    has trouble if not guarded for loss of numpy dimensions.
    """
    b = asarray(b).reshape(-1,)
    if b.shape == (1,):
        return col(b.squeeze() / A[0, 0])
    else:
        return spsolve(A, b)


def sp_factiz(A):
    """
    Helper function to behave the same way scipy.sparse.factorized does, but
    for to allow for matrix right hand sides.
    """
    from scipy.sparse.linalg import factorized, splu
    from scipy.sparse.linalg.dsolve import _superlu
    A = csc_matrix(A)
    M, N = A.shape
    #Afactsolve = factorized(A)
    Afactsolve = splu(A, options={'IterRefine' : 'DOUBLE', 'SymmetricMode' : True}).solve

    def solveit(b):
        """
        Mainly lifted from scipy.sparse.linalg.spsolve, which doesn't allow for
        the factorization of A to be saved, which should happen in this usage.
        """
        b_is_vector = (max(b.shape) == prod(b.shape))
        if b_is_vector:
            if isspmatrix(b):
                b = b.toarray()
            b = asarray(b)
            b = b.squeeze()
        else:
            if isspmatrix(b):
                b = csc_matrix(b)
            if b.ndim != 2:
                raise ValueError("b must be either a vector or a matrix")

        if M != b.shape[0]:
            raise ValueError("matrix - rhs dimension mismatch (%s - %s)"
                            % (A.shape, b.shape[0]))

        if b_is_vector:
            b = asarray(b, dtype=A.dtype)
            options = dict(ColPerm='COLAMD', IterRefine='DOUBLE', SymmetricMode=True)
            x = _superlu.gssv(N, A.nnz, A.data, A.indices, A.indptr, b, 1,
                            options=options)[0]
        else:
            # Cover the case where b is also a matrix
            tempj = empty(M, dtype=int)
            x = A.__class__(b.shape)
            mat_flag = False
            if b.__class__ is matrix:
                mat_flag = True
            for j in range(b.shape[1]):
                if mat_flag:
                    xj = Afactsolve(squeeze(asarray(b[:, j])))
                else:
                    xj = Afactsolve(squeeze(b[:, j].toarray()))
                w = where(xj != 0.0)[0]
                tempj.fill(j)
                x = x + A.__class__((xj[w], (w, tempj[:len(w)])),
                                    shape=b.shape, dtype=A.dtype)
        return x
    return solveit


def col(x):
    """
    Helper function to return x as a n x 1 numpy matrix.
    """
    return matrix(densify(x)).reshape(-1, 1)


def row(x):
    """
    Helper function to return x as a 1 x n numpy matrix.
    """
    return matrix(x).reshape(1, -1)


def vec_clamp(x, clamp_tol=100.):
    """
    Helper function to clamp a numpy object x to 0 if it is within a tolerance.
    """
    return multiply(x, (abs(x) > clamp_tol * eps))


def extract_row(x, row_list):
    """
    Helper function to deal with pulling out rows for possibly 0xN matrices.
    """
    return x[row_list]


def extract_col(x, col_list):
    """
    Helper function to deal with pulling out cols for possibly Nx0 matrices.
    """
    return x[:, col_list]


def sp_extract_row(x, row_list):
    """
    Helper function to deal with pulling out rows for possibly 0xN sparse
    matrices.
    """
    try:
        to_return = x[row_list]
    except:
        to_return = coo_matrix((0, x.shape[1]), np.float64)
    return to_return


def sp_extract_col(x, col_list):
    """
    Helper function to deal with pulling out cols for possibly Nx0 sparse
    matrices.
    """
    try:
        to_return = x[:, row_list]
    except:
        to_return = x[:, 0:0]
    return to_return


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
            jac = csc_matrix(rows)
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



