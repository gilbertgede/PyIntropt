import numpy as np
import scipy.sparse as sp

from numpy import bmat, nonzero, maximum, asarray, count_nonzero
from numpy.linalg import LinAlgError
from scipy.sparse.linalg import eigsh
from pyintropt.functions import col, eps, big, vec_clamp, get_independent_rows
from pyintropt.functions import sp_extract_row as extract, sp_solve as solve
from pyintropt.ss_spqr import QR

import time

mbmat = lambda x: sp.bmat(x).tocsc()

class ___QP_STALLING(Exception):
    def __str__(self):
        return("QP stalled, wasn't caught right")
    __repr__ = __str__
class ___QP_INFEASIBLE(Exception):
    def __str__(self):
        return("QP is infeasible, wasn't caught right")
    __repr__ = __str__
class ___QP_MAXITER(Exception):
    pass


def qp_dispatch(x, c, f_xx, c_x, c_l, c_u, x_l, x_u, active_set=[]):
    """
    Dispatch to QP algorithm taking care of slack variables, infeasabilities, etc.
    Also handles sparsity issues, and proper shapes of input data.
    Takes in problem of the form:

        min   c.T * x + 1/2 * x.T * H * x
        s.t.  c_l <= c_x * x <= c_u,
              x_l <=   I * x <= x_u
    """

    # Auto-checking for sparse
    if not (sp.isspmatrix(f_xx) and sp.isspmatrix(c_x)):
        raise ValueError('Need to supply sparse matrices')

    n = max(x.shape)
    m = c_x.shape[0]
    active_set = set(active_set)

    x = col(x)
    c = col(c)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)
    c_x  = sp.csc_matrix(c_x)
    f_xx = sp.csc_matrix(f_xx)

    # Check to see if we should even be solving this problem
    # TODO: add check for positive definiteness
    if m > n:
        raise LinAlgError('Overdefined A not working yet')
    Q, R, rank = QR(c_x)
    if rank != m:
        raise LinAlgError('Constraint matrix is not full row rank')
    """ # TODO add this for sparse
        if not (eigvals(f_xx) > 0).all():
            raise LinAlgError('Not positive definite')
    """

    """
    testing linear programming feasibility phase
    ""
    from pyintropt.sqp.lp_solver import lp_dispatch
    for i in range(len(x)):
        if x[i] < x_l[i]:
            x[i] = x_l[i]
        if x[i] > x_u[i]:
            x[i] = x_u[i]
    s = c_x * x
    for i in range(m):
        if s[i] < c_l[i]:
            s[i] = c_l[i]
        if s[i] > c_u[i]:
            s[i] = c_u[i]
    v = np.zeros((m, 1))
    w = np.zeros((m, 1))
    temp = c_x * x - s
    for i in range(m):
        if temp[i] > 0:
            v[i] = temp[i]
        if temp[i] < 0:
            w[i] = -temp[i]
    eye = sp.eye(m, m)
    zeros = np.zeros((m, 1))
    ones = np.ones((m, 1))
    temp_x = bmat([[x], [s], [v], [w]])
    temp_A = mbmat([[c_x, -eye, -eye, eye]])
    temp_l = bmat([[x_l], [c_l], [zeros], [zeros]])
    temp_u = bmat([[x_u], [c_u], [big * ones], [big * ones]])
    temp_c = bmat([[np.zeros((n + m, 1))], [ones], [ones]])

    print(abs(temp_A * temp_x).max())
    fixed        = ((temp_l == temp_x) | (temp_x == temp_u)).squeeze().tolist()
    active_setlp   = set(nonzero(fixed)[0].tolist())
    temp_c[:] = 1
    xlp, active_setlp = lp_dispatch(temp_x, temp_c, temp_A, temp_l, temp_u, active_setlp)
    temp_c = bmat([[np.zeros((n + m, 1))], [ones], [ones]])
    xlp, active_setlp = lp_dispatch(xlp, temp_c, temp_A, temp_l, temp_u, active_setlp)
    print('LP result: ' + str(temp_c.T * xlp))
    x = xlp[:n]


    ""
    end testing linear programming feasibility phase
    """


    # Transforming to using slack variables, with only x and s bounded on both
    # sides
    xp, cp, Hp, Ap, lp, up = slackify(x, c, f_xx, c_x, x_l, x_u, c_l, c_u)

    # The solve part, and transforming back from slackified
    x, active_set, mu = qp_antistall(xp * 0, cp + Hp * xp, Hp, Ap, vec_clamp(lp - xp), vec_clamp(up - xp), active_set)
    x = vec_clamp(xp + x)[:n, 0]
    mu_ineq = c + f_xx * x - c_x.T * mu
    return x, active_set, mu, mu_ineq


def slackify(x, c, H, c_x, x_l, x_u, c_l, c_u):
    m, n = c_x.shape

    zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
    eye   = sp.eye(m, m).tocsc()

    xp = bmat([[                 x],
               [           c_x * x]])
    lp = bmat([[               x_l],
               [               c_l]])
    up = bmat([[               x_u],
               [               c_u]])
    cp = bmat([[                 c],
               [ col(zeros[:m, 0])]])
    Hp   = mbmat([[             H, zeros[:n, :m]],
                  [ zeros[:m, :n], zeros[:m, :m]]]).tocsc()
    Ap   = mbmat([[c_x, -eye[:m, :m]]]).tocsc()

    return xp, cp, Hp, Ap, lp, up


def qp_antistall(x, c, H, A, x_l, x_u, active_set):
    """
    Used to prevent stalling.
    """
    while True:
        try:
            #print('pre stalling exception')
            x, active_set, mu = qp_feasdispatch(x, c, H, A, x_l, x_u, active_set)
            break
        except ___QP_STALLING as e:
            x, active_set, mu = e.args
            #print('stalled')
            x_l2 = x_l.copy()
            x_u2 = x_u.copy()
            rand_perturb = abs(np.random.randn(*x_l.shape)) / 10000.
            x_l2 -= rand_perturb
            x_u2 += rand_perturb
            x, active_set, mu = qp_antistall(x.copy(), c, H, A, x_l2, x_u2, active_set)
            #print('post antistall call')
    return x, active_set, mu


def qp_feasdispatch(x, c, H, A, x_l, x_u, active_set):
    """
    Used to call appropriate methods for finding a feasible point.
    """
    m, n = A.shape
    zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
    I = sp.eye(len(x), len(x)).tocsc()
    ones  = np.matrix(np.ones((n + m, 1)))

    run = False
    # Check to see if bounds have been violated
    if (x < x_l).any() or (x > x_u).any():
            run = True
    for i in range(len(x)):
        if x[i, 0] < x_l[i, 0]:
            x[i, 0] = x_l[i, 0]
        elif x[i, 0] > x_u[i, 0]:
            x[i, 0] = x_u[i, 0]
    # Check to see if EQ constraints are violated
    if abs(A * x).max() > 1.e3 * eps:
        run = True
    if run:
        while True:
            #print('pre feas search')
            try:
                out = qp_ns(x * 0, c * 0, I, A, vec_clamp(x_l - x), vec_clamp(x_u - x), active_set, ress0=A*x)
                x = vec_clamp(x + out[0])
                active_set = out[1]
                mu = out[2]
                # At this point, feasibility is assumed
                break
            except ___QP_STALLING as e:
                x_error, active_set, mu = e.args
                x = vec_clamp(x + x_error)
                raise ___QP_STALLING(x, active_set, mu)
            except ___QP_INFEASIBLE as e:
                rho = 1.e6
                #print('infeasibility detected')
                x_error, active_set, mu = e.args
                x = vec_clamp(x + x_error)
                x2 = col(zeros[:m + n, 0])
                I2 = mbmat([[I, None], [None, rho * I[:m, :m]]])
                xl2 = bmat([[vec_clamp(x_l - x)], [-big * ones[:m, 0]]])
                xu2 = bmat([[vec_clamp(x_u - x)], [ big * ones[:m, 0]]])
                A2 = mbmat([[A, I[:m, :m]]])
                c2 = col(zeros[:m + n, 0])
                out = qp_ns(x2, c2, I2, A2, xl2, xu2, [], ress0=A*x)#, infeasfix=True)
                x = vec_clamp(x + out[0][:n, 0])
                active_set = out[1]
                #print('infeas_fix')

    for i in range(n):
        if abs(x[i] - x_l[i]) < 1.e3 * eps:
            x[i] = x_l[i]
        elif abs(x[i] - x_u[i]) < 1.e3 * eps:
            x[i] = x_u[i]

    out = qp_ns(x, c, H, A, x_l, x_u, active_set)
    x = vec_clamp(out[0])
    active_set = out[1]
    mu = out[2]
    return x, active_set, mu


def qp_ns(x, c, H, A, x_l, x_u, active_set, ress0=None, infeas_fix=False):
    """
    Quadratic programming solver.

    Note that the function takes in c, not f_x - do the appropriate conversion
    before hand.
    Assumes problem of the form:

    min   c.T * x + 1/2 * x.T * H * x
    s.t.  A * x = 0
          x_l <= x <= x_u

    """

    def _print_step(i, x, num_actv, alpha):
        if find_feas:
            outstr = '    F:'
        else:
            outstr = '      '
        blue = '\033[34m'
        good = '\033[32m'
        bad  = '\033[31m'
        end  = '\033[0m'
        outstr += ' Iter: %7d   ' % i
        val = (c.T * x + 0.5 * x.T * H * x)[0, 0]
        outstr += 'Obj. val: ' + blue + '%4.6e    ' % val + end
        outstr += 'Active set size: %7d  ' % num_actv

        leader = bad
        if ress0 is not None:
            if max(abs(A * x + ress0)) < 1.e3 * eps:
                leader = good
            outstr += '|Ax+r|= ' + leader + '%2.3e ' % max(abs(A * x + ress0)) + end + '    '
        else:
            if max(abs(A * x)) < 1.e3 * eps:
                leader = good
            outstr += '  |Ax|= ' + leader + '%2.3e ' % max(abs(A * x)) + end + '    '

        l = (x_l <= x).all()
        u = (x_u >= x).all()
        llead = bad
        ulead = bad
        if l:
            llead = good
        if u:
            ulead = good
        outstr += llead + 'x_l <= x' + end + ', ' + ulead + 'x <= x_u' + end
        outstr += '      \u03B1: %3.7f' % alpha
        return outstr

    def _min_ratio_test(x, p, l, u):
        """
        Steplength procedure already in place, placed into a function to maybe
        utilize methods in [Gill1988].
        """

        p2 = p + eps * (abs(p) < eps)
        alpha_l = (l - x) / p2
        alpha_u = (u - x) / p2
        tol = eps**(2. / 3.)    # from [Gill1988], TODO re-examine maybe?
        alphas = (asarray(p < -tol) * asarray(alpha_l) +
                  asarray(p >  tol) * asarray(alpha_u) +
                  asarray(p <  tol) * asarray(p > -tol) * asarray(ones[:n, 0] * big))
        alphas = np.matrix(alphas).reshape(-1, 1)
        alpha = min(alphas.min(), 1)
        ind = asarray(nonzero(alphas == alpha)[0][0]).squeeze().tolist()
        try:
            ind = list(ind)
        except:
            ind = [ind]
        return (alpha, ind)

    def _free(x, active_set, free_set, mu_ineq, stalling_count):
        """
        Putting the code used to move variables between sets when freeing in
        one place.
        """
        freed = False
        active_list = sorted(list(active_set))
        mu_ineq = asarray(mu_ineq)
        at_upper = asarray(abs(x - x_u) < eps)[active_list]
        at_lower = asarray(abs(x - x_l) < eps)[active_list]
        upper = at_upper * (mu_ineq > 0) * abs(mu_ineq) * (at_upper != at_lower)
        lower = at_lower * (mu_ineq < 0) * abs(mu_ineq) * (at_upper != at_lower)
        maxval = maximum(upper, lower)
        if max(maxval) > 0:
            max_ind1 = nonzero(maxval == max(maxval))[0]
            for ii in range(max(max_ind1.shape)):
                ind = active_list[max_ind1[ii]]
                #print('freeing: ' + str(ind))
                free_set.add(ind)
                active_set -= {ind}
                stalling_count[ind] += 1
                freed = True
        return freed

    def _fix(x, active_set, free_set, alpha_ind, stalling_count):
        """
        Putting the code to move variables between sets when fixing in one
        place.
        """
        for ii in range(len(alpha_ind)):
            ind = alpha_ind[ii]
            if (ind in active_set) or (ind in fixed_set):
                raise ValueError('That variable is already fixed...')
            if abs(x[ind, 0] - x_l[ind, 0]) < 1.e3 * eps:
                x[ind, 0] = x_l[ind, 0]
            elif abs(x[ind, 0] - x_u[ind, 0]) < 1.e3 * eps:
                x[ind, 0] = x_u[ind, 0]
            else:
                raise ValueError("That index isn't actually close...")

            #print('fixing:  ' + str(ind))
            stalling_count[ind] += 1
            active_set.add(ind)
            free_set  -= {ind}

    def _solve_KKT(H, A, g, ress, C_f, C_nfx):
        """    Putting code to solve KKT system in one place.    """
        # TODO fix it so the solve doesn't sometimes get errors?
        o = A.shape[0]
        K = mbmat([[ H,          A.T],
                   [ A, zeros[:o, :o]]])
        f = bmat([[        g],
                  [col(ress)]])
        xx = col(solve(K, f))
        p = - xx[:n]
        p = C_f.T * C_f * p
        mu = xx[n : n + m]

        if abs(K * xx - f).max() > 1e5 * eps:
            print('solve failed')
            rows_to_keep = get_independent_rows(A, 1e3*eps)
            ress = ress[rows_to_keep, 0]
            A = extract(A, rows_to_keep)
            o = A.shape[0]
            K = mbmat([[ H,          A.T],
                       [ A, zeros[:o, :o]]])
            f =  bmat([[         g],
                       [ col(ress)]])
            xx = col(solve(K, f))
            p = - xx[:n]
            p = C_f.T * C_f * p
            mu = xx[n : n + m]

        if abs(K * xx - f).max() > 1e5 * eps:
            print('solve failed')
            #raise Exception('Solve Still Failed!!!')

        return p, mu

    def _get_BSN(B, S, N, A, free_set, active_set):
        """    Find independent, fixed, extra columns of A    """
        for i in free_set:
            if i in N:
                N -= {i}
                S.add(i)
        for i in active_set:
            if i in S:
                S -= {i}
                N.add(i)

        for i in active_set:
            if i in B and len(S) != 0:
                tempB = A[:, sorted(list(B.difference({i,})))].tocsc()
                tempS = A[:, sorted(list(S))]
                Q, R, rank = QR(tempB.tocsc())
                Z = Q[:, -1]
                outside_null = Z.T * tempS
                try:
                    ind = nonzero(vec_clamp(asarray(outside_null).squeeze()))[0][0]
                    B -= {i}
                    N.add(i)
                    B.add(ind)
                    S -= {ind}
                except:
                    pass
        return B, S, N


    m, n = A.shape

    if ress0 is not None:
        find_feas = True
    else:
        find_feas = False

    zeros = sp.coo_matrix((2 * n, 2 * n), np.float64).tocsc()
    eye   = sp.eye(n, n).tocsc()
    ones  = np.matrix(np.ones((n + m, 1)))
    Cmat = eye[:n, :n]

    stalling_count = [0]*n

    fixed        = asarray(x_l == x_u).squeeze().tolist()
    fixed_set    = set(nonzero(fixed)[0].tolist())
    free_set     = set(list(range(n)))
    active_set   = set(active_set).union(fixed_set)
    free_set    -= active_set

    B  = set(get_independent_rows(A.T))
    S  = set(list(range(n)))
    S -= B
    N  = set()


    n0 = len(free_set)
    #print('\n\nInitial Sizes - n: %8d, m: %8d, n0: %8d' % (n, m, n0))
    if find_feas:
        #print('Initial Feasibility Violation: ' + str(abs(ress0).max()))
        pass

    g = col(c + H * x) * (not find_feas)
    mu = 0
    mu_ineq = 0
    i = 0
    alpha = 0
    while True:
        B, S, N = _get_BSN(B, S, N, A, free_set, active_set)
        # Need to compute some transformation and sizes for this iteration
        C_f   = extract(Cmat, sorted(list(free_set)))       # Current free (all) variables
        C_aN  = extract(Cmat, sorted(list(N)))              # Current fixed (activated) variables
        C_a   = extract(Cmat, sorted(list(active_set)))     # Current fixed (activated) variables
        num_actv = len(N)     # no. of fixed variables
        print(_print_step(i, x, len(active_set), alpha))

        A_hat = mbmat([[   A],
                       [C_aN]])
        if ress0 is None:
            ress_hat = zeros[: m + num_actv, 0]
        else:
            ress = ress0 + A * x
            ress_hat = bmat([[ress], [col(zeros[:num_actv, 0])]])

        # Compute step, and step length
        p, mu = _solve_KKT(H, A_hat, g, ress_hat, C_f, Cmat)
        alpha, alpha_ind = _min_ratio_test(x, p, x_l, x_u)

        # take the step
        x = x + alpha * p
        g = col(c + H * x) * (not find_feas)
        mu_ineq = C_a * g - C_a * A.T * mu
        i += 1

        if find_feas:
            if abs(A * x + ress0).max() < 1.e3 * eps:
                break
        # Can't take full step, take largest without crossing a bound
        if alpha < 1:
            if find_feas:               # Iff feasibility phase, we will 'shed'
                if num_actv > 0:        # bad parts of active set on partial steps
                    _free(x, active_set, free_set, mu_ineq, stalling_count)
            _fix(x, active_set, free_set, alpha_ind, stalling_count)

        # Taking full step, checking if any constraints need to be freed
        else:
            if num_actv == 0:           # If no active constraints, and a full
                break                   # step, finished

            freed = _free(x, active_set, free_set, mu_ineq, stalling_count)
            if not freed:               # If nothing was freed, finished
                if find_feas:
                    raise ___QP_INFEASIBLE(x, active_set, mu)
                #print('All lagrange multipliers are good!')
                break

        if max(stalling_count) > 5:
            raise ___QP_STALLING(x, active_set, mu)

        if i > 2 * (n):
            raise ___QP_MAXITER(x, active_set, mu)


    print(_print_step(i, x, len(active_set), alpha))
    return x, active_set, mu, False

