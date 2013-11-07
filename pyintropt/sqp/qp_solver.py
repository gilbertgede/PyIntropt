import numpy as np
import scipy.sparse as sp

from numpy import bmat, nonzero, maximum, multiply, asarray, matrix, count_nonzero
from numpy.linalg import eigvals, matrix_rank, LinAlgError # TODO should be doing sparse versions of these...
from numpy.random import randint
from pyintropt.functions import col, eps, big, densify, vec_clamp, get_independent_rows
from pyintropt.functions import sp_extract_row as extract, sp_factiz as factorized, sp_solve as solve
from scipy.sparse.linalg import eigsh, svds, inv

mbmat = lambda x: sp.bmat(x).tocsc()


def qp_dispatch(x, c, f_xx, c_x, c_l, c_u, x_l, x_u, active_set=[]):
    """
    Dispatch to QP algorithm taking care of slack variables, infeasabilities, etc.
    Also handles sparsity issues, and proper shapes of input data.
    Takes in problem of the form:

        min   c.T * x + 1/2 * x.T * H * x
        s.t.  c_l <= c_x * x <= c_u,
                x_l <= x <= x_u

    """

    # Auto-checking for sparse
    if not (sp.isspmatrix(f_xx) and sp.isspmatrix(c_x)):
        raise ValueError('Need to supply sparse matrices')

    n = max(x.shape)
    m = c_x.shape[0]
    active_set = set(active_set)

    if c_x.shape != (m, n):
        raise LinAlgError('Linear constraint matrix is the incorrect shape')
    elif m == n:
        raise LinAlgError('Not working yet')

    """ TODO check if this is needed after stalling fix, as slackify does this already """
    x = col(x)
    c = col(c)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)
    c_x  = sp.csc_matrix(c_x)
    f_xx = sp.csc_matrix(f_xx)
    """ End check section """

    zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
    eye   = sp.eye(n + m, n + m).tocsc()
    ones  = np.matrix(np.ones((n + m, 1)))

    # Check to see if we should even be solving this problem, and if it's
    # initially feasible
    # TODO: add checks for sparse matrices, maybe?
    """
        if matrix_rank(Ap) != min(Ap.shape):
            raise LinAlgError('Bad rank for initial constraint matrix')
        if not (eigvals(f_xx) > 0).all():
            raise LinAlgError('Not positive definite')
    """
    # Transforming to using slack variables, with only x and s bounded on both
    # sides
    s_init = densify(c_x * x)
    xp, cp, Hp, Ap, lp, up = slackify(x, c, f_xx, c_x, x_l, x_u, c_l, c_u, s_init)

    # Feasibility part
    out = qp_subdispatch(xp * 0, cp + Hp * xp, Hp, Ap, vec_clamp(lp - xp), vec_clamp(up - xp), active_set, feas=qp_feasdispatch)
    xp  = vec_clamp(xp + out[0])
    active_set = out[1]

    # Final solve
    x, active_set, mu = qp_subdispatch(xp * 0, cp + Hp * xp, Hp, Ap, vec_clamp(lp - xp), vec_clamp(up - xp), active_set, func=qp_ns, feas=qp_feasdispatch)
    x = vec_clamp(xp + x)

    return x[:n, 0], active_set, mu


def slackify(x, c, H, c_x, x_l, x_u, c_l, c_u, s_init):
    n = max(x.shape)
    m = c_x.shape[0]

    zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
    eye   = sp.eye(m, m).tocsc()

    xp = bmat([[         col(x)],
               [densify(s_init)]])
    lp = bmat([[col(x_l)],
               [col(c_l)]])
    up = bmat([[col(x_u)],
               [col(c_u)]])
    cp = bmat([[               col(c)],
               [densify(zeros[:m, 0])]])

    c_x  = sp.csc_matrix(c_x)
    H = sp.csc_matrix(H)
    Hp = mbmat([[            H, zeros[:n, :m]],
                [zeros[:m, :n], zeros[:m, :m]]]).tocsc()
    Ap = mbmat([[c_x, -eye[:m, :m]]]).tocsc()

    return xp, cp, Hp, Ap, lp, up


"""
as_feasdispatch
    feas
    if stall -> perturb -> feas
    post stall -> feas

as_dispatch
    run
    if stall -> perturb -> run
    post stall -> as_feasdispatch -> run
"""



def qp_subdispatch(x, c, H, A, x_l, x_u, active_set, func=None, feas=None):
    if func is None:
        callfunc = feas
    else:
        callfunc = func
    x, active_set, mu, stalled = callfunc(x, c, H, A, x_l, x_u, active_set)
    if stalled:
        x_l2 = x_l.copy()
        x_u2 = x_u.copy()
        while stalled == True:   # means it stalled
            rand_perturb = abs(np.random.randn(*x_l.shape)) / 10000.
            x_l2 -= rand_perturb
            x_u2 += rand_perturb
            x, active_set, mu, stalled = callfunc(x, c, H, A, x_l2, x_u2, active_set)
        if func is not None:
            x, active_set, mu = qp_subdispatch(x, c, H, A, x_l, x_u, active_set, feas=feas)
            x = vec_clamp(x)
        x, active_set, mu, stalled = callfunc(x, c, H, A, x_l, x_u, active_set)
    return x, active_set, mu


def qp_feasdispatch(x, c, H, A, x_l, x_u, active_set):
    """
    Used to call appropriate methods for finding a feasible point.
    """
    run = False
    mu = 0
    stalling = False
    # Check to see if bounds have been violated
    for i in range(len(x)):
        if x[i, 0] < x_l[i, 0]:
            x[i, 0] = x_l[i, 0]
            run = True
        elif x[i, 0] > x_u[i, 0]:
            x[i, 0] = x_u[i, 0]
            run = True
    # Check to see if EQ constraints are violated
    if abs(A * x).max() > 1.e3 * eps:
        run = True
    if run:
        I = sp.eye(len(x), len(x)).tocsc()
        out = qp_ns(x * 0, c * 0, I, A, vec_clamp(x_l - x), vec_clamp(x_u - x), active_set, find_feas=True, ress0=A*x)
        x = vec_clamp(x + out[0])
        active_set = out[1]
        mu = out[2]
        stalling = out[3]
    return x, active_set, mu, stalling


def qp_ns(x, c, H, A, x_l, x_u, active_set, find_feas=False, ress0=None):
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
        # TODO figure out how to do this without the re-casting
        alphas = matrix(alphas).reshape(-1, 1)
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
        at_upper = (abs(x - x_u) < eps)[active_list]
        at_lower = (abs(x - x_l) < eps)[active_list]
        upper = multiply(multiply(at_upper, (mu_ineq > 0)), abs(mu_ineq))
        lower = multiply(multiply(at_lower, (mu_ineq < 0)), abs(mu_ineq))
        maxval = maximum(upper, lower)
        if max(maxval) > 0:
            max_ind1 = nonzero(maxval == max(maxval))[0]
            for ii in range(max(max_ind1.shape)):
                ind = active_list[max_ind1[0, ii]]
                print('freeing: ' + str(ind))
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

            print('fixing:  ' + str(ind))
            stalling_count[ind] += 1
            active_set.add(ind)
            free_set  -= {ind}


    n = x.shape[0]
    m = A.shape[0]

    zeros = sp.coo_matrix((n, n), np.float64).tocsc()
    eye   = sp.eye(n, n).tocsc()
    ones  = np.matrix(np.ones((n + m, 1)))

    stalling_count = dict(zip(list(range(n)), [0]*n))

    fixed = asarray(x_l == x_u).squeeze().tolist()
    fixed_list = []
    for i, v in enumerate(fixed):
        if v:
            fixed_list += [i]
    fixed_set     = set(fixed_list)
    free_set      = set(list(range(n)))
    active_set    = set(active_set)
    nonfixed_set = free_set - fixed_set
    active_set   -= fixed_set
    free_set     -= active_set
    free_set     -= fixed_set

    # Transformation matrices between x` and working/free sets
    Cmat = eye[:n, :n]
    C_nfx = extract(Cmat, sorted(list(nonfixed_set)))
    g = densify(c + H * x)

    n0 = len(free_set)
    print('\n\nInitial Sizes - n: %8d, m: %8d, n0: %8d' % (n, m, n0))
    if find_feas:
        print('Initial Feasibility Violation: ' + str(abs(ress0).max()))
        pass

    mu = 0
    mu_ineq = 0
    i = 0
    stalling = False
    while True:
        working_set = active_set.union(fixed_set)
        # Need to compute some transformation and sizes for this iteration
        C_f  = extract(Cmat, sorted(list(free_set)))       # Current free (all) variables
        C_a  = extract(Cmat, sorted(list(active_set)))     # Current fixed (activated) variables
        C_w  = extract(Cmat, sorted(list(working_set)))     # Current fixed (all) variables
        num_actv = len(active_set)     # no. of fixed variables
        num_work = len(working_set)     # no. of fixed variables


        if ress0 is None:
            ress_hat = zeros[: m + num_actv, 0]
        else:
            ress = ress0 + A * x
            ress_hat = bmat([[ress], [zeros[:num_actv, 0].todense()]])

        A_hat = mbmat([[A * C_nfx.T], [C_a * C_nfx.T]])


        # TODO fix it so the solve doesn't sometimes get errors
        K = mbmat([[C_nfx * H * C_nfx.T, A_hat.T], [A_hat, zeros[:A_hat.shape[0], :A_hat.shape[0]]]])
        f = bmat([[C_nfx * g], [densify(ress_hat)]])
        xx = col(solve(K, f))
        p = - xx[: n - len(fixed_set)]
        p = C_nfx.T * p
        p = C_f.T * C_f * p
        mu = xx[n - len(fixed_set) : n - len(fixed_set) + m]
        #mu_ineq = xx[n - len(fixed_set) + m :]


        if abs(K * xx - f).max() > 1e5 * eps:
            print('Solve Failed')
            rows_to_keep = get_independent_rows(A_hat, 1e3*eps)
            ress_hat = ress_hat[rows_to_keep, 0]
            A_hat = extract(A_hat, rows_to_keep)
            K = mbmat([[C_nfx * H * C_nfx.T, A_hat.T], [A_hat, zeros[:A_hat.shape[0], :A_hat.shape[0]]]])
            f = bmat([[C_nfx * g], [densify(ress_hat)]])
            xx = col(solve(K, f))
            p = - xx[: n - len(fixed_set)]
            p = C_nfx.T * p
            p = C_f.T * C_f * p
            mu = xx[n - len(fixed_set) : n - len(fixed_set) + m]
            #if i > n:
            #    import ipdb
            #    ipdb.set_trace()


        if not find_feas and (abs(A * p).max() > 2.e3 * eps):
            print('\033[31mAp max: ' + str((abs(A * p).max())) + '\033[0m')
            pass
        print('Found point Ax + ress0: ' + str(abs(A * x + ress0).max()))

        # Portion where the step size is calculated
        alpha, alpha_ind = _min_ratio_test(x, p, x_l, x_u)

        print(_print_step(i, x, num_actv, alpha))

        # take the step
        x = x + alpha * p
        g = densify(c + H * x)
        mu_ineq = C_a * g - C_a * A.T * mu # Direct solve mu_ineq calc
        i += 1

        if find_feas:
            if abs(A * x + ress).max() < 1.e3 * eps:
                print('Found feasible point: ' + str(abs(A * x + ress0).max()))
                break

        #if max(list(stalling_count.values())) > (n - m) / 2:
        if max(list(stalling_count.values())) > 20:
            stalling = True
            print('Stalling!!!')
            break

        if i > 2 * (n):
            raise Exception('Too many iterations')

        if alpha < 1:
            """ If we're not taking a full step, we add the limiting index to
            the active set, and pin x` to the appropriate boundary. """
            if find_feas:
                if num_actv > 0:
                    _free(x, active_set, free_set, mu_ineq, stalling_count)

            _fix(x, active_set, free_set, alpha_ind, stalling_count)

        else:
            """ If we are taking a full step, we need to check the lagrange
            multipliers on the inequalities to make sure they have the correct
            sign, and remove them from the active set if necessary."""
            if num_actv == 0:
                break

            freed = _free(x, active_set, free_set, mu_ineq, stalling_count)
            if not freed: # if nothing was freed, we are done!
                if find_feas:
                    print('Found feasible point: ' + str(abs(A * x + ress0).max()))
                else:
                    print('All lagrange multipliers are good!')
                    break

    mu_ineq = g - A.T * mu # final mu_ineq to return
    return x, active_set, mu_ineq, stalling

