from numpy import bmat, clip, nonzero, nanmin, maximum, multiply
from numpy.linalg import eigvals, matrix_rank, LinAlgError
from scipy.sparse import isspmatrix
from pyintropt.functions import col, eps, big, densify, vec_clamp


def qp_dispatch(x, c, f_xx, c_x, c_l, c_u, x_l, x_u, active_set=[], find_feas=False, relax=False):
    """
    Dispatch to QP algorithm taking care of slack variables, infeasabilities, etc.
    Also handles sparsity issues, and proper shapes of input data.
    Takes in problem of the form:

            min   c.T * x + 1/2 * x.T * H * x
            s.t.  c_l <= c_x * x <= c_u,
                  x_l <= x <= x_u

    """

    # Auto-checking for sparse
    if isspmatrix(f_xx) and isspmatrix(c_x):
        sparse = True
    elif isspmatrix(f_xx) or isspmatrix(c_x):
        raise ValueError('Either both f_xx and c_x need to be sparse, or neither')
    else:
        sparse = False

    n = max(x.shape)
    m = c_x.shape[0]
    active_set = set(active_set)

    if c_x.shape != (m, n):
        raise LinAlgError('Linear constraint matrix is the incorrect shape')
    elif m == n:
        raise LinAlgError('Not working yet')

    x = col(x)
    c = col(c)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)
    if sparse:
        import numpy as np
        import scipy.sparse as sp
        from pyintropt.functions import sp_solve as solve
        c_x  = sp.csc_matrix(c_x)
        zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
        eye   = sp.eye(n + 2 * m, n + m).tocsc()
        ones  = np.matrix(np.ones((n + m, 1)))
        mbmat = lambda x: sp.bmat(x).tocsc()
    else:
        import numpy as np
        from numpy.linalg import solve
        c_x  = np.matrix(c_x)
        zeros = np.matrix(np.zeros((n + m, n + m)))
        eye   = np.matrix(np.eye(n + 2 * m, n + m))
        ones  = np.matrix(np.ones((n + m, 1)))
        from numpy import bmat as mbmat


    # Check to see if we should even be solving this problem, and if it's
    # initially feasible
    if not sparse:
        if matrix_rank(Ap) != min(Ap.shape):
            raise LinAlgError('Bad rank for initial constraint matrix')
        if not (eigvals(f_xx) > 0).all():
            raise LinAlgError('Not positive definite')
    else:
        pass # TODO: add checks for sparse matrices, maybe?


    # Transforming to using slack variables, with only x and s bounded on both
    # sides
    s_init = densify(c_x * x)
    xp, cp, Hp, Ap, lp, up = slackify(x, c, f_xx, c_x, x_l, x_u, c_l, c_u, s_init, sparse)

    # TODO feasibility/s_init calc
    run = False
    for i in range(len(xp)):
        if xp[i, 0] < lp[i, 0]:
            xp[i, 0] = lp[i, 0]
            run = True
        if xp[i, 0] > up[i, 0]:
            xp[i, 0] = up[i, 0]
            run = True
    # TODO "shed" inconsistent constraints
    if run:
        Hp2 = eye[: n + m, : n + m]
        # is this right - assume no initial active set for feasibility?
        # out = qp_sc(xp * 0, cp * 0, Hp2, Ap, lp - xp, up - xp, active_set, sparse, find_feas=True, ress = Ap * xp)
        out = qp_sc(xp * 0, cp * 0, Hp2, Ap, lp - xp, up - xp, active_set, sparse, find_feas=True, ress0=Ap*xp)
        xp  = vec_clamp(xp + out[0])
        active_set = active_set.union(out[1])

    # TODO "shed" inconsistent constraints
    # Final solve
    x, active_set = qp_sc(xp, cp, Hp, Ap, lp, up, active_set, sparse)
    return x[:n, 0], active_set


def slackify(x, c, H, c_x, x_l, x_u, c_l, c_u, s_init, sparse):
    n = max(x.shape)
    m = c_x.shape[0]

    x = col(x)
    c = col(c)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)

    if sparse:
        import numpy as np
        import scipy.sparse as sp
        c_x  = sp.csc_matrix(c_x)
        H = sp.csc_matrix(H)
        zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
        eye   = sp.eye(n + 2 * m, n + m).tocsc()
        cp = bmat([[c], [zeros[:m, 0].todense()]]) # slack transformation
        mbmat = lambda x: sp.bmat(x).tocsc()
    else:
        import numpy as np
        from numpy import bmat as mbmat
        c_x  = np.matrix(c_x)
        H = np.matrix(H)
        zeros = np.matrix(np.zeros((n + m, n + m)))
        eye   = np.matrix(np.eye(n + 2 * m, n + m))
        cp = bmat([[c], [zeros[:m, 0]]]) # slack transformation

    Hp = mbmat([[            H, zeros[:n, :m]],
                [zeros[:m, :n], zeros[:m, :m]]])
    Ap = mbmat([[c_x, -eye[:m, :m]]])
    lp = bmat([[x_l], [c_l]])
    up = bmat([[x_u], [c_u]])
    s_init = densify(s_init)
    xp = bmat([[x], [s_init]])

    return xp, cp, Hp, Ap, lp, up



def qp_sc(x, c, H, A, x_l, x_u, active_set, sparse, find_feas=False, ress0=None):
    """
    Quadratic programming solver.

    Note that the function takes in c, not f_x - do the appropriate conversion
    before hand.
    Using the Schur-Complement method, solves quadratic programming problems in
    the form:

    min   c.T * x + 1/2 * x.T * H * x
    s.t.  A * x = 0
          x_l <= x <= x_u

    """

    def _print_step(i, x, num_actv, num_actvd):
        good = '\033[32m'
        bad  = '\033[31m'
        end  = '\033[0m'
        outstr = 'Iter: %7d   ' % i
        val = (c.T * x + 0.5 * x.T * H * x)[0, 0]
        outstr += 'Obj. val: ' + good + '%4.6e    ' % val + end
        outstr += 'Activ(ated) set size: %7d (%7d)     ' % (num_actv, num_actvd)
        leader = bad
        if max(abs(A * x)) < 1.e3 * eps:
            leader = good
        temps = leader + '%2.3e ' % max(abs(A * x)) + end + '    '
        outstr += '|Ax|= %s' % temps

        l = (x_l <= x).all()
        u = (x_u >= x).all()
        llead = bad
        ulead = bad
        if l:
            llead = good
        if u:
            ulead = good
        outstr += 'x_l <= x: ' + llead + str(l) + end + ' x <= x_u: ' + ulead + str(u) + end
        return outstr


    n = x.shape[0]
    m = A.shape[0]
    if sparse:
        import numpy as np
        import scipy.sparse as sp
        from pyintropt.functions import sp_extract_row as extract, sp_factiz as factorized, sp_solve as solve
        zeros = sp.coo_matrix((n, n), np.float64).tocsc()
        eye   = sp.eye(n, n).tocsc()
        mbmat = lambda x: sp.bmat(x).tocsc()
    else:
        import numpy as np
        from numpy import bmat as mbmat
        from numpy.linalg import solve
        from pyintropt.functions import extract_row as extract, factiz as factorized
        zeros = np.matrix(np.zeros((n, n)))
        eye   = np.matrix(np.eye(n, n))


    free_set      = set(list(range(n)))
    freed_set     = set()
    active_set    = set(active_set)
    activated_set = set()
    free_set     -= active_set

    n0 = len(free_set)
    free0_set = free_set.copy()
    active0_set = active_set.copy()

    # Transformation matrices between x` and working/free sets
    Cmat = eye
    C_f0 = extract(Cmat, sorted(list(free_set)))         # Initial free variables
    C_w0 = extract(Cmat, sorted(list(active0_set)))      # Initial fixed variables

    A0 = A * C_f0.T
    K0 = mbmat([[C_f0 * H * C_f0.T,         A0.T],
                [                A0, zeros[:m, :m]]])
    K0fact = factorized(K0)

    # Faster to calculate worst possible K0_inv * U, and pull out relevant
    # parts later.
    U_max1 = mbmat([[C_f0 * eye[: n, : n]], [zeros[: m, : n]]]) # Left part, fixed variables
    U_max2 = mbmat([[C_f0 * H], [A]])                         # Right part, freed variables
    K_inv_U_max1 = K0fact(U_max1)
    K_inv_U_max2 = K0fact(U_max2)

    g = densify(c + H * x)
    print('\nInitial Sizes - n: %8d, m: %8d, n0: %8d' % (n, m, n0))
    mu = 0
    i = 0
    while True:
        # Need to compute some transformation and sizes for this iteration
        C_f  = extract(Cmat, sorted(list(free_set)))       # Current free (all) variables
        C_fd = extract(Cmat, sorted(list(freed_set)))      # Current free (not initially free) variables
        C_w  = extract(Cmat, sorted(list(active_set)))     # Current fixed (all) variables
        C_wd = extract(Cmat, sorted(list(activated_set)))  # Current fixed (not-initially fixed) variables
        num_actvd = len(activated_set) # no. of initially free variables, fixed
        num_actv = len(active_set)     # no. of fixed variables
        num_frd   = len(freed_set)     # no. of initially fixed variables, freed

        if i > 2 * n:
            break #TODO add error return or something?
            raise Exception('Too many iterations')

        print(_print_step(i, x, num_actv, num_actvd))
        i += 1

        # Increasing the size of the system to solve
        # Utilizing the Schur Complement approach
        # TODO still not convinced this is the correct U, from initial
        # constraints concerns
        U = mbmat([[ mbmat([[C_f0 * eye[: n, : n]], [zeros[: m, : n]]]) * C_wd.T,
                     mbmat([[C_f0 * H * C_fd.T], [A * C_fd.T]])]])
        V = mbmat([[zeros[:num_actvd, :num_actvd], zeros[:num_actvd, :num_frd]],
                   [  zeros[:num_frd, :num_actvd],          C_fd * H * C_fd.T]])
        #C = V - U.T * K0fact(U) # Original (slow) C calculation
        C = V - U.T * mbmat([[K_inv_U_max1 * C_wd.T, K_inv_U_max2 * C_fd.T]])


        if ress0 is None:
            ress = zeros[:m, 0]
        else:
            ress = ress0 + A * x

        f = bmat([[C_f0 * g], [densify(ress)]])
        w = bmat([[densify(zeros[:num_actvd, 0])], [C_fd * g]])
        v = col(K0fact(f))
        z = col(solve(C, w - U.T * v))
        y = col(K0fact(f - U * z))

        """
        K = mbmat([[K0, U], [U.T, V]])
        bb = bmat([[f], [w]])
        xx = col(solve(K, bb))
        y = col(xx[:n0 + m, 0])
        z = col(xx[n0 + m:, 0])
        """

        p = C_f0.T * -y[:n0, 0]        # portion from initially free variables
        if num_frd != 0:               # portion from initially fixed variables
            p += C_fd.T * -z[-num_frd:, 0]
        mu = y[n0 : n0 + m, 0]          # Equality multipliers
        p  = C_f.T * C_f * p

        if not find_feas and (abs(A * p).max() > 1.e3 * eps):
            print('\033[31mAp max: ' + str((abs(A * p).max())) + '\033[0m')


        # Portion where the step size is calculated
        alpha = 0
        alpha_ind = 0
        alpha_l = C_f.T * C_f * (x_l - x) / p
        alpha_u = C_f.T * C_f * (x_u - x) / p
        alphas = maximum(alpha_l, alpha_u)
        alpha_unclip = nanmin(alphas)
        alpha_ind = nonzero(alphas == alpha_unclip)[0][0][0, 0]
        alpha = clip(alpha_unclip, 0, 1)

        # take the step
        x = x + alpha * p
        g = densify(c + H * x)

        #TODO needs to be after x = alpha + xp
        if find_feas:
            if abs(A * x + ress).max() < 1.e3 * eps:
                break

        if alpha < 1:
            """ If we're not taking a full step, we add the limiting index to
            the active set, and pin x` to the appropriate boundary. """
            # check to see if it variable to be fixed has already been fixed
            if alpha_ind in active_set:
                raise ValueError('That variable is already fixed...')
            # Update the sets
            active_set.add(alpha_ind)
            free_set  -= {alpha_ind}
            freed_set -= {alpha_ind}
            if alpha_ind not in active0_set: # only "activated" if not
                activated_set.add(alpha_ind) # initially in fixed set

            if alpha_unclip in alpha_l:             # fixing on a lower bound
                x[alpha_ind, 0] = x_l[alpha_ind, 0]
            else:                                   # fixing on an upper bound
                x[alpha_ind, 0] = x_u[alpha_ind, 0]
            print('reduced alpha: %2.7f, fixing variable %d' % (alpha, alpha_ind))
        else:
            """ If we are taking a full step, we need to check the lagrange
            multipliers on the inequalities to make sure they have the correct
            sign, and remove them from the active set if necessary."""
            if len(active0_set) == 0:   # there was no initial active set
                if len(active_set) == 0:
                    break
                mu_ineq = z[:len(active_set), 0] # multipliers from fixed vars
            else:                       # some initial active set was provided
                #TODO it's wrong...? Currently, _seems_ right, but inefficient
                #mu_ineq1 = z[:len(active_set), 0] # fixed during iterations
                #C_temp = extract(Cmat, sorted(list(active_set.intersection(active0_set))))
                #mu_ineq2 = C_temp * g - C_temp * A.T * mu
                #mu_ineq = C_wd.T * mu_ineq1 + C_temp.T * mu_ineq2
                mu_ineq = C_w * g - C_w * A.T * mu

            active_list = sorted(list(active_set))
            at_upper = (abs(x - x_u) < eps)[active_list]
            at_lower = (abs(x - x_l) < eps)[active_list]
            upper = multiply(multiply(at_upper, (mu_ineq > 0)), abs(mu_ineq))
            lower = multiply(multiply(at_lower, (mu_ineq < 0)), abs(mu_ineq))
            maxval = maximum(upper, lower)

            # if nothing has a bad multiplier, we're done!
            if max(maxval) <= 0:
                print('full alpha     All lagrange multipliers are good!')
                break
            else: # otherwise, remove offending element from active set
                max_ind = nonzero(maxval == max(maxval))[0][0][0, 0]
                max_ind = active_list[max_ind]
                print('full alpha     ind to remove: ' + str(max_ind))
                free_set.add(max_ind)
                active_set -= {max_ind}
                activated_set -= {max_ind}
                if max_ind not in free0_set: # only "freed" if not
                    freed_set.add(max_ind)   # initially in free set
    return x, active_set
