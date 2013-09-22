from numpy import bmat, clip, nonzero, nanmin, maximum, multiply
from numpy.linalg import eigvals, matrix_rank, LinAlgError
from scipy.sparse import isspmatrix
from pyintropt.functions import col, eps, big, densify


def qp_sc(x, c, f_xx, c_x, c_l, c_u, x_l, x_u, active_set=[],
          find_feas=False, s_init=None):
    """
    Quadratic programming solver.

    Note that the function takes in c, not f_x - do the appropriate conversion
    before hand.
    Using the Schur-Complement method, solves quadratic programming problems in
    the form:

    min   c.T * x + 1/2 * x.T * H * x
    s.t.  c_l <= c_x * x <= c_u,
          x_l <= x <= x_u

    Uses slack variables to transform [x.T, s.T] into x`, redefining the
    problem as:

    min   c`.T * x` + 1/2 * x`.T * H` * x`
    s.t.  A` * x` = 0
          x_l` <= x` <= x_u`

    """
    # Auto-checking for sparse
    if isspmatrix(f_xx) and isspmatrix(c_x):
        sparse = True
    elif isspmatrix(f_xx) or isspmatrix(c_x):
        raise ValueError('Either both f_xx and c_x need to be sparse, or neither')
    else:
        sparse = False

    # Getting things into the right form
    n = max(x.shape)
    m = max(c_l.shape)
    x = col(x)
    c = col(c)
    x_l = col(x_l)
    x_u = col(x_u)
    c_l = col(c_l)
    c_u = col(c_u)
    if sparse:
        import numpy as np
        import scipy.sparse as sp
        from pyintropt.functions import sp_extract_row as extract, sp_factiz as factorized, sp_solve as solve
        c_x  = sp.csc_matrix(c_x)
        f_xx = sp.csc_matrix(f_xx)
        zeros = sp.coo_matrix((n + m, n + m), np.float64).tocsc()
        eye   = sp.eye(n + 2 * m, n + m).tocsc()
        ones  = np.matrix(np.ones((n + m, 1)))
        cp = bmat([[c], [zeros[:m, 0].todense()]]) # slack transformation
        mbmat = lambda x: sp.bmat(x).tocsc()
    else:
        import numpy as np
        from numpy import bmat as mbmat
        from numpy.linalg import solve
        from pyintropt.functions import extract_row as extract, factiz as factorized
        c_x  = np.matrix(c_x)
        f_xx = np.matrix(f_xx)
        zeros = np.matrix(np.zeros((n + m, n + m)))
        eye   = np.matrix(np.eye(n + 2 * m, n + m))
        ones  = np.matrix(np.ones((n + m, 1)))
        cp = bmat([[c], [zeros[:m, 0]]]) # slack transformation


    # Transforming to using slack variables, with only x and s bounded on both
    # sides
    Hp = mbmat([[         f_xx, zeros[:n, :m]],
                [zeros[:m, :n], zeros[:m, :m]]])
    Ap = mbmat([[c_x, -eye[:m, :m]]])
    lp = bmat([[x_l], [c_l]])
    up = bmat([[x_u], [c_u]])


    #TODO probably should have a more intelligent initial feasibility stage
    c_cur = c_x * x
    if not find_feas:
        if (x_l > x).any() or (x > x_u).any() or (c_l > c_cur).any() or (c_cur > c_u).any():
            x, active_set = qp_sc(x, 0 * c, eye[:n, :n], c_x, c_l, c_u, x_l, x_u, active_set, True)
    if s_init is None:
        s_init = c_x * x


    xp = bmat([[x], [s_init]])
    if find_feas:
        gp = zeros[: n + m, 0]
    else:
        gp = cp + Hp * xp
    gp = densify(gp)

    free_set      = set(list(range(n + m)))
    freed_set     = set()
    active_set    = set(active_set)
    activated_set = set()
    free_set     -= active_set

    n0 = len(free_set)
    free0_set = free_set.copy()
    active0_set = active_set.copy()

    # Transformation matrices between x` and working/free sets
    Cmat = eye[: n + m, : n + m]
    C_f0 = extract(Cmat, list(free_set))         # Initial free variables
    C_w0 = extract(Cmat, list(active0_set))      # Initial fixed variables

    num_actvd = len(activated_set) # no. of initially free variables, fixed
    num_frd   = len(freed_set)     # no. of initially fixed variables, freed

    # Check to see if we should even be solving this problem, and if it's
    # initially feasible
    A0 = Ap * C_f0.T
    if not sparse:
        if matrix_rank(A0) != min(A0.shape):
            raise LinAlgError('Bad rank for initial constraint matrix')
        if not (eigvals(f_xx) > 0).all():
            raise LinAlgError('Not positive definite')
    else:
        pass # add checks for sparse matrices, maybe?
    if c_x.shape != (m, n):
        raise LinAlgError('Linear constraint matrix is the incorrect shape')
    #if (lp > xp).any() or (xp > up).any():
    #    raise ValueError('Outside bounds')

    K0 = mbmat([[C_f0 * Hp * C_f0.T,         A0.T],
                [                A0, zeros[:m, :m]]])
    K0fact = factorized(K0)

    # Faster to calculate worst possible K0_inv * U, and pull out relevant
    # parts later.
    U_max1 = eye[: n0 + m, : n + m]          # Left part, fixed variables
    U_max2 = mbmat([[C_f0 * Hp], [Ap]])      # Right part, freed variables
    K_inv_U_max1 = K0fact(U_max1)
    K_inv_U_max2 = K0fact(U_max2)


    print('Initial Sizes - n: %8d, m: %8d, n0: %8d' % (n, m, n0))
    print('gp sum: %f c sum: %f K0 sum: %f' %(gp.sum(), c.sum(), K0.sum()))
    mu = 0
    i = 0
    cyc = 0
    while True:
        # Need to compute some transformation and sizes for this iteration
        C_f  = extract(Cmat, list(free_set))       # Current free (all) variables
        C_fd = extract(Cmat, list(freed_set))      # Current free (not initially free) variables
        C_w  = extract(Cmat, list(active_set))     # Current fixed (all) variables
        C_wd = extract(Cmat, list(activated_set))  # Current fixed (not-initially fixed) variables
        num_actvd = len(activated_set) # no. of initially free variables, fixed
        num_frd   = len(freed_set)     # no. of initially fixed variables, freed

        cur_eq = Ap * xp
        print('\nIteration: %8d' % i)
        print('Current objective value: %g' % (c.T * xp[:n, 0] + 0.5 * xp[:n, 0].T * f_xx * xp[:n, 0])[0, 0])
        if find_feas:
            print('feasibility search')
        print('Activated set size: %4d' % num_actvd)
        print('Max EQ constraint violation: \033[31m%g\033[0m' % max(cur_eq))
        print("Lower Bound OK? \033[31m%5s\033[0m Upper Bound OK? \033[31m%5s\033[0m" % ((lp - xp < eps).all(), (up - xp > eps).all()))
        i += 1

        #if (max(abs(cur_eq)) > 1.e3 * eps) and not find_feas:
        #    x, active_set = qp_sc(xp[:n], 0 * c, eye[:n, :n], c_x, c_l, c_u, x_l, x_u, active_set, True)
        #    return qp_sc(x, c, f_xx, c_x, c_l, c_u, x_l, x_u, active_set)


        # Increasing the size of the system to solve
        # Utilizing the Schur Complement approach
        U = mbmat([[ eye[: n0 + m, : n + m] * C_wd.T,
                     mbmat([[C_f0 * Hp * C_fd.T], [Ap * C_fd.T]])]])
        V = mbmat([[zeros[:num_actvd, :num_actvd], zeros[:num_actvd, :num_frd]],
                   [  zeros[:num_frd, :num_actvd],          C_fd * Hp * C_fd.T]])
        # C = V - U.T * K0fact(U) # Original (slow) C calculation
        C = V - U.T * mbmat([[K_inv_U_max1 * C_wd.T, K_inv_U_max2 * C_fd.T]])


        if find_feas:
            ress = Ap * xp
        else:
            ress = zeros[:m, 0]
        ress = densify(ress)

        f = bmat([[C_f0 * gp], [ress]])
        w = bmat([[densify(zeros[:num_actvd, 0])], [C_fd * gp]])
        v = col(K0fact(f))
        z = col(solve(C, w - U.T * v))
        y = col(K0fact(f - U * z))
        """
        K = mbmat([[K0, U], [U.T, V]])
        bb = bmat([[f], [w]])
        xx = col(solve(K, bb))
        y = xx[:n0, 0]
        z = xx[n0:, 0]
        """

        p = C_f0.T * -y[:n0, 0] # portion from initially free variables
        if num_frd != 0:               # portion from initially fixed variables
            p += C_fd.T * -z[-num_frd:, 0]
        p  = C_f.T * C_f * p
        mu = y[n0 : n0 + m, 0]          # Equality multipliers

        print('Ap max')
        print(abs(Ap * p).max())


        # Portion where the step size is calculated
        alpha_l = C_f.T * C_f * (lp - xp) / p
        alpha_u = C_f.T * C_f * (up - xp) / p
        alphas = maximum(alpha_l, alpha_u)
        alpha_unclip = nanmin(alphas)
        alpha_ind = nonzero(alphas == alpha_unclip)[0][0][0, 0]
        alpha = clip(alpha_unclip, 0, 1)

        # take the step
        xp = xp + alpha * p
        if not find_feas:
            gp = cp + Hp * xp
        gp = densify(gp)

        if alpha < 1:
            """ If we're not taking a full step, we add the limiting index to
            the active set, and pin x` to the appropriate boundary. """
            # check to see if it variable to be fixed has already been fixed
            if alpha_ind in active_set:
                raise ValueError('That variable is already fixed...')
            # Update the sets
            active_set.add(alpha_ind)
            free_set -= {alpha_ind}
            freed_set -= {alpha_ind}
            if alpha_ind not in active0_set: # only "activated" if not
                activated_set.add(alpha_ind) # initially in fixed set

            if alpha_unclip in alpha_l:             # fixing on a lower bound
                xp[alpha_ind, 0] = lp[alpha_ind, 0]
            else:                                   # fixing on an upper bound
                xp[alpha_ind, 0] = up[alpha_ind, 0]
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
                mu_ineq1 = z[:len(active_set), 0] # fixed during iterations
                C_temp = extract(Cmat, list(active_set.intersection(active0_set)))
                mu_ineq2 = C_temp * gp - C_temp * Ap.T * mu
                #mu_ineq = C_wd.T * mu_ineq1 + C_temp.T * mu_ineq2
                mu_ineq = C_w * gp - C_w * Ap.T * mu

            active_list = list(active_set)
            at_upper = (abs(xp - up) < eps)[active_list]
            at_lower = (abs(xp - lp) < eps)[active_list]

            upper = multiply(multiply(at_upper, (mu_ineq > 0)), abs(mu_ineq))
            lower = multiply(multiply(at_lower, (mu_ineq < 0)), abs(mu_ineq))
            maxval = maximum(upper, lower)

            print('full alpha')
            # if nothing has a bad multiplier, we're done!
            if max(maxval) == 0:
                print('All lagrange multipliers are good!')
                break
            else: # otherwise, remove offending element from active set
                max_ind = nonzero(maxval == max(maxval))[0][0][0, 0]
                max_ind = active_list[max_ind]
                print('ind to remove: ' + str(max_ind))
                free_set.add(max_ind)
                active_set -= {max_ind}
                activated_set -= {max_ind}
                if max_ind not in free0_set: # only "freed" if not
                    freed_set.add(max_ind)   # initially in free set

    return xp[:n, 0], active_set
