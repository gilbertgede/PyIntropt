from numpy import bmat, finfo, clip, nonzero, matrix, nanmin, maximum
from numpy.linalg import eigvals, matrix_rank, inv, solve, LinAlgError
eps = finfo(float).eps
big = 1 / eps**2

def qp_sc(x, f_x, f_xx, c_x, c_l, c_u, x_l, x_u, active_set=[],
          find_feas=False, sinit=None, sparse=False):
    """
    Quadratic programming solver.

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

    n = max(x.shape)
    m = max(c_l.shape)
    x = matrix(x).reshape(-1, 1)
    f_x = matrix(f_x).reshape(-1, 1)
    x_l = matrix(x_l).reshape(-1, 1)
    x_u = matrix(x_u).reshape(-1, 1)
    c_l = matrix(c_l).reshape(-1, 1)
    c_u = matrix(c_u).reshape(-1, 1)
    #TODO add provisions for sparse matrices
    if c_x.shape != (m, n):
        raise LinAlgError('Linear constraint matrix is the incorrect shape')
    c_x = matrix(c_x)
    f_xx = matrix(f_xx)

    # TODO Need to add provisions for sparse matrices
    import numpy as np
    ones  = matrix(np.ones((n + m, 1)))
    zeros = matrix(np.zeros((n + m, n + m)))
    eye   = matrix(np.eye(n + 2 * m, n + m))


    # Transforming to using slack variables, with only x and s bounded on both
    # sides
    c  = f_x - f_xx * x
    cp = bmat([[c], [zeros[:m, 0]]])
    Hp = bmat([[         f_xx, zeros[:n, :m]],
               [zeros[:m, :n], zeros[:m, :m]]])
    Ap = bmat([[c_x, -eye[:m, :m]]])
    lp = bmat([[x_l], [c_l]])
    up = bmat([[x_u], [c_u]])

    #TODO probably should have a more intelligent initial feasibility stage
    s_init = c_x * x
    for i in range(m):
        if (s_init[i, 0] > c_u[i, 0]) or (s_init[i, 0] < c_l[i, 0]):
            if abs(c_l[i, 0]) < abs(c_u[i, 0]):
                s_init[i, 0] = c_l[i, 0]
            else:
                s_init[i, 0] = c_u[i, 0]
    if sinit is not None:
        s_init = sinit

    xp = bmat([[x], [s_init]])
    if find_feas:
        gp = zeros[: n + m, 0]
    else:
        gp = cp + Hp * xp

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
    C_f0 = Cmat[list(free_set)]         # Initial free variables

    num_actvd = len(activated_set) # no. of initially free variables, fixed
    num_frd   = len(freed_set)     # no. of initially fixed variables, freed

    # Check to see if we should even be solving this problem, and if it's
    # initially feasible
    A0 = Ap * C_f0.T
    if matrix_rank(A0) != min(A0.shape):
        raise LinAlgError('Bad rank for initial constraint matrix')
    # TODO should this area be cleaner? or stronger for boundary cases?
    if not (eigvals(f_xx) > 0).all():
        raise LinAlgError('Not positive definite')
    if (lp > xp).any() or (xp > up).any():
        raise ValueError('Outside bounds')

    K0 = bmat([[C_f0 * Hp * C_f0.T,         A0.T],
               [                A0, zeros[:m, :m]]])
    Kinv = inv(K0)

    print('Initial Sizes')
    print('n = %d' % n)
    print('m = %d' % m)
    print('n0 = %d' % n0)

    mu = 0
    i = 0
    while True:
        # Need to compute some transformation and sizes for this iteration
        C_f  = Cmat[list(free_set)]         # Current free (all) variables
        C_fd = Cmat[list(freed_set)]        # Current freed (not-initially free)
        C_w  = Cmat[list(active_set)]       # Current fixed (all) variables
        C_wd = Cmat[list(activated_set)]    # Current fixed (not-initially fixed)
        num_actvd = len(activated_set) # no. of initially free variables, fixed
        num_frd   = len(freed_set)     # no. of initially fixed variables, freed

        print('\n\nIteration: %4d' % i)
        print('Current objective value: %g' % (c.T * xp[:n, 0] + 0.5 * xp[:n, 0].T * f_xx * xp[:n, 0])[0, 0])
        print('Activated set size: %4d' % num_actvd)
        print('Max EQ constraint violation: \033[31m%g\033[0m' % max(Ap * xp))
        print("Lower Bound OK? \033[31m%5s\033[0m Upper Bound OK? \033[31m%5s\033[0m" % ((lp - xp < eps).all(), (up - xp > eps).all()))
        i += 1

        # Increasing the size of the system to solve
        # Utilizing the Schur Complement approach
        U = bmat([[            eye[: n0 + m, : n + m] * C_w.T,
                   bmat([[C_f0 * Hp * C_fd.T], [Ap * C_fd.T]])]])
        V = bmat([[zeros[:num_actvd, :num_actvd], zeros[:num_actvd, :num_frd]],
                  [  zeros[:num_frd, :num_actvd],          C_fd * Hp * C_fd.T]])
        C = V - U.T * Kinv * U

        if find_feas:
            ress = Ap * xp
        else:
            ress = zeros[:m, 0]

        f = bmat([[C_f0 * gp], [ress]])
        w = bmat([[zeros[0, :num_actvd], (gp.T * C_fd.T)]]).T
        v = Kinv * f
        z = solve(C, w - U.T * v)
        y = Kinv * (f - U * z)

        p = C_f0.T * C_f0 * -y[:n0, 0] # portion from initially free variables
        if num_frd != 0:               # portion from initially fixed variables
            p += C_fd.T * C_fd * -z[-num_frd:, 0]

        p  = C_f.T * C_f * p
        mu = v[n0 : n0 + m, 0] # Equality multipliers


        # Portion where the step size is calculated
        alpha_l = C_f.T * C_f * (lp - xp) / p
        alpha_u = C_f.T * C_f * (up - xp) / p
        alphas = maximum(alpha_l, alpha_u)
        alpha_unclip = nanmin(alphas)
        alpha_ind = nonzero(alphas == alpha_unclip)[0][0][0, 0]
        alpha = clip(alpha_unclip, 0, 1)


        xp = xp + alpha * p
        if not find_feas:
            gp = cp + Hp * xp

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
                mu_ineq = z[:len(active_set), 0]
            else:                       # some initial active set was provided
                #TODO it's wrong...
                mu_ineq = array(dot(C_w, gp) - dot(dot(Ap, C_w.T).T, mu)).flatten()

            active_list = list(active_set)
            upper = (abs(xp - up) < eps)[active_list]
            lower = (abs(xp - lp) < eps)[active_list]

            val_to_remove = 0
            ind_to_remove = -1
            # TODO Could probably redo w/ matrix operations?
            for ii in range(len(mu_ineq)):
                if upper[ii]:
                    if mu_ineq[ii] >= 0:
                        if abs(mu_ineq[ii]) >= val_to_remove:
                            ind_to_remove = active_list[ii]
                            val_to_remove = abs(mu_ineq[ii])
                elif lower[ii]:
                    if mu_ineq[ii] <= 0:
                        if abs(mu_ineq[ii]) >= val_to_remove:
                            ind_to_remove = active_list[ii]
                            val_to_remove = abs(mu_ineq[ii])
            print('full alpha')
            # if nothing has a bad multiplier, we're done!
            if ind_to_remove == -1:
                print('All lagrange multipliers are good!')
                break
            else: # otherwise, remove offending element from active set
                free_set.add(ind_to_remove)
                active_set -= {ind_to_remove}
                activated_set -= {ind_to_remove}
                if ind_to_remove not in free0_set: # only "freed" if not
                    freed_set.add(ind_to_remove)   # initially in free set

    return xp[:n, 0], active_set
