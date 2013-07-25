from numpy import zeros, finfo, vstack, ones
from scipy.sparse import csc_matrix, coo_matrix, bmat, vstack as svstack
from scipy.sparse import eye as seye


class problem:
    """
    The object which stores a problem's definition.

    It is supposed to take in however the user has defined the problem, and
    transform it into a format for the solver to use.
    """

    def __init__(self, n, f, **kwargs):
        """
        Initializer for the problem class.

        The user will supply functions and vectors as keyword arguements. The
        input vector x is assumed to be of length n.

        Each function is assumed to take in only the vector x (with the
        exception of the hessian function). Any user arguements must be wrapped
        first instead.

        Parameters
        ==========
            n : integer, length of input vector for NLP problem
            f :   function
                  takes in x
                  returns scalar

        Keyword args
        ============
            f_x : function
                  takes in x
                  returns numpy column array of n x 1
            x_l : a numpy column array of n x 1
            x_u : a numpy column array of n x 1


          and:


            g   : function
                  takes in x
                  returns numpy column array
            h   : function
                  takes in x
                  returns numpy column array
            g_x : function
                  takes in x
                  returns scipy sparse matrix of n x len(g)
            h_x : function
                  takes in x
                  returns scipy sparse matrix of n x len(h)
            hessian : function
                      takes in x, v_g, v_h
                      where v are the lagrange multipliers of the lagrangian,
                      assumed to be of the form:
                      f(x) + v_g^T g + v_h^T h
                      returns scipy sparse matrix of n x n
          or:
            c   : function
                  takes in x
                  returns numpy column array
            c_x : function
                  takes in x
                  returns scipy sparse matrix of n x len(g)
            c_l : numpy column array of len(c) x 1
            c_u : numpy column array of len(c) x 1
            hessian : function
                      takes in x, v
                      where v are the lagrange multipliers (as a numpy column
                      vector) of the lagrangian, assumed to be of the form:
                      f(x) + v^T c
                      returns scipy sparse matrix of n x n

        Methods generated for the solver are:
        f
        f_x
        c_e
        c_i
        A_e
        A_i
        H

        """
        self.n = n
        # First check to make sure problem hasn't been incorrectly supplied
        combined  = ['c', 'cl', 'cu', 'c_x']
        separated = ['g', 'h', 'g_x', 'h_x']
        # now check if components from either style are in the kwargs
        check1 = max([(i in kwargs.keys()) for i in combined])
        check2 = max([(i in kwargs.keys()) for i in separated])
        # Raise error if 2 styles combined, or no constraints supplied
        if check1 and check2:
            raise ValueError('Problem supplied incorrectly (constraint style)')
        elif check1:
            style = 'c'
        elif check2:
            style = 's'
        else:
            raise ValueError('Only constrained problems supported')
        # Also need to create settings for finite differencing
        try:
            # This is equivalent to max(f^y (x0)) where y = order + 1
            # Used to calculate optimal finite difference step size
            self._fdscale = kwargs['fin_diff_scale']
        except:
            self._fdscale = 1. / 3.
        try: # Finite difference order - forward h, central h^2, central h^4
            self._order = kwargs['fin_diff_order']
        except:
            self._order = 2

        ##########################################
        # Functions are now going to be defined. #
        ##########################################

        ########################
        # Common to both forms #
        ########################
        # objective function definition
        resh = lambda x: x.reshape(-1, 1)
        self.f = lambda x: resh(f(x))
        # objective function gradient
        try:
            self.f_x = kwargs['f_x']
        except:
            self.f_x = lambda x: self.approx_jacobian(x, f)

        ##################
        # Separated form #
        ##################
        if style == 's':
            # equality constraint function
            try:
                self.c_e = kwargs['g']
            except:
                self.c_e = lambda x: self.empty_f(x)
            # inequality constraint function
            try:
                self.c_i = kwargs['h']
            except:
                self.c_i = lambda x: self.empty_f(x)
            # equality constraint gradient
            try:
                self.A_e = kwargs['g_x']
            except:
                self.A_e = lambda x: csc_matrix(self.approx_jacobian(x, self.c_e))
            # inequality constraint gradient
            try:
                self.A_i = kwargs['h_x']
            except:
                self.A_i = lambda x: csc_matrix(self.approx_jacobian(x, self.c_i))
            # hessian function
            try:
                self.hessian = kwargs['hessian']
            except:
                self.hessian = None
        #################
        # Combined form #
        #################
        else:
            ######## long and awkward... ###########
            try:
                self.xl = kwargs['xl']
            except:
                self.xl = ones((n, 1)) * -1e20
            try:
                self.xu = kwargs['xu']
            except:
                self.xu = ones((n, 1)) * 1e20
            self.c  = kwargs['c']
            self.cl = kwargs['cl']
            self.cu = kwargs['cu']
            try:
                self.c_x = kwargs['c_x']
            except:
                self.c_x = lambda x: csc_matrix(self.approx_jacobian(x, self.c))
            o = n
            mn = len(self.cl)
            I = seye(o, o).tocsc()

            (coo_xer, coo_xec, coo_xed, coo_xir, coo_xic, coo_xid, coo_xlr,
             coo_xlc, coo_xld, coo_xur, coo_xuc, coo_xud, coo_cer, coo_cec,
             coo_ced, coo_cir, coo_cic, coo_cid, coo_clr, coo_clc, coo_cld,
             coo_cur, coo_cuc, coo_cud) = ([], [], [], [], [], [], [], [], [],
                                           [], [], [], [], [], [], [], [], [],
                                           [], [], [], [], [], [])

            ############## BOUNDS ################
            xl = self.xl
            xu = self.xu
            c = self.c
            cl = self.cl
            cu = self.cu
            c_x = self.c_x
            xm = 0
            xn = 0
            for i in range(o):
                if xl[i] == xu[i]:
                    coo_xer += [xm]
                    coo_xec += [i]
                    coo_xed += [1]
                    xm += 1
                else:
                    coo_xir += [xn]
                    coo_xic += [i]
                    coo_xid += [1]
                    xn += 1
            l = 0
            u = 0
            for i in range(xn):
                if xl[coo_xic[i]] >= -1e19:
                    coo_xlr += [l]
                    coo_xlc += [coo_xic[i]]
                    coo_xld += [coo_xid[i]]
                    l += 1
                if xu[coo_xic[i]] <= 1e19:
                    coo_xur += [u]
                    coo_xuc += [coo_xic[i]]
                    coo_xud += [coo_xid[i]]
                    u += 1
            try:
                Kxe = coo_matrix((coo_xed, (coo_xer, coo_xec)), shape=(xm, o)).tocsc()
            except:
                Kxe = None
            try:
                Kxl = coo_matrix((coo_xld, (coo_xlr, coo_xlc)), shape=(l, o)).tocsc()
            except:
                Kxl = None
            try:
                Kxu = coo_matrix((coo_xud, (coo_xur, coo_xuc)), shape=(u, o)).tocsc()
            except:
                Kxu = None

            ############## CONSTRAINTS ################
            cm = 0
            cn = 0
            for i in range(mn):
                if cl[i] == cu[i]:
                    coo_cer += [cm]
                    coo_cec += [i]
                    coo_ced += [1]
                    cm += 1
                else:
                    coo_cir += [cn]
                    coo_cic += [i]
                    coo_cid += [1]
                    cn += 1

            l = 0
            u = 0
            for i in range(cn):
                if cl[coo_cic[i]] >= -1e19:
                    coo_clr += [l]
                    coo_clc += [coo_cic[i]]
                    coo_cld += [coo_cid[i]]
                    l += 1
                if cu[coo_cic[i]] <= 1e19:
                    coo_cur += [u]
                    coo_cuc += [coo_cic[i]]
                    coo_cud += [coo_cid[i]]
                    u += 1
            try:
                Kce = coo_matrix((coo_ced, (coo_cer, coo_cec)), shape=(cm, mn)).tocsc()
            except:
                Kce = None
            try:
                Kcl = coo_matrix((coo_cld, (coo_clr, coo_clc)), shape=(l, mn)).tocsc()
            except:
                Kcl = None
            try:
                Kcu = coo_matrix((coo_cud, (coo_cur, coo_cuc)), shape=(u, mn)).tocsc()
            except:
                Kcu = None

            ############## COMBINING ################
            # Equality
            if (Kxe is not None) and (Kce is not None):
                Ke  = bmat([[Kxe, None],
                            [None, Kce]])
                ce  = vstack([Kxe * xl, Kce * cl])
                eq  = lambda x: Ke * vstack([resh(x), c(x)]) - ce
                jeq = lambda x: (Ke * svstack([I, c_x(x)]))
                num_x_eq = len(Kxe * xl)
            elif Kxe is not None:
                Ke  = Kxe
                ce  = Kxe * xl
                eq  = lambda x: Ke * resh(x) - ce
                jeq = lambda x: Ke
                num_x_eq = len(Kxe * xl)
            elif Kce is not None:
                Ke  = Kce
                ce  = Kce * cl
                eq  = lambda x: Ke * c(x) - ce
                jeq = lambda x: (Ke * c_x(x))
                num_x_eq = 0
            else:
                Ke  = None
                ce  = None
                eq  = None
                jeq = None
                num_x_eq = 0
            # Bounds
            if (Kxl is not None) and (Kxu is not None):
                Kiu = bmat([[-Kxl],
                            [ Kxu]])
                ciu = vstack([Kxl * xl, -Kxu * xu])
            elif Kxl is not None:
                Kiu = -Kxl
                ciu = Kxl * xl
            elif Kxu is not None:
                Kiu = Kxu
                ciu = -Kxu * xu
            else:
                Kiu = None
                ciu = None
            # Constraints
            if (Kcl is not None) and (Kcu is not None):
                Kil = bmat([[-Kcl],
                            [ Kcu]])
                cil = vstack([Kcl * cl, -Kcu * cu])
            elif Kcl is not None:
                Kil = -Kcl
                cil = Kcl * cl
            elif Kcu is not None:
                Kil = Kcu
                cil = -Kcu * cu
            else:
                Kil = None
                cil = None
            # Bounds + Constraints
            if (Kiu is not None) and (Kil is not None):
                Ki    = bmat([[ Kiu, None],
                            [None,  Kil]])
                ci    = vstack([ciu, cil])
                ineq  = lambda x: ci + Ki * vstack([resh(x), c(x)])
                jineq = lambda x: (Ki * svstack([I, c_x(x)]))
                num_x_bound = len(ciu)
            elif Kil is not None:
                Ki    = Kil
                ci    = cil
                ineq  = lambda x: ci + Ki * c(x)
                jineq = lambda x: (Ki * c_x(x))
                num_x_bound = 0
            elif Kiu is not None:
                Ki    = Kiu
                ci    = ciu
                ineq  = lambda x: ci + Ki * resh(x)
                jineq = lambda x: Ki
                num_x_bound = len(ciu)
            else:
                Ki    = None
                ci    = None
                ineq  = None
                jineq = None
                num_x_bound = 0

            ############# HESSIAN ###################
            try:
                hess_func = kwargs['hessian']
                self.c_hessian = hess_func
                def hess(x, lam_e, lam_i):
                    lam_e = resh(lam_e)
                    lam_i = resh(lam_i)
                    lam = 0
                    try:
                        lam += Kce.transpose() * lam_e[num_x_eq:]
                    except:
                        pass
                    try:
                        lam += Kcl.transpose() * lam_i[num_x_bound:]
                    except:
                        pass
                    try:
                        lam += Kcu.transpose() * lam_i[num_x_bound:]
                    except:
                        pass
                    return hess_func(x, lam)
            except:
                hess = None

            self.c_e = eq
            self.c_i = ineq
            self.A_e = jeq
            self.A_i = jineq
            self.hessian = hess





    def empty_f(x):
        return zeros((0, 1))

    def approx_jacobian(self, x0, f):
        # The code below finds the 'optimal' step size, h, for finite
        # differencing. The function for the appropriate order is also
        # generated.
        # See Course Notes: University of Washington, AMATH 301 Beginning
        # Scientific Computing. J. N. Kutz, 2009
        M = self._fdscale # eq. to max(f'''(x0)), for 2nd order
        order = self._order
        eps = finfo(float).eps
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
        self._h = h  # Store this, for other uses
        f0 = f(x0)
        jac = zeros([len(x0), len(f0)])
        dx = zeros((len(x0), 1))
        for i in range(len(x0)):
            dx[i] = h
            jac[i] = row(dx)
            dx[i] = 0
        if jac.shape[0] == 1:
            return jac.T
        elif jac.shape[1] == 1:
            return jac
        elif jac.shape[0] == len(x0):
            return jac.T
        else:
            return jac







