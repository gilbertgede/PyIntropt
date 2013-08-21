from numpy import zeros, finfo, vstack, ones
from numpy.linalg import norm, cholesky
from scipy.sparse import csc_matrix, coo_matrix, bmat, vstack as svstack
from scipy.sparse import spdiags, eye as seye
from scipy.sparse.linalg import factorized, eigsh


class iterate:
    """
    The object which stores a problem's state(s).

    Stores all relevant information for an iteration of a problem.
    """

    def __init__(self, x, problem, **kwargs):
        try:
            self.old = kwargs['old']
        except:
            self.old = None
        self.problem = problem
        self.x = x
        self.hessian_modified = False
        self.hessian_original = None

        self.mu     = kwargs['mu']
        self.delta  = kwargs['delta']
        self.tau    = kwargs['tau']
        self.eps_mu = kwargs['eps_mu']

        self.f   = problem.f(x)
        self.f_x = problem.f_x(x)
        self.c_e = problem.c_e(x)
        self.c_i = problem.c_i(x)
        self.A_e = problem.A_e(x)
        self.A_i = problem.A_i(x)

        self.n = len(x)
        self.m = len(self.c_i)
        self.t = len(self.c_e)
        self.e = ones((self.m, 1))

        if self.A_e.shape == (self.t, self.n):
            self.A_e = self.A_e.transpose()
        elif self.A_e.shape == (self.n, self.t):
            pass
        else:
            raise ValueError('Wrong shape for equality jacobian matrix')

        if self.A_i.shape == (self.m, self.n):
            self.A_i = self.A_i.transpose()
        elif self.A_i.shape == (self.n, self.m):
            pass
        else:
            raise ValueError('Wrong shape for inequality jacobian matrix')


        try:
            self.s = kwargs['s']
        except:
            s      = abs(self.c_i.reshape(-1, 1))
            self.s = s + (1. - self.tau) * (s == 0) # prevents div by 0 later

        # Combination objects, calculated once for convenience/speed
        self.fx_mu = vstack((self.f_x, -self.mu * self.e))
        self.ce_cis = vstack((self.c_e, self.c_i + self.s))
        self.A = bmat([[self.A_e,                                  self.A_i],
                       [    None,    spdiags(self.s.T, [0], self.m, self.m)]])
        self.A_aug = bmat([[seye(self.n + self.m, self.n + self.m), self.A],
                           [self.A.transpose(), None]]).tocsc()
        self.A_aug_fact = factorized(self.A_aug)
        self.A_c = self.A * self.ce_cis

        # Lagrange multipliers and Hessian
        try:
            oldie = kwargs['lm_mu']
        except:
            oldie = None
        self.l_e, self.l_i, s_sigma_s = kwargs['update_lambdas'](self, oldie)
        self.Aele_Aili = self.A_e * self.l_e + self.A_i * self.l_i
        try: # Either use actual hessian, or ...
            self.hessian = problem.hessian(x, self.l_e, self.l_i)
        except: # start with I for approximations
            try:
                self.hessian = kwargs['hessian_approx'](self)
            except:
                self.hessian = seye(self.n, self.n)

        self.hessian_original = self.hessian

        i = 0.001
        while (eigsh(self.hessian, k=1, which='SA', maxiter=100000,
               return_eigenvectors=False)[0] < 0):
            self.hessian_modified = True
            self.hessian = (self.hessian + i * seye(*self.hessian.shape)).tocsc()
            i *= 10

        self.G = bmat([[self.hessian, None],
                       [None,    s_sigma_s]]).tocsc()

        # Error values
        self.E, self.E_type = kwargs['error_func'](extra=True, it=self)
        self.E_mu = kwargs['error_func'](self.mu, it=self)


    def post_next(self, mu, tau, eps_mu, delta=None, **kwargs):
        self.mu = mu
        self.tau = tau
        self.eps_mu = eps_mu
        if delta:
            self.delta = delta
        try:
            self.E, self.E_type = kwargs['error_func'](extra=True, it=self)
            self.E_mu = kwargs['error_func'](self.mu, it=self)
        except:
            pass


    def next(self, d_x, d_s, **kwargs):
        # Resizing the trust region radius
        gamma = kwargs['gamma']
        d_norm = (norm(d_x)**2 + norm(d_s)**2)**0.5
        if gamma >= 0.9:
            delta = max(7. * d_norm, self.delta * 1.05)
        elif gamma >= 0.3:
            delta = max(2. * d_norm, self.delta * 1.05)
        elif gamma >= 0:
            delta = self.delta * 1.05
        else:
            delta = self.delta * 0.4

        return iterate(self.x + d_x, self.problem, s=self.s + d_s, delta=delta,
                       mu=self.mu, tau=self.tau, eps_mu=self.eps_mu,
                       error_func=kwargs['error_func'],
                       update_lambdas=kwargs['update_lambdas'],
                       hessian_approx=kwargs['hessian_approx'], old=self, lm_mu=self.mu)
