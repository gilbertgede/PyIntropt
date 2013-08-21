from numpy import zeros, ones, eye, dot, vstack, log, finfo, empty, inf, isnan
from numpy.linalg import norm
from scipy.sparse import csc_matrix, spdiags, bmat, eye as seye
from scipy.sparse.linalg import factorized
from .problem import problem
from .iterate import iterate


class nitro:
    """
    Solver based on the NITRO algorithm.

    References:
    [Byrd1999] R. H. Byrd, M. E. Hribar, and J. Nocedal, An interior point
    method for large scale nonlinear programming, SIAM Journal on
    Optimization, 9 (1999), pp. 877-900.
    [Nocedal2006] J. Nocedal and S. J. Wright. Numerical Optimization, Second
    Edition. Springer Series in Operations Research, Springer Verlag, 2006.
    [SciPy2011] SciPy's Optimization Module, Oct. 2011
    """

    def __init__(self, x0, problem, mu=0.1, delta=1., delta_min=1.e-20):
        """
        Initializer for nitro class instance.

        Takes initial point, problem, and initial barrier and trust-region values.
        """

        self.hess_approx = 'SR1'
        if mu > 1.:
            raise ValueError('mu cannot be above 1.')
        tau = 0.995
        eps_mu = mu * 1.

        self.eta = 1.e-8
        self.zeta = 0.8
        self.rho = 0.3
        self.normal_tol = 1e-5
        self.delta_min = delta_min
        self.nu = 1.
        self.qp_k = 1
        self.mu_k = 1

        self.problem = problem
        self.it_list = [iterate(x0, problem, mu=mu, delta=delta, tau=tau,
                                eps_mu=eps_mu, error_func=self._error_func,
                                hessian_approx=self._update_hessian,
                                update_lambdas=self._update_lambdas)]


    def run(self, max_iter=50, stopping_tol=1.e-7, printing=True):
        running = True
        while ( (self.qp_k < max_iter) and (self.mu_k < max_iter) and
                (self.it_list[-1].delta > self.delta_min) and running):
            it = self.it_list[-1]

            if it.E < stopping_tol:
                running = False
                continue
            elif it.E_mu < it.eps_mu:
                self.mu_k += 1
                mu = max(1.e-8, min(0.2 * it.mu, (it.E_mu / self.it_list[0].E_mu)**1.7))
                eps_mu = max(1.e-8, min(1.e-2, 30 * it.mu))
                tau = 1. - min(0.005, it.E_mu / self.it_list[0].E_mu)
                self._print_major(mu, it.mu)
                it.post_next(mu, tau, eps_mu, error_func=self._error_func)
                continue

            d_x, d_s, gamma, step_flag = self.step()
            self.qp_k += 1
            self.it_list.append(it.next(d_x, d_s, gamma=gamma,
                                        error_func=self._error_func,
                                        hessian_approx=self._update_hessian,
                                        update_lambdas=self._update_lambdas))
            self._print_minor(step_flag, d_x, d_s)


    def step(self):
        """
        Compute single step for an iterate.
        """
        it = self.it_list[-1]
        (v_x, v_s) = self._normal_step(self.zeta, it.delta, it.tau, it.A,
                                       it.ce_cis, it.n, it.m, it.A_aug_fact,
                                       it.A_c, it.s, self.normal_tol)
        (w_x, w_s) = self._tangential_step(v_x, v_s, it.n, it.m, it.t,
                                           it.delta, it.tau, it.s, it.fx_mu,
                                           it.G, it.A_aug_fact)
        d_x = w_x + v_x
        d_s = w_s + v_s

        # compute penalty parameter nu, actual and predicted merit
        # function reductions
        if it.m == 0:
            v_bar = v_x
            w_bar = w_x
        else:
            v_bar = vstack((v_x, v_s / it.s))
            w_bar = vstack((w_x, w_s / it.s))
        v_w_bar = v_bar + w_bar
        A_T_v_bar = it.A.transpose() * v_bar
        vpred = (norm(it.ce_cis) - norm(it.ce_cis + A_T_v_bar))
        self._update_penalty_parameter(A_T_v_bar, v_w_bar, vpred,
                                       it.ce_cis, it.G, self.nu,
                                       it.fx_mu, self.rho)
        pred_parts = self._compute_pred(v_w_bar, vpred, it)
        ared_parts = self._compute_ared(d_x, d_s, it)
        pred = sum(pred_parts)
        ared = sum(ared_parts)

        step_flag = 0   # Type of step: 0 (none), 1 (normal), 2 (corrected)

        if ared >= self.eta * pred:
            # Take step as computed
            step_flag = 1
        else:
            # attempt second order correction
            y_x, y_s = self._second_order_correction(v_bar, w_bar, d_x, d_s)
            if (y_x is not None) and (y_s is not None):
                d_x += y_x
                d_s += y_s
                #slack_vio = (it.s + d_s >= (1. - it.tau) * it.s).all()
                slack_vio = (it.e + d_s / it.s >= (1. - it.tau)).all()
                ared = sum(self._compute_ared(d_x, d_s, it))
                if slack_vio and (ared >= self.eta * pred):
                    # Take step with second order correction
                    step_flag = 2
        if step_flag == 0:
            d_x *= 0
            d_s *= 0

        gamma = ared / pred
        return (d_x, d_s, gamma, step_flag)

    def _print_major(self, mu, mu_old):
        print(('\033[95m\033[1m%6d|  Barrier Parameter Iteration  ' % self.mu_k +
              '\t     \u03BC: %10.6e -> %10.6e\033[0;0m\x1b[0;0m' % (mu_old, mu)))

    def _print_minor(self, step_flag, d_x, d_s):
        it = self.it_list[-1]
        if step_flag == 0:
            temp_str = 'No Step '
        elif step_flag == 1:
            temp_str = 'Step    '
        elif step_flag == 2:
            temp_str = 'S.O.C.  '
        elif step_flag == 3:
            temp_str = 'NM Step '
        step_len = (norm(d_x)**2 + norm(d_s)**2)**0.5
        print(('%6d|  ' % self.qp_k + temp_str +
               '    \u0394: %10.6e' % it.delta +
               '    Step length: %10.6e' % step_len +
               '    Error %d: %10.6e' % (it.E_type, it.E_mu)))


    def _error_func(self, mu=0, extra=False, it=None):
        """
        Function for computing optimality condition of the barrier problem.

        Returns ||L_x, s l_i - mu, c_e, c_i + s||inf
        From [Byrd1999] equation (2.3)
        """
        one   = abs(it.f_x + it.Aele_Aili)
        two   = abs(it.s * it.l_i - mu * it.e)
        three = abs(it.c_e)
        four  = abs(it.c_i + it.s)
        l = [one.max(), two.max(), three.max(), four.max()]
        m = max(l)
        i = l.index(m) + 1
        if extra == False:
            return m
        else:
            return (m, i)


    def _update_lambdas(self, it, oldie):
        # Function to update lagrange multipliers
        # [Byrd1999] equations (3.12, 3.15)
        if oldie is not None:
            mu = oldie
        else:
            mu = it.mu
        fx_mu = vstack([it.f_x, -it.e * mu])
        b = (it.A.transpose() * -fx_mu).reshape(-1, 1)
        temp = vstack((zeros((it.n + it.m, 1)), -b)).squeeze()
        lambdas = it.A_aug_fact(temp).reshape(-1, 1)[it.n + it.m:]
        l_e = lambdas[:it.t]
        l_i = lambdas[it.t:]
        sigma = empty((it.m, 1))
        for i in range(len(l_i)):
            if l_i[i] > 0:
                sigma[i] = l_i[i] / it.s[i]
            else:
                sigma[i] = mu * it.s[i]**-2
        return (l_e, l_i, spdiags((it.s * sigma * it.s).T, [0], it.m, it.m))


    def _update_hessian(self, it):
        """
        Provides quasi-newton approximation to the Hessian matrix.
        """
        it = self.it_list[-1]
        it_old = it.old
        s = it.x - it_old.x
        y = it.f_x + it.Aele_Aili - (it.old.f_x + it.old.Aele_Aili)
        hess = it.hessian

        if self.hess_approx == 'SR1':
            temp = y - it.hess * s
            temp2 = dot(temp.T, s)
            if abs(temp2) >= self.eta * norm(s) * norm(temp):
                temp = dot(temp, temp.T) / temp2
                temp = csc_matrix(temp)
                hess = hess + temp
        elif self.hess_approx == 'BFGS':
            H_s = hess * s
            s_T_H_s = dot(s.T, H_s)
            s_T_y = dot(s.T, y)
            # Choose size for theta
            if s_T_y >= 0.2 * s_T_H_s:
                theta = 1.
            else:
                theta = ((0.8 * s_T_H_s) / (s_T_H_s - s_T_y))
            r = theta * y + (1. - theta) * H_s
            temp = csc_matrix( - dot(H_s, H_s.T) / s_T_H_s + dot(r, r.T) / dot(s.T, r))
            hess = hess + temp
        else:
            raise ValueError('Invalid Hessian approximation style chosen')

        return hess


    def _normal_step(self, zeta, delta, tau, A, ce_cis, n, m, A_aug_fact,
                     A_c, s, normal_tol):
        """
        The normal step seeks to improve constraint violations

        originally from:
        [Byrd1999] Dogleg Procedure, equations (3.19 - 3.24)
        possibly modified by work in :
        [Liu1999]
        """
        # Tests if the step is feasible
        def feas(v_x, v_s):
            v_norm = (norm(v_x)**2 + norm(v_s)**2)**0.5
            first = (v_norm <= zeta * delta)
            second = (v_s >= -tau / 2.).all()
            return (first and second)
        # Model for approximation of quadratic subproblem
        def model_m(v):
            A_T_v = A.transpose() * v
            return dot(2. * ce_cis.T + A_T_v.transpose(), A_T_v)

        # Newton step
        res = -A_aug_fact(vstack((zeros((n + m, 1)),
                                  ce_cis)).squeeze()).reshape(-1, 1)
        v_n = -A * res[n + m:]
        v_n_x = v_n[:n]
        v_n_s = v_n[n:]
        theta1 = 1.
        while ((not feas(theta1 * v_n_x, theta1 * v_n_s)) and
               (theta1 > normal_tol)):
            theta1 *= 0.8
        # If Newton step is feasible, take that
        if theta1 == 1:
            v = v_n
        # If Newton step is not feasible:
        else:
            temp = A.transpose() * A_c
            alpha = norm(A_c)**2. / dot(temp.T, temp)
            # Cauchy point step
            v_cp = -alpha * A_c
            v_cp_x = v_cp[:n]
            v_cp_s = v_cp[n:]
            theta2 = 1.
            while ((not feas((1. - theta2) * v_cp_x + theta2 * v_n_x, (1. -
                   theta2) * v_cp_s + theta2 * v_n_s)) and (theta2 >
                       normal_tol)):
                theta2 *= 0.8
            if theta2 < normal_tol:
                # If Cauchy point step is too big, scale it down
                theta3 = 1.
                while ((not feas(theta3 * v_cp_x, theta3 * v_cp_s)) and
                       (theta3 > normal_tol)):
                    theta3 *= 0.8
                v_dl = theta3 * v_cp
            else:
                # Otherwise, take combination of Cauchy point and Newton steps
                v_dl = (1. - theta2) * v_cp + theta2 * v_n
            if (model_m(v_dl) < model_m(theta1 * v_n)):
                v = v_dl
            else:
                v = theta1 * v_n
        # Untransform from v_bar to v
        return (v[:n], s * v[n:])


    def _tangential_step(self, v_x, v_s, n, m, t, delta, tau, s,
                         fx_mu, G, A_aug_fact):
        """
        Tangent step stays at curent constraints and searchs for optimality

        from:
        [Liu1999] PCG Proceduce, strategy 2
        """
        if m != 0:
            v_bar = vstack((v_x, v_s / s))
        else:
            v_bar = v_x
        v_s_bar = v_bar[n:]

        qmod = lambda w: dot(fx_mu.T, v_bar + w) + 0.5 * dot((v_bar + w).T, G * (v_bar + w))

        w = zeros((n + m, 1))
        r = fx_mu + G * v_bar
        # skipping that inverse
        temp_0 = zeros((t + m, 1))
        temp = vstack((r, temp_0)).squeeze()
        g = A_aug_fact(temp)[:n + m].reshape(-1, 1)
        # end of skipping inverse
        p = -g
        w_c = w.copy()
        i = 0
        dot_rtg = dot(r.T, g)
        tol = 0.01 * abs(dot_rtg)

        while abs(dot_rtg) >= tol:
            ptGp = dot(p.T, (G * p))
            rtg = dot_rtg.copy()
            alpha = rtg / ptGp
            w += alpha * p
            if i == 0:
                w_c = w.copy()
            r_plus = r + alpha * (G * p)
            # also skipping inverse
            temp = vstack((r_plus, temp_0)).squeeze()
            g_plus = A_aug_fact(temp)[:n + m].reshape(-1, 1)
            # done skipping use of inverted matrix
            dot_rtg = dot(r_plus.T, g_plus)
            beta =  dot_rtg / rtg
            p = -g_plus + beta * p
            r = r_plus
            g = g_plus
            i += 1
            # strategy 2 test and infinite loop avoidance check
            if (((norm(w) >= delta) or (i > 2 * (n - t)) or (ptGp < 0)) and
                (i > 1)):
                """
                if norm(w) >= delta:
                    print 'norm'
                if i > 2 * (n-t):
                    print 'i'
                if ptGp < 0:
                    print 'neg'
                """
                break
        # Figure out the boundary violation status of the results
        l_slack = False
        l_trust = False
        vs_tau = v_s_bar + tau
        if ((w[n:] + vs_tau) < 0).any():
            l_slack = True
        if norm(w) >= delta:
            l_trust = True

        # fix any potential slack violation
        slack_violation = w_c[n:] + vs_tau
        slack_violation = slack_violation.clip(-inf, 0)
        #TODO line below
        scale = min((w_c[n:] - slack_violation) / w_c[n:]) # slow line
        scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
        w_c *= scale
        if norm(w_c) > delta: # trust region violation, scale down Cauchy point
            w_c *= delta / norm(w_c)
        if i == 1: # only Cauchy point was found
            return (w_c[:n], s * w_c[n:])

        # Now choose the actual point
        if l_trust and l_slack: # step is very bad
            w_c_norm = norm(w_c)
            w_norm = norm(w)
            dist   = w_norm - w_c_norm
            over   = w_norm - delta
            scale  = (dist - over) / dist
            scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
            w_dl = w - w_c
            w_i = w_c + scale * w_dl
            # now need to compare w, w_i, w_c
            # first, make sure nothing violates slack
            slack_violation = w[n:] + vs_tau
            slack_violation = slack_violation.clip(-inf, 0)
            scale = min((w[n:] - slack_violation) / w[n:])
            scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
            w *= scale
            slack_violation = w_i[n:] + vs_tau
            slack_violation = slack_violation.clip(-inf, 0)
            scale = min((w_i[n:] - slack_violation) / w_i[n:])
            scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
            w_i *= scale
            # actually comparing
            w *= delta / norm(w)
            qmod_w_c = qmod(w_c)
            qmod_w_i = qmod(w_i)
            qmod_w   = qmod(w)
            temp = [qmod_w, qmod_w_i, qmod_w_c]
            if min(temp) == qmod_w_c:
                w = w_c
            if min(temp) == qmod_w_i:
                w = w_i
            return (w[:n], s * w[n:])
        if l_slack: # step violates slack only
            slack_violation = w[n:] + vs_tau
            slack_violation = slack_violation.clip(-inf, 0)
            #TODO line below
            scale = min((w[n:] - slack_violation) / w[n:]) # slow line
            scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
            w *= scale
            qmod_w_c = qmod(w_c)
            qmod_w   = qmod(w)
            if qmod_w_c < qmod_w:
                return (w_c[:n], s * w_c[n:])
            else:
                return (w[:n], s * w[n:])
        if l_trust: # step only violates trust region
            w_c_norm = norm(w_c)
            w_norm = norm(w)
            dist   = w_norm - w_c_norm
            over   = w_norm - delta
            scale  = (dist - over) / dist
            scale = (1e-8 if (isnan(scale) or (scale is 0)) else scale)
            w_dl = w - w_c
            w = w_c + scale * w_dl
            return (w[:n], s * w[n:])
        else: # everything's good
            return (w[:n], s * w[n:])


    # Update the penalty parameter used to determine if a step should be taken
    # [Byrd1999] Penalty Parameter Prodcedure
    def _update_penalty_parameter(self, A_T_v_bar, v_w_bar, vpred, ce_cis,
                                  G, nu, fx_mu, rho):
        def model_m():
            first = 2 * dot(ce_cis.T, A_T_v_bar)
            second = dot(A_T_v_bar.T, A_T_v_bar)
            return first + second

        top = (dot(fx_mu.T, v_w_bar) +
               0.5 * dot(v_w_bar.T, (G * v_w_bar)))
        # Possibly increase nu if it is not big enough
        if model_m() != 0:
            nu = max(nu, top / ((1 - rho) * vpred))


    # Predicted reduction in merit function
    # [Byrd1999] equations (3.32, 3.49, 3.50)
    def _compute_pred(self, v_w_bar, vpred, it):
        temp = 0.5 * it.G * v_w_bar
        f_red = dot(it.f_x.T, v_w_bar[:it.n]) + dot(v_w_bar[:it.n].T, temp[:it.n])
        s_red = -it.mu * sum(v_w_bar[it.n:]) + dot(v_w_bar[it.n:].T, temp[it.n:])
        return (-f_red, -s_red, self.nu * vpred)


    # Actual reduction in merit function
    # [Byrd1999] equations (2.8, 3.53)
    def _compute_ared(self, d_x, d_s, it):
        problem = self.problem
        f_x = it.f
        f_xdx = problem.f(it.x + d_x)
        c_e_x = it.c_e
        c_e_xdx = problem.c_e(it.x + d_x)
        c_i_x = it.c_i
        c_i_xdx = problem.c_i(it.x + d_x)
        cs_norm = (norm(c_e_x)**2 + norm(c_i_x + it.s)**2)**0.5
        cs_norm_dxds = (norm(c_e_xdx)**2 + norm(c_i_xdx + it.s + d_s)**2)**0.5
        return (f_x - f_xdx,
                - it.mu * (sum(log(abs(it.s))) - sum(log(abs(it.s + d_s)))),
                self.nu * (cs_norm - cs_norm_dxds))


    # Makes a second order correction if necessary
    # [Byrd1999] Procedure SOC
    def _second_order_correction(self, v_bar, w_bar, d_x, d_s):
        problem = self.problem
        it = self.it_list[-1]
        v_bar_norm = norm(v_bar)
        w_bar_norm = norm(w_bar)
        if v_bar_norm <= 0.1 * w_bar_norm:
            xdx = it.x + d_x
            c_e_xdx = problem.c_e(xdx)
            c_i_xdx = problem.c_i(xdx)
            temp = vstack((c_e_xdx, c_i_xdx + it.s + d_s))
            temp = vstack((zeros((it.n + it.m, 1)), temp)).squeeze()
            res = -it.A_aug_fact(temp).reshape(-1, 1)
            y = it.A * res[it.n + it.m:]
            return (y[:it.n], y[it.n:])
        else:
            return (None, None)


