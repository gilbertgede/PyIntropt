from pyintropt import problem
from numpy import sin, cos, tan, array, ones, eye, arange, vstack
from scipy.sparse import csc_matrix, bmat


def test_problem():
    # The following code is... disgusting.
    # It was generated with SymPy originally, but the SymPy -> NumPy
    # translation wasn't playing nice, so it was copied/pasted.
    # The equations themselves aren't anything specific.
    xn = 10
    gn = 4
    hn = 5
    f = lambda x: array([[x[2, 0] * x[3, 0]**2 + x[4, 0] * x[5, 0] * x[6, 0] +
                          x[7, 0] * x[8, 0] / x[9, 0] + sin(x[0, 0] * x[1,
                          0])]])
    # First, the separated constraints, which is the format the NITRO algorithm
    # is written for.
    g = lambda x: array([[x[2, 0] + x[3, 0] + 1.], [x[0, 0] * x[1, 0] / 4.],
                         [x[3, 0]**3], [tan(x[8, 0] * x[9, 0] / 20.)]])
    h = lambda x: array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[5, 0] / x[6, 0]],
                         [cos(x[7, 0]**2)]])
    f_x = lambda x: array([[x[1, 0] * cos(x[0, 0] * x[1, 0])],
                           [x[0, 0] * cos(x[0, 0] * x[1, 0])], [x[3, 0]**2],
                           [2. * x[2, 0] * x[3, 0]], [x[5, 0] * x[6, 0]],
                           [x[4, 0] * x[6, 0]], [x[4, 0] * x[5, 0]],
                           [x[8, 0] / x[9, 0]], [x[7, 0] / x[9, 0]],
                           [-x[7, 0] * x[8, 0] / x[9, 0]**2]])
    g_x = lambda x: csc_matrix(array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                      [x[1, 0] / 4., x[0, 0] / 4., 0 ,0 ,0 ,0,
                                       0 ,0, 0, 0],
                                      [0, 0, 0, 3. * x[3, 0]**2, 0, 0, 0, 0, 0,
                                       0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, x[9] * (tan(x[8,
                                       0] * x[9, 0] / 20.)**2 + 1.) / 20., x[8,
                                       0] * (tan(x[8, 0] * x[9, 0] / 20.)**2 +
                                       1.) / 20.]]))
    h_x = lambda x: csc_matrix(array([[1., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1., 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1. / x[6, 0], -x[5, 0] /
                                       x[6, 0]**2, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, -2. * x[7, 0] *
                                       sin(x[7, 0]**2), 0, 0]]))
    hessian = lambda x, ve, vi: csc_matrix(array([
                                    [-x[1, 0]**2 * sin(x[0, 0] * x[1, 0]),
                                     ve[1, 0] / 4. - x[0, 0] * x[1, 0] *
                                     sin(x[0, 0] * x[1, 0]) + cos(x[0, 0]
                                     * x[1, 0]), 0, 0, 0, 0, 0, 0, 0, 0],
                                    [ve[1, 0] / 4. - x[0, 0] * x[1, 0] *
                                     sin(x[0, 0] * x[1, 0]) + cos(x[0, 0] *
                                     x[1, 0]), -x[0, 0]**2 * sin(x[0, 0] * x[1,
                                     0]), 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 2. * x[3, 0], 0, 0, 0, 0, 0, 0],
                                    [0, 0, 2. * x[3, 0], 6. * ve[2, 0] * x[3, 0] + 2. *
                                     x[2, 0], 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, x[6, 0], x[5, 0], 0, 0, 0],
                                    [0, 0, 0, 0, x[6, 0], 0, -vi[3, 0] / x[6,
                                     0]**2 + x[4, 0], 0, 0, 0],
                                    [0, 0, 0, 0, x[5, 0], -vi[3, 0] / x[6,
                                     0]**2 + x[4, 0], 2. * vi[3, 0] * x[5, 0] /
                                     x[6, 0]**3, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, -4. * vi[4, 0] * x[7,
                                     0]**2 * cos(x[7, 0]**2) - 2. * vi[4, 0] *
                                     sin(x[7, 0]**2), 1. / x[9, 0], -x[8, 0] /
                                     x[9, 0]**2],
                                    [0, 0, 0, 0, 0, 0, 0, 1. / x[9, 0], ve[3,
                                     0] * x[9, 0]**2 * (tan(x[8, 0] * x[9, 0] /
                                     20.)**2 + 1) * tan(x[8, 0] * x[9, 0] /
                                     20.) / 200., ve[3, 0] * x[8, 0] * x[9, 0]
                                     * (tan(x[8, 0] * x[9, 0] / 20.)**2 + 1.) *
                                     tan(x[8, 0] * x[9, 0] / 20.) / 200. +
                                     ve[3, 0] * (tan(x[8, 0] * x[9, 0] /
                                     20.)**2 + 1.) / 20. - x[7, 0] / x[9,
                                     0]**2],
                                    [0, 0, 0, 0, 0, 0, 0, -x[8, 0] / x[9,
                                     0]**2, ve[3, 0] * x[8, 0] * x[9, 0] *
                                     (tan(x[8, 0] * x[9, 0] / 20.)**2 + 1.) *
                                     tan(x[8, 0] * x[9, 0] / 20.) / 200. +
                                     ve[3, 0] * (tan(x[8, 0] * x[9, 0] /
                                     20.)**2 + 1.) / 20. - x[7, 0] / x[9,
                                     0]**2, ve[3, 0] * x[8, 0]**2 * (tan(x[8,
                                     0] * x[9, 0] / 20.)**2 + 1.) * tan(x[8, 0]
                                     * x[9, 0] / 20.) / 200. + 2. * x[7, 0]
                                     * x[8, 0] / x[9, 0]**3]]))

    # Putting the above together to test the combined constraint case.
    c = lambda x: vstack([g(x), h(x)])
    c_x = lambda x: bmat([[g_x(x)], [h_x(x)]])
    cl = array([0, 0, 0, 0, -1e20, -1e20, -1e20, -1e20, -1e20]).reshape(-1, 1)
    cu = array([0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    c_hessian = lambda x, v: hessian(x, v[:gn], v[gn:])

    # Numerical values to test with
    x0 = arange(1, xn + 1).reshape(-1, 1)
    vg0 = ones((gn, 1))
    vh0 = ones((hn, 1))

    # Testing the test functions (making sure they output the correct shapes)
    assert f(x0).shape == (1, 1)
    assert g(x0).shape == (gn, 1)
    assert h(x0).shape == (hn, 1)
    assert f_x(x0).shape == (xn, 1)
    assert g_x(x0).shape == (gn, xn)
    assert h_x(x0).shape == (hn, xn)
    assert hessian(x0, vg0, vh0).shape == (xn, xn)
    assert (hessian(x0, vg0, vh0) - hessian(x0, vg0, vh0).transpose()).nnz == 0
    assert c(x0).shape == (gn + hn, 1)
    assert cl.shape == cu.shape == (gn + hn, 1)
    assert c_x(x0).shape == (gn + hn, xn)

    # Test separated constraints, with exact derivatives
    prob = problem(n=xn, f=f, g=g, h=h, f_x=f_x, g_x=g_x, h_x=h_x, hessian=hessian)
    assert (prob.f(x0) == f(x0)).all()
    assert (prob.c_e(x0) == g(x0)).all()
    assert (prob.c_i(x0) == h(x0)).all()
    assert (prob.f_x(x0) == f_x(x0)).all()
    assert (prob.A_e(x0) - g_x(x0)).nnz == 0
    assert (prob.A_i(x0) - h_x(x0)).nnz == 0
    assert (prob.hessian(x0, vg0, vh0) - hessian(x0, vg0, vh0)).nnz == 0

    # Test separated constraints, with approximated derivatives
    prob = problem(n=xn, f=f, g=g, h=h)
    assert (prob.f(x0) == f(x0)).all()
    assert (prob.c_e(x0) == g(x0)).all()
    assert (prob.c_i(x0) == h(x0)).all()
    assert (abs(prob.f_x(x0) - f_x(x0)) < 10 * prob._h).all()
    assert (abs(prob.A_e(x0) - g_x(x0)).todense() < 10 * prob._h).all()
    assert (abs(prob.A_i(x0) - h_x(x0)).todense() < 10 * prob._h).all()
    assert prob.hessian is None

    # Test combined constraints, with exact derivatives
    xl = -1e20 * ones((len(x0), 1))
    xu = 1e20 * ones((len(x0), 1))
    prob = problem(n=xn, f=f, f_x=f_x, c=c, cl=cl, cu=cu, c_x=c_x, xl=xl,
                   xu=xu, hessian=c_hessian)
    assert (prob.f(x0) == f(x0)).all()
    assert (prob.c_e(x0) == g(x0)).all()
    assert (prob.c_i(x0) == h(x0)).all()
    assert (prob.f_x(x0) == f_x(x0)).all()
    assert (prob.A_e(x0) - g_x(x0)).nnz == 0
    assert (prob.A_i(x0) - h_x(x0)).nnz == 0
    assert (prob.hessian(x0, vg0, vh0) - c_hessian(x0, vstack([vg0, vh0]))).nnz == 0

    # Testing combined constraints, with approximated derivatives
    prob = problem(n=xn, f=f, c=c, cl=cl, cu=cu)
    assert (prob.f(x0) == f(x0)).all()
    assert (prob.c_e(x0) == g(x0)).all()
    assert (prob.c_i(x0) == h(x0)).all()
    assert (abs(prob.f_x(x0) - f_x(x0)) < 10 * prob._h).all()
    assert (abs(prob.A_e(x0) - g_x(x0)).todense() < 10 * prob._h).all()
    assert (abs(prob.A_i(x0) - h_x(x0)).todense() < 10 * prob._h).all()
    assert prob.hessian is None



