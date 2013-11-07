PyIntropt
=========

(What started out as:)
A Python Interior Point Optimization Solver

Originally, another nonlinear interior-point trust region solver, based on the
NITRO (Nonlinear Interior point Trust Region Optimizer) algorithm developed by
Byrd et al. [Byrd1999].

The robustness of this implementation of the NITRO algorithm against the CUTEr
[Gould2005] test suite, as compared to published results [Byrd1999] did not
seem favourable. As such, the interior point algorithm isn't being worked on
currently.

The NLP active set method described by Betts [Betts2010] is instead being
implemented. It appears more appropriate for optimal control problems, and its
core is a quadratic programming (QP) solver, which is easier to test in
isolation. 


References (Interior Point):
[Byrd1999] R. H. Byrd, M. E. Hribar, and J. Nocedal, An interior point method
for large scale nonlinear programming, SIAM Journal on Optimization, 9 (1999),
pp. 877-900.
[Liu1999] Liu, G. (1999). Design Issues in Algorithms for Large Scale Nonlinear
Programming
[Nocedal2006] J. Nocedal and S. J. Wright. Numerical Optimization, Second
Edition. Springer Series in Operations Research, Springer Verlag, 2006.

References (Active Set SQP):
[Gill1987] Gill, Philip E., et al. A Schur-complement method for sparse
quadratic programming. No. SOL-87-12. Stanford Univ., CA (USA). Systems
Optimization Lab., 1987.
[Gill1988] Gill, Philip E., and Walter Murray. A practical anti-cycling
procedure for linear and nonlinear programming. No. SOL-88-4. Stanford Univ. CA
(USA). Systems Optimization Lab, 1988.
[Betts1994] Betts, J. T., & Frank, P. D. (1994). A sparse nonlinear
optimization algorithm. Journal of Optimization Theory and Applications, 82(3),
519-541.
[Betts2010] Betts, J. T. (2010). Practical Methods for Optimal Control and
Estimation Using Nonlinear Programming. Control. 3600 Market Street, 6th Floor
Philadelphia, PA 19104-2688: Siam.
[Gill2005] Gill, P., Murray, W., & Saunders, M. (2005). SNOPT : An SQP
Algorithm for Large-Scale Constrained Optimization. SIAM review, 47(1), 99–131.
Retrieved from http://epubs.siam.org/doi/abs/10.1137/S0036144504446096

References (General):
[Gould2005] Gould, N. I. M., Orban, D., & Toint, P. L. (2005). General CUTEr
documentation.
