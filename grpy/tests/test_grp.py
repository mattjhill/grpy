from __future__ import division, print_function
import numpy as np

from .. import _grp 

def test_solve(m=5, N=500):
	alpha = np.random.random(m)
	beta = np.random.random(m)
	t = np.random.random(N)
	t.sort()
	yerr = np.random.random(N)
	y = np.random.random(N)

	A = np.diag(yerr**2)
	for i in range(m):
		A += alpha[i]*np.exp(-beta[i]*abs(np.subtract.outer(t,t)))

	print("assembled matrix")
	grp = _grp.GRPSolver(alpha, beta)
	grp.compute(t, yerr)
	assert np.allclose(grp.solve(y), np.linalg.solve(A,y))

def test_determinant(m=5, N=500):
	alpha = np.random.random(m)
	beta = np.random.random(m)
	t = np.random.random(N)
	t.sort()
	yerr = np.random.random(N)

	A = np.diag(yerr**2)
	for i in range(m):
		A += alpha[i]*np.exp(-beta[i]*abs(np.subtract.outer(t,t)))

	print("assembled matrix")
	grp = _grp.GRPSolver(alpha, beta)
	grp.compute(t, yerr)
	assert np.allclose(grp.log_determinant, np.linalg.slogdet(A)[1])