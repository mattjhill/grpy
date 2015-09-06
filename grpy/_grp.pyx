import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "../include/solver.cpp":
    cdef cppclass Solver:
        Solver(const int N, const int m, double* alpha_in, double* beta_in, double* t_in, double* d)
        void assemble_Matrix()
        void factor_Matrix()
        void solve(double* rhs_in, double* sol_in)
        double get_logdet()

cdef class GRPSolver:
    """
    A solver using Sivaram Amambikasaran's 
    `GRP library
    <https://github.com/sivaramambikasaran/ESS>`_, which implements an
    O(n) algorithm for solving systems and calculating determinants of sum of 
    exponentional covariance matrices (e.g. CARMA models).

    Parameters
    ----------
    alpha : ndarray
        the amplitudes of exponentionals in the covariance matrices

    beta : ndarray
        the factor in the arguments of the exponentionals
    """
    cdef Solver* solver
    cdef unsigned int N
    cdef unsigned int m
    cdef double* alpha
    cdef double* beta

    def __cinit__(self, np.ndarray[DTYPE_t, ndim=1] alpha, np.ndarray[DTYPE_t, ndim=1] beta):
        self.m = beta.shape[0]
        self.alpha = <double*> alpha.data
        self.beta = <double*> beta.data

    def __dealloc__(self):
        del self.solver

    def compute(self, np.ndarray[DTYPE_t, ndim=1] t, np.ndarray[DTYPE_t, ndim=1] yerr):
        """
        compute(t, yerr)

        Assemble and factor the covariance matrix.

        Parameters
        ----------
        t : ndarray
            the times at which to evaluate the covariance matrix

        d : ndarray
            the values of diagonals of covariance matrix
        """
        self.N = t.shape[0]
        self.solver = new Solver(self.N, self.m, self.alpha, self.beta, <double*> t.data, <double*> yerr.data)
        self.solver.assemble_Matrix()
        self.solver.factor_Matrix()

    def solve(self, np.ndarray[DTYPE_t, ndim=1] y):
        """
        solve(y)

        Solve the equation A x = rhs.

        Parameters
        ----------
        y : ndarray
            the right hand side of equation

        Returns
        -------
        out : ndarray
            the solution
        """

        cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty_like(y, dtype=DTYPE)
        self.solver.solve(<double*>y.data, <double*>out.data)
        
        return out

    property log_determinant:
        """
        The log of the absolute value of the determinant
        """
        def __get__(self):
            return self.solver.get_logdet()

