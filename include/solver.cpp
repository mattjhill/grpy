// _solvers.h
// Created by Matthew Hill on September 5, 2015

#include "GRP.hpp"
#include <iostream>

class Solver {
	int N;		//	Number of unknowns.
	int m;		//	Rank of the separable part.
	//	The semi-separable matrix is of the form diag(d) + triu(U*V,1) + tril((U*V)',-1).
	Eigen::VectorXd alpha;
	Eigen::VectorXd beta;
	Eigen::VectorXd t;
	Eigen::MatrixXd gamma;
	Eigen::VectorXd d;	//	Diagonal entries of the matrix.
	double logdet;	//	Determinant of the extended sparse matrix.
	
private:
	GRP* grp_;

public:
	Solver(const int N, const int m, double* alpha_in, double* beta_in, double* t_in, double* yerr) {

		
		Eigen::Map<Eigen::VectorXd> alpha(alpha_in, m);
		Eigen::Map<Eigen::VectorXd> beta(beta_in, m);
		Eigen::Map<Eigen::VectorXd> t(t_in, N);
		Eigen::Map<Eigen::VectorXd> d(yerr, N);
		for (int i=0; i<N; i++) {
			d(i) = alpha.sum() + yerr[i]*yerr[i];
		}
		this->N = N;
		this->m = m;
		grp_ = new GRP(N, m, alpha, beta, t, d);

	}; 
	~Solver(){
		delete grp_;
	};                     //	Constructor gets all the desired quantities.
	void assemble_Matrix() {
		grp_->assemble_Extended_Matrix();
	};                        //	Assembles the extended sparse matrix.
	void factor_Matrix() {
		grp_->factorize_Extended_Matrix();
	};                       //	Factorizes the extended sparse matrix.
	void solve(double* rhs_in, double* sol_in) {
		Eigen::Map<Eigen::VectorXd> rhs(rhs_in, N);
		Eigen::VectorXd solution;
		Eigen::VectorXd solutionex;
		grp_->obtain_Solution(rhs, solution, solutionex);
		for (int i=0; i<N; i++) {
			sol_in[i] = solution(i);
		}
	};	//	Obtains the solution.
	double get_logdet() {
		logdet = grp_->obtain_Determinant();
		return logdet;
	};	//	Obtains the determinant of the extended sparse matrix.
};

