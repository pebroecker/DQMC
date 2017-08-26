#ifndef DETS_HPP
#define DETS_HPP

#include <cmath>

#include <iomanip>

namespace dqmc {
    namespace la {
	inline void inv_log_sum(pvec_t& D, double& det) {
	    det = 0;
	    for (int i = 0; i < D.size(); ++i) { det -= log(D(i)); };
	} 

	inline void log_sum(pvec_t& D, double& det) {
	    det = 0;
	    for (int i = 0; i < D.size(); ++i) { det += log(D(i)); };
	}

	inline void log_det(pmat_t M, double& det,
			    double& sign) {
	    // PrintMatrix(sites, M, 24, 16);
	    ivec ipiv(M.rows());
	    sign = 1.;
	    det = 0.;
	    int info;
	    LAPACKE_dgetrf(LAPACK_COL_MAJOR, M.rows(), M.cols(),
	    		   M.data(), M.rows(), ipiv.data());

	    for (int i = 0; i < M.rows(); i++) {
		if (ipiv[i] != i + 1) {
		    sign *= -1;
		}
		if (M(i, i) < 0) {
		    sign *= -1;
		}
		det += std::log(std::fabs(M(i, i)));
	    }	 
	}


	inline void det_sign(pmat_t M,  double& sign) {
	    sign = 1;
	    Eigen::PartialPivLU<mat> lu(M);
	    lu.compute(M);
	    sign = lu.determinant();

	    if (fabs(fabs(sign) - 1.) > 1e-10) {
		std::cout << std::setprecision(12) << sign << std::endl;
		// throw std::runtime_error("This matrix is not orthogonal");
	    }
	    sign < 0 ? sign = -1. : sign = 1;
	    return;
	    
	    // std::cout << lu.permutationP().indices() << std::endl;
	    for (int i = 0; i < M.rows(); ++i) {
		// std::cout << ipiv[i] << std::endl;
		if (lu.permutationP().indices()[i] != i) {
		    sign *= -1;
		}
		
		if (lu.matrixLU()(i, i) < 0) {
		    sign *= -1;
		}
	    }	 
	    return;
	    
	    ivec ipiv(M.rows());
	    sign = 1.;
	    int info;
	    
	    // arma::dgetrf_(&sites, &sites, &M(0, 0), &sites, &ipiv(0), &info);
	    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, M.rows(), M.cols(),
	    		   M.data(), M.rows(), ipiv.data());
	    //PrintMatrix(sites, M, 16, 8);	    
	    for (int i = 0; i < M.rows(); ++i) {
		// std::cout << ipiv[i] << std::endl;
		if (ipiv[i] != i + 1) {
		    sign *= -1;
		}
		
		if (M(i, i) < 0) {
		    sign *= -1;
		}
	    }	 
	}
    }
}
#endif
