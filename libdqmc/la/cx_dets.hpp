#ifndef CX_DETS_HPP
#define CX_DETS_HPP

#include <cmath>

#include <iomanip>

namespace dqmc {
    namespace la {
	inline void cx_log_det(cx_mat M, double& det,
			       double& sign) {
	    // PrintMatrix(sites, M, 24, 16);
	    ivec ipiv(M.rows());
	    sign = 1.;
	    det = 0.;
	    int info;
	    LAPACKE_zgetrf(LAPACK_COL_MAJOR, M.rows(), M.cols(),
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


	inline void cx_det_sign(cx_mat M,  double& sign) {
	    sign = 1;
	    Eigen::PartialPivLU<cx_mat> lu(M);
	    lu.compute(M);
	    sign = lu.determinant();

	    if (fabs(fabs(sign) - 1.) > 1e-10) {
		std::cout << std::setprecision(12) << sign << std::endl;
	    }
	    sign < 0 ? sign = -1. : sign = 1;
	    return;
	}
    }
}
#endif
