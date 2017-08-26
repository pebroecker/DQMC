#ifndef CX_SOLVES_HPP
#define CX_SOLVES_HPP

#include "print.hpp"
//#include "cx_modify.hpp"
#include "sub.hpp"
#include "../tools.hpp"
// #include "cx_multiply.hpp"
// #include "cx_special.hpp"

#include <mkl.h>

#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include <vector>
#include <utility>
#include <complex>


namespace dqmc {
    namespace la {
	inline void solve_qr_col_piv(cx_mat&__restrict__ A,
				     const cx_mat&__restrict__ B,
				     cx_mat&__restrict__ X) {
	    using namespace std;
	    using namespace Eigen;

	    cx_vec_t taus(A.cols());
	    ivec iwork(A.cols()); iwork.setZero();
	    
	    int lwork = -1;
	    int M = A.rows();
	    int N = A.cols();
	    
	    perm_mat row_sort_perm = cx_row_sort(A, X);
	    lapack_int info = LAPACKE_zgeqp3( LAPACK_COL_MAJOR, M, N, X.data(),
					      M, iwork.data(), taus.data() );
	    ivec col_ids(A.cols());
	    for (int i = 0; i < A.cols(); ++i) { col_ids(i) = iwork(i) - 1; }	    
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lapack_col_perm(col_ids);

	    cx_mat_t upper = X.triangularView<Upper>();
	    lwork = -1;

	    info = LAPACKE_zungqr( LAPACK_COL_MAJOR, M, N, N,
				   X.data(), M, taus.data() );
	    // cout << "Reconstructed matrix " << endl
	    // 	 << row_sort_perm.transpose() * X * upper * lapack_col_perm.transpose() << endl << endl;
	    // solving
	    A = X.adjoint() * row_sort_perm * B;
	    X = lapack_col_perm * upper.triangularView<Upper>().solve(A);
	}
    }
}

#endif
