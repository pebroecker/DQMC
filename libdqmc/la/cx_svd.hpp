#ifndef CX_SVD_HPP
#define CX_SVD_HPP

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

// inline bool comparator ( const max_pair& l, const max_pair& r) { return l.first > r.first; }

namespace dqmc {
    namespace la {

	inline perm_mat cx_find_col_perm(cx_mat&__restrict in,
					 cx_mat&__restrict out) {
	    using namespace std;
	    using namespace Eigen;
	    mat_t::Index min_col;
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.rows(); ++i) {
		in.col(i).cwiseAbs().minCoeff(&min_col);
		cout << min_col << endl;
		col_ids[i] = int(min_col) - 1;
	    }
	    // for (std::vector<max_pair>::iterator it=min_indices.begin();
	    // 	 it != min_indices.end(); ++it) {
	    // 	col_ids((*it).second) = std::distance(min_indices.begin(), it);
	    // 	// cout << row_ids((*it).second) <<  " ";
	    // }
	    // // cout << endl;
	    perm_mat col_perm(col_ids);
	    
	    out = in * col_perm.transpose();
	    return col_perm;
	}
	
	inline perm_mat cx_row_sort(cx_mat&__restrict in, cx_mat&__restrict out) {
	    using namespace std;
	    using namespace Eigen;
	    vector<int> max_index(in.rows());
	    vector<max_pair> max_indices;
	    vec max_vals(in.rows());
	    mat A = in.cwiseAbs().real().cast<double>();
	    
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = A.cwiseAbs().real().row(i).maxCoeff( &max_index[i] );
		max_indices.push_back(max_pair(max_vals(i), i));
	    }
	    std::sort(max_indices.begin(), max_indices.end(), comparator);

	    ivec row_ids(in.rows());
	    for (std::vector<max_pair>::iterator it=max_indices.begin(); it != max_indices.end(); ++it) {
		row_ids((*it).second) = std::distance(max_indices.begin(), it);
		// cout << row_ids((*it).second) <<  " ";
	    }
	    // cout << endl;
	    perm_mat row_perm(row_ids);
	    
	    out = row_perm * in;
	    return row_perm;
	}


	inline perm_mat cx_col_sort(cx_mat&__restrict in,
				    cx_mat&__restrict out) {
	    using namespace std;
	    using namespace Eigen;
	    vector<int> max_index(in.cols());
	    vector<max_pair> max_indices;
	    vec max_vals(in.cols());
	    mat A = in.cwiseAbs().real().cast<double>();
	    for(int i = 0; i < in.cols(); ++i) {
		max_vals(i) = A.col(i).maxCoeff( &max_index[i] );
		max_indices.push_back(max_pair(max_vals(i), i));
	    }
	    std::sort(max_indices.begin(), max_indices.end(), comparator);

	    ivec col_ids(in.cols());
	    for (std::vector<max_pair>::iterator it=max_indices.begin();
		 it != max_indices.end(); ++it) {
		col_ids((*it).second) = std::distance(max_indices.begin(), it);
		// cout << row_ids((*it).second) <<  " ";
	    }
	    // cout << endl;
	    perm_mat col_perm(col_ids);
	    
	    out = in * col_perm;
	    return col_perm;
	}


	inline void decompose_svd(cx_mat&__restrict__ in,
				  cx_mat&__restrict__ U, 
				  vec&__restrict__ D,
				  cx_mat&__restrict__ T) {
	    int lwork = -1;
	    int m = in.rows();
	    int n = in.cols();
	    int lda = m;
	    int ldu = m;
	    int ldvt = n;
	    cx_double wkopt;
	    vec_t rwork(5 * m);
	    int info = -1;

	    cx_mat_t U_temp(m, m);
	    vec D_temp(m);
	    vec_t superb(std::min(m, n) - 1);
	    info = LAPACKE_zgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, in.data(), m,
				   D_temp.data(), U_temp.data(), m, T.data(), n, superb.data());
	    U = U_temp.block(0, 0, m, n);
	    D = D_temp.segment(0, n);
	    // lwork = -1;
	    // zgesvd( "All", "All", &m, &n, in.data(), &lda, D.data(), U.data(), &ldu, T.data(),
	    // 	    &ldvt, &wkopt, &lwork,
	    // 	    rwork.data(), &info );
	    // lwork = int(wkopt.real());
	    // cx_vec_t work(lwork);
	    // /* Compute SVD */
	    // zgesvd( "All", "All", &m, &n, in.data(), &lda, D.data(), U.data(), &ldu, T.data(), &ldvt, work.data(), &lwork,
	    // 	    rwork.data(), &info );

	    
	}

	inline void decompose_udt_full_piv(cx_mat&__restrict__ in,
					   cx_mat&__restrict__ U, 
					   vec&__restrict__ D,
					   cx_mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;

	    if (U.rows() != in.rows() || U.cols() != in.cols()) {
		cout << U.rows() << " vs. " << in.rows() << "\tand\t"
		     << U.cols() << " vs. " << in.cols() << endl;
		throw std::runtime_error("dimensions of U are wrong");
	    }
	    
	    if (D.size() != in.cols()) {
		cout << D.size() << endl;
		throw std::runtime_error("dimensions of D are wrong");
	    }
	    
	    if (T.rows() != in.cols() || U.cols() != in.cols()) {
		cout << T.rows() << " vs. " << in.cols() << "\tand\t"
		     << U.cols() << " vs. " << in.cols() << endl;
		throw std::runtime_error("dimensions of T are wrong");
	    }

	    cx_mat_t scaled_in = in;
	    // perm_mat row_perm = dqmc::la::cx_row_sort(in, scaled_in);
	    // perm_mat col_perm = dqmc::la::cx_col_sort(scaled_in, in);
	    
	    Eigen::FullPivHouseholderQR<cx_mat> qr(in);
	     
	    qr.compute(in);
	    // U = row_perm.inverse() * qr.matrixQ().block(0, 0, in.rows(), in.cols());
	    U = qr.matrixQ().block(0, 0, in.rows(), in.cols());
	    
	    cx_mat upper = qr.matrixQR().triangularView<Upper>();// .block(0, 0, in.cols(), in.cols());
	    // cout << "upper" << endl << upper << endl << endl;
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal().real().cast<double>();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    // std::cout << "culprit " << D(i) << " was "
		    // 	      << upper(i, i) << std::endl;
		    dqmc::tools::abort("Invalid upper triangle"); }
		    // throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    // cout << "D " << endl << D.transpose() << endl << endl;
	    // cout << "inv D " << endl << d_inv.transpose() << endl << endl;
	    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
	    	* qr.colsPermutation().transpose(); // * col_perm.inverse();
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}

	inline void decompose_udt_full_sort(cx_mat&__restrict__ in,
					    cx_mat&__restrict__ U,
					    vec&__restrict__ D,
					    cx_mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;

	    if (U.rows() != in.rows() || U.cols() != in.cols())
		throw std::runtime_error("dimensions of U are wrong");
	    
	    if (D.size() != in.cols()) {
		cout << D.size() << endl;
		throw std::runtime_error("dimensions of D are wrong");
	    }
	    
	    if (T.rows() != in.cols() || U.cols() != in.cols())
		throw std::runtime_error("dimensions of T are wrong");

	    cx_mat_t scaled_in;// = cx_mat_t::Zero(in.rows(), in.cols());
	    scaled_in.fill(cx_double(0., 0.));
	    perm_mat row_perm = dqmc::la::cx_row_sort(in, scaled_in);
	    perm_mat col_perm = dqmc::la::cx_col_sort(scaled_in, in);
	    
	    // Very crude scaling...
	    //	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.5);
	    double scale = 1.;

	    cx_vec_t taus(in.cols());
	    ivec iwork(in.cols());
	    
	    // vec work(in.rows() * in.rows() * in.rows());
	    int lwork = -1;
	    // int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    
	    scaled_in.block(0, 0, in.rows(), in.cols()) = in;

	    iwork.setZero();
	    lapack_int info = LAPACKE_zgeqp3( LAPACK_COL_MAJOR, M, N, scaled_in.data(),
					      M, iwork.data(), taus.data() );

	    // cout << iwork.size() << " " << work.size() << " " << taus.size() << endl;
	    // lwork = -1;
	    // zgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
	    // 	    taus.data(), taus.data(),
	    // 	    &lwork, &info);
	    // lwork = taus(0);
	    // cx_vec_t work(lwork);
	    // work.setZero();
	    // taus.setZero();
	    
	    // zgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
	    // 	    taus.data(), work.data(),
	    // 	    &lwork, &info);

	    cx_mat_t upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal().real().cast<double>();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    // std::cout << "culprit " << D(i) << " was "
		    // 	      << upper(i, i) << std::endl;
		    dqmc::tools::abort("Invalid upper triangle"); }
		    // throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;

	    info = LAPACKE_zungqr( LAPACK_COL_MAJOR, M, N, N,
				   scaled_in.data(), M, taus.data() );

	    // zungqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
	    // 	    work.data(), &lwork, &info);
	    // // if (work(0) > work.size()) {
	    // // 	cout << "Need more lwork for dorgqr" << endl;
	    // // }
	    // lwork = work(0);
	    // work.resize(lwork);	    
	    // // cout << "in" << endl << scaled_in << endl << endl;
	    // zungqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
	    // 	    work.data(), &lwork, &info);
	    // cout << "out" << endl << scaled_in << endl << endl;
	    
	    U = row_perm.inverse() * scaled_in.block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
	    }
	    
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lapack_col_perm(col_ids);
	    T = (d_inv.cast<cx_double_t>().asDiagonal()
		 * upper.block(0, 0, in.cols(), in.cols()))
		* lapack_col_perm.transpose() * col_perm.transpose();
	    D *= scale;
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}

	inline void decompose_udt_col_piv(cx_mat&__restrict__ in,
					    cx_mat&__restrict__ U,
					    vec&__restrict__ D,
					    cx_mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;
	    if (U.rows() != in.rows() || U.cols() != in.cols() || U.cols() == 0) {
		cout << U.rows() << " " << U.cols() << endl;
		throw std::runtime_error("dimensions of U are wrong");
	    }
	    
	    if (D.size() != in.cols() || D.size() == 0) {
		cout << D.size() << endl;
		throw std::runtime_error("dimensions of D are wrong");
	    }
	    
	    if (T.rows() != in.cols() || T.cols() == 0)
		throw std::runtime_error("dimensions of T are wrong");

	    
	    cx_mat_t scaled_in = cx_mat_t::Zero(in.rows(), in.cols());
	    scaled_in.fill(cx_double(0., 0.));

	    cx_vec_t taus(in.cols());
	    ivec iwork(in.cols());
	    
	    int lwork = -1;
	    int M = in.rows();
	    int N = in.cols();
	    
	    perm_mat row_sort_perm = cx_row_sort(in, scaled_in);
	    // scaled_in.block(0, 0, in.rows(), in.cols()) = in;
	    iwork.setZero();
	    lapack_int info = LAPACKE_zgeqp3( LAPACK_COL_MAJOR, M, N, scaled_in.data(),
					      M, iwork.data(), taus.data() );
	    
	    cx_mat_t upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal().real().cast<double>();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    // std::cout << "culprit " << D(i) << " was "
		    // 	      << upper(i, i) << std::endl;
		    dqmc::tools::abort("Invalid upper triangle");
		    // throw std::runtime_error("Invalid upper triangle"); 
		}	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;

	    info = LAPACKE_zungqr( LAPACK_COL_MAJOR, M, N, N,
				   scaled_in.data(), M, taus.data() );

	    U = (row_sort_perm.inverse() * scaled_in).block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
	    }
	    
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lapack_col_perm(col_ids);
	    T = (d_inv.cast<cx_double_t>().asDiagonal()
		 * upper.block(0, 0, in.cols(), in.cols()))
		* lapack_col_perm.transpose();
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}
	

	inline void decompose_udt(cx_mat&__restrict__ in,
				  cx_mat&__restrict__ U,
				  vec&__restrict__ D,
				  cx_mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;
	    if (U.rows() != in.rows() || U.cols() != in.cols() || U.cols() == 0) {
		cout << U.rows() << " " << U.cols() << endl;
		throw std::runtime_error("dimensions of U are wrong");
	    }
	    
	    if (D.size() != in.cols() || D.size() == 0) {
		cout << D.size() << endl;
		throw std::runtime_error("dimensions of D are wrong");
	    }
	    
	    if (T.rows() != in.cols() || T.cols() == 0)
		throw std::runtime_error("dimensions of T are wrong");

	    
	    cx_mat_t scaled_in = cx_mat_t::Zero(in.rows(), in.cols());
	    scaled_in.fill(cx_double(0., 0.));

	    cx_vec_t taus(in.cols());
	    ivec iwork(in.cols());
	    
	    int lwork = -1;
	    int M = in.rows();
	    int N = in.cols();
	    
	    scaled_in.block(0, 0, in.rows(), in.cols()) = in;
	    iwork.setZero();

	    lapack_int info = LAPACKE_zgeqrf( LAPACK_COL_MAJOR, M, N,
					      scaled_in.data(),
					      M, taus.data() );
	    
	    cx_mat_t upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal().real().cast<double>();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		 
		    dqmc::tools::abort("Invalid upper triangle");
		}   
		    // throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;

	    info = LAPACKE_zungqr( LAPACK_COL_MAJOR, M, N, N,
				   scaled_in.data(), M, taus.data() );

	    U = scaled_in.block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
	    }
	    
	    T = (d_inv.cast<cx_double_t>().asDiagonal()
		 * upper.block(0, 0, in.cols(), in.cols()));
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}
    }	
}
#endif
