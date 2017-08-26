#ifndef SVD_HPP

#define SVD_HPP

#include "print.hpp"
#include "modify.hpp"
#include "sub.hpp"
#include "multiply.hpp"
#include "special.hpp"
#include "../workspace.hpp"
#include "../tools.hpp"
#include <mkl.h>

#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include <vector>
#include <utility>

inline bool comparator ( const max_pair& l, const max_pair& r) { return l.first > r.first; }

namespace dqmc {
    namespace la {

	// extern "C" {
	//     void dgejsv_( char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
	// 		  char* jobp,
	// 		  const int* m, const int* n, double* a, const int* lda,
	// 		  double* sva, double* u, const int* ldu,
	// 		  double* v, const int* ldv, double* work,
	// 		  const int* lwork, int* iwork, int* info );

	//     int LAPACKE_dgejsv( int matrix_layout, char joba, char jobu, char jobv, 
	//     			char jobr, char jobt, char jobp, int m, 
	//     			int n, const double* a, 
	//     			int lda, double* sva, double* u,
	//     			int ldu, double* v, int ldv,
	//     			double* stat, int* istat );

	//     void dgesvj_( const char* joba, const char* jobu, const char* jobv,
	// 		  const int* m, const int* n, double* a,
	// 		  const int* lda, double* sva, const int* mv, double* v,
	// 		  const int* ldv, double* work, int* lwork,
	// 		  int* info );
	// }
	
	template <typename M, typename V>
	inline void close_svd(int sites, M&__restrict__ U, V&__restrict__ D,
			      M&__restrict__ T, 
			      M&__restrict__ out) {
	    using namespace std;
	    
	    for (int col = sites - 1; col >= 0; col--) {
		for (int row = sites - 1; row >= 0; row--) {
		    out(row, col) = 0.;
		    for (int k = sites - 1; k >= 0; k--) {
			if (U(row, k) != 0 && T(k, col) != 0 && D(k) != 0) 
			    out(row, col) += ( U(row, k) * T(k, col)) * D(k);
		    }
		}
	    }
	}

	template <typename M, typename V>
	inline void close_svd_inv_diag(int sites, M&__restrict__ U, V&__restrict__ D,
				       M&__restrict__ T,
				       M& out) {
	    using namespace std;
	    
	    for (int col = sites - 1; col >= 0; col--) {
		for (int row = sites - 1; row >= 0; row--) {
		    out(row, col) = 0.;
		    for (int k = sites - 1; k >= 0; k--) {
			if (U(row, k) != 0 && T(k, col) != 0 && D(k) != 0) 
			    out(row, col) += ( U(row, k) * T(k, col)) / D(k);
		    }
		}
	    }
	}

	inline int decompose_dgejsv_nt(pmat_t&__restrict__ in, pmat_t&__restrict__ U,
				       pvec_t&__restrict__ D, pmat_t&__restrict__ Tt,
				       pvec_t& work, ivec& ivywork) {
	    using namespace std;

	    int info;
	    int lwork = work.size() - 10;

	    char f = 'F';
	    char j = 'J';
	    char n = 'N';
	    char p = 'P';
	    int rows = U.rows();
	    int cols = U.cols();
	    int T_rows = Tt.rows();
	    // arma::vec work(6);
	    ivec iwork(in.rows() + 3 * in.cols());
	    work.setZero();
	    iwork.setZero();

	    dgejsv_( &f, &f, &j, &n, &n, &p, 
	    	     &rows, &cols, in.data(), &rows,
	    	     D.data(), U.data(), &rows,
	    	     Tt.data(), &T_rows, work.data(),
	    	     &lwork, iwork.data(), &info);
	    return info;
	    // info = LAPACKE_dgejsv_work( 102, f, f, j, n, n, p, 
	    // 				rows, cols, in.data(), rows,
	    // 				D.data(), U.data(), rows,
	    // 				Tt.data(), T_rows, work.data(),
	    // 				work.size(),
	    // 				iwork.data());

	    // return info;
	}


	inline int decompose_dgejsv_col_nt(pmat_t&__restrict__ in, pmat_t&__restrict__ U,
				       pvec_t&__restrict__ D, pmat_t&__restrict__ Tt,
				       pvec_t& work, ivec& iwork) {
	    using namespace std;

	    int info;
	    int lwork = work.size() - 10; //std::max(1024, int(in.rows()*in.rows()*in.rows()));
	    // pvec_t work(lwork);
	    // ivec iwork(std::max(1024, int(in.rows() * in.cols())));

	    char c = 'C';
	    char u = 'U';
	    char j = 'J';
	    char n = 'N';
	    char p = 'P';
	    int rows = U.rows();
	    int cols = U.cols();
	    int T_rows = Tt.rows();

	    work.setZero();
	    iwork.setZero();
	    // info = LAPACKE_dgejsv_work( 102, c, j, j, n, n, p, 
	    // 			   rows, cols, in.data(), rows,
	    // 			   D.data(), U.data(), rows,
	    // 			   Tt.data(), T_rows, work.data(),
	    // 				work.size(),
	    // 			   iwork.data());

	    dgejsv_( &c, &u, &j, &n, &n, &p, // 6
	    	     &rows, &cols, in.data(), &rows, //10
	    	     D.data(), U.data(), &rows, // 13
	    	     Tt.data(), &T_rows, work.data(),
	    	     &lwork, iwork.data(), &info); // 17
	    // std::cout << info << std::endl;
	    return info;
	}

       
	inline int decompose_jsvd_nt(pmat_t&__restrict__ in, pmat_t&__restrict__ U,
				     pvec_t&__restrict__ D, pmat_t&__restrict__ Tt) {
	    using namespace std;

	    char joba = 'G';
	    char jobu = 'U';
	    char jobv = 'V';

	    U = in;

	    int rows = in.rows();
	    int cols = in.cols();
	    int U_rows = in.rows();
	    int T_rows = Tt.rows();

	    int lwork = max(1024, rows * cols);
	    pvec_t work(lwork);
	    work.setOnes();
	    int mv = 0;
	    int info;
	    
	    dgesvj_(&joba, &jobu, &jobv, &rows, &cols, U.data(), &rows,
		    D.data(), &mv, Tt.data(), &T_rows, work.data(), &lwork, &info);
			  
	    if (work(0) != 1.) {
		cout << "MKL ERROR: SVD HAVE TO BE SCALED! " << work(0) << endl;
		throw std::runtime_error("SVD Error");
	    }
	    
	    return info;
	}


	inline int decompose_jsvd_nt(pmat_t&__restrict__ in, pmat_t&__restrict__ U,
				     pvec_t&__restrict__ D, pmat_t&__restrict__ Tt,
				     pvec_t&__restrict__ work) {
	    using namespace std;

	    char joba = 'G';
	    char jobu = 'U';
	    char jobv = 'V';

	    U = in;

	    int rows = in.rows();
	    int cols = in.cols();
	    int U_rows = in.rows();
	    int T_rows = Tt.rows();

	    int lwork = work.size();
	    //pvec_t work(lwork);
	    work.setOnes();
	    int mv = 0;
	    int info;
	    
	    dgesvj_(&joba, &jobu, &jobv, &rows, &cols, U.data(), &rows,
		    D.data(), &mv, Tt.data(), &T_rows, work.data(), &lwork, &info);
			  
	    if (work(0) != 1.) {
		cout << "MKL ERROR: SVD HAVE TO BE SCALED! " << work(0) << endl;
		throw std::runtime_error("SVD Error");
	    }
	    
	    return info;
	}

	inline perm_mat row_sort(mat&__restrict in, mat&__restrict out) {
	    using namespace std;
	    using namespace Eigen;
	    vector<int> max_index(in.rows());
	    vector<max_pair> max_indices;
	    vec max_vals(in.rows());
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = in.cwiseAbs().row(i).maxCoeff( &max_index[i] );
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


	inline perm_mat col_sort(mat&__restrict in, mat&__restrict out) {
	    using namespace std;
	    using namespace Eigen;
	    vector<int> max_index(in.cols());
	    vector<max_pair> max_indices;
	    vec max_vals(in.cols());
	    for(int i = 0; i < in.cols(); ++i) {
		max_vals(i) = in.cwiseAbs().col(i).maxCoeff( &max_index[i] );
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


	inline void decompose_udt_full_piv(mat&__restrict__ in, mat&__restrict__ U, vec&__restrict__ D, mat&__restrict__ T) {
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

	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.5);
	    Eigen::FullPivHouseholderQR<mat> qr(in);
	    in /= scale;
	     
	    qr.compute(in);
	    U = qr.matrixQ().block(0, 0, in.rows(), in.cols());
	    mat upper = qr.matrixQR().triangularView<Upper>();// .block(0, 0, in.cols(), in.cols());
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    std::cout << "culprit " << D(i) << " was "
			      << upper(i, i) << std::endl;
		    throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) {
		    D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    
	    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
		* qr.colsPermutation().transpose();
	    D *= scale;
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}



	inline void decompose_udt_full_sort(mat&__restrict__ in, mat&__restrict__ U, vec&__restrict__ D, mat&__restrict__ T) {
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

	    mat scaled_in = mat::Zero(in.rows(), in.cols());
	    perm_mat row_perm = dqmc::la::row_sort(in, scaled_in);
	    perm_mat col_perm = dqmc::la::col_sort(scaled_in, in);
	    
	    // Very crude scaling...
	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.5);

	    vec taus(in.cols());
	    ivec iwork(in.cols());
	    
	    // vec work(in.rows() * in.rows() * in.rows());
	    int lwork = -1;
	    int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    
	    scaled_in.block(0, 0, in.rows(), in.cols()) = in/scale;
	    iwork.setZero();
	    // cout << iwork.size() << " " << work.size() << " " << taus.size() << endl;
	    lwork = -1;
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), taus.data(),
		    &lwork, &info);
	    lwork = taus(0);
	    vec work(lwork);
	    work.setZero();
	    taus.setZero();
	    
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), work.data(),
		    &lwork, &info);

	    mat upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    std::cout << "culprit " << D(i) << " was "
			      << upper(i, i) << std::endl;
		    throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;
	    dorgqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
		    work.data(), &lwork, &info);
	    // if (work(0) > work.size()) {
	    // 	cout << "Need more lwork for dorgqr" << endl;
	    // }
	    lwork = work(0);
	    work.resize(lwork);	    
	    // cout << "in" << endl << scaled_in << endl << endl;
	    dorgqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
		    work.data(), &lwork, &info);
	    // cout << "out" << endl << scaled_in << endl << endl;
	    U = row_perm.inverse() * scaled_in.block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
	    }
	    
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lapack_col_perm(col_ids);
	    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
		* lapack_col_perm.transpose() * col_perm.transpose();
	    D *= scale;
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}
	
	inline void decompose_udt_row_sort(mat&__restrict__ in, mat&__restrict__ U, vec&__restrict__ D, mat&__restrict__ T) {
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

	    
	    vector<int> max_index(in.rows());
	    vector<max_pair> max_indices;
	    vec max_vals(in.rows());
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = in.cwiseAbs().row(i).maxCoeff( &max_index[i] );
		max_indices.push_back(max_pair(max_vals(i), i));
	    }
	    std::sort(max_indices.begin(), max_indices.end(), comparator);

	    ivec row_ids(in.rows());
	    for (std::vector<max_pair>::iterator it=max_indices.begin(); it != max_indices.end(); ++it) {
		row_ids((*it).second) = std::distance(max_indices.begin(), it);
		// cout << row_ids((*it).second) <<  " ";
	    }
	    // cout << endl;
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> row_perm(row_ids);

	    // Very crude scaling...
	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.5);

	    vec taus(in.cols());
	    ivec iwork(in.cols());
	    
	    // vec work(in.rows() * in.rows() * in.rows());
	    int lwork = -1;
	    int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    mat scaled_in = mat::Zero(in.rows(), in.cols());
	    
	    scaled_in.block(0, 0, in.rows(), in.cols()) = (row_perm * in)/scale;
	    iwork.setZero();
	    // cout << iwork.size() << " " << work.size() << " " << taus.size() << endl;
	    lwork = -1;
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), taus.data(),
		    &lwork, &info);
	    lwork = taus(0);
	    vec work(lwork);
	    work.setZero();
	    taus.setZero();
	    
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), work.data(),
		    &lwork, &info);

	    mat upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    std::cout << "culprit " << D(i) << " was "
			      << upper(i, i) << std::endl;
		    throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;
	    dorgqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
		    work.data(), &lwork, &info);
	    // if (work(0) > work.size()) {
	    // 	cout << "Need more lwork for dorgqr" << endl;
	    // }
	    lwork = work(0);
	    work.resize(lwork);	    
	    // cout << "in" << endl << scaled_in << endl << endl;
	    dorgqr_(&M, &N, &N, scaled_in.data(), &M, taus.data(),
		    work.data(), &lwork, &info);
	    // cout << "out" << endl << scaled_in << endl << endl;
	    U = row_perm.inverse() * scaled_in.block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
	    }
	    
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> col_perm(col_ids);
	    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
		* col_perm.transpose();
	    D *= scale;
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}

	inline void decompose_udt_row_sort_sqr_q(mat&__restrict__ in, mat&__restrict__ U, vec&__restrict__ D, mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;
	    
	    vector<int> max_index(in.rows());
	    vector<max_pair> max_indices;
	    vec max_vals(in.rows());
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = in.cwiseAbs().row(i).maxCoeff( &max_index[i] );
		max_indices.push_back(max_pair(max_vals(i), i));
	    }
	    std::sort(max_indices.begin(), max_indices.end(), comparator);

	    ivec row_ids(in.rows());
	    for (std::vector<max_pair>::iterator it=max_indices.begin(); it != max_indices.end(); ++it) {
		row_ids((*it).second) = std::distance(max_indices.begin(), it);
		// cout << row_ids((*it).second) <<  " ";
	    }
	    // cout << endl;
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> row_perm(row_ids);

	    // Very crude scaling...
	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.3);

	    vec taus(in.rows());
	    // vec work(in.rows() * in.rows() * in.rows());
	    int lwork = -1;
	    ivec iwork(in.rows());
	    int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    mat scaled_in = mat::Zero(in.rows(), in.rows());
	    
	    scaled_in.block(0, 0, in.rows(), in.cols()) = (row_perm * in)/scale;
	    iwork.setZero();
	    // cout << iwork.size() << " " << work.size() << " " << taus.size() << endl;
	    lwork = -1;
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), taus.data(),
		    &lwork, &info);
	    lwork = taus(0);
	    vec work(lwork);
	    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
		    taus.data(), work.data(),
		    &lwork, &info);

	    {
		{		
		    mat upper = scaled_in.triangularView<Upper>();
		    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
		    vec d_inv = D;
	    
		    for (int i = 0; i < in.cols(); ++i) {
			if (D(i) == 0.) {		    
			    std::cout << "culprit " << D(i) << " was "
				      << upper(i, i) << std::endl;
			    throw std::runtime_error("Invalid upper triangle"); }	
			d_inv(i) = 1./D(i);
			if (D(i) < 0) { D(i) *= -1.;
			    d_inv(i) *= -1.; }
		    }	    	    
		    dorgqr_(&M, &M, &N, scaled_in.data(), &M, taus.data(),
			    work.data(), &lwork, &info);
		    U = row_perm.inverse() * scaled_in.block(0, 0, in.rows(), in.cols());  
		    ivec col_ids(in.cols());
		    for (int i = 0; i < in.cols(); ++i) {
			col_ids(i) = iwork(i) - 1;
		    }
	    
		    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> col_perm(col_ids);
		    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
			* col_perm.transpose();
		    D *= scale;
		    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
		}
	    }	    
	}

#ifdef USE_DD
	inline void decompose_udt_row_sort_dd(mat&__restrict__ in, mat&__restrict__ U, vec&__restrict__ D, mat&__restrict__ T) {
	    using namespace std;
	    using namespace Eigen;
	    
	    vector<int> max_index(in.rows());
	    vector<max_pair> max_indices;
	    vec max_vals(in.rows());
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = in.cwiseAbs().row(i).maxCoeff( &max_index[i] );
		max_indices.push_back(max_pair(max_vals(i), i));
	    }
	    std::sort(max_indices.begin(), max_indices.end(), comparator);

	    ivec row_ids(in.rows());
	    for (std::vector<max_pair>::iterator it=max_indices.begin(); it != max_indices.end(); ++it) {
		row_ids((*it).second) = std::distance(max_indices.begin(), it);
		// cout << row_ids((*it).second) <<  " ";
	    }
	    // cout << endl;
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> row_perm(row_ids);

	    // Very crude scaling...
	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.5);

	    vec taus(in.rows());
	    // vec work(in.rows() * in.rows() * in.rows());
	    int lwork = -1;
	    ivec iwork(in.rows());
	    int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    mat temp = mat::Zero(in.rows(), in.cols());
	    dd_mat dd_temp = dd_mat::Zero(in.cols(), in.cols());
	    dd_mat scaled_in = dd_mat::Zero(in.rows(), in.cols());
	    temp = ((row_perm * in)/scale);
	    scaled_in.block(0, 0, in.rows(), in.cols()) = temp.cast<ddouble>();
	    Eigen::FullPivHouseholderQR<dd_mat> qr(scaled_in);
	    // cout << scaled_in << endl << endl;
	    qr.compute(scaled_in);
	    dd_temp = qr.matrixQR().block(0, 0, in.cols(), in.cols()).triangularView<Upper>();
	    mat upper = mat::Zero(in.cols(), in.cols());	    
	    // cout << "from dd_temp" << endl << endl << upper << endl << endl;
	    // cout << dd_temp.rows() << " x " << dd_temp.cols() << " - " <<
	    // 	upper.rows() << " x " << upper.cols() << endl;
	    dqmc::la::copy_from_dd(dd_temp, upper);
	    // cout << "from dd_temp" << endl << endl << upper << endl << endl;
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    std::cout << "culprit " << D(i) << " was "
			      << upper(i, i) << std::endl;
		    throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }	    	    
	    //dd_mat qrQ  = qr.householderQ();
	    dd_mat qrQ  = qr.matrixQ();
	    dd_temp = qrQ.block(0, 0, in.rows(), in.cols());

	    dqmc::la::copy_from_dd(dd_temp, temp);
	    // temp = dd_temp.cast<double>();
	    U = row_perm.inverse() * temp;
	    
	    T = (d_inv.asDiagonal() * upper.block(0, 0, in.cols(), in.cols()))
		* qr.colsPermutation().transpose();
	    D *= scale;
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}
#endif
	
	inline void decompose_udt_row_sort(mat &__restrict__ in, mat &__restrict__ U, vec &__restrict__ D, mat &__restrict__ T,
					   mat &__restrict__ scaled_in, mat &__restrict__ T_temp,					   
					   ivec &__restrict__ row_ids, ivec &__restrict__ col_ids,
					   ivec &__restrict__ max_index, vec &__restrict__ max_vals, std::vector<max_pair>& max_pairs,  
					   vec &__restrict__ taus, vec &__restrict__ work_old, ivec &__restrict__ iwork_old ) {
	    using namespace std;
	    using namespace Eigen;

	    max_pairs.clear();	    
	    for(int i = 0; i < in.rows(); ++i) {
		max_vals(i) = in.cwiseAbs().row(i).maxCoeff( &max_index(i) );
		max_pairs.push_back(max_pair(max_vals(i), i));
		// cout << i << endl;
	    }

	    // for (std::vector<max_pair>::iterator it=max_pairs.begin(); it != max_pairs.end(); ++it) {
	    // 	row_ids((*it).second) = std::distance(max_pairs.begin(), it);
	    // 	cout << (*it).second << " | " << std::distance(max_pairs.begin(), it) << "   ";
	    // }
	    // cout << endl;

	    std::sort(max_pairs.begin(), max_pairs.end(), comparator);
	    // cout << "Sorted" << endl;
	    // cout << row_ids.size();
	    for (std::vector<max_pair>::iterator it=max_pairs.begin(); it != max_pairs.end(); ++it) {
		row_ids((*it).second) = std::distance(max_pairs.begin(), it);
		// cout << (*it).second << " | " << std::distance(max_pairs.begin(), it) << "   ";
	    }
	    // cout << endl;
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> row_perm(row_ids);
	    // cout << "permutation" << endl;

	    
	    // Very crude scaling...
	    double scale = pow(in.cwiseAbs().maxCoeff(), 0.3);
	    int lwork; // = work.size();
	    // vec work(in.rows() * in.rows() * in.rows());
	    // int lwork = work.size();

	    int info = 0;
	    int M = in.rows();
	    int N = in.cols();
	    ivec iwork(in.cols());

	    // cout << "Scaled" << endl;
	    // cout << row_perm.indices().transpose() << endl << endl;
	    // cout << row_perm.toDenseMatrix() << endl << endl;
	    // cout << in << endl << endl;
	    {
		{
		    scaled_in = (row_perm * in)/scale;
		    // cout << scaled_in << endl << endl;
		    iwork.setZero();
		    // cout << "QRing" << endl;		    
		    // cout << iwork.size() << " " << work.size() << " " << taus.size() << endl;
		    // ivec iwork(4*in.rows());
		    lwork = -1;
		    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
			    taus.data(), taus.data(),
			    &lwork, &info);
		    lwork = taus(0);
		    vec work(lwork);

		    dgeqp3_(&M, &N, scaled_in.data(), &M, iwork.data(),
			    taus.data(), work.data(),
			    &lwork, &info);		    
		}
	    }
	    {
		{
		
		    // cout << "To zero!" << endl;
		    // cout << "Getting diagonal" << endl;
		    D = scaled_in.diagonal();
		    vec d_inv = D;
		    // cout << "Gitgotthat" << endl;
	    
		    for (int i = 0; i < in.cols(); ++i) {
			if (D(i) == 0.) {		    
			    std::cout << "culprit " << D(i) << std::endl;
			    throw std::runtime_error("Invalid upper triangle"); }
		
			d_inv(i) = 1./D(i);
			if (D(i) < 0) { D(i) *= -1.;
			    d_inv(i) *= -1.; }
		    }	    	    
	    
		    for (int i = 0; i < in.cols(); ++i) col_ids(i) = iwork(i) - 1;	    
		    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> col_perm(col_ids);
		    T_temp = scaled_in.triangularView<Upper>();
		    T = (d_inv.asDiagonal() * T_temp) * col_perm.transpose();
		    // cout << T << endl << endl;

		    D *= scale;
		    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
		    lwork = -1;
		    dorgqr_(&M, &M, &N, scaled_in.data(), &M, taus.data(), taus.data(), &lwork, &info);
		    lwork = taus(0);
		    vec work(lwork);
		    dorgqr_(&M, &M, &N, scaled_in.data(), &M, taus.data(), work.data(), &lwork, &info);
		    U = row_perm.inverse() * scaled_in;
		}
	    }
	    return;
	}


	inline void decompose_udt_col_piv(mat&__restrict__ in,
					  mat&__restrict__ U,
					  vec&__restrict__ D,
					  mat&__restrict__ T) {
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

	    
	    mat_t scaled_in = mat_t::Zero(in.rows(), in.cols());

	    vec_t taus(in.cols());
	    ivec iwork(in.cols());
	    
	    int lwork = -1;
	    int M = in.rows();
	    int N = in.cols();
	    
	    perm_mat row_sort_perm = row_sort(in, scaled_in);
	    // scaled_in.block(0, 0, in.rows(), in.cols()) = in;
	    iwork.setZero();
	    lapack_int info = LAPACKE_dgeqp3( LAPACK_COL_MAJOR, M, N, scaled_in.data(),
					      M, iwork.data(), taus.data() );
	    
	    mat_t upper = scaled_in.triangularView<Upper>();
	    D = upper.block(0, 0, in.cols(), in.cols()).diagonal();
	    vec d_inv = D;
	    
	    for (int i = 0; i < in.cols(); ++i) {
		if (D(i) == 0.) {		    
		    // std::cout << "culprit " << D(i) << " was "
		    // 	      << upper(i, i) << std::endl;
		    throw std::runtime_error("Invalid upper triangle"); }	
		d_inv(i) = 1./D(i);
		if (D(i) < 0) { D(i) *= -1.;
		    d_inv(i) *= -1.; }
	    }
	    lwork = -1;

	    info = LAPACKE_dorgqr( LAPACK_COL_MAJOR, M, N, N,
				   scaled_in.data(), M, taus.data() );

	    U = (row_sort_perm.inverse() * scaled_in).block(0, 0, in.rows(), in.cols());  
	    ivec col_ids(in.cols());
	    for (int i = 0; i < in.cols(); ++i) {
		col_ids(i) = iwork(i) - 1;
		if (col_ids(i) == -1) {
		    dqmc::tools::abort("Permutation fails");
		}
	    }
	    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> lapack_col_perm(col_ids);
	    
	    T = (d_inv.asDiagonal()
		 * upper.block(0, 0, in.cols(), in.cols()))
		* lapack_col_perm.transpose();
	    if (D(0) != D(0)) std::cout << "NaN alert" << endl;
	}

    }
}
#endif
