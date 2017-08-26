#ifndef CX_DQMC_HOPPING_HPP
#define CX_DQMC_HOPPING_HPP

#include "la.hpp"
#include <exception>
#include <fstream>

namespace cx_dqmc {
    namespace hopping {

	inline void hopping_matrix(dqmc::parameters& p, cx_dqmc::workspace& ws,
				   mat_t& hopping, mat_t& hopping_inv,
				   cx_mat_t& cx_hopping, cx_mat_t& cx_hopping_inv) {
	    using namespace std;
	    alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    hopping.setZero();
	    hopping_inv.setZero();

	    cx_hopping.setZero();
	    cx_hopping_inv.setZero();

	    for (boost::tie(itr1, itr1_end) = p.graph.bonds();
		 itr1 != itr1_end; ++itr1) {	   
		//itr1 = itr1_end; --itr1;
		int s = p.graph.source(*itr1);
		int t = p.graph.target(*itr1);
		hopping(s, t) = p.ts[p.graph.bond_type(*itr1)] * p.delta_tau ;
		hopping(t, s) = p.ts[p.graph.bond_type(*itr1)] * p.delta_tau ;

		if (p.complex_hoppings == true) {
		    cx_hopping(s, t) =
			std::complex<double>(p.ts[p.graph.bond_type(*itr1)] * p.delta_tau,
					     p.im_ts[p.graph.bond_type(*itr1)] * p.delta_tau);
		    cx_hopping(t, s) =
			std::complex<double>(p.ts[p.graph.bond_type(*itr1)] * p.delta_tau,
					     -p.im_ts[p.graph.bond_type(*itr1)] * p.delta_tau);
		}
	    }		    
	    
	    // cout << "hopping matrix" << endl << endl << hopping << endl;;
	    ofstream mat_file("hopping_matrix.mat", ios::out | ios::trunc);
	    mat_file << hopping;
	    mat_file.close();
	    
	    pvec_t eigval(p.N);
	    pvec_t eigval_inv(p.N);
	    pmat_t eigvec = hopping;

	    Eigen::SelfAdjointEigenSolver<mat> eig(hopping);
	    eig.compute(hopping);
	    eigval = eig.eigenvalues();
	    eigvec = eig.eigenvectors();
	    
	    for (int i = 0; i < p.N; i++) {
		eigval_inv(i) = exp(-eigval(i));		
		eigval(i) = exp(eigval(i));		
	    }
	    
	    mat e = mat::Zero(p.N, p.N);
	    mat e_inv = mat::Zero(p.N, p.N);
	    for (int i = 0; i < p.N; ++i) {
		e(i, i) = eigval(i);
		e_inv(i, i) = eigval_inv(i);
	    }
	    
	    hopping = eigvec * e * eigvec.transpose();
	    hopping_inv = eigvec * e_inv * eigvec.transpose();

	    // COMPLEX
	    ofstream cx_mat_file("cx_hopping_matrix.mat", ios::out | ios::trunc);
	    cx_mat_file << cx_hopping;
	    cx_mat_file.close();

	    vec_t cx_eigval(p.N);
	    vec_t cx_eigval_inv(p.N);
	    cx_mat_t cx_eigvec = cx_hopping;

	    Eigen::SelfAdjointEigenSolver<cx_mat_t> cx_eig(cx_hopping);
	    cx_eig.compute(cx_hopping);
	    cx_eigval = cx_eig.eigenvalues();
	    cx_eigvec = cx_eig.eigenvectors();
	    
	    for (int i = 0; i < p.N; i++) {
		cx_eigval_inv(i) = exp(-cx_eigval(i));		
		cx_eigval(i) = exp(cx_eigval(i));		
	    }
	    
	    cx_mat_t cx_e = cx_mat_t::Zero(p.N, p.N);
	    cx_mat_t cx_e_inv = cx_mat_t::Zero(p.N, p.N);
	    for (int i = 0; i < p.N; ++i) {
		cx_e(i, i) = cx_eigval(i);
		cx_e_inv(i, i) = cx_eigval_inv(i);
	    }
	    
	    cx_hopping = cx_eigvec * cx_e * cx_eigvec.adjoint();
	    cx_hopping_inv = cx_eigvec * cx_e_inv * cx_eigvec.adjoint();

	}

	inline void hopping_matrix_renyi(dqmc::parameters& p, cx_dqmc::workspace& ws,
					 mat& hopping, mat& hopping_inv,
					 int replica) {
	    using namespace std;
	    
	    alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    hopping.setZero();
	    hopping_inv.setZero();
	    
	    int b, s1, s2, s1a, s2a;
	    
	    for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end; ++itr1) {	   
		//itr1 = itr1_end; --itr1;
		b = p.graph.index(*itr1);
		s1 = p.graph.source(*itr1);
		s2 = p.graph.target(*itr1);
		s1a = s1;
		s2a = s2;
		
		if (replica == 1) {
		    if (s1 >= p.n_A) {
			s1a = s1 + p.n_B;
		    }
		    if (s2 >= p.n_A) {
			s2a = s2 + p.n_B;
		    }
		}

		hopping(s1a, s2a) = p.ts[p.graph.bond_type(*itr1)] * p.delta_tau ;
		hopping(s2a, s1a) = p.ts[p.graph.bond_type(*itr1)] * p.delta_tau ;

		if (replica == 0 && s1 >= p.n_A && s2 >= p.n_A) {
		    hopping(s1 + p.n_B, s1 + p.n_B)
			= 0;
		    hopping(s2 + p.n_B, s2 + p.n_B)
			= 0;
		}
		else if (replica == 1 && s1 >= p.n_A && s2 >= p.n_A) {
		    hopping(s1, s1)
			= 0;
		    hopping(s2, s2)
			= 0.;
		}		    
	    }		    
	    
	    pvec_t eigval(p.N + p.n_B);
	    pvec_t eigval_inv(p.N + p.n_B);
	    pmat_t eigvec = hopping;
	    
	    Eigen::SelfAdjointEigenSolver<mat> eig(hopping);
	    eig.compute(hopping);
	    eigval = eig.eigenvalues();
	    eigvec = eig.eigenvectors();

	    for (int i = 0; i < p.N + p.n_B; i++) {
		eigval(i) = exp(eigval(i));
		eigval_inv(i) = 1/eigval(i);
	    }
	    sp_mat e(p.N + p.n_B, p.N + p.n_B);
	    sp_mat e_inv(p.N + p.n_B, p.N + p.n_B);
	    for (int i = 0; i < p.N + p.n_B; ++i) {
		e.insert(i, i) = eigval(i);
		e_inv.insert(i, i) = eigval_inv(i);
	    }
	    
	    hopping = eigvec * e * eigvec.transpose();
	    hopping_inv = eigvec * e_inv * eigvec.transpose();

	}
    }
}
#endif
