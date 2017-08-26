#ifndef CX_CALCULATE_GREENS_FT_HPP
#define CX_CALCULATE_GREENS_FT_HPP

#include "la.hpp"
#include <vector>

namespace dqmc {
    namespace calculate_greens {

	template<typename workspace>
	inline void basic_udt_col_piv_qr_partial_piv_lu(cx_mat_t& U,
					  vec_t& D,
					  cx_mat_t& T,
					  workspace& ws,
					  cx_mat_t& greens) {
	    using namespace std;

	    ws.mat_1 = T.adjoint().partialPivLu().solve(U);
	    ws.mat_5 = ws.mat_1.adjoint();
	    ws.mat_5.diagonal().real() += D;

	    dqmc::la::decompose_udt_col_piv(ws.mat_5, ws.mat_2, ws.re_vec_2, ws.mat_3);
	
	    ws.mat_4 = ws.mat_3 * T;
	    ws.mat_5 = ws.re_vec_2.asDiagonal().inverse() * ws.mat_2.adjoint() * U.adjoint();

	    greens = ws.mat_4.partialPivLu().solve(ws.mat_5);
	}

	
	template<typename workspace>
	inline void basic_udt_col_piv_qr_col_piv_qr(cx_mat_t& U,
					  vec_t& D,
					  cx_mat_t& T,
					  workspace& ws,
					  cx_mat_t& greens) {
	    using namespace std;
	    // perm_mat col_perm = dqmc::la::cx_find_col_perm(T, ws.mat_2);
	    // cout << "ws mat 2" << endl << ws.mat_2 << endl << endl;
	    // cx_mat_t& temp = ws.mat_2;
	    // ws.mat_1 = temp.adjoint().triangularView<Eigen::Lower>().solve( U   );
	    // // ws.mat_1 = T.adjoint().triangularView<Eigen::Lower>().solve( U * D.asDiagonal().inverse() );
	    ws.mat_2 = T.adjoint();
	    dqmc::la::solve_qr_col_piv(ws.mat_2, U, ws.mat_1);
	    ws.mat_5 = ws.mat_1.adjoint();
	    ws.mat_5.diagonal().real() += D;
	    
	    dqmc::la::decompose_udt_col_piv(ws.mat_5, ws.mat_2, ws.re_vec_2, ws.mat_3);
	    ws.mat_4 = ws.mat_3 * T;
	    ws.mat_5 = ws.re_vec_2.asDiagonal().inverse() * ws.mat_2.adjoint() * U.adjoint();
	    // cout << "mat_4 " << endl << ws.mat_4 << endl << endl;
	    // cout << "mat_5 " << endl << ws.mat_5 << endl << endl;
	    dqmc::la::solve_qr_col_piv(ws.mat_4, ws.mat_5, greens);
	    // greens = ws.mat_4.partialPivLu().solve(ws.mat_5);
	}

	
	template<typename workspace>
	inline void full_piv_qr_full_piv_lu(std::vector<cx_mat_t*> Us,
					    std::vector<vec_t*> Ds,
					    std::vector<cx_mat_t*> Ts,
					    std::vector<cx_mat_t*>
					    large_mats,
					    cx_mat_t& U_mat_vec,
					    cx_mat_t& T_mat_vec,
					    std::vector<vec_t*>
					    large_vecs,
					    workspace& ws,
					    cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];
		
		large_mats[0]->block(i * vol, i * vol, vol, vol) =
		    ((Us)[left])->fullPivLu().inverse()
		    * ((Ts)[right])->fullPivLu().inverse();
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal().cwiseAbs().maxCoeff() << endl;
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal() << endl;
	    // cout << "Decomposing" << endl
	    // 	 << large_mats[0]->format(CleanFmt) << endl << endl;
	    // cx_mat_t to_decompose = *large_mats[0];
	    
	    dqmc::la::decompose_udt_full_piv(*large_mats[0], 
	    				     *large_mats[1],
	    				     *large_vecs[0],
	    				     *large_mats[2]);
	    // cout << "Decomposition error\t" << (*large_mats[1] *
	    // 					large_vecs[0]->asDiagonal() *
	    // 					*large_mats[2]
	    // 					- to_decompose).cwiseAbs().maxCoeff()
	    	 // << endl;
	    
	    *large_mats[0] = large_vecs[0]->asDiagonal()
	    	* *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    // *large_mats[1] =
	    // 	large_mats[0]->fullPivLu().solve(large_mats[2]->adjoint());
	    *large_mats[1] = large_mats[0]->fullPivLu().inverse() * large_mats[2]->fullPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);
	    return;
	    

	    cx_mat_t to_decompose = large_mats[0]->transpose();

	    // cout << U_mat_vec * *large_mats[0] * T_mat_vec << endl << endl;
	    
	    dqmc::la::decompose_udt_full_piv(to_decompose, 
					     *large_mats[1],
					     *large_vecs[0],
					     *large_mats[2]);
	    
	    *large_mats[0] = large_vecs[0]->asDiagonal()
		* *(large_mats)[2] * U_mat_vec.transpose();
	    *large_mats[2] = T_mat_vec.transpose() * (*large_mats[1]);
	    *large_mats[1] = large_mats[0]->fullPivLu().inverse() * large_mats[2]->fullPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);



	    // cout << endl << greens << endl << endl;
								  
	    // large_mats[0] = large_vecs[0]->asDiagonal().inverse()
	    // 	* large_mats[1]->transpose() 
	    // 	* large_vecs[1]->asDiagonal().inverse()
	    // 	* U_mat_vec.transpose();

	    // large_mats[1] = large_mats[2] * large_vecs[2]->asDiagonal() * T_mat_vec;

	    // large_mats[2] = large_mats[1]->fullPivLu().solve(large_mats[0]);
	    // greens = large_mats[2]->block(0, 0, vol, vol);
	}       

	template<typename workspace>
	inline void full_piv_qr_full_piv_lu(std::vector<cx_mat_t*> Us,
					    std::vector<int> U_is_unitary,
					    std::vector<vec_t*> Ds,
					    std::vector<cx_mat_t*> Ts,
					    std::vector<int> T_is_unitary,
					    std::vector<cx_mat_t*>
					    large_mats,
					    cx_mat_t& U_mat_vec,
					    cx_mat_t& T_mat_vec,
					    std::vector<vec_t*>
					    large_vecs,
					    workspace& ws,
					    cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("full_piv_qr_full_piv_lu: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];

		if (U_is_unitary[left] == 0 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->fullPivLu().inverse()
			* ((Ts)[right])->fullPivLu().inverse();
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			//((Us)[left])->fullPivLu().inverse()
			(((Ts)[right])->adjoint().fullPivLu().solve(*(Us)[left])).adjoint();
		} else if (U_is_unitary[left] == 0 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->fullPivLu().solve((Ts)[right]->adjoint());
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->adjoint()
			* ((Ts)[right])->adjoint();
		}

		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    
	    dqmc::la::decompose_udt_full_piv(*large_mats[0], 
	    				     *large_mats[1],
	    				     *large_vecs[0],
	    				     *large_mats[2]);
	    *large_mats[0] = large_vecs[0]->asDiagonal()
	    	* *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    *large_mats[1] = large_mats[0]->fullPivLu().inverse() * large_mats[2]->fullPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);
	    return;
	}       

	template<typename workspace>
	inline void col_piv_qr_full_piv_lu(std::vector<cx_mat_t*> Us,
					    std::vector<vec_t*> Ds,
					    std::vector<cx_mat_t*> Ts,
					    std::vector<cx_mat_t*>
					    large_mats,
					    cx_mat_t& U_mat_vec,
					    cx_mat_t& T_mat_vec,
					    std::vector<vec_t*>
					    large_vecs,
					    workspace& ws,
					    cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];
		
		large_mats[0]->block(i * vol, i * vol, vol, vol) =
		    ((Us)[left])->fullPivLu().inverse()
		    * ((Ts)[right])->fullPivLu().inverse();
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    
	    dqmc::la::decompose_udt_col_piv(*large_mats[0], 
	    				     *large_mats[1],
	    				     *large_vecs[0],
	    				     *large_mats[2]);
	    
	    *large_mats[0] = large_vecs[0]->asDiagonal()
	    	* *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    *large_mats[1] = large_mats[0]->fullPivLu().inverse() * large_mats[2]->fullPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);
	    return;
	}


	template<typename workspace>
	inline void full_piv_qr_partial_piv_lu(std::vector<cx_mat_t*> Us,
					    std::vector<vec_t*> Ds,
					    std::vector<cx_mat_t*> Ts,
					    std::vector<cx_mat_t*>
					    large_mats,
					    cx_mat_t& U_mat_vec,
					    cx_mat_t& T_mat_vec,
					    std::vector<vec_t*>
					    large_vecs,
					    workspace& ws,
					    cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];
		
		large_mats[0]->block(i * vol, i * vol, vol, vol) =
		    ((Us)[left])->partialPivLu().inverse()
		    * ((Ts)[right])->partialPivLu().inverse();
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    
	    dqmc::la::decompose_udt_full_piv(*large_mats[0], 
	    				     *large_mats[1],
	    				     *large_vecs[0],
	    				     *large_mats[2]);
	    
	    *large_mats[0] = large_vecs[0]->asDiagonal()
	    	* *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);

	    *large_mats[1] = large_mats[0]->partialPivLu().inverse() * large_mats[2]->partialPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);
	}       

	template<typename workspace>
	inline void col_piv_qr_partial_piv_lu(std::vector<cx_mat_t*> Us,
					    std::vector<vec_t*> Ds,
					    std::vector<cx_mat_t*> Ts,
					    std::vector<cx_mat_t*>
					    large_mats,
					    cx_mat_t& U_mat_vec,
					    cx_mat_t& T_mat_vec,
					    std::vector<vec_t*>
					    large_vecs,
					    workspace& ws,
					    cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];
		
		large_mats[0]->block(i * vol, i * vol, vol, vol) =
		    ((Us)[left])->partialPivLu().inverse()
		    * ((Ts)[right])->partialPivLu().inverse();
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal().cwiseAbs().maxCoeff() << endl;
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal() << endl;
	    // cout << "Decomposing" << endl
	    // 	 << large_mats[0]->format(CleanFmt) << endl << endl;
	    // cx_mat_t to_decompose = *large_mats[0];
	    
	    dqmc::la::decompose_udt_col_piv(*large_mats[0], 
	    				     *large_mats[1],
	    				     *large_vecs[0],
	    				     *large_mats[2]);
	    // cout << "Decomposition error\t" << (*large_mats[1] *
	    // 					large_vecs[0]->asDiagonal() *
	    // 					*large_mats[2]
	    // 					- to_decompose).cwiseAbs().maxCoeff()
	    	 // << endl;
	    
	    *large_mats[0] = large_vecs[0]->asDiagonal()
	    	* *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    // *large_mats[1] =
	    // 	large_mats[0]->partialPivLu().solve(large_mats[2]->adjoint());
	    *large_mats[1] = large_mats[0]->partialPivLu().inverse() * large_mats[2]->partialPivLu().inverse();
	    greens = large_mats[1]->block(0, 0, vol, vol);
	}       


	template<typename workspace>
	inline void col_piv_qr_partial_piv_lu(std::vector<cx_mat_t*> Us,
					      std::vector<int>& U_is_unitary,
					      std::vector<vec_t*> Ds,
					      std::vector<cx_mat_t*> Ts,
					      std::vector<int>& T_is_unitary,
					      std::vector<cx_mat_t*>
					      large_mats,
					      cx_mat_t& U_mat_vec,
					      cx_mat_t& T_mat_vec,
					      std::vector<vec_t*>
					      large_vecs,
					      workspace& ws,
					      cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];

		if (U_is_unitary[left] == 0 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->partialPivLu().inverse()
			* ((Ts)[right])->partialPivLu().inverse();
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			//((Us)[left])->partialPivLu().inverse()
			(((Ts)[right])->adjoint().partialPivLu().solve((*Us[left]))).adjoint();
		} else if (U_is_unitary[left] == 0 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->partialPivLu().solve((Ts)[right]->adjoint());
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->adjoint()
			* ((Ts)[right])->adjoint();
		}
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    
	    dqmc::la::decompose_udt_col_piv(*large_mats[0], 
					    *large_mats[1],
					    *large_vecs[0],
					    *large_mats[2]);

	    *large_mats[0] = // large_vecs[0]->asDiagonal()
	    	 *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    // *large_mats[1] =
	    // 	large_mats[0]->partialPivLu().solve(large_mats[2]->adjoint());
	    // cout << endl << *large_mats[0] * large_mats[0]->partialPivLu().inverse() << endl << endl;
	    // cout << endl << *large_mats[2] * large_mats[2]->partialPivLu().inverse();
	    *large_mats[1] = large_mats[0]->partialPivLu().solve(
								 large_vecs[0]->asDiagonal().inverse()
								 * large_mats[2]->partialPivLu().inverse());
	    greens = large_mats[1]->block(0, 0, vol, vol);
	}       


	template<typename workspace>
	inline void partial_piv_lu(std::vector<cx_mat_t*> Us,
					      std::vector<int>& U_is_unitary,
					      std::vector<vec_t*> Ds,
					      std::vector<cx_mat_t*> Ts,
					      std::vector<int>& T_is_unitary,
					      std::vector<cx_mat_t*>
					      large_mats,
					      cx_mat_t& U_mat_vec,
					      cx_mat_t& T_mat_vec,
					      std::vector<vec_t*>
					      large_vecs,
					      workspace& ws,
					      cx_mat_t& greens) {
	    using namespace std;
	    Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
		
	    int order = Us.size();
	    int vol = Us[0]->rows();
	    int large_vol = large_mats[0]->rows();

	    if (large_vol % vol != 0 || large_vol / vol != order) {
		cout << "vol\t" << vol << endl;
		cout << "large_vol\t" << large_vol << endl;
		cout << "order\t" << order << endl;
		throw std::runtime_error("row_sort_qr: invalid matrix sizes");
	    }
		
	    int left, right;

	    large_mats[0]->setZero();

	    U_mat_vec.setZero();
	    T_mat_vec.setZero();
	    large_mats[0]->setZero();
	    
	    for (int i = 0; i < order; ++i) {
		left = (i - 1 + order) % order;
		right = i;

		U_mat_vec.block(i * vol, i * vol, vol, vol) = *Us[left];
		T_mat_vec.block(i * vol, i * vol, vol, vol) = *Ts[right];

		if (U_is_unitary[left] == 0 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->partialPivLu().inverse()
			* ((Ts)[right])->partialPivLu().inverse();
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 0) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			//((Us)[left])->partialPivLu().inverse()
			(((Ts)[right])->adjoint().partialPivLu().solve((*Us[left]))).adjoint();
		} else if (U_is_unitary[left] == 0 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->partialPivLu().solve((Ts)[right]->adjoint());
		} else if (U_is_unitary[left] == 1 && T_is_unitary[right] == 1) {
		    large_mats[0]->block(i * vol, i * vol, vol, vol) =
			((Us)[left])->adjoint()
			* ((Ts)[right])->adjoint();
		}
		
		if (i == 0) {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = *Ds[left];
		}
		else {
		    large_mats[0]->block(i * vol, left * vol,
					 vol, vol).diagonal().real() = - *Ds[left];
		}
	    }
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal().cwiseAbs().maxCoeff() << endl;
	    // cout << (U_mat_vec * *large_mats[0] * T_mat_vec - cx_mat_t::Identity(large_vol, large_vol)).diagonal() << endl;
	    // cout << "Decomposing" << endl
	    // 	 << large_mats[0]->format(CleanFmt) << endl << endl;
	    // cx_mat_t to_decompose = *large_mats[0];
	    
	    dqmc::la::decompose_udt(*large_mats[0], 
				    *large_mats[1],
				    *large_vecs[0],
				    *large_mats[2]);
	    // cout << "Decomposition error\t" << (*large_mats[1] *
	    // 					large_vecs[0]->asDiagonal() *
	    // 					*large_mats[2]
	    // 					- to_decompose).cwiseAbs().maxCoeff()
	    	 // << endl;
	    
	    *large_mats[0] = // large_vecs[0]->asDiagonal()
	    	 *(large_mats)[2] * T_mat_vec;
	    *large_mats[2] = U_mat_vec * (*large_mats[1]);
	    // *large_mats[1] =
	    // 	large_mats[0]->partialPivLu().solve(large_mats[2]->adjoint());
	    // cout << endl << *large_mats[0] * large_mats[0]->partialPivLu().inverse() << endl << endl;
	    // cout << endl << *large_mats[2] * large_mats[2]->partialPivLu().inverse();
	    *large_mats[1] = large_mats[0]->partialPivLu().solve(
								 large_vecs[0]->asDiagonal().inverse()
								 * large_mats[2]->partialPivLu().inverse());
	    greens = large_mats[1]->block(0, 0, vol, vol);
	}       
    }
}
#endif
