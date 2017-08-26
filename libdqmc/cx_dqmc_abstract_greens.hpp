#ifndef CX_DQMC_ABSTRACT_GREENS_HPP
#define CX_DQMC_ABSTRACT_GREENS_HPP

#include "la.hpp"
#include "parameters.hpp"
#include "cx_workspace.hpp"

#include <alps/alea.h>
#include <boost/multi_array.hpp>
#include <boost/timer/timer.hpp>
#include <exception>
#include <cstring>
#include <valarray>
#include <algorithm>

typedef boost::multi_array<double, 3> aux_spin_3t;

namespace cx_dqmc {
    
    class abstract_greens {
	
    public:	
	cx_mat X, Y, X_sls, Y_sls, Y_sls_row_1, Y_sls_row_2;
	int delayed_buffer_size;
	
	cx_mat greens, temp_greens, prop_greens;
	bool first_initialization;

	double spin, delta_tau;
	cx_double_t det_sign, phase;
	pdouble_t det, log_det, curr_log_det, diff;
	int N, n_A, n_B, particles, eff_particles;
	int vol, sites, replica;
    	int direction, current_slice, current_bond,
	    slices, n_elements, safe_mult, chunks;
	int start, stop, idx;
	int indices[2];
	bool changed_direction, new_sign, fresh_det, dir_switch;
	double lowest_sv;

	std::vector<cx_mat_t> u_stack, t_stack;
	std::vector<vec_t> d_stack;

	std::vector<cx_mat_t*> Us, Ts, large_mats;
	std::vector<vec_t*> Ds, large_vecs;
	
	std::vector<int> U_is_unitary, T_is_unitary;
	
	std::valarray<double> svds;
	std::valarray<double> svd_example;
	std::valarray<double> stability_checks;
	std::valarray<double> stability;

	abstract_greens(dqmc::parameters * p_, cx_dqmc::workspace * ws_,
			cx_dqmc::workspace * reg_ws_, aux_spin_3t * s)
	    : p(p_), ws(ws_), reg_ws(reg_ws_), aux_spins(s) {};

	// abstract_greens(dqmc::parameters * p_, cx_dqmc::workspace * ws_,
	// 			     aux_spin_3t * s)
	//     : p(p_), ws(ws_), reg_ws(ws_), aux_spins(s) {};
	
	cx_dqmc::workspace * ws;
	cx_dqmc::workspace * reg_ws;
	dqmc::parameters * p;	
	aux_spin_3t * aux_spins;

	virtual void initialize() = 0;
	virtual void build_stack() = 0;
	virtual int propagate(alps::Observable& stability) = 0;
	virtual void log_weight() = 0;
	virtual void log_weight_full_piv() = 0;
	virtual void update_remove_interaction() = 0;
	virtual void update_add_interaction() = 0;
	

	int get_direction() { return direction; };	
	int get_slice() { return current_slice; }
	void print_diag() { std::cout << greens.diagonal().transpose() << std::endl; }
	cx_mat& get_greens() { return greens; }
	cx_mat * get_greens_ptr() { return &greens; }
	void save_curr_det() { curr_log_det = log_det; }
	void update_det() {
	    curr_log_det = log_det;
	}

	double get_log_det() { 
	    if (fresh_det == false) {
		throw std::runtime_error("det is not fresh");
	    }
	    return log_det; }
	
	cx_double get_det_sign() { return det_sign; }   
	cx_double& at(int i, int j) { return greens(i, j); }    
	cx_double get_tr() {
	    return greens.trace();
	    // throw std::runtime_error("not available yet");
	    //return {0., 0.}; 
	}
	
    };
}
#endif
