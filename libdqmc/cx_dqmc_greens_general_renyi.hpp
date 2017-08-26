#ifndef CX_DQMC_GREENS_GENERAL_RENYI_HPP
#define CX_DQMC_GREENS_GENERAL_RENYI_HPP

#include <complex>
#include <libdqmc/cx_dqmc_abstract_greens.hpp>
#include <libdqmc/cx_dqmc_greens_general.hpp>
#include <libdqmc/la.hpp>
#include <libdqmc/cx_density.hpp>
#include <libdqmc/parameters.hpp>
#include <libdqmc/cx_workspace.hpp>
#include <libdqmc/cx_checkerboard.hpp>
#include <libdqmc/cx_interactions.hpp>
#include <libdqmc/cx_calculate_greens_ft.hpp>
#include <alps/alea.h>
#include <boost/multi_array.hpp>
#include <exception>
#include <cstring>

typedef boost::multi_array<double, 3> aux_spin_3t;

namespace cx_dqmc {
    
    class greens_general_renyi : public cx_dqmc::abstract_greens {

    public:       
	bool first_initialization, dir_switch, active, swept_on;
	cx_mat_t u_temp, t_temp;
	cx_mat ket, ket_sqr, bra, bra_sqr;
	vec ket_d, bra_d, d_temp;

	cx_mat_t Ul, Um, Ur, Tl, Tm, Tr;
	vec_t Dl, Dm, Dr;
	
	int det_method;
	bool fresh_sign;

	cx_dqmc::greens_general * gs_0;
	cx_dqmc::greens_general * gs_1;
	
	greens_general_renyi(dqmc::parameters* p_,
			     cx_dqmc::workspace* ws_,
			     cx_dqmc::workspace* reg_ws_,
			     aux_spin_3t* s)
	    : cx_dqmc::abstract_greens(p_, ws_, reg_ws_, s) {};
	
	void initialize();
	void set_greens(cx_dqmc::greens_general * gs_0,
			cx_dqmc::greens_general * gs_1);	    
	
	void build_stack();
	int propagate(alps::Observable& stability);
		
	void update_remove_interaction();
	void update_add_interaction();

	void slice_sequence(int start, int stop) {}; 
	void slice_sequence_left(int start, int stop, cx_mat& M);
	void slice_sequence_left_renyi(int start, int stop, cx_mat& M);
	void slice_sequence_left_t(int start, int stop, cx_mat& M);
	void slice_sequence_left_renyi_t(int start, int stop, cx_mat& M);
	void slice_sequence_right(int start, int stop, cx_mat& M);
	
	void slice_matrix_left(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_left_t(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_left_renyi_t(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_right(int slice, cx_mat& M, bool inv = false);

	void slice_matrix_left_renyi(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_right_renyi(int slice, cx_mat& M, bool inv = false);

	void hopping_matrix_left(int slice, cx_mat& M, bool inv = false);
	void hopping_matrix_right(int slice, cx_mat& M, bool inv = false);
	void hopping_matrix_left_renyi(int slice, cx_mat& M, bool inv = false);
	void hopping_matrix_right_renyi(int slice, cx_mat& M, bool inv = false);

	void interaction_matrix_left(int slice, cx_mat& M,
				     int bond_group, bool inv = false);
	void interaction_matrix_right(int slice, cx_mat& M,
				      int bond_group, bool inv = false);

	void regularize_svd(vec& in, vec& out);

	inline void prep_bra(cx_mat& col, vec& diag, cx_mat& sqr) {	    
	    using namespace std;
	    bra = col.transpose();
	    bra_d = diag;
	    bra_sqr = sqr.transpose();

	    dqmc::la::thin_col_to_invertible(col, ws->mat_1);
	    Tr = ws->mat_1.transpose().eval();
	    regularize_svd(diag, Dr);
	    Ur.setIdentity();
	    Ur.block(0, 0, ws->particles, ws->particles) = sqr.transpose().eval();

	    
	    // dqmc::la::thin_col_to_invertible(col, Ul);	    
	    // regularize_svd(diag, Dl);
	    // Tl.setIdentity();
	    // Tl.block(0, 0, ws->particles, ws->particles) = sqr;	    

	}

	inline void prep_ket(cx_mat& col, vec& diag,
			     cx_mat& sqr) {
	    using namespace std;
	    ket = col;
	    ket_d = diag;
	    ket_sqr = sqr;

	    // dqmc::la::thin_col_to_invertible(col, ws->mat_1);
	    // Tr = ws->mat_1.transpose().eval();
	    // regularize_svd(diag, Dr);
	    // Ur.setIdentity();
	    // Ur.block(0, 0, ws->particles, ws->particles) = sqr.transpose().eval();
	    
	    dqmc::la::thin_col_to_invertible(col, Ul);	    
	    regularize_svd(diag, Dl);
	    Tl.setIdentity();
	    Tl.block(0, 0, ws->particles, ws->particles) = sqr;	    
	}

	void log_weight();
	void calculate_greens_exact(int slice);
	void calculate_greens_general();
	void calculate_greens_basic();
	inline void calculate_greens() {
	    using namespace std;

	    if (current_slice % slices == 0) {
		calculate_greens_general();
		return;
	    }

	    idx = (current_slice) / safe_mult;
	    
	    if (stability[idx] == 0) {
		calculate_greens_basic();
	    }
	    else {
		calculate_greens_general();
	    }

	    if (check_stability() > 1e-2) {
		if (stability[idx] != 1) {
		    cout << p->outp << "renyi - Escalating stability at " << idx << " to "
			 << stability[idx] + 1 << endl;
		}
		stability[idx] = std::min(1, stability[idx] + 1);
		if (idx - direction >= 0 && idx - direction < n_elements) {
		    stability[idx - direction] = std::min(1, stability[idx - direction] + 1);
		}
	    }
	}


    	double check_stability();	    

	void enlarge(int replica, cx_mat_t& in, cx_mat_t& out);
	void enlarge(int replica, vec_t& in, vec_t& out);
	void enlarge_thin(int replica, cx_mat_t& in, cx_mat_t& out);
	void enlarge_thin(int replica, vec_t& in, vec_t& out);
	void enlarge_thin_ized_col(int replica, cx_mat_t& in, cx_mat_t& out);
	void enlarge_thin_ized_row(int replica, cx_mat_t& in, cx_mat_t& out);
    };
}
#endif
