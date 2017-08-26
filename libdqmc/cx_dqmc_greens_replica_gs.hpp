#ifndef CX_DQMC_GREENS_REPLICA_GS_HPP
#define CX_DQMC_GREENS_REPLICA_GS_HPP

#include <complex>
#include <libdqmc/cx_dqmc_abstract_greens.hpp>
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
    
    class greens_replica_gs : public cx_dqmc::abstract_greens {

    public:       
	bool first_initialization;

	cx_mat_t u_temp, t_temp;
	vec_t d_temp;
	cx_mat_t Ul, Um, Ur, Tl, Tm, Tr;
	vec_t Dl, Dm, Dr;
	
	greens_replica_gs(dqmc::parameters * p_, cx_dqmc::workspace * ws_,
			aux_spin_3t * s)
	    : cx_dqmc::abstract_greens(p_, ws_, ws_, s) {};

	void initialize();	
	void build_stack();
	int propagate(alps::Observable& stability);
	
	void ket_update(int start, int stop,
			int idx, int target);
	void bra_update(int start, int stop,
			int idx, int target);

	void update_remove_interaction();
	void update_add_interaction();

	void slice_sequence(int start, int stop) {}; 
	void slice_sequence_left(int start, int stop, cx_mat& M);
	void slice_sequence_left_t(int start, int stop, cx_mat& M);
	void slice_sequence_right(int start, int stop, cx_mat& M);
	
	void slice_matrix_left(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_left_t(int slice, cx_mat& M, bool inv = false);
	void slice_matrix_right(int slice, cx_mat& M, bool inv = false);

	void hopping_matrix_left(int slice, cx_mat& M, bool inv = false);
	void hopping_matrix_right(int slice, cx_mat& M, bool inv = false);

	void interaction_matrix_left(int slice, cx_mat& M,
				     int bond_group, bool inv = false);
	void interaction_matrix_right(int slice, cx_mat& M,
				      int bond_group, bool inv = false);

	void regularize_svd(vec& in, vec& out);
	
	void log_weight();	
	void log_weight_full_piv();	

	void calculate_greens_basic();
	void calculate_greens_half_compressed();
	void calculate_greens_general();
	void calculate_greens_exact(int slice);

	inline void calculate_greens() {
	    using namespace std;

	    if (p->escalate_stability == false) {
		if (p->basic_stability == true) {
		    calculate_greens_basic();
		} else if (p->half_stability == true) {
		    calculate_greens_half_compressed();
		} else if (p->full_stability == true) {
		    calculate_greens_general();
		}
	    } else {
		if (int(stability[idx]) == 0) {
		    calculate_greens_basic();
		} else if (int(stability[idx]) == 1) {
		    calculate_greens_half_compressed();
		} else if (int(stability[idx]) == 2) {
		    calculate_greens_general();
		}
	    }


	    // if (p->full_piv_stable == true) {
	    // 	calculate_greens_general();
	    // 	if (current_slice % slices == 0) return;
	    // } else {
	    // 	if (current_slice % slices == 0) {
	    // 	    calculate_greens_basic();
	    // 	    return;
	    // 	}
	    
	    // 	idx = current_slice / safe_mult;
	    
	    // 	if (stability[idx] == 0) {
	    // 	    calculate_greens_basic();
	    // 	}
	    // 	else {
	    // 	    calculate_greens_general();
	    // 	}
	    // }

	    if (current_slice % slices == 0) return;

	    double stability_check = check_stability();
	    stability_checks[idx] = stability_check;
	    
	    if (stability_check > 1e-2) {
		if (int(stability[idx]) != 2) {
		    cout << p->outp << "greens - Escalating stability at " << idx << " to "
			 << int(stability[idx] + 1) << endl;
		}
		stability[idx] = std::min(2, int(stability[idx]) + 1);
		if (idx - direction >= 0 && idx - direction < n_elements) {
		    stability[idx - direction] = std::min(2, int(stability[idx - direction]) + 1);
		}
	    }
	}
    	double check_stability();	    
    };
}
#endif
