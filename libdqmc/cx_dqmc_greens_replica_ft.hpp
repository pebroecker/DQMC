#ifndef CX_DQMC_GREENS_REPLICA_FT_HPP
#define CX_DQMC_GREENS_REPLICA_FT_HPP

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
    
    class greens_replica_ft : public cx_dqmc::abstract_greens {

    public:       
	bool first_initialization, dir_switch;

	cx_mat_t u_temp, t_temp;
	vec_t d_temp;
	cx_mat_t Ul, Um, Ur, Tl, Tm, Tr;
	vec_t Dl, Dm, Dr;

	greens_replica_ft(dqmc::parameters * p_, cx_dqmc::workspace * ws_,
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

	void calculate_greens(int method = 0);
	void log_weight();
	void calculate_greens_general();
	void calculate_greens_exact(int slice);
	void calculate_greens(cx_mat U, cx_mat& G);

    	double check_stability();	    
    };
}
#endif
