#ifndef CX_DQMC_GREENS_REPLICA_RENYI_FT_HPP
#define CX_DQMC_GREENS_REPLICA_RENYI_FT_HPP

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
    
    class greens_replica_renyi_ft : public cx_dqmc::abstract_greens {

    public:       
	bool first_initialization, dir_switch;

	cx_mat_t u_temp, t_temp;
	cx_mat ket, ket_sqr, bra, bra_sqr;
	vec ket_d, bra_d, d_temp;

	cx_mat_t Ul, Um, Ur, Tl, Tm, Tr;
	vec_t Dl, Dm, Dr;

	int det_method;
	bool fresh_sign;
	
	greens_replica_renyi_ft(dqmc::parameters* p_,
				cx_dqmc::workspace* ws_,
				cx_dqmc::workspace* reg_ws_,
				aux_spin_3t* s)
	    : cx_dqmc::abstract_greens(p_, ws_, reg_ws_, s) {};
	
	void initialize();	
	void build_stack();
	int propagate(alps::Observable& stability);

	void log_weight();
	void calculate_greens(int direction, int slice) {
	    using namespace std;
	    // boost::timer::auto_cpu_timer t;
	    // cout << "greens_replica_renyi_ft::calculate_greens(int, int)\t";
	    calculate_greens_general();
	};
	void calculate_greens_exact(int slice);
	
	void update_remove_interaction();
	void update_add_interaction();

	void ket_update(int start, int stop, int i, int j);
	void bra_update(int start, int stop, int i, int j);
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

	void calculate_greens(int method = 0);
	void calculate_greens_general();
	void calculate_greens(cx_mat U, cx_mat& G);

    	double check_stability();	    

	void enlarge(int replica, cx_mat_t& in, cx_mat_t& out);
	void enlarge(int replica, vec_t& in, vec_t& out);
    };
}
#endif
