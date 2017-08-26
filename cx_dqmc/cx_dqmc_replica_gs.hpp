#ifndef CX_DQMC_REPLICA_GS_HPP
#define CX_DQMC_REPLICA_GS_HPP
#include <boost/mpi.hpp>

#include <libdqmc/la.hpp>
#include <libdqmc/cx_dqmc_greens_general.hpp>
#include <libdqmc/cx_dqmc_greens_replica_gs.hpp>
#include <libdqmc/cx_dqmc_greens_replica_renyi_gs.hpp>
#include <libdqmc/global.hpp>
#include <libdqmc/prefixed_out.hpp>
#include <libdqmc/cx_updates.hpp>
#include <libdqmc/tools.hpp>
#include <libdqmc/temper_package.hpp>
#include <libdqmc/cx_hopping.hpp>
#include <libdqmc/cx_density.hpp>
#include <libdqmc/cx_checkerboard.hpp>
#include <libdqmc/cx_compare_hopping_checkerboard.hpp>
// #include <libdqmc/cx_checkerboard_manual.hpp>
// #include <libdqmc/cx_measurements.hpp>
#include <complex>
// ALPS INCLUDES
#include <alps/lattice.h>
#include <alps/parapack/worker.h>
#include <alps/alea.h>
#include <alps/osiris.h>
#include <alps/osiris/dump.h>
#include <alps/expression.h>

#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

// Standard includes
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <list>
#include <vector>
#include <ctime>
#include <climits>
#include <limits>
#include <cmath>
#include <numeric>
#include <map>
#include <exception>

#include <boost/multi_array.hpp>
#include <boost/filesystem.hpp>


using namespace std;


class cx_dqmc_replica_gs : public alps::parapack::mc_worker {
    
public:
    
    ///@{
    ~cx_dqmc_replica_gs() { }
	
    cx_dqmc_replica_gs(const alps::Parameters &);
    cx_dqmc_replica_gs(boost::mpi::communicator const& comm, const alps::Parameters &);

    void start(const alps::Parameters& params);
    double work_done(alps::ObservableSet& obs);
    bool is_thermalized() const { return steps.is_thermalized(); } ;
    void init_observables(alps::Parameters const &, alps::ObservableSet& obs);
    void run(alps::ObservableSet& obs);
    double progress() const { return steps.progress(); } ;

    alps::mc_steps steps;
    dqmc::parameters cx_dqmc_params;
    alps::RealObservable dummy;
    cx_dqmc::workspace * ws;
    cx_dqmc::workspace * sqr_ws;
    cx_dqmc::workspace * r_ws;
    cx_dqmc::workspace * r_ws_0;

    temper_package outgoing, incoming;
    
    /// The auxiliary spin arrays
    boost::multi_array<double, 3 > aux_spins;
    boost::multi_array<double, 3 > aux_spins_0;
    boost::multi_array<double, 3 > aux_spins_1;
    boost::multi_array<double, 3 > aux_spins_temp;

    int particles;
    int rank;
    double gamma;
    double T;
    double beta;
    double delta_tau;
    int slices;
    double hopping, hopping_perp;
    double interaction;
    int sweeps_measurement; 

    double lambda;
    double gamma_table[2];

    // int bonds;
    int slice_steps;
    int start_slice;
    int end_slice;
    int volume;

    uint connected_0, connected_1, disconnected_0, disconnected_1;
    uint n_A_0, n_A_1, n_B_0, n_B_1;
    int max_flips;
    bool first_run;
    cx_dqmc::abstract_greens * gs_0_0;
    cx_dqmc::abstract_greens * gs_0_1;
    cx_dqmc::abstract_greens * gs_0_0_gen;
    cx_dqmc::abstract_greens * gs_0;
    cx_dqmc::abstract_greens * gs_1;

    int state;
    double log_det_ratio;
    pdouble_t det_up_0, det_down_0, det_up_1, det_down_1;
    pdouble_t det_up_0_0, det_down_0_0, det_up_0_1, det_down_0_1;
    pdouble_t last_det_0, last_det_1;
    cx_double sign_0_0, avg_sign_0_0;
    cx_double sign_0_1, avg_sign_0_1;
    cx_double sign_0, avg_sign_0;
    cx_double sign_1, avg_sign_1;

    pdouble_t log_ratio, ratio;
    pdouble_t local_moment, double_occupancy;
    int meas_counter;
    bool flipped;
    int test_runs;
    double alpha;
    double alpha_0;
    double alpha_1;

    bool state_0_done;
    bool state_1_done;

    bool grover;
    
    void switch_state(alps::ObservableSet& obs);
    bool gs_sweep(alps::ObservableSet& obs);
    bool gs_sweep_continuously_slice(alps::ObservableSet& obs);
    bool gs_sweep_continuously_chunk(alps::ObservableSet& obs);
    bool gs_sweep_continuously_complete(alps::ObservableSet& obs);
    double gs_delayed_spinful_flip(alps::ObservableSet& obs, int slice, bool at_once = true);
    double gs_delayed_continuous_spinful_flip(alps::ObservableSet& obs, int slice, vector<double>& switch_probs);
    double gs_simple_spinless_flip(alps::ObservableSet& obs, int slice);
    double gs_delayed_spinless_flip(alps::ObservableSet& obs, int slice);
    double gs_delayed_continuous_spinless_flip(alps::ObservableSet& obs, int slice, vector<double>& switch_probs);
    double gs_attractive_spinless_flip(alps::ObservableSet& obs, int slice);
    double gs_delayed_attractive_spinless_flip(alps::ObservableSet& obs, int slice);
    int gs_spinful_naive_flip(alps::ObservableSet& obs, int slice);
    bool package_ok;
    void TiePackage();
    void ComputeRatio();
    void Decide();
    void HandlePackage();

    void measure(alps::ObservableSet& obs);

    void randomize_spins() {
	for (uint s = 0; s < cx_dqmc_params.slices; s++) {
	    for (int i = 0; i < cx_dqmc_params.num_aux_spins; i++) {
		uniform_01() > 0.5 ?
		    aux_spins[0][s][i] = -1:
		    aux_spins[0][s][i] = 1;
		aux_spins_0[0][s][i] = aux_spins[0][s][i];
		aux_spins_1[0][s][i] = aux_spins[0][s][i];

		aux_spins[1][s][i] = aux_spins[0][s][i];;
		aux_spins_0[1][s][i] = aux_spins[1][s][i];
		aux_spins_1[1][s][i] = aux_spins[1][s][i];

	    }
	}
    };

    void ordered_spins() {
	for (uint s = 0; s < cx_dqmc_params.slices; s++) {
	    for (int i = 0; i < cx_dqmc_params.num_aux_spins; i++) {
		i % 2 == 0 ?
		    aux_spins[0][s][i] = -1:
		    aux_spins[0][s][i] = 1;

		aux_spins_0[0][s][i] = aux_spins[0][s][i];
		aux_spins_1[0][s][i] = aux_spins[0][s][i];

		aux_spins[1][s][i] = aux_spins[0][s][i];;
		aux_spins_0[1][s][i] = aux_spins[1][s][i];
		aux_spins_1[1][s][i] = aux_spins[1][s][i];

	    }
	}
    };

    
    void copy_aux_spins(boost::multi_array<double,3>& in,
			boost::multi_array<double, 3>& out) {
	for (int r = 0; r < 2; r++) {
	    for (int s = 0; s < cx_dqmc_params.slices; s++) {
		for (int l = 0; l < cx_dqmc_params.num_aux_spins; l++) {
		    out[r][s][l] = in[r][s][l];
		}
	    }
	}
    }

    std::string TimeString() {
	using namespace std;
	
	time_t timestamp;
	tm *now;
	timestamp = time(0);
	now = localtime(&timestamp);
	
	stringstream time_string;
	
	time_string << "[" << now->tm_year + 1900 << "-";
	if (now->tm_mon + 1 < 10) time_string << "0";
	time_string << now->tm_mon+1 << "-";
	if (now->tm_mday < 10) time_string << "0";
	time_string << now->tm_mday << " ";
	if (now->tm_hour < 10) time_string << "0";
	time_string << now->tm_hour << ":";
	if (now->tm_min < 10) time_string << "0";
	time_string << now->tm_min << ":";
	if (now->tm_sec < 10) time_string << "0";
	time_string << now->tm_sec << "]   ";
	
	return time_string.str();
    }


    void copy_to_gs_0() {
	for (int i = 0; i < gs_0_0->chunks; ++i) {
	    gs_0->u_stack[i] = gs_0_0->u_stack[i];
	    gs_0->d_stack[i] = gs_0_0->d_stack[i];
	    gs_0->t_stack[i] = gs_0_0->t_stack[i];
	}
	for (int i = 0; i < gs_0_1->chunks; ++i) {
	    gs_0->u_stack[i + gs_0_0->chunks] = gs_0_1->u_stack[i];
	    gs_0->d_stack[i + gs_0_0->chunks] = gs_0_1->d_stack[i];
	    gs_0->t_stack[i + gs_0_0->chunks] = gs_0_1->t_stack[i];
	}
	gs_0->idx = 0;
	gs_0->current_slice = 0;
	gs_0->direction = 1;
	gs_0->dir_switch = true;
	gs_0->propagate(dummy);
    }


    void copy_to_gs_1() {
	for (int i = 0; i < gs_0_0->chunks; ++i) {
	    gs_1->u_stack[i] = gs_0_0->u_stack[i];
	    gs_1->d_stack[i] = gs_0_0->d_stack[i];
	    gs_1->t_stack[i] = gs_0_0->t_stack[i];
	}
	for (int i = 0; i < gs_0_1->chunks; ++i) {
	    gs_1->u_stack[i + gs_0_0->chunks] = gs_0_1->u_stack[i];
	    gs_1->d_stack[i + gs_0_0->chunks] = gs_0_1->d_stack[i];
	    gs_1->t_stack[i + gs_0_0->chunks] = gs_0_1->t_stack[i];
	}
	gs_1->idx = 0;
	gs_1->current_slice = 0;
	gs_1->direction = 1;
	gs_1->dir_switch = true;
	gs_1->propagate(dummy);
    }


    void copy_to_gs_0x() {
	for (int i = 0; i < gs_0_0->chunks; ++i) {
	    gs_0_0->u_stack[i] = gs_1->u_stack[i];
	    gs_0_0->d_stack[i] = gs_1->d_stack[i];
	    gs_0_0->t_stack[i] = gs_1->t_stack[i];
	}
	for (int i = 0; i < gs_0_1->chunks; ++i) {
	    gs_0_1->u_stack[i] = gs_1->u_stack[i + gs_0_0->chunks];
	    gs_0_1->d_stack[i] = gs_1->d_stack[i + gs_0_0->chunks];
	    gs_0_1->t_stack[i] = gs_1->t_stack[i + gs_0_0->chunks];
	}
	gs_0_0->idx = 0;
	gs_0_0->current_slice = 0;
	gs_0_0->direction = 1;
	gs_0_0->dir_switch = true;
	gs_0_0->propagate(dummy);

	gs_0_1->idx = 0;
	gs_0_1->current_slice = 0;
	gs_0_1->direction = 1;
	gs_0_1->dir_switch = true;
	gs_0_1->propagate(dummy);
    }

    void load(alps::IDump& dump) {
	print_memory_usage();
	dump >> steps >> state;
	print_memory_usage();

	for (int r = 0; r < 2; r++) {
	    for (int s = 0; s < cx_dqmc_params.slices; s++) {
		for (int l = 0; l < cx_dqmc_params.num_aux_spins; l++) {
		    dump >> aux_spins_0[r][s][l];
		    dump >> aux_spins_1[r][s][l];
		}
	    }
	}

	if (state == 0) {
	    for (int a = 0; a < 2; a++) {
		for (uint s = 0; s < cx_dqmc_params.slices; s++) {
		    for (int i = 0; i < cx_dqmc_params.num_aux_spins; i++) {
			aux_spins[a][s][i] = aux_spins_0[a][s][i];
		    }
		}	    
	    }
	} else {
	    for (int a = 0; a < 2; a++) {
		for (uint s = 0; s < cx_dqmc_params.slices; s++) {
		    for (int i = 0; i < cx_dqmc_params.num_aux_spins; i++) {
			aux_spins[a][s][i] = aux_spins_1[a][s][i];
		    }
		}
	    }
	}

	out << "Monte Carlo Steps " << steps() << endl;
	print_memory_usage();

	gs_0_0->build_stack();
	gs_0_1->build_stack();
	gs_1->build_stack();
	copy_to_gs_1();
	// gs_1->propagate(dummy);

	if (cx_dqmc_params.partial_cuts == true) {
	    gs_0->build_stack();
	    copy_to_gs_0();
	    // gs_0->propagate(dummy);
	}

	out << "Retesting sweep stability - state is " << state << endl;
	{
	    boost::timer::auto_cpu_timer t;
	    for (uint i = 0; i < cx_dqmc_params.slices; i++) {
		gs_0_0->propagate(dummy);
		gs_0_1->propagate(dummy);
	    }
	}

	{
	    boost::timer::auto_cpu_timer t;
	    for (uint i = 0; i < cx_dqmc_params.slices; i++) {
		cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
		cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;	
		gs_1->propagate(dummy);
		gs_1->propagate(dummy);
	    }
	}

	if (cx_dqmc_params.partial_cuts == true) {
	    {
		cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
		cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;	
	    
		boost::timer::auto_cpu_timer t;
		for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
		    gs_0->propagate(dummy);
		    gs_0->propagate(dummy);
		}
	    }
	}

	if (state == 0 && cx_dqmc_params.partial_cuts == true) {
	    cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	    cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;	
	} else if (state == 1 && cx_dqmc_params.partial_cuts == true) {
	    cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	    cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;	
	}   
    }


    void save(alps::ODump& dump) const {
	dump << steps << state;

	for (int r = 0; r < 2; r++) {
	    for (int s = 0; s < cx_dqmc_params.slices; s++) {
		for (int l = 0; l < cx_dqmc_params.num_aux_spins; l++) {
		    dump << aux_spins_0[r][s][l];
		    dump << aux_spins_1[r][s][l];
		}
	    }
	}
    }

    void calculate_weights(bool recalculate = true) {
	if (recalculate == true) {
	    gs_0_0->build_stack();
	    gs_0_1->build_stack();
	    copy_to_gs_1();

	    if (cx_dqmc_params.partial_cuts == true) {
		copy_to_gs_0();
	    }
	}

	{
	    if (state == 0 && cx_dqmc_params.partial_cuts == false) {
		if (gs_0_0->get_slice() != 0) 
		    dqmc::tools::abort("Called calculate_weights on the wrong slice of gs_0_0");
		if (gs_0_1->get_slice() != 0) 
		    dqmc::tools::abort("Called calculate_weights on the wrong slice of gs_0_1");
		
		copy_to_gs_1();
		gs_1->log_weight();
	    } else if (state == 0 && cx_dqmc_params.partial_cuts == true) {
		if (gs_0->get_slice() != 0) 
		    dqmc::tools::abort("Called measure on the wrong slice of gs_0");
		
		gs_0->log_weight();
		cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
		cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
		for (int i = 0; i < gs_0->chunks; ++i) {
		    gs_1->u_stack[i] = gs_0->u_stack[i];
		    gs_1->d_stack[i] = gs_0->d_stack[i];
		    gs_1->t_stack[i] = gs_0->t_stack[i];
		}
		gs_1->idx = 0;
		gs_1->current_slice = 0;
		gs_1->direction = 1;
		gs_1->dir_switch = true;
		gs_1->propagate(dummy);
		gs_1->log_weight();
		cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
		cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
	    } else if (state == 1) {
		if (gs_1->get_slice() != 0) 
		    dqmc::tools::abort("Called measure on the wrong slice of gs_1");

		gs_1->log_weight();

		if (cx_dqmc_params.partial_cuts == true) {
		    cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
		    cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
		    for (int i = 0; i < gs_1->chunks; ++i) {
			gs_0->u_stack[i] = gs_1->u_stack[i];
			gs_0->d_stack[i] = gs_1->d_stack[i];
			gs_0->t_stack[i] = gs_1->t_stack[i];
		    }
		    gs_0->idx = 0;
		    gs_0->current_slice = 0;
		    gs_0->direction = 1;
		    gs_0->dir_switch = true;
		    gs_0->propagate(dummy);
		    gs_0->log_weight();
		    cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
		    cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
		}
		else {
		    copy_to_gs_0x();
		    gs_0_0->log_weight();
		    gs_0_1->log_weight();
		}
	    }
	}

	if (cx_dqmc_params.model_id <= cx_dqmc_params.CX_SPINFUL_HUBBARD_ALTERNATIVE) {
	    if (cx_dqmc_params.partial_cuts == false) {
		sign_0_0 = gs_0_0->det_sign * gs_0_0->det_sign * gs_0_0->phase;
		sign_0_1 = gs_0_1->det_sign * gs_0_1->det_sign * gs_0_1->phase;
		sign_0 = sign_0_0 * sign_0_1;
		sign_1 = gs_1->det_sign * gs_1->det_sign * gs_1->phase;
		det_up_0_0 = gs_0_0->get_log_det();
		det_up_0_1 = gs_0_1->get_log_det();
		det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
		det_up_1 = 2 * gs_1->get_log_det();
	    } else {
		sign_0 = gs_0->det_sign * gs_0->det_sign * gs_0->phase;
		sign_1 = gs_1->det_sign * gs_1->det_sign * gs_1->phase;
		det_up_0 = 2 * gs_0->get_log_det();
		det_up_1 = 2 * gs_1->get_log_det();
	    }
	} else {
	    if (cx_dqmc_params.partial_cuts == false) {
		sign_0_0 = gs_0_0->det_sign * gs_0_0->phase;
		sign_0_1 = gs_0_1->det_sign * gs_0_1->phase;
		sign_0 = sign_0_0 * sign_0_1;
		sign_1 = gs_1->det_sign * gs_1->phase;
		det_up_0_0 = gs_0_0->get_log_det();
		det_up_0_1 = gs_0_1->get_log_det();
		det_up_0 = (det_up_0_0 + det_up_0_1);
		det_up_1 = gs_1->get_log_det();
	    } else {
		sign_0 = gs_0->det_sign * gs_0->phase;
		sign_1 = gs_1->det_sign * gs_1->phase;
		det_up_0 = gs_0->get_log_det();
		det_up_1 = gs_1->get_log_det();
	    }
	}
    }
	
    inline void print_memory_usage() {
	using namespace std;
	if (cx_dqmc_params.rank < 1) {
	    try {
		double vm, rss;
		dqmc::tools::process_mem_usage(vm, rss);
		out << "VM: " << vm << setw(5) << "" << "RSS: " << rss << endl;
	    } catch(std::exception const& ex) {
		out << "Determining memory usage failed\t" << ex.what() << endl;
	    }
	}
    }
    
    boost::mpi::communicator world;
    PrefixedOut out;
    PrefixedOut err;
};

#endif
 
