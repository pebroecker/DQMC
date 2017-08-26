#ifndef CX_DQMC_HPP
#define CX_DQMC_HPP
#include <boost/mpi.hpp>

#include <libdqmc/la.hpp>
#include <libdqmc/cx_dqmc_greens_general.hpp>
#include <libdqmc/cx_dqmc_greens_general_renyi.hpp>
#include <libdqmc/global.hpp>
#include <libdqmc/prefixed_out.hpp>
#include <libdqmc/cx_updates.hpp>
#include <libdqmc/tools.hpp>
#include <libdqmc/temper_package.hpp>
// #include <libdqmc/cx_dqmc_hopping.hpp>
#include <libdqmc/cx_density.hpp>
#include <libdqmc/cx_checkerboard.hpp>
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


class cx_dqmc_worker : public alps::parapack::mc_worker {
    
public:
    
    ///@{
    ~cx_dqmc_worker() { }
	
    cx_dqmc_worker(const alps::Parameters &);
    cx_dqmc_worker(boost::mpi::communicator const& comm, const alps::Parameters &);

    void start(const alps::Parameters& params);
    double work_done(alps::ObservableSet& obs);
    bool is_thermalized() const { return steps.is_thermalized(); } ;
    void init_observables(alps::Parameters const &, alps::ObservableSet& obs);
    void run(alps::ObservableSet& obs);
    double progress() const { return steps.progress(); } ;

    alps::mc_steps steps;
    dqmc::parameters cx_dqmc_params;
    
    cx_dqmc::workspace * ws;
    cx_dqmc::workspace * sqr_ws;
    cx_dqmc::workspace * r_ws;

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
    int volume_0, volume_1;
    int max_flips;
    bool first_run;
    cx_dqmc::greens_general * gs_0_0;
    cx_dqmc::greens_general * gs_0_1;
    cx_dqmc::greens_general_renyi * gs_1;

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
    bool gs_sweep_continuously(alps::ObservableSet& obs);
    double gs_delayed_spinful_flip(alps::ObservableSet& obs, int slice);
    double gs_delayed_continuous_spinful_flip(alps::ObservableSet& obs, int slice, vector<double>& switch_probs);
    double gs_simple_spinless_flip(alps::ObservableSet& obs, int slice);
    double gs_delayed_spinless_flip(alps::ObservableSet& obs, int slice);
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


    void load(alps::IDump& dump) {
	dump >> steps >> state;

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

	gs_0_0->build_stack();
	gs_0_1->build_stack();
	gs_1->build_stack();
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

    void calculate_weights() {
	if (state == 0) {
	    gs_0_0->build_stack();
	    gs_0_1->build_stack();

	    det_up_0_0 = gs_0_0->get_log_det();
	    det_up_0_1 = gs_0_1->get_log_det();

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
	    } else {
		det_up_0 = (det_up_0_0 + det_up_0_1);
	    }
	} else {
	    gs_1->build_stack();

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		det_up_1 = 2 * gs_1->get_log_det();
	    } else {
		det_up_1 =  gs_1->get_log_det();	
	    }
	}
    }
	

    boost::mpi::communicator world;
    PrefixedOut out;
    PrefixedOut err;
};

#endif
 
