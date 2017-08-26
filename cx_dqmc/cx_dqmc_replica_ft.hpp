#ifndef CX_DQMC_REPLICA_FT_HPP
#define CX_DQMC_REPLICA_FT_HPP
#include <boost/mpi.hpp>

#include <libdqmc/la.hpp>
#include <libdqmc/cx_dqmc_greens_replica_ft.hpp>
#include <libdqmc/cx_dqmc_greens_replica_renyi_ft.hpp>
#include <libdqmc/global.hpp>
#include <libdqmc/prefixed_out.hpp>
#include <libdqmc/cx_updates.hpp>
#include <libdqmc/tools.hpp>
#include <libdqmc/temper_package.hpp>
#include <libdqmc/cx_density.hpp>
#include <libdqmc/cx_checkerboard.hpp>
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
#include <map>
#include <exception>

#include <boost/multi_array.hpp>
#include <boost/filesystem.hpp>


using namespace std;

class cx_dqmc_replica_ft : public alps::parapack::mc_worker {
    
public:
    
    ///@{
    ~cx_dqmc_replica_ft() { }
	
    cx_dqmc_replica_ft(const alps::Parameters &);
    cx_dqmc_replica_ft(boost::mpi::communicator const& comm, const alps::Parameters &);

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
    
    cx_dqmc::abstract_greens * gs_0_0;
    cx_dqmc::abstract_greens * gs_0_1;
    cx_dqmc::abstract_greens * gs_1;

    int state;
    pdouble_t det_up_0, det_down_0, det_up_1, det_down_1;
    pdouble_t det_up_0_0, det_down_0_0, det_up_0_1, det_down_0_1;
    pdouble_t last_det_0, last_det_1;
    cx_double sign_0_0, sign_0_0_if_changed, avg_sign_0_0;
    cx_double sign_0_1, sign_0_1_if_changed, avg_sign_0_1;
    cx_double sign_0, sign_0_if_changed, avg_sign_0;
    cx_double sign_1, sign_1_if_changed, avg_sign_1;

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
    int gs_delayed_spinful_flip(alps::ObservableSet& obs, int slice);
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
		// // start in "ground state"
		// i % 2 == 0 ?
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


    boost::mpi::communicator world;
    PrefixedOut out;
    PrefixedOut err;
};

#endif
 
