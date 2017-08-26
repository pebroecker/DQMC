#include "cx_dqmc_replica_ft.hpp"

using namespace dqmc::la;

cx_dqmc_replica_ft::cx_dqmc_replica_ft( alps::Parameters const& p) 
    : alps::parapack::mc_worker(p), cx_dqmc_params(p), steps(p) {
    rank = -1;
    if (global::use_mpi) rank = world.rank();
    out.SetRankPrefix("worker", rank);
    start(p);
    cx_dqmc_params.rank = rank;
}


cx_dqmc_replica_ft::cx_dqmc_replica_ft(boost::mpi::communicator const& comm,  alps::Parameters const& p) 
    : alps::parapack::mc_worker(p), cx_dqmc_params(p, true), steps(p), world(comm) {
    if (global::use_mpi) rank = world.rank();
    out.SetRankPrefix("worker", rank);
    start(p);    
    cx_dqmc_params.rank = rank;
}


void cx_dqmc_replica_ft::start(const alps::Parameters& p) {
    ws = new cx_dqmc::workspace();    
    ws->parameters(cx_dqmc_params);
    ws->particles = cx_dqmc_params.N;
    ws->eff_particles = cx_dqmc_params.N;
    ws->large_vol = 2 * ws->vol;
    ws->initialize_gs();

    sqr_ws = new cx_dqmc::workspace();    
    sqr_ws->parameters(cx_dqmc_params);
    sqr_ws->particles = cx_dqmc_params.N;
    sqr_ws->eff_particles = cx_dqmc_params.N;
    sqr_ws->large_vol = 2 * sqr_ws->vol;
    sqr_ws->initialize_gs();
    
    r_ws = new cx_dqmc::workspace();
    r_ws->parameters(cx_dqmc_params);
    r_ws->particles = cx_dqmc_params.N;
    r_ws->eff_particles = cx_dqmc_params.n_A + 2 * cx_dqmc_params.n_B;
    r_ws->renyi = true;
    r_ws->vol = cx_dqmc_params.n_A + 2 * cx_dqmc_params.n_B;
    r_ws->large_vol = 2 * r_ws->vol;    
    r_ws->initialize_gs();

    cx_dqmc::checkerboard::graph_to_checkerboard(cx_dqmc_params, *ws);
    cx_dqmc::checkerboard::graph_to_checkerboard(cx_dqmc_params, *sqr_ws);
    cx_dqmc::checkerboard::graph_to_checkerboard_renyi(cx_dqmc_params, *r_ws);

    aux_spins.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_0.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_1.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_temp.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    randomize_spins();
    
    cout << cx_dqmc_params.slices << " Slices"
	 << "    Delta Tau: " << cx_dqmc_params.delta_tau
	 << "    b: " << cx_dqmc_params.beta
	 << "    safe: " << cx_dqmc_params.safe_mult
	 << " cut: " << cx_dqmc_params.cut_step
	 << "  (" << 0 << ", "
	 << cx_dqmc_params.N << ") <=> (" << cx_dqmc_params.n_A << ", "
	 << cx_dqmc_params.n_B << ")" << endl;

    volume_0 = cx_dqmc_params.N;
    volume_1 = 2 * cx_dqmc_params.n_B + cx_dqmc_params.n_A;

    gs_0_0 = new cx_dqmc::greens_replica_ft(&cx_dqmc_params, ws, &aux_spins);
    gs_0_1 = new cx_dqmc::greens_replica_ft(&cx_dqmc_params, ws, &aux_spins);
    gs_1 = new cx_dqmc::greens_replica_renyi_ft(&cx_dqmc_params, r_ws, sqr_ws, &aux_spins);
    
    cx_dqmc_params.replica = 0;        
    gs_0_0->initialize();
    gs_0_0->build_stack();

    cx_dqmc_params.replica = 1;
    gs_0_1->initialize();
    gs_0_1->build_stack();
    out << "Building non-renyi stacks done" << endl;

    cout << "Initializing gs_1" << endl;
    gs_1->initialize();
    cout << "Building gs_1" << endl;
    gs_1->build_stack();
    out << "Building Renyi stack done" << endl;
    
    vector<double> rands;
    rands.resize(ws->num_bonds);
    
    for (int i = 0; i < ws->num_bonds; ++i) {
	rands[i] = uniform_01();
    }

    cx_double avg_sign;
    double alpha_0 = 0;
    
    alps::ObservableSet obs;
    state = 1;

    out << "Done - sweeping" << endl;

    if (cx_dqmc_params.use_tempering == true) {
	outgoing.renyi = 2;
	outgoing.slices = cx_dqmc_params.slices;
	outgoing.sites = cx_dqmc_params.num_aux_spins;
	incoming.renyi = 2;
	incoming.slices = cx_dqmc_params.slices;
	incoming.sites = cx_dqmc_params.num_aux_spins;

	outgoing.aux_spins.resize(boost::extents[outgoing.renyi][outgoing.slices][outgoing.sites]);
	incoming.aux_spins.resize(boost::extents[incoming.renyi][incoming.slices][incoming.sites]);
    }
}


void cx_dqmc_replica_ft::init_observables(alps::Parameters const& p, alps::ObservableSet& obs) {
    std::vector<std::string> real_obs = { "Temper Label", "S2", "Swap To 1", "Swap To 0",
					  "Sign 0 0", "Sign 0 1", "Sign 1",
					  "Sign 0 0 Recalc", "Sign 0 1 Recalc", "Sign 1 Recalc",
					  "Re Phase 0 0", "Re Phase 0 1", "Re Phase 1",
					  "Re Phase 0 0 Recalc", "Re Phase 0 1 Recalc",
					  "Re Phase 0 Recalc", "Re Phase 1 Recalc",
					  "Im Phase 0 0", "Im Phase 0 1", "Im Phase 1",
					  "Im Phase 0 0 Recalc", "Im Phase 0 1 Recalc",
					  "Im Phase 0 Recalc", "Im Phase 1 Recalc",
					  "Sweep Time 0", "Sweep Time 1",
					  "Alpha 0", "Alpha 1", "Flip Rate", "Flip Rate 0",
					  "Flip Rate 1",
					  "Stability 0", "Stability 1", "Log Weight 0",
					  "Log Weight 1" };
					  

    if (cx_dqmc_params.use_tempering == true) {
	for (int i = 0; i < world.size(); ++i) {
	    for (int j = 0; j < real_obs.size(); ++j) {
		obs << alps::RealObservable(real_obs[j]
					    + std::string(" [" + dqmc::tools::to_str(i) + "]"));
	    }
	    
	    obs << alps::RealVectorObservable("SVD Stack 0 [" + dqmc::tools::to_str(i) + "]");
	    obs << alps::RealVectorObservable("SVD Stack 1 [" + dqmc::tools::to_str(i) + "]");
	    obs << alps::RealVectorObservable("Stability Slice 0 [" + dqmc::tools::to_str(i) + "]");
	    obs << alps::RealVectorObservable("Stability Slice 1 [" + dqmc::tools::to_str(i) + "]");
	}
	obs << alps::HistogramObservable<double>("N Up", 0, world.size(), 1);
	obs << alps::HistogramObservable<double>("N Down", 0, world.size(), 1);

    } else {
	for (int i = 0; i < real_obs.size(); ++i) {
	    obs << alps::RealObservable(real_obs[i]);
	}
	obs << alps::RealVectorObservable("SVD Stack 0");	
	obs << alps::RealVectorObservable("SVD Stack 1");	
    }
}


void cx_dqmc_replica_ft::switch_state(alps::ObservableSet& obs) {
    if (state == 1) {
	state = 0;

	copy_aux_spins(aux_spins, aux_spins_1);
	if (is_thermalized()) {
	    copy_aux_spins(aux_spins_0, aux_spins);
	}

	gs_0_0->build_stack();
	gs_0_1->build_stack();

	sign_0_0 = gs_0_0->get_det_sign();
	sign_0_1 = gs_0_1->get_det_sign();
	sign_0 = sign_0_0 * sign_0_1;
	    
    } else {
	state = 1;

	copy_aux_spins(aux_spins, aux_spins_0);
	if (is_thermalized()) {
	    copy_aux_spins(aux_spins_1, aux_spins);
	}

	gs_1->build_stack();
	sign_1 = gs_1->get_det_sign();
    }
}


void cx_dqmc_replica_ft::run(alps::ObservableSet& obs) {
    ++steps;
    ws->step = steps();
    
    alps::RealObsevaluator alpha_0_obs(obs["Alpha 0" + cx_dqmc_params.obs_suffix]);
    if (alpha_0_obs.count() != 0) {
    	alpha_0 = alpha_0_obs.mean();
    } else {
    	obs["Alpha 0" + cx_dqmc_params.obs_suffix] << alpha_0;
    }

    alps::RealObsevaluator alpha_1_obs(obs["Alpha 1" + cx_dqmc_params.obs_suffix]);
    if (alpha_1_obs.count() != 0) {
    	alpha_1 = alpha_1_obs.mean();
    } else {
    	obs["Alpha 1" + cx_dqmc_params.obs_suffix] << alpha_1;
    }
    alpha_0 = 1.;
    alpha_1 = 1.;
    
    gs_sweep(obs);
}


double cx_dqmc_replica_ft::work_done(alps::ObservableSet& obs ) {
    try {
	double renyi, renyi_alt;
	alps::RealObsevaluator swap_to_0(obs["Swap To 0"
					     + cx_dqmc_params.obs_suffix]);
	if (swap_to_0.count() == 0) {
	    out << "No obs for 'Swap To 0' after "
		<< steps() << " obs" << endl;
	    return 0.01; } 

	alps::RealObsevaluator swap_to_1(obs["Swap To 1"
					     + cx_dqmc_params.obs_suffix]);
	if (swap_to_1.count() == 0) {
	    out << "No obs for 'Swap To 1' after "
		<< steps() << " obs"  << endl;
	    return 0.01; } 

	alps::RealObsevaluator flip_rate(obs["Flip Rate" + cx_dqmc_params.obs_suffix]);

	if (swap_to_0.mean() == 0 || swap_to_1.mean() == 0) 
	    renyi = 0;
	else
	    renyi = 1;
	
	renyi = -1 * log  (swap_to_1.mean() 
			   / swap_to_0.mean());

	cout.setf(ios::scientific, ios::floatfield);
	cout.precision(6);

	out << setw(5) <<  cx_dqmc_params.osi;
	cout << " " 
	     << left << setw(5) << cx_dqmc_params.n_A
	     << setprecision(5) << "(" << swap_to_1.mean() << " \u00b1 "
	     << setprecision(5) << swap_to_1.error() << ")    " 
	     << setprecision(5) << "(" << swap_to_0.mean() << " \u00b1 "
	     << setprecision(5) << swap_to_0.error() << ")    " 
	     << "Result: \033[1m" << setprecision(5) << renyi << "\033[0m    "
	     << "(" << setprecision(5) << swap_to_1.count() << ", " << setprecision(5)
	     << swap_to_0.count() << ") ";
		    
	cout << "  Flips and Alphas " << setw(5) << setprecision(5);
	dqmc::tools::print_real_obs_mean(obs, "Flip Rate 0");
	cout <<  "   " << alpha_0 << "   " << endl;
	dqmc::tools::print_real_obs_mean(obs, "Flip Rate 1");
	cout <<  "   " << alpha_1 << "   " << endl;

	dqmc::tools::print_real_obs_mean(obs, "Re Phase 0 Recalc");
	dqmc::tools::print_real_obs_mean(obs, "Im Phase 0 Recalc");
	dqmc::tools::print_real_obs_mean(obs, "Re Phase 1 Recalc");
	dqmc::tools::print_real_obs_mean(obs, "Im Phase 1 Recalc");
	cout << endl;
    } catch (std::exception const& ex) {
	err << "Exception in work_done() - but hey, it's optional anyway, right?\t"
	    << ex.what() << endl;
    } catch (...) {
	err << "Exception in work_done() - but hey, it's optional anyway, right?" << endl;
    }
}


bool cx_dqmc_replica_ft::gs_sweep(alps::ObservableSet& obs) {
    // boost::timer::auto_cpu_timer t;
    
    int flip_counter = 0;
    static int replica;
    static int slice; 
    static int delay_measurement = 0;
    static int sweeps_measurement = 10;
    static int recalculated;
    static int sweep_type = 1;
    double sweep_sites = 0;
    
    if (steps() % 1000 == 0) {
	time_t timestamp;
	tm *now;
	timestamp = time(0);
	now = localtime(&timestamp);
		
	out << "Switching after " << steps()
	    << " sweeps, which is equivalent to " 
	    << (steps() / sweeps_measurement) << " - "
	    << state << " -> ";
	switch_state(obs);
	cout << state << endl;
	work_done(obs);
    }

    // cout << "Sweeping " << state << endl;

    if (state == 0) {	    
	for (uint i = 0; i < 4 * cx_dqmc_params.slices; i++) {
	    // cout << "0 Sweeping"
	    // 	 << gs_0_0->get_slice() << " / " <<  cx_dqmc_params.slices << endl;
	    flip_counter += gs_delayed_spinful_flip(obs, gs_0_0->get_slice());
	    gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);		
	   
	    flip_counter += gs_delayed_spinful_flip(obs, gs_0_1->get_slice() + cx_dqmc_params.slices);
	    gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

	    sweep_sites += 2 * cx_dqmc_params.N;
	}
    }
    else if (state == 1) {
	// cout << "Sweep" << endl;
	for (uint i = 0; i < 4 * cx_dqmc_params.slices; i++) {
	    // cout << "1 Sweeping"
	    // 	 << gs_1->get_slice() << " / " <<  2 * cx_dqmc_params.slices << endl;
	    flip_counter += gs_delayed_spinful_flip(obs, gs_1->get_slice());
	    gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);
	    sweep_sites += cx_dqmc_params.N;
	}

	// cout << flip_counter << " \t on sweep sites\t"  << sweep_sites << "\t" << 4 * cx_dqmc_params.slices * cx_dqmc_params.N << endl;
    }
    
    if (delay_measurement >= sweeps_measurement && is_thermalized()) {	    
	delay_measurement = 0;
	measure(obs);
    }

    if (is_thermalized())
	measure(obs);

    if (state == 0) {
	obs["Flip Rate 0" + cx_dqmc_params.obs_suffix]
	    << flip_counter/sweep_sites;
    } else if (state == 1) {
	obs["Flip Rate 1" + cx_dqmc_params.obs_suffix]
	    << flip_counter/sweep_sites;
    }
    ++delay_measurement;
    return false;
}

    
void cx_dqmc_replica_ft::measure(alps::ObservableSet& obs) {    
    try {
	gs_0_0->build_stack();
	gs_0_1->build_stack();
	gs_1->build_stack();

	sign_0_0 = gs_0_0->det_sign * gs_0_0->det_sign * gs_0_0->phase;
	sign_0_1 = gs_0_1->det_sign * gs_0_1->det_sign * gs_0_1->phase;
	sign_0 = sign_0_0 * sign_0_1;
	sign_1 = gs_1->det_sign * gs_1->det_sign * gs_1->phase;
	det_up_0_0 = gs_0_0->get_log_det();
	det_up_0_1 = gs_0_1->get_log_det();
	det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
	det_up_1 = 2 * gs_1->get_log_det();

	// cout << gs_0_0->phase * gs_0_1->phase << " vs. " << gs_1->phase << endl;
	// cout << det_up_0_0 << " + " << det_up_0_1
	//      << "\t vs. \t" << det_up_1 << endl;
	
	if (state == 0) {
	    obs["Sign 0 0 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0_0));
	    obs["Sign 0 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0_1));

	    obs["Re Phase 0 0 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0_0));
	    obs["Re Phase 0 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0_1));
	    obs["Re Phase 0 Recalc" + cx_dqmc_params.obs_suffix]
	    	<< double(std::real(sign_0));

	    obs["Im Phase 0 0 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(sign_0_0));
	    obs["Im Phase 0 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(sign_0_1));
	    obs["Im Phase 0 Recalc" + cx_dqmc_params.obs_suffix]
	    	<< double(std::imag(sign_0));	    
	}
	else if (state == 1) {
	    obs["Sign 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_1));

	    obs["Re Phase 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_1));
	    obs["Im Phase 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(sign_1));
	}

	if (state == 0) {	    
	    log_ratio = - det_up_0 + det_up_1;
	    ratio = exp(log_ratio);		
	    obs["Swap To 1" + cx_dqmc_params.obs_suffix]
		<< min(1., ratio);
	} else {
	    // cout << "Meas" << endl;
	    log_ratio = det_up_0 - det_up_1;
	    ratio = exp(log_ratio);		
	    obs["Swap To 0" + cx_dqmc_params.obs_suffix]
		<< min(1., ratio);
	}       
    }
    catch(std::exception& ex) {
	cout << "Error while trying to measure" << endl
	     << ex.what() << endl;
    }   
}

int cx_dqmc_replica_ft::gs_delayed_spinful_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    int flips_accepted = 0;
    cx_double avg_sign = 0;

    vector<double> rands;
    rands.resize(ws->num_sites);
    
    for (int i = 0; i < ws->num_sites; ++i) {
	rands[i] = uniform_01();
    }

    if (state == 0) {
	if (rep == 0) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
								rep, rep_slice,
								gs_0_0, rands, alpha_0,
								sign_0_0, avg_sign, flips_accepted);
	} else if (rep == 1) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
								rep, rep_slice,
								gs_0_1, rands, alpha_0,
								sign_0_1, avg_sign, flips_accepted);
	}	    
    }
    else if (state == 1) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, rep, 
							    rep_slice,
							    gs_1, rands, alpha_0,
							    sign_1, avg_sign,
							    flips_accepted);
    }
    return flips_accepted;
}


int cx_dqmc_replica_ft::gs_spinful_naive_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    int flips_accepted = 0;
    cx_double avg_sign = 0;

    vector<double> rands;
    rands.resize(ws->num_sites);
    
    for (int i = 0; i < ws->num_sites; ++i) {
	rands[i] = uniform_01();
    }

    if (state == 0) {
	if (rep == 0) {
	    flips_accepted = cx_dqmc::update::spinful_naive_flip(obs, aux_spins, state, 
								rep, rep_slice,
								gs_0_0, rands, alpha_0,
								sign_0_0, avg_sign, flips_accepted);
	} else if (rep == 1) {
	    flips_accepted = cx_dqmc::update::spinful_naive_flip(obs, aux_spins, state, 
								rep, rep_slice,
								gs_0_1, rands, alpha_0,
								sign_0_1, avg_sign, flips_accepted);
	}	    
    }
    else if (state == 1) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	flips_accepted = cx_dqmc::update::spinful_naive_flip(obs, aux_spins, state, rep, 
							    rep_slice,
							    gs_1, rands, alpha_0,
							    sign_1, avg_sign,
							    flips_accepted);
    }
    return flips_accepted;
}
