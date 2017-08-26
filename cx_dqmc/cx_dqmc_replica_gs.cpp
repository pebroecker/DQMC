#include "cx_dqmc_replica_gs.hpp"

using namespace dqmc::la;

cx_dqmc_replica_gs::cx_dqmc_replica_gs( alps::Parameters const& p) 
    : alps::parapack::mc_worker(p), cx_dqmc_params(p), steps(p), dummy(alps::RealObservable("dummy")) {
    rank = -1;
    if (global::use_mpi) rank = world.rank();
    out.SetRankPrefix("worker", rank);
    start(p);
    cx_dqmc_params.rank = rank;
}


cx_dqmc_replica_gs::cx_dqmc_replica_gs(boost::mpi::communicator const& comm,  alps::Parameters const& p) 
	: alps::parapack::mc_worker(p),
	cx_dqmc_params(p, true, comm.rank()),
	steps(p), world(comm),
	  dummy(alps::RealObservable("dummy")) {    
    if (global::use_mpi) rank = world.rank();
    out.SetRankPrefix("worker", rank);
    start(p);    
    cx_dqmc_params.rank = rank;
}


void cx_dqmc_replica_gs::start(const alps::Parameters& p) {

    print_memory_usage();

    ws = new cx_dqmc::workspace();    
    cx_dqmc_params.particles = cx_dqmc_params.real_particles;
    ws->parameters(cx_dqmc_params);
    ws->particles = cx_dqmc_params.real_particles;
    ws->eff_particles = ws->particles;
    ws->large_vol = 2 * ws->vol;
    ws->initialize_gs();

    print_memory_usage();
    sqr_ws = new cx_dqmc::workspace();    
    cx_dqmc_params.particles = cx_dqmc_params.real_particles;
    sqr_ws->parameters(cx_dqmc_params);
    print_memory_usage();
    sqr_ws->particles = cx_dqmc_params.real_particles;
    sqr_ws->eff_particles = sqr_ws->particles;
    sqr_ws->large_vol = 2 * sqr_ws->vol;
    sqr_ws->initialize_gs();
    print_memory_usage();
    
    r_ws = new cx_dqmc::workspace();
    r_ws->parameters(cx_dqmc_params);
    r_ws->renyi = true;
    cx_dqmc_params.particles = cx_dqmc_params.real_particles;
    cx_dqmc_params.eff_particles = cx_dqmc_params.real_particles + cx_dqmc_params.n_B;
    r_ws->particles = cx_dqmc_params.real_particles;
    r_ws->eff_particles = ws->particles + cx_dqmc_params.n_B;
    r_ws->vol = cx_dqmc_params.n_A + 2 * cx_dqmc_params.n_B;
    r_ws->large_vol = 2 * r_ws->vol;
    print_memory_usage();
    r_ws->initialize_gs();

    if (cx_dqmc_params.partial_cuts == true) {
	r_ws_0 = new cx_dqmc::workspace();
	// if (cx_dqmc_params.rank < 1)
	//     out << cx_dqmc_params.particles << " - "
	// 	<< cx_dqmc_params.n_A_0 << 
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;	
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
	r_ws_0->parameters(cx_dqmc_params);
	r_ws_0->renyi = true;
	cx_dqmc_params.particles = cx_dqmc_params.real_particles;
	cx_dqmc_params.eff_particles = cx_dqmc_params.real_particles + cx_dqmc_params.n_B;
	r_ws_0->particles = cx_dqmc_params.real_particles;
	r_ws_0->eff_particles = ws->particles + cx_dqmc_params.n_B;
	r_ws_0->vol = cx_dqmc_params.n_A + 2 * cx_dqmc_params.n_B;
	r_ws_0->large_vol = 2 * r_ws_0->vol;    
	if (cx_dqmc_params.rank < 2) out << "Initializing addition workspace" << endl;
	r_ws_0->initialize_gs();
	
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
	cx_dqmc_params.particles = cx_dqmc_params.real_particles;
	cx_dqmc_params.eff_particles = cx_dqmc_params.real_particles + cx_dqmc_params.n_B;
	if (cx_dqmc_params.rank < 2) out << "initialization done" << endl;
    }


    if(cx_dqmc_params.rank < 2) out << "Density matrices" << endl;

    if (cx_dqmc_params.complex_hoppings == false) {
	cx_dqmc::density::density_matrix(cx_dqmc_params, *ws);
	cx_dqmc::density::density_matrix(cx_dqmc_params, *sqr_ws);
	cx_dqmc::density::density_matrix(cx_dqmc_params, *r_ws);
    }
    else {
	cx_dqmc::density::cx_density_matrix(cx_dqmc_params, *ws);
	cx_dqmc::density::cx_density_matrix(cx_dqmc_params, *sqr_ws);
	cx_dqmc::density::cx_density_matrix(cx_dqmc_params, *r_ws);
    }


    if (cx_dqmc_params.partial_cuts == true) {
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
	cx_dqmc::density::density_matrix(cx_dqmc_params, *r_ws_0);
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;

    }
    
    if(cx_dqmc_params.rank < 2) out << "Checkerboard" << endl;
    cx_dqmc::checkerboard::graph_to_checkerboard(cx_dqmc_params, *ws);
    cx_dqmc::checkerboard::graph_to_checkerboard(cx_dqmc_params, *sqr_ws);
    cx_dqmc::checkerboard::graph_to_checkerboard_renyi(cx_dqmc_params, *r_ws);


    if(cx_dqmc_params.rank < 2) out << "Hopping matrices" << endl;
    cx_dqmc::hopping::hopping_matrix(cx_dqmc_params, *ws, ws->re_mat_1,
				     ws->re_mat_2,
				     ws->mat_1, ws->mat_2);

    if (cx_dqmc_params.complex_hoppings == false) {
	cx_dqmc::checks::compare_hopping_checkerboard(cx_dqmc_params, *ws,
						      ws->re_mat_1,
						      ws->re_mat_2,
						      ws->mat_3, ws->mat_4);
    } else {
	cx_dqmc::checks::compare_hopping_checkerboard(cx_dqmc_params, *ws,
						      ws->mat_1,
						      ws->mat_2,
						      ws->mat_3, ws->mat_4);
    }

    cx_dqmc::hopping::hopping_matrix_renyi(cx_dqmc_params, *r_ws, r_ws->re_mat_1,
					   r_ws->re_mat_2, 1);
    cx_dqmc::checks::compare_hopping_checkerboard_renyi(cx_dqmc_params, *r_ws,
							r_ws->re_mat_1,
							r_ws->re_mat_2,
							r_ws->mat_3, 
							r_ws->mat_4);

    if (cx_dqmc_params.partial_cuts == true) {
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
	cx_dqmc::checkerboard::graph_to_checkerboard_renyi(cx_dqmc_params, *r_ws_0);
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
    }

    // ws->mat_1.setIdentity();
    // cx_dqmc::checkerboard::hop_right(ws, ws->mat_1, 1);
    // cout << "Some hopping matrix" << endl << ws->mat_1 << endl << endl;

    // cx_dqmc::checkerboard::hop_right(ws, ws->mat_1, -1);
    // cout << "Some identity matrix" << endl << ws->mat_1 << endl << endl;
    
    aux_spins.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_0.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_1.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);
    aux_spins_temp.resize(boost::extents[2][cx_dqmc_params.slices][cx_dqmc_params.num_aux_spins]);

    randomize_spins();

    if (cx_dqmc_params.rank < 1) {
	out << cx_dqmc_params.slices << " Slices"
	    << "    Delta Tau: " << cx_dqmc_params.delta_tau
	    << "    b: " << cx_dqmc_params.beta
	    << "    safe: " << cx_dqmc_params.safe_mult
	    << " cut: " << cx_dqmc_params.cut_step;

	if (cx_dqmc_params.partial_cuts == false) {
	    cout << "  (" << 0 << ", "
		 << cx_dqmc_params.N << ") <=> (" << cx_dqmc_params.n_A << ", "
		 << cx_dqmc_params.n_B << ")\t"
		 << cx_dqmc_params.measure_continuously << endl;
	} else {
	    cout << "  (" << cx_dqmc_params.n_B_0 << ", "
		 << cx_dqmc_params.n_A_0 << ") <=> ("
		 << cx_dqmc_params.n_B_1 << ", "
		 << cx_dqmc_params.n_A_1 << ")\t"
		 << cx_dqmc_params.measure_continuously << endl;	
	}
    }
    
    gs_0_0 = new cx_dqmc::greens_replica_gs(&cx_dqmc_params, ws, &aux_spins);
    gs_0_1 = new cx_dqmc::greens_replica_gs(&cx_dqmc_params, ws, &aux_spins);
    gs_1 = new cx_dqmc::greens_replica_renyi_gs(&cx_dqmc_params, r_ws, sqr_ws, &aux_spins);

    cx_dqmc_params.replica = 0;       
    out << "gs_0_0->initialize()" << endl;
    gs_0_0->initialize();
    out << "gs_0_0->build_stack()" << endl;
    gs_0_0->build_stack();

    cx_dqmc_params.replica = 1;
    out << "gs_0_1->initialize()" << endl;
    gs_0_1->initialize();
    gs_0_1->build_stack();
    gs_0_1->save_curr_det();

    gs_1->initialize();
    gs_1->build_stack();
    copy_to_gs_1();

    if (cx_dqmc_params.partial_cuts == true) {
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;	
	gs_0 = new cx_dqmc::greens_replica_renyi_gs(&cx_dqmc_params, r_ws_0, sqr_ws, &aux_spins);
	gs_0->initialize();
	gs_0->build_stack();
	copy_to_gs_0();	
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
    }


    out << "Done building" << endl;
    print_memory_usage();

    out << "Testing sweep stability" << endl;
    {
	boost::timer::auto_cpu_timer t;
	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    gs_0_0->propagate(dummy);
	    gs_0_1->propagate(dummy);
	}
    }
    print_memory_usage();
    
    {
	boost::timer::auto_cpu_timer t;
	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    gs_1->propagate(dummy);
	    gs_1->propagate(dummy);
	}
    }
    print_memory_usage();

    if (cx_dqmc_params.partial_cuts == true) {
	{
	    cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	    cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;	
	    
	    boost::timer::auto_cpu_timer t;
	    for (uint i = 0; i < cx_dqmc_params.slices; i++) {
		gs_0->propagate(dummy);
		gs_0->propagate(dummy);
	    }
	}
	cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;	
    }
    // cout << "Propagate\t" << gs_1->get_slice() << endl;
    // gs_1->propagate(dummy);
    // cout << "Propagate\t" << gs_1->get_slice() << endl;

    state = 0;

    if (cx_dqmc_params.simple_adapt_mu == true || cx_dqmc_params.adapt_mu == false) {
	state = 1;
    }
    state = 1;
    
    // out << "Using mu " << cx_dqmc_params.mu_site << endl;
    out << "Done - sweeping in state " << state << endl;
    // out << gs_0_0->current_slice << " " << gs_1->current_slice << endl;
    if (cx_dqmc_params.use_tempering == true
	|| cx_dqmc_params.use_internal_tempering == true) {
	outgoing.renyi = 2;
	outgoing.slices = cx_dqmc_params.slices;
	outgoing.sites = cx_dqmc_params.num_aux_spins;
	incoming.renyi = 2;
	incoming.slices = cx_dqmc_params.slices;
	incoming.sites = cx_dqmc_params.num_aux_spins;

	outgoing.aux_spins.resize(boost::extents[outgoing.renyi][outgoing.slices][outgoing.sites]);
	incoming.aux_spins.resize(boost::extents[incoming.renyi][incoming.slices][incoming.sites]);
    }

    if (cx_dqmc_params.use_tempering == false
	&& cx_dqmc_params.simple_adapt_mu == true
	&& global::use_mpi == true) {
	double old_mu_site = cx_dqmc_params.mu_site;
	double new_mu_site;

	for (int i = 0; i < world.size(); ++i) {
	    if (i == cx_dqmc_params.rank) {
		boost::mpi::broadcast(world, old_mu_site, i);
	    } else {
		boost::mpi::broadcast(world, new_mu_site, i);
	    }
	    cx_dqmc_params.mu_site = std::max(cx_dqmc_params.mu_site,
					      new_mu_site);
	}
    }
	
    out << "Using mu " << cx_dqmc_params.mu_site << endl;
    first_run = true;
}


void cx_dqmc_replica_gs::init_observables(alps::Parameters const& p, alps::ObservableSet& obs) {
    std::vector<std::string> real_obs = { "Temper Label", "S2", "Swap To 1", "Swap To 0",
					  "Cont Swap To 1", "Cont Swap To 0",
					  "Sign 0 0", "Sign 0 1", "Sign 0", "Sign 1",
					  "Sign 0 0 Recalc", "Sign 0 1 Recalc", "Sign 0 Recalc", "Sign 1 Recalc",
					  "Re Phase 0 0", "Re Phase 0 1", "Re Phase 0", "Re Phase 1",
					  "Re Phase 0 0 Recalc", "Re Phase 0 1 Recalc",
					  "Re Phase 0 Recalc", "Re Phase 1 Recalc",
					  "Im Phase 0 0", "Im Phase 0 1", "Im Phase 0", "Im Phase 1",
					  "Im Phase 0 0 Recalc", "Im Phase 0 1 Recalc",
					  "Im Phase 0 Recalc", "Im Phase 1 Recalc",
					  "Sweep Time 0", "Sweep Time 1",
					  "Alpha 0", "Alpha 1", "Flip Rate", "Flip Rate 0",
					  "Flip Rate 1",
					  "Stability 0", "Stability 1", "Log Weight 0",
					  "Log Weight 1", "Internal to 1", "Internal to 0", "Mu Adapt" };
					  
    std::vector<std::string> real_vector_obs = { "SVD Stack 0", "SVD Stack 1", "SVD Stack Example 0",
						 "SVD Stack Example 1", "Stability Slice 0", "Stability Slice 1",
						 "Log Det Cont Example 0 0", "Log Det Cont Example 0 1",
						 "Log Det Cont Example 1 0", "Log Det Cont Example 1 1",
						 "Log Det Meas Example 0 0", "Log Det Meas Example 0 1"
						 "Log Det Meas Example 1 0", "Log Det Meas Example 1 1"
    };
    
    if (cx_dqmc_params.use_tempering == true) {
	for (int i = 0; i < world.size(); ++i) {
	    for (int j = 0; j < real_obs.size(); ++j) {
		obs << alps::RealObservable(real_obs[j]
					    + std::string(" [" + dqmc::tools::to_str(i) + "]"));
	    }

	    for (int j = 0; j < real_vector_obs.size(); ++j) {
		obs << alps::RealVectorObservable(real_vector_obs[j]
						  + std::string(" [" + dqmc::tools::to_str(i) + "]"));
	    }
	}
	obs << alps::HistogramObservable<double>("N Up", 0, world.size(), 1);
	obs << alps::HistogramObservable<double>("N Down", 0, world.size(), 1);

    } else {
	for (int i = 0; i < real_obs.size(); ++i) { obs << alps::RealObservable(real_obs[i]); }
	for (int i = 0; i < real_vector_obs.size(); ++i) { obs << alps::RealVectorObservable(real_vector_obs[i]); }
    }
}


void cx_dqmc_replica_gs::switch_state(alps::ObservableSet& obs) {
    if (state == 1) {
	try {
	    alps::RealObsevaluator swap_to_1(obs["Swap To 1" + cx_dqmc_params.obs_suffix]);
	    if (fabs(swap_to_1.mean() - 1) < 1e-8 && swap_to_1.count() > 20
		&& cx_dqmc_params.use_tempering == false) {
		out << "Not switching to 0" << endl;
	    }
	} catch(std::exception& ex) {
	    out << "No values for swap_to_1" << endl
		 << ex.what() << endl;
	}   
		
	state = 0;

	copy_aux_spins(aux_spins, aux_spins_1);
	copy_aux_spins(aux_spins_0, aux_spins);

	gs_0_0->build_stack();
	gs_0_1->build_stack();
	
	if (cx_dqmc_params.partial_cuts == true) {
	    cx_dqmc_params.n_A = cx_dqmc_params.n_A_0;
	    cx_dqmc_params.n_B = cx_dqmc_params.n_B_0;
	    copy_to_gs_0();
	    sign_1 = gs_0->get_det_sign();
	}
	else {
	    sign_0_0 = gs_0_0->get_det_sign();
	    sign_0_1 = gs_0_1->get_det_sign();
	    sign_0 = sign_0_0 * sign_0_1;
	}	    
    } else {
	try {
	    alps::RealObsevaluator swap_to_0(obs["Swap To 0" + cx_dqmc_params.obs_suffix]);
	    if (fabs(swap_to_0.mean() - 1) < 1e-8 && swap_to_0.count() > 20
		&& cx_dqmc_params.use_tempering == false) {
		out << "Not switching to 1" << endl;
		// return;
	    }
	} catch(std::exception& ex) {
	    out << "No values for swap_to_0" << endl
		 << ex.what() << endl;
	}   
	state = 1;

	copy_aux_spins(aux_spins, aux_spins_0);
	copy_aux_spins(aux_spins_1, aux_spins);

	gs_0_0->build_stack();
	gs_0_1->build_stack();


	copy_to_gs_1();
	sign_1 = gs_1->get_det_sign();

	if (cx_dqmc_params.partial_cuts == true) {
	    cx_dqmc_params.n_A = cx_dqmc_params.n_A_1;
	    cx_dqmc_params.n_B = cx_dqmc_params.n_B_1;
	}
    }
    out << "Switched states" << endl;
}


void cx_dqmc_replica_gs::run(alps::ObservableSet& obs) {
    if (ws->num_sites > 1000) {
	out << steps() << endl;
    }

    // cout << "Quick check" << endl;
    // {
    // 	boost::timer::auto_cpu_timer t;
    // 	for (uint i = 0; i < 2* cx_dqmc_params.slices; i++) {
    // 	    gs_1->propagate(dummy);
    // 	    gs_delayed_attractive_spinless_flip(obs, gs_1->get_slice());
    // 	}
    // }

    // {
    // 	boost::timer::auto_cpu_timer t;
    // 	for (uint i = 0; i < 2* cx_dqmc_params.slices; i++) {
    // 	    gs_1->propagate(dummy);
    // 	    gs_attractive_spinless_flip(obs, gs_1->get_slice());
    // 	}
    // }

    if (cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
	if (first_run == true || steps() == cx_dqmc_params.thermalization / 2 - 1) {
	    first_run = false;
	    alps::RealObsevaluator mu_obs(obs["Mu Adapt" + cx_dqmc_params.obs_suffix]);
	    out << "Trying to reset mu" << endl;
	    if (mu_obs.count() != 0) {
		out << "Average is " << mu_obs.mean() << " from " << mu_obs.count()
		    << " measurements" << endl;
		// out << "Max is " << mu_obs.max() << endl;
		cx_dqmc_params.mu_site = double(int(mu_obs.mean() * 1000)) / 1000.;
		out << "Resetting mu " << cx_dqmc_params.mu_site << endl;
		cx_dqmc_params.mu_site_factor[0] = exp(-cx_dqmc_params.delta_tau * cx_dqmc_params.mu_site);
		cx_dqmc_params.mu_site_factor[1] = NAN;
		cx_dqmc_params.mu_site_factor[2] = exp(cx_dqmc_params.delta_tau * cx_dqmc_params.mu_site);

		if (state == 0) {
		    gs_0_0->build_stack();
		    gs_0_1->build_stack();

		    out << "Result " << gs_0_0->d_stack[0][cx_dqmc_params.particles - 1] << endl;
		    out << "Result " << gs_0_0->d_stack[gs_0_0->chunks - 1][cx_dqmc_params.particles - 1] << endl;
		} else if (state == 1) {
		    gs_1->build_stack();
		}	    
	    }
	}
    }

    ++steps;
    ws->step = steps();
    r_ws->step = steps();
    r_ws_0->step = steps();
	    
	
    alps::RealObsevaluator alpha_0_obs(obs["Alpha 0" + cx_dqmc_params.obs_suffix]);
    if (alpha_0_obs.count() != 0) {
    	alpha_0 = alpha_0_obs.mean();
    } else {
	for (int i = 0; i < 100; ++i) 
	    obs["Alpha 0" + cx_dqmc_params.obs_suffix] << alpha_0;
    }

    alps::RealObsevaluator alpha_1_obs(obs["Alpha 1" + cx_dqmc_params.obs_suffix]);
    if (alpha_1_obs.count() != 0) {
    	alpha_1 = alpha_1_obs.mean();
    } else {
	for (int i = 0; i < 100; ++i) 
	    obs["Alpha 1" + cx_dqmc_params.obs_suffix] << alpha_1;
    }
    alpha_0 = 1.;
    alpha_1 = 1.;
    // {
    // 	boost::timer::auto_cpu_timer t;
    // 	cout << steps();

    if (cx_dqmc_params.measure_continuously == true) {
	if (cx_dqmc_params.partial_cuts == true) {
	    dqmc::tools::abort("Partial cuts and continuous sweeping are not supported");
	}
	// cout << "Continuous Sweep\t" << gs_0_0->get_slice() << " " << gs_0_1->get_slice() << " " << gs_1->get_slice() << endl;
	// gs_sweep_continuously_slice(obs);
	// gs_sweep_continuously_chunk(obs);
	gs_sweep_continuously_complete(obs);
    }
    else {
	gs_sweep(obs);
    }
    
    if (is_thermalized() && cx_dqmc_params.adapt_mu == true
	&& cx_dqmc_params.simple_adapt_mu == false) {
    }

    static int temper_direction = (cx_dqmc_params.rank % 2 == 0) ? 1 : -1;

    if (cx_dqmc_params.use_internal_tempering == true) {
	if (steps() % 10 != 0 || cx_dqmc_params.rank == -1) {
	    return;
	}

	int exchange_partner = cx_dqmc_params.rank + temper_direction;

	if (exchange_partner == -1) {
	    exchange_partner = world.size() - 1;
	} else if (exchange_partner == world.size()) {
	    exchange_partner = 0;
	}

	calculate_weights(false);
	copy_aux_spins(aux_spins, outgoing.aux_spins);
	
	if (cx_dqmc_params.rank + 1 == world.size()) {
	    outgoing.direction = -1;
	} else if (cx_dqmc_params.rank == 0) {
	    outgoing.direction = 1;
	}

	double initial_weight = (state == 0)
	    ? det_up_0 : det_up_1;

	if (cx_dqmc_params.rank < exchange_partner) {
	    world.send(exchange_partner, 0, outgoing);
	    world.recv(exchange_partner, 0, incoming);
	} else {
	    world.recv(exchange_partner, 0, incoming);
	    world.send(exchange_partner, 0, outgoing);	    
	}
	// double initial_weight = outgoing.weight;

	copy_aux_spins(incoming.aux_spins, aux_spins);
	calculate_weights(true);
	
	double final_weight = (state == 0)
	    ? det_up_0 : det_up_1;

	if (cx_dqmc_params.rank < exchange_partner) {
	    world.send(exchange_partner, 0, outgoing);
	    world.recv(exchange_partner, 0, incoming);
	} else {
	    world.recv(exchange_partner, 0, incoming);
	    world.send(exchange_partner, 0, outgoing);	    
	}
	double ratio = final_weight - initial_weight;
	bool decision;

	// out << exchange_partner << " Using this @ " << exp(final_weight - initial_weight) << endl;
	decision =  (uniform_01() < exp(final_weight - initial_weight))
	    ? decision = true
	    : decision = false;
	
	if (decision == false) {
	    copy_aux_spins(outgoing.aux_spins, aux_spins);
	    calculate_weights();
	} 
	temper_direction *= -1;
    }
    
    if (cx_dqmc_params.use_tempering == true) {
	if (steps() % cx_dqmc_params.temper_freq == 0) {
	    int exchange_partner = cx_dqmc_params.rank + temper_direction;

	    if (exchange_partner == -1) {
		exchange_partner = world.size() - 1;
	    } else if (exchange_partner == world.size()) {
		exchange_partner = 0;
	    }

	    calculate_weights();
	    copy_aux_spins(aux_spins, outgoing.aux_spins);
	
	    if (cx_dqmc_params.rank + 1 == world.size()) {
		outgoing.direction = -1;
	    } else if (cx_dqmc_params.rank == 0) {
		outgoing.direction = 1;
	    }

	    if (state == 0) { outgoing.weight = det_up_0 + det_down_0;
	    } else if (state == 1) { outgoing.weight = det_up_1 + det_up_1;	}
	
	    if (cx_dqmc_params.rank < exchange_partner) {
		world.send(exchange_partner, 0, outgoing);
		world.recv(exchange_partner, 0, incoming);
	    } else {
		world.recv(exchange_partner, 0, incoming);
		world.send(exchange_partner, 0, outgoing);	    
	    }
	    double initial_weight = outgoing.weight + incoming.weight;

	    copy_aux_spins(incoming.aux_spins, aux_spins);
	    calculate_weights();
	
	    if (state == 0) { outgoing.weight = det_up_0 + det_down_0;
	    } else if (state == 1) { outgoing.weight = det_up_1 + det_up_1;	}
	    if (cx_dqmc_params.rank < exchange_partner) {
		world.send(exchange_partner, 0, outgoing);
		world.recv(exchange_partner, 0, incoming);
	    } else {
		world.recv(exchange_partner, 0, incoming);
		world.send(exchange_partner, 0, outgoing);	    
	    }

	    double final_weight = outgoing.weight + incoming.weight;
	    double ratio = final_weight - initial_weight;
	    bool decision;

	    if (cx_dqmc_params.rank < exchange_partner) {
		world.recv(exchange_partner, 0, decision);
	    } else {
		if (uniform_01() < exp(final_weight - initial_weight)) {
		    decision = true;
		} else {
		    decision = false;
		}
		world.send(exchange_partner, 0, decision);	    
	    }
	
	    if (decision == false) {
		copy_aux_spins(outgoing.aux_spins, aux_spins);
		calculate_weights();
	    } else {
		if (incoming.direction != 0) {
		    outgoing.direction = incoming.direction; }
	    }

	    if (outgoing.direction == 1) {
		if (!obs.has("N Up")) {
		    out << "Added histogram observable N Up" << endl;
		    obs << alps::HistogramObservable<double>("N Up", 0, world.size() - 1, 1); }
		// out << "up package" << endl;
		obs["N Up"] << double(cx_dqmc_params.rank);
	    } else if (outgoing.direction == -1) {
		if (!obs.has("N Up")) {
		    out << "Added histogram observable N Down" << endl;
		    obs << alps::HistogramObservable<double>("N Down", 0, world.size() - 1, 1); }
		obs["N Down"] << double(cx_dqmc_params.rank);
	    }
	    temper_direction *= -1;
	}
    }
}


double cx_dqmc_replica_gs::work_done(alps::ObservableSet& obs ) {
    try {
	double renyi, renyi_alt;

	try {
	    alps::RealObsevaluator cont_swap_to_0(obs["Cont Swap To 0"
						      + cx_dqmc_params.obs_suffix]);
	    alps::RealObsevaluator cont_swap_to_1(obs["Cont Swap To 1"
						      + cx_dqmc_params.obs_suffix]);

	    if (cont_swap_to_0.count() != 0 && cont_swap_to_1.count() != 0) {
		if (cont_swap_to_0.mean() == 0 || cont_swap_to_1.mean() == 0) 
		    renyi = 0;
		else
		    renyi = 1;
	    
		renyi = -1 * log  (cont_swap_to_1.mean() 
				   / cont_swap_to_0.mean());
		out << setw(5) <<  cx_dqmc_params.osi;
		cout << " " 
		     << left << "Cont" << setw(5) << cx_dqmc_params.n_A
		     << setprecision(5) << "(" << cont_swap_to_1.mean() << " \u00b1 "
		     << setprecision(5) << cont_swap_to_1.error() << ")    " 
		     << setprecision(5) << "(" << cont_swap_to_0.mean() << " \u00b1 "
		     << setprecision(5) << cont_swap_to_0.error() << ")    " 
		     << "Result: \033[1m" << setprecision(5) << renyi << "\033[0m    "
		     << "(" << setprecision(5) << cont_swap_to_1.count() << ", " << setprecision(5)
		     << cont_swap_to_0.count() << ") " << endl;

	    }
	} catch (std::exception const& ex) {
	    cout << ex.what() << endl; }

	alps::RealObsevaluator swap_to_0(obs["Swap To 0"
					     + cx_dqmc_params.obs_suffix]);
	alps::RealObsevaluator swap_to_1(obs["Swap To 1"
					     + cx_dqmc_params.obs_suffix]);

	if (swap_to_0.count() != 0 && swap_to_1.count() != 0) {
	    if (swap_to_0.mean() == 0 || swap_to_1.mean() == 0) 
		renyi = 0;
	    else
		renyi = 1;
	    
	    renyi = -1 * log  (swap_to_1.mean() 
			       / swap_to_0.mean());
	    out << setw(5) <<  cx_dqmc_params.osi;
	    cout << " " 
		 << left << "Swap" << setw(5) << cx_dqmc_params.n_A
		 << setprecision(5) << "(" << swap_to_1.mean() << " \u00b1 "
		 << setprecision(5) << swap_to_1.error() << ")    " 
		 << setprecision(5) << "(" << swap_to_0.mean() << " \u00b1 "
		 << setprecision(5) << swap_to_0.error() << ")    " 
		 << "Result: \033[1m" << setprecision(5) << renyi << "\033[0m    "
		 << "(" << setprecision(5) << swap_to_1.count() << ", " << setprecision(5)
		 << swap_to_0.count() << ") " << endl;
	}


	else {
	    renyi = 0.;
	}


	alps::RealObsevaluator flip_rate(obs["Flip Rate" + cx_dqmc_params.obs_suffix]);

	cout.setf(ios::scientific, ios::floatfield);
	cout.precision(6);
		    
	out << "  Flips and Alphas " << setw(5) << setprecision(5);
	dqmc::tools::print_real_obs_mean(obs, "Flip Rate 0" + cx_dqmc_params.obs_suffix );
	dqmc::tools::print_real_obs_mean(obs, "Flip Rate 1" + cx_dqmc_params.obs_suffix);
	// cout <<  "   " << alpha_1 << "   " << endl;
	cout << endl;

	out << "Ph Rec\t";
	dqmc::tools::print_real_obs_mean(obs, "Re Phase 0 Recalc" + cx_dqmc_params.obs_suffix);
	dqmc::tools::print_real_obs_mean(obs, "Re Phase 1 Recalc" + cx_dqmc_params.obs_suffix);
	cout << endl;

	out << "Phase\t";
	dqmc::tools::print_real_obs_mean(obs, "Re Phase 0"+ cx_dqmc_params.obs_suffix);
	dqmc::tools::print_real_obs_mean(obs, "Re Phase 1" + cx_dqmc_params.obs_suffix);
	cout << endl;
	
	// out << "Weight\t";
	// dqmc::tools::print_real_obs_mean(obs, "Log Weight 0");
	// dqmc::tools::print_real_obs_mean(obs, "Log Weight 1");
	// cout << endl;

	// out << "Internal exchange\t";
	// dqmc::tools::print_real_obs_mean(obs, "Internal to 0");
	// dqmc::tools::print_real_obs_mean(obs, "Internal to 1");
	// cout << endl;
    } catch (std::exception const& ex) {
	err << "Exception in work_done() - but hey, it's optional anyway, right?\t"
	    << ex.what() << endl;
    } catch (...) {
	err << "Exception in work_done() - but hey, it's optional anyway, right?" << endl;
    }
}


bool cx_dqmc_replica_gs::gs_sweep(alps::ObservableSet& obs) {
    // cout << "Sweep" << endl;
    // boost::timer::auto_cpu_timer t;
    
    double flip_counter = 0;
    static int replica;
    static int slice; 
    static int delay_measurement = 0;
    static int sweeps_measurement = (cx_dqmc_params.use_tempering == true)
	? cx_dqmc_params.temper_freq : 1;
    sweeps_measurement = 1;
    static int recalculated;
    static int sweep_type = 1;
    double sweep_slices = 0;
    std::vector<int> swept_on;
    for (int i = 0; i < 2 * cx_dqmc_params.slices; ++i) { swept_on.push_back(0); }
	
    int switch_frequency = (cx_dqmc_params.N < 100 ) ? 1000 : 100;
    
    if ((steps() % switch_frequency == 0 && is_thermalized())
	|| steps() == cx_dqmc_params.thermalization/2 && !is_thermalized()) {
	time_t timestamp;
	tm *now;
	timestamp = time(0);
	now = localtime(&timestamp);
		
	out << "Switching after " << steps()
	    << " sweeps, which is equivalent to " 
	    << (steps() / sweeps_measurement) << " - "
	    << state << endl;
	switch_state(obs);
	// cout << state << endl;
	work_done(obs);
    }

    if (state == 0 && cx_dqmc_params.partial_cuts == false) {
	if (gs_0_0->get_slice() != 0 || gs_0_1->get_slice() != 0)
	    dqmc::tools::abort("gs_0_0 or gs_0_1 not on slice 0");

	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    ++swept_on[gs_0_0->get_slice()];
	    ++swept_on[gs_0_0->get_slice() + cx_dqmc_params.slices];
	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		flip_counter += gs_delayed_spinful_flip(obs, gs_0_0->get_slice());
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		if (cx_dqmc_params.nni < 0) {
		    flip_counter += gs_delayed_attractive_spinless_flip(obs, gs_0_0->get_slice());
		} else {
		    flip_counter += gs_delayed_spinless_flip(obs, gs_0_0->get_slice());
		}
	    } else { dqmc::tools::abort("Unknown update type"); }

	    int ret = gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);		

	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
	    }
	    
	    ret = gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

	    if (ret == 3 && steps() > 500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);
	    }
	    if (gs_0_0->get_slice() == 0) break;
	    ++sweep_slices;
	    ++sweep_slices;
	}
    }

    else if (state == 0 && cx_dqmc_params.partial_cuts == true) {
	if (gs_0->get_slice() != 0)
	    dqmc::tools::abort("gs_0 not on slice 0");

	// out << "Propagating 0" << endl;
	// out << cx_dqmc_params.N << " " << cx_dqmc_params.n_A << " " << cx_dqmc_params.n_B << endl;
	// out << r_ws_0->vol << " " << gs_0->vol << endl;
	for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
	    ++swept_on[gs_0->get_slice()];
	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		flip_counter += gs_delayed_spinful_flip(obs, gs_0->get_slice());
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		if (cx_dqmc_params.nni < 0) {
		    flip_counter += gs_delayed_attractive_spinless_flip(obs, gs_0->get_slice());
		    // flip_counter += gs_attractive_spinless_flip(obs, gs_0->get_slice());
		} else {
		    flip_counter += gs_delayed_spinless_flip(obs, gs_0->get_slice());
		}
	    } else { dqmc::tools::abort("Unknown update type"); }
	    // out << "Actually Propagating 0" << "\t";
	    // out << cx_dqmc_params.N << " " << cx_dqmc_params.n_A << " " << cx_dqmc_params.n_B << "\t";
	    // out << r_ws_0->vol << " " << gs_0->vol << endl;
	    // out << gs_0->get_slice() << endl;
	    gs_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);
	    if (gs_0->get_slice() == 0) break;
	    ++sweep_slices;
	}
    }


    else if (state == 1) {
	if (gs_1->get_slice() != 0)
	    dqmc::tools::abort("gs_1 not on slice 0");

	for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
	    ++swept_on[gs_1->get_slice()];
	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		flip_counter += gs_delayed_spinful_flip(obs, gs_1->get_slice());
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		if (cx_dqmc_params.nni < 0) {
		    flip_counter += gs_delayed_attractive_spinless_flip(obs, gs_1->get_slice());
		    // flip_counter += gs_attractive_spinless_flip(obs, gs_1->get_slice());
		} else {
		    flip_counter += gs_delayed_spinless_flip(obs, gs_1->get_slice());
		}
	    } else { dqmc::tools::abort("Unknown update type"); }
	    gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);
	    if (gs_1->get_slice() == 0) break;
	    ++sweep_slices;
	}
    }

    if (*std::min_element(swept_on.begin(), swept_on.end()) == 0) {
	dqmc::tools::abort("Not every element was swept on");
    }

    ++delay_measurement;
    if (delay_measurement >= sweeps_measurement) {	    
	delay_measurement = 0;
	measure(obs);
    }

    if (state == 0) {
	if (obs.has("Stability Slice 0" + cx_dqmc_params.obs_suffix)) {
	    try {
		if (cx_dqmc_params.partial_cuts == false) {
		    obs["Stability Slice 0" + cx_dqmc_params.obs_suffix] << gs_0_0->stability_checks;
		    obs["Stability Slice 0" + cx_dqmc_params.obs_suffix] << gs_0_1->stability_checks;
		} else {
		    obs["Stability Slice 0" + cx_dqmc_params.obs_suffix] << gs_0->stability_checks;
		}		    
	    } catch (std::exception const& ex) {}
	}
	obs["Flip Rate 0" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
	
	
    } else if (state == 1) {
	if (obs.has("Stability Slice 1" + cx_dqmc_params.obs_suffix)) {
	    try {
		obs["Stability Slice 1" + cx_dqmc_params.obs_suffix] << gs_1->stability_checks;
	    } catch (std::exception const& ex) {}
	}

	obs["Flip Rate 1" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
    }
    dqmc::tools::set_obs_constant(obs, "Mu Adapt", cx_dqmc_params.mu_site);
    
    return false;
}


bool cx_dqmc_replica_gs::gs_sweep_continuously_slice(alps::ObservableSet& obs) {
    // out << "Performing a continuous sweep on one slice only" << endl;

    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE)
    	dqmc::tools::abort("Not prepared for continuous update");
    
    double flip_counter = 0;
    static int replica, rep_slice, slice;
    static int delay_measurement = 0;
    static int sweeps_measurement = (cx_dqmc_params.use_tempering == true)
    	? cx_dqmc_params.temper_freq : 10;
    static int recalculated;
    static int sweep_type = 1;
    double sweep_slices = 0;

    vector<int> swept_on;
    vector<double> switch_probs;
    
    if ((steps() % 1000 == 0 && is_thermalized())
    	|| steps() == cx_dqmc_params.thermalization/2 && !is_thermalized()) {
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
    
    swept_on.resize(2 * cx_dqmc_params.slices);
    std::fill(swept_on.begin(), swept_on.end(), 0);

    gs_0_0->build_stack();
    gs_0_1->build_stack();
    copy_to_gs_1();
    gs_1->log_weight();

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
    det_up_1 = 2 * gs_1->get_log_det();	

    if (state == 0) {
    	log_det_ratio = det_up_0 - det_up_1;
	obs["Swap To 1" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));
    } else {
    	log_det_ratio = det_up_1 - det_up_0;
	obs["Swap To 0" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));

    }    

    if (gs_0_0->get_slice() != 0) dqmc::tools::abort("gs_0_0 is not at slice 0");
    if (gs_0_1->get_slice() != 0) dqmc::tools::abort("gs_0_1 is not at slice 0");
    if (gs_1->get_slice() != 0)	dqmc::tools::abort("gs_1 is not at slice 0");
    
    if (state == 0) {	    
    	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
    	    if (gs_0_0->get_slice() != gs_1->get_slice() && gs_0_0->get_slice() == 0) {
    		dqmc::tools::abort("gs_0_0 and gs_1 not on same slice");
    	    }

    	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
    		if (gs_0_0->get_slice () == 0) {
    		    flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_0_0->get_slice(), switch_probs);
    		} else {
    		    flip_counter += gs_delayed_spinful_flip(obs, gs_0_0->get_slice(), false);
    		}
    	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
    		flip_counter += gs_delayed_spinless_flip(obs, gs_0_0->get_slice());
    	    } else { dqmc::tools::abort("Unknown update type"); }

    	    ++swept_on[gs_0_0->get_slice()];
	    
    	    int ret = gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

    	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
    		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
    		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
    		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
    	    }
    	    ++sweep_slices;
    	}
	
	copy_to_gs_1();
	gs_1->log_weight();
	gs_1->current_slice = cx_dqmc_params.slices;
	gs_1->idx = gs_1->chunks / 2 - 1;
	gs_1->dir_switch = true;
	gs_1->propagate(dummy);
	
	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
    	    if (gs_0_1->get_slice() + gs_0_1->slices != gs_1->get_slice() && gs_0_1->get_slice() == 0) {
    		dqmc::tools::abort("gs_0_1 and gs_1 not on same slice");
    	    }

    	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
    		if (gs_0_1->get_slice() == 0) {
    		    flip_counter +=
    			gs_delayed_continuous_spinful_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, switch_probs);
    		} else {
    		    flip_counter +=
    			gs_delayed_spinful_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, false);
    		}
    	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
    		flip_counter += gs_delayed_spinless_flip(obs, gs_0_1->get_slice() + gs_0_1->slices);
    	    } else { dqmc::tools::abort("Unknown update type"); }

    	    ++swept_on[gs_0_1->get_slice() + gs_0_1->slices];
	    
    	    int ret = gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);
	    
    	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
    		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
    		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
    		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
    	    }
    	    ++sweep_slices;
    	}
    }

    else if (state == 1) {
    	for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
    	    rep_slice = gs_1->get_slice() % cx_dqmc_params.slices;
    	    replica = gs_1->get_slice() / cx_dqmc_params.slices;

    	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		if (gs_1->get_slice() == 0 || gs_1->get_slice() == cx_dqmc_params.slices) {
		    if ((replica == 0 && rep_slice != gs_0_0->get_slice())
			|| (replica == 1 && rep_slice != gs_0_1->get_slice())) {
			out << replica << "\t| " << rep_slice << " "
			    << gs_0_0->get_slice() << " - " << gs_0_1->get_slice() << endl;
			dqmc::tools::abort("gs_1 - wrong reference slice in state 1");		
		    }

		    if (gs_1->get_slice() == cx_dqmc_params.slices) {
			gs_0_0->build_stack();
		    }

		    if (gs_1->get_slice() == cx_dqmc_params.slices) {
			gs_1->log_weight();
		    }
		    
		    flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_1->get_slice(), switch_probs);
		} else {
		    flip_counter += gs_delayed_spinful_flip(obs, gs_1->get_slice(), false);
		}
    	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
    		flip_counter += gs_delayed_spinless_flip(obs, gs_1->get_slice());
    	    } else { dqmc::tools::abort("Unknown update type"); }

    	    ++swept_on[gs_1->get_slice()];
	    gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);
    	    ++sweep_slices;

	    if (gs_1->get_slice() == slices) {
		double old_log_weight = gs_0_0->get_log_det();
		gs_0_0->build_stack();
		if (std::fabs(old_log_weight - gs_0_0->get_log_det()) > 1e-9) {
		    cout << old_log_weight << " vs " << gs_0_0->get_log_det() << "\t"
			 << std::fabs(old_log_weight - gs_0_0->get_log_det()) << endl;
		    dqmc::tools::abort("Log weight is incorrect for 0_0 after continuous update");
		}
			
	    } else if (gs_1->get_slice() == 0) {
		double old_log_weight_0_1 = gs_0_1->get_log_det();
		gs_0_1->build_stack();
		if (std::fabs(old_log_weight_0_1 - gs_0_1->get_log_det()) > 1e-9) {
		    dqmc::tools::abort("Log weight is incorrect for 0_1 after continuous update");
		}
	    }
    	}
    }

    if (*std::max_element(swept_on.begin(), swept_on.end()) != 1
    	|| *std::min_element(swept_on.begin(), swept_on.end()) != 1) {
    	for (auto i = swept_on.begin(); i != swept_on.end(); ++i)
    	    std::cout << *i << ' ';
    	dqmc::tools::abort("Not all slices have been swept on or some have been swept on twice");
    }

    if (state == 0) {	    
    	obs["Cont Swap To 1" + cx_dqmc_params.obs_suffix]
    	    << switch_probs[0];
    } else {
    	obs["Cont Swap To 0" + cx_dqmc_params.obs_suffix]
    	    << switch_probs[0];
    }
    
    if (state == 0) {
    	obs["Flip Rate 0" + cx_dqmc_params.obs_suffix]
    	    << flip_counter / sweep_slices;
    } else if (state == 1) {
    	obs["Flip Rate 1" + cx_dqmc_params.obs_suffix]
    	    << flip_counter / sweep_slices;
    }
    return false;
}


bool cx_dqmc_replica_gs::gs_sweep_continuously_chunk(alps::ObservableSet& obs) {

    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
	dqmc::tools::abort("Not prepared for continuous update");
    }
    
    double flip_counter = 0;
    static int replica, rep_slice, slice;
    static int delay_measurement = 0;
    static int sweeps_measurement = (cx_dqmc_params.use_tempering == true)
	? cx_dqmc_params.temper_freq : 10;
    static int recalculated;
    static int sweep_type = 1;
    double sweep_slices = 0;

    vector<int> swept_on;
    vector<double> switch_probs;
    
    if ((steps() % 1000 == 0 && is_thermalized())
	|| steps() == cx_dqmc_params.thermalization/2 && !is_thermalized()) {
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
    
    swept_on.resize(2 * cx_dqmc_params.slices);
    std::fill(swept_on.begin(), swept_on.end(), 0);

    gs_0_0->build_stack();
    gs_0_1->build_stack();
    copy_to_gs_1();
    gs_1->log_weight();

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
    det_up_1 = 2 * gs_1->get_log_det();	

    if (state == 0) {
    	log_det_ratio = det_up_0 - det_up_1;
	obs["Swap To 1" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));
    } else {
    	log_det_ratio = det_up_1 - det_up_0;
	obs["Swap To 0" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));
    }

    if (gs_0_0->get_slice() != 0) dqmc::tools::abort("gs_0_0 is not at slice 0");
    if (gs_0_1->get_slice() != 0) dqmc::tools::abort("gs_0_1 is not at slice 0");
    if (gs_1->get_slice() != 0) dqmc::tools::abort("gs_1 is not at slice 0");
    
    if (state == 0) {	    
	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    if (gs_0_0->get_slice() != gs_1->get_slice()
		&& gs_0_0->get_slice() < gs_0_0->safe_mult) {
		dqmc::tools::abort("gs_0_0 and gs_1 not on same slice");
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		if (gs_0_0->get_slice() < gs_0_0->safe_mult) {
		    flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_0_0->get_slice(), switch_probs);
		} else {
		    flip_counter += gs_delayed_spinful_flip(obs, gs_0_0->get_slice(), false);
		}
			   
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_spinless_flip(obs, gs_0_0->get_slice());
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_0_0->get_slice()];

	    if (gs_0_0->get_slice() < gs_0_0->safe_mult) {
		int ret_1 = gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);   
	    }
	    int ret = gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
	    }
	    ++sweep_slices;
	}
	copy_to_gs_1();
	gs_1->log_weight();
	gs_1->current_slice = cx_dqmc_params.slices;
	gs_1->idx = gs_1->chunks / 2 - 1;
	gs_1->dir_switch = true;
	gs_1->propagate(dummy);

	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    if (gs_0_1->get_slice() + gs_0_1->slices != gs_1->get_slice()
		&& gs_0_1->get_slice() < gs_0_1->safe_mult) {
		dqmc::tools::abort("gs_0_1 and gs_1 not on same slice");
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		if (gs_0_1->get_slice() < gs_0_1->safe_mult) {
		    flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, switch_probs);
		} else {
		    flip_counter += gs_delayed_spinful_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, false);
		}
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_spinless_flip(obs, gs_0_1->get_slice() + gs_0_1->slices);
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_0_1->get_slice() + gs_0_1->slices];

	    if (gs_0_1->get_slice() < gs_0_1->safe_mult) {
		int ret_1 = gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);	    
	    }
	    int ret = gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);
	    
	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
	    }
	    ++sweep_slices;
	}
    }
    else if (state == 1) {
	for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
	    rep_slice = gs_1->get_slice() % cx_dqmc_params.slices;
	    replica = gs_1->get_slice() / cx_dqmc_params.slices;

	    if ((replica == 0 && rep_slice != gs_0_0->get_slice()
		 && gs_1->get_slice() < gs_1->safe_mult)
		|| (replica == 1 && rep_slice != gs_0_1->get_slice()
		    && gs_1->get_slice() < gs_1->safe_mult)) {
		out << replica << "\t| " << rep_slice << " "
		    << gs_0_0->get_slice() << " - " << gs_0_1->get_slice() << endl;
		dqmc::tools::abort("gs_1 - wrong reference slice in state 1");		
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		if (gs_1->get_slice() < gs_1->safe_mult
		    || (gs_1->get_slice() >= gs_1->slices && gs_1->get_slice() < gs_1->slices + gs_1->safe_mult)) {


		    if (gs_1->get_slice() == cx_dqmc_params.slices) {
			gs_1->log_weight();
		    }

		    flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_1->get_slice(), switch_probs);
		} else {
		    flip_counter += gs_delayed_spinful_flip(obs, gs_1->get_slice(), false);
		}
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_spinless_flip(obs, gs_1->get_slice());
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_1->get_slice()];

	    if (replica == 0) {
		if (gs_1->get_slice() < gs_1->safe_mult) {
		    gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);		    
		}

		if (gs_0_0->get_slice() == gs_0_0->safe_mult) {
		    double ow = gs_0_0->get_log_det();
		    gs_0_0->build_stack();
		    if (std::fabs(ow - gs_0_0->get_log_det()) > 1e-4) {
			cout << ow << " vs " << gs_0_0->get_log_det()
			     << std::fabs(ow - gs_0_0->get_log_det()) << endl;
			dqmc::tools::abort("Log weight is incorrect for 0_0 after continuous update");
		    }
		}

	    } else if (replica == 1) {
		if (gs_1->get_slice() - gs_1->slices < gs_1->safe_mult) {
		    gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);
		}

		if (gs_0_1->get_slice() == gs_0_1->safe_mult) {
		    double ow = gs_0_1->get_log_det();
		    gs_0_1->build_stack();
		    if (std::fabs(ow - gs_0_1->get_log_det()) > 1e-4) {
			cout << ow << " vs " << gs_0_1->get_log_det()
			     << std::fabs(ow - gs_0_1->get_log_det()) << endl;
			dqmc::tools::abort("Log weight is incorrect for 0_1 after continuous update");
		    }
		}
	    }
	    
	    // cout << "Propagating from " << gs_1->get_slice() << endl;
	    gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);
	    // cout << "Propagating to " << gs_1->get_slice() << endl;
		
	    ++sweep_slices;
	}
    }
    if (*std::max_element(swept_on.begin(), swept_on.end()) != 1
	|| *std::min_element(swept_on.begin(), swept_on.end()) != 1) {
	for (auto i = swept_on.begin(); i != swept_on.end(); ++i)
	    std::cout << *i << ' ';
	dqmc::tools::abort("Not all slices have been swept on or some have been swept on twice");
    }

    // cout << "Final " << setw(15) << right << exp(-log_det_ratio) << "\t";
    
    double sum = std::accumulate(switch_probs.begin(), switch_probs.end(), 0.0);
    double switch_prob = sum / double(switch_probs.size());

    // cout << "Measurement " << setw(15) << right << switch_prob << endl;
    if (state == 0) {	    
	obs["Cont Swap To 1" + cx_dqmc_params.obs_suffix]
	    << switch_prob;
    } else {
	obs["Cont Swap To 0" + cx_dqmc_params.obs_suffix]
	    << switch_prob;
    }

    if (state == 0) {
	obs["Flip Rate 0" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
    } else if (state == 1) {
	obs["Flip Rate 1" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
    }
    return false;
}


bool cx_dqmc_replica_gs::gs_sweep_continuously_complete(alps::ObservableSet& obs) {

    // if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
    // 	dqmc::tools::abort("Not prepared for continuous update");
    // }
    
    double flip_counter = 0;
    static int replica, rep_slice, slice;
    static int delay_measurement = 0;
    static int sweeps_measurement = (cx_dqmc_params.use_tempering == true)
	? cx_dqmc_params.temper_freq : 10;
    sweeps_measurement = 1;
    static int recalculated;
    static int sweep_type = 1;
    double sweep_slices = 0;

    vector<int> swept_on;
    vector<double> switch_probs;
    
    if ((steps() % 1000 == 0 && is_thermalized())
	|| steps() == cx_dqmc_params.thermalization/2 && !is_thermalized()) {
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
    
    swept_on.resize(2 * cx_dqmc_params.slices);
    std::fill(swept_on.begin(), swept_on.end(), 0);

    gs_0_0->build_stack();
    gs_0_1->build_stack();
    copy_to_gs_1();
    gs_1->log_weight();

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();

    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
	det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
	det_up_1 = 2 * gs_1->get_log_det();
    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
	det_up_0 = (det_up_0_0 + det_up_0_1);
	det_up_1 = gs_1->get_log_det();
    }

    if (state == 0) {
    	log_det_ratio = det_up_0 - det_up_1;
	obs["Swap To 1" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));
    } else {
    	log_det_ratio = det_up_1 - det_up_0;
	obs["Swap To 0" + cx_dqmc_params.obs_suffix]
	    << min(1., exp(-log_det_ratio));
    }

    if (gs_0_0->get_slice() != 0) dqmc::tools::abort("gs_0_0 is not at slice 0");
    if (gs_0_1->get_slice() != 0) dqmc::tools::abort("gs_0_1 is not at slice 0");
    if (gs_1->get_slice() != 0) dqmc::tools::abort("gs_1 is not at slice 0");
    
    if (state == 0) {	    
	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    if (gs_0_0->get_slice() != gs_1->get_slice()) {
		dqmc::tools::abort("gs_0_0 and gs_1 not on same slice");
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_0_0->get_slice(), switch_probs);
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_continuous_spinless_flip(obs, gs_0_0->get_slice(), switch_probs);
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_0_0->get_slice()];

	    int ret_1 = gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);   
	    int ret = gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
	    }
	    ++sweep_slices;
	}
	double ow = gs_1->get_log_det();
	gs_1->build_stack();
	if (std::fabs(ow - gs_1->get_log_det())/std::fabs(ow + gs_1->get_log_det()) > 1e-4) {
	    cout << ow << " vs " << gs_1->get_log_det()  << "\t"
		 << std::fabs(ow - gs_1->get_log_det()) << endl;
	    dqmc::tools::abort("Log weight is incorrect for _1 after half the continuous update");
	}

	copy_to_gs_1();
	gs_1->log_weight();
	gs_1->current_slice = cx_dqmc_params.slices;
	gs_1->idx = gs_1->chunks / 2 - 1;
	gs_1->dir_switch = true;
	gs_1->propagate(dummy);

	for (uint i = 0; i < cx_dqmc_params.slices; i++) {
	    if (gs_0_1->get_slice() + gs_0_1->slices != gs_1->get_slice()) {
		dqmc::tools::abort("gs_0_1 and gs_1 not on same slice");
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, switch_probs);
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_continuous_spinless_flip(obs, gs_0_1->get_slice() + gs_0_1->slices, switch_probs);
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_0_1->get_slice() + gs_0_1->slices];

	    int ret_1 = gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);	    
	    int ret = gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);
	    
	    if (ret == 3 && steps() >  500 && steps() < 1000 && !is_thermalized()
		&& cx_dqmc_params.adapt_mu == true && cx_dqmc_params.simple_adapt_mu == false) {
		obs["Mu Adapt" + cx_dqmc_params.obs_suffix] <<
		    2 / cx_dqmc_params.beta * log(cx_dqmc_params.mu_target / gs_0_1->lowest_sv);	  
	    }
	    ++sweep_slices;
	}
	ow = gs_1->get_log_det();
	copy_to_gs_1();
	gs_1->log_weight();
	if (std::fabs(ow - gs_1->get_log_det())/std::fabs(ow + gs_1->get_log_det()) > 1e-4) {
	    cout << ow << " vs " << gs_1->get_log_det() << "\t"
		 << std::fabs(ow - gs_1->get_log_det()) << endl;
	    dqmc::tools::warn("Log weight is incorrect for _1 after other half the continuous update");
	}
    }
    else if (state == 1) {
	for (uint i = 0; i < 2 * cx_dqmc_params.slices; i++) {
	    rep_slice = gs_1->get_slice() % cx_dqmc_params.slices;
	    replica = gs_1->get_slice() / cx_dqmc_params.slices;

	    if ((replica == 0 && rep_slice != gs_0_0->get_slice())		 
		|| (replica == 1 && rep_slice != gs_0_1->get_slice())) {
		out << replica << "\t| " << rep_slice << " "
		    << gs_0_0->get_slice() << " - " << gs_0_1->get_slice() << endl;
		dqmc::tools::abort("gs_1 - wrong reference slice in state 1");		
	    }

	    if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINFUL_UPDATE) {
		if (gs_1->get_slice() == cx_dqmc_params.slices) {
		    gs_1->log_weight();
		    gs_0_0->log_weight();
		}
		
		flip_counter += gs_delayed_continuous_spinful_flip(obs, gs_1->get_slice(), switch_probs);
	    } else if (cx_dqmc_params.update_type == cx_dqmc_params.CX_SPINLESS_UPDATE) {
		flip_counter += gs_delayed_continuous_spinless_flip(obs, gs_1->get_slice(), switch_probs);
	    } else { dqmc::tools::abort("Unknown update type"); }

	    ++swept_on[gs_1->get_slice()];

	    if (replica == 0) {
		gs_0_0->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);		    

		if (gs_0_0->get_slice() == 0) {
		    // cout << "Is this ever true?" << endl;
		    double ow = gs_0_0->get_log_det();
		    gs_0_0->build_stack();
		    if (std::fabs(ow - gs_0_0->get_log_det()) / std::fabs(ow + gs_0_0->get_log_det()) > 1e-4) {
			cout << ow << " vs " << gs_0_0->get_log_det() << "\t"
			     << std::fabs(ow - gs_0_0->get_log_det()) << endl;
			dqmc::tools::warn("Log weight is incorrect for 0_0 after continuous update");
		    }
		}

	    } else if (replica == 1) {
		gs_0_1->propagate(obs["Stability 0" + cx_dqmc_params.obs_suffix]);

		if (gs_0_1->get_slice() == 0) {
		    // cout << "Is this ever true? part 2" << endl;
		    double ow = gs_0_1->get_log_det();
		    gs_0_1->build_stack();
		    if (std::fabs(ow - gs_0_1->get_log_det()) / std::fabs(ow + gs_0_1->get_log_det()) > 1e-4) {
			cout << ow << " vs " << gs_0_1->get_log_det() << "\t"
			     << std::fabs(ow - gs_0_1->get_log_det()) << endl;
			dqmc::tools::warn("Log weight is incorrect for 0_1 after continuous update");
		    }
		}
	    }
	    
	    // cout << "Propagating from " << gs_1->get_slice() << endl;
	    gs_1->propagate(obs["Stability 1" + cx_dqmc_params.obs_suffix]);
	    // cout << "Propagating to " << gs_1->get_slice() << endl;
		
	    ++sweep_slices;
	}
    }
    if (*std::max_element(swept_on.begin(), swept_on.end()) != 1
	|| *std::min_element(swept_on.begin(), swept_on.end()) != 1) {
	for (auto i = swept_on.begin(); i != swept_on.end(); ++i)
	    std::cout << *i << ' ';
	dqmc::tools::abort("Not all slices have been swept on or some have been swept on twice");
    }

    // cout << "Final " << setw(15) << right << exp(-log_det_ratio) << "\t";
    // cout << "Switch probs" << endl;
    // for (auto i = switch_probs.begin(); i != switch_probs.end(); ++i) {
    // 	std::cout << *i << ' ';
    // }
    // cout << std::endl;

    double sum = std::accumulate(switch_probs.begin(), switch_probs.end(), 0.0);
    double switch_prob = sum / double(switch_probs.size());

    // cout << "Measurement " << setw(15) << right << switch_prob << endl;
    if (state == 0) {	    
	obs["Cont Swap To 1" + cx_dqmc_params.obs_suffix]
	    << switch_prob;
    } else {
	obs["Cont Swap To 0" + cx_dqmc_params.obs_suffix]
	    << switch_prob;
    }

    if (state == 0) {
	obs["Flip Rate 0" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
    } else if (state == 1) {
	obs["Flip Rate 1" + cx_dqmc_params.obs_suffix]
	    << flip_counter / sweep_slices;
    }
    return false;
}

    
void cx_dqmc_replica_gs::measure(alps::ObservableSet& obs) {    
    // cout << "Measuring" << endl;

    calculate_weights(false);
    
    try {
	if (state == 0) {
	    if (cx_dqmc_params.partial_cuts == false) {
		try {
		    if (obs.has("SVD Stack 0" + cx_dqmc_params.obs_suffix)) {
			try {
			    obs["SVD Stack 0" + cx_dqmc_params.obs_suffix] << gs_0_0->svds;
			    obs["SVD Stack 0" + cx_dqmc_params.obs_suffix] << gs_0_1->svds;
			} catch (std::exception const& ex) {
			}
		    }

		    if (obs.has("SVD Stack Example 0" + cx_dqmc_params.obs_suffix)) {
			dqmc::tools::set_vector_constant(obs, "SVD Stack Example 0" + cx_dqmc_params.obs_suffix, gs_0_0->svd_example);
		    }
		} catch (std::exception const& ex) {
		}

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

		obs["Log Weight 0" + cx_dqmc_params.obs_suffix]
		    << double(gs_0_0->get_log_det());
		obs["Log Weight 0" + cx_dqmc_params.obs_suffix]
		    << double(gs_0_1->get_log_det());
	    }
	    else {
		try {
		    if (obs.has("SVD Stack 0" + cx_dqmc_params.obs_suffix)) {
			try {
			    obs["SVD Stack 0" + cx_dqmc_params.obs_suffix] << gs_0->svds;
			} catch (std::exception const& ex) { }
		    }

		    if (obs.has("SVD Stack Example 0" + cx_dqmc_params.obs_suffix)) {
			dqmc::tools::set_vector_constant(obs, "SVD Stack Example 0" + cx_dqmc_params.obs_suffix, gs_0->svd_example);
		    }
		} catch (std::exception const& ex) {}

		obs["Sign 0 Recalc" + cx_dqmc_params.obs_suffix]
		    << double(std::real(sign_0));

		obs["Re Phase 0 Recalc" + cx_dqmc_params.obs_suffix]
		    << double(std::real(sign_0));

		obs["Im Phase 0 Recalc" + cx_dqmc_params.obs_suffix]
		    << double(std::imag(sign_0));

		obs["Log Weight 0" + cx_dqmc_params.obs_suffix]
		    << double(gs_0->get_log_det());
	    }
	}
	else if (state == 1) {
	    try {
		if (obs.has("SVD Stack 1" + cx_dqmc_params.obs_suffix)) {
		    try {
			obs["SVD Stack 1" + cx_dqmc_params.obs_suffix] << gs_1->svds;
		    } catch(std::exception const& ex) {}
		}
		if (obs.has("SVD Stack Example 1" + cx_dqmc_params.obs_suffix)) {
		    dqmc::tools::set_vector_constant(obs, "SVD Stack Example 1" + cx_dqmc_params.obs_suffix, gs_1->svd_example);
		}
	    } catch (std::exception& ex) {
	    }

	    obs["Sign 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_1));

	    obs["Re Phase 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_1));
	    obs["Im Phase 1 Recalc" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(sign_1));

	    obs["Log Weight 1" + cx_dqmc_params.obs_suffix]
		<< double(gs_1->get_log_det());
	}

	if (state == 0) {	    
	    log_ratio = - det_up_0 + det_up_1;
	    ratio = exp(log_ratio);		
	    obs["Swap To 1" + cx_dqmc_params.obs_suffix]
		<< min(1., ratio);
	} else {
	    log_ratio = det_up_0 - det_up_1;
	    ratio = exp(log_ratio);		
	    obs["Swap To 0" + cx_dqmc_params.obs_suffix]
		<< min(1., ratio);
	}

	// finally attempt internal exchange
	// if (state == 0) {
	//     gs_0_0->save_curr_det();
	//     gs_0_1->save_curr_det();

	//     obs["Internal to 1" + cx_dqmc_params.obs_suffix] << min(1., exp(2 * (gs_1->log_det - gs_1->curr_log_det)));
	//     if (uniform_01() < exp(2 * (gs_1->log_det - gs_1->curr_log_det))) {
	// 	copy_aux_spins(aux_spins,
	// 		       aux_spins_1);
	// 	gs_1->update_det();
	//     }
	// } 
	// else {
	//     gs_1->save_curr_det();
	//     obs["Internal to 0" + cx_dqmc_params.obs_suffix] <<
	// 	min(1., exp(2 * (gs_0_0->log_det + gs_0_1->log_det
	// 			 - gs_0_0->curr_log_det - gs_0_1->curr_log_det)));
	//     if (uniform_01() < exp(2 * (gs_0_0->log_det + gs_0_1->log_det
	// 				- gs_0_0->curr_log_det - gs_0_1->curr_log_det))) {
	// 	copy_aux_spins(aux_spins,
	// 		       aux_spins_0);
	// 	gs_0_0->update_det();
	// 	gs_0_1->update_det();
	//     }
	// }
		
	    
    }
    catch(std::exception& ex) {
	cout << "Error while trying to measure" << endl
	     << ex.what() << endl;
    }   
}


double cx_dqmc_replica_gs::gs_delayed_spinful_flip(alps::ObservableSet& obs, int s, bool at_once) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1, log_ratios;
    
    // int flip_runs = std::max(std::ceil(100. / ws->num_sites), 3 * std::ceil(ws->num_sites / 100.));
    // int target_flushes = std::max(1, std::ceil(p->N / 100.));
 
    rands_0.resize(10 * std::max(100, ws->num_sites));
    rands_1.resize(10 * std::max(100, ws->num_sites));
    log_ratios.resize(ws->num_sites);
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0 && cx_dqmc_params.partial_cuts == false) {

	if (at_once == true) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
								   0, rep_slice,
								   gs_0_0, rands_0, alpha_0,
								   avg_sign_0);
	    
	    flips_accepted += cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
								    1, rep_slice,
								    gs_0_1, rands_1, alpha_0,
								    avg_sign_1);       

	    sign_0 = avg_sign_0 * avg_sign_1;

	    obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_0));

	    obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));		
	
	    obs["Sign 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0));	
	
	    obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(sign_0));
	
	    obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(sign_0));
	} else {
	    if (rep == 0) {
		flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
								       0, rep_slice,
								       gs_0_0, rands_0, alpha_0,
								       avg_sign_0);
		obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
		    << double(std::real(avg_sign_0));	    
		obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
		    << double(std::real(avg_sign_0));	    
		obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
		    << double(std::imag(avg_sign_0));

	    } else if (rep == 1) {
		flips_accepted += cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, 
									1, rep_slice,
									gs_0_1, rands_1, alpha_0,
									avg_sign_1);       
		obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
		    << double(std::real(avg_sign_1));	    
		obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
		    << double(std::real(avg_sign_1));	    
		obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
		    << double(std::imag(avg_sign_1));		
	    }
	}

    }
    else if (state == 1 || cx_dqmc_params.partial_cuts == true) {
	if (state == 0) {
	    // cout << "Delayed Flip in 0 " << gs_0->get_slice() << endl;
	    flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, 1, rep, 
								   rep_slice,
								   gs_0, rands_1, alpha_0,
								   avg_sign_1, false);

	    obs["Sign 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	}
	else {
	    flips_accepted = cx_dqmc::update::spinful_delayed_flip(obs, aux_spins, state, rep, 
								   rep_slice,
								   gs_1, rands_1, alpha_0,
								   avg_sign_1);

	    obs["Sign 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	}
    }
    return flips_accepted;
}


double cx_dqmc_replica_gs::gs_delayed_continuous_spinful_flip(alps::ObservableSet& obs, int s,
							      vector<double>& switch_probs) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;

    vector<double> rands_0, rands_1, prob_log_ratios,
	weight_log_ratios, dummy_log_ratios;
    
    rands_0.resize(10 * std::max(100, ws->num_sites));
    rands_1.resize(10 * std::max(100, ws->num_sites));    
    prob_log_ratios.reserve(2 * ws->num_sites + 1);

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
    det_up_1 = 2 * gs_1->get_log_det();	

    if (state == 0) {
	log_det_ratio = det_up_0 - det_up_1;
    } else {
	log_det_ratio = det_up_1 - det_up_0;
    }

    prob_log_ratios.push_back(log_det_ratio);
    weight_log_ratios.reserve(2 * ws->num_sites + 1);
    dummy_log_ratios.reserve(2 * ws->num_sites + 1);

    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0) {
	if (rep == 0) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_continuous_flip(obs, aux_spins, state, 
									      0, rep_slice,
									      gs_0_0, gs_1, rands_0, alpha_0,
									      avg_sign_0, prob_log_ratios,
									      weight_log_ratios, dummy_log_ratios);

	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_0_0->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_1->log_det += *i; }

	    obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_0));

	}
	else if (rep == 1) {
	    flips_accepted += cx_dqmc::update::spinful_delayed_continuous_flip(obs, aux_spins, state, 
									       1, rep_slice,
									       gs_0_1, gs_1, rands_1, alpha_0,
									       avg_sign_1, prob_log_ratios,
									       weight_log_ratios, dummy_log_ratios);       

	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_0_1->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_1->log_det += *i; }

	    obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));		

	}
	else {
	    dqmc::tools::abort("No replica higher than 1 allowed");
	}

	sign_0 = avg_sign_0 * avg_sign_1;
		
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));	
	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));
	
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(sign_0));

    }
    else if (state == 1) {
	if (rep == 0) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_continuous_flip(obs, aux_spins, state, rep, 
									      rep_slice,
									      gs_1, gs_0_0, rands_1, alpha_0,
									      avg_sign_1, prob_log_ratios,
									      weight_log_ratios, dummy_log_ratios);

	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_1->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_0_0->log_det += *i; }
	    
	} else if (rep == 1) {
	    flips_accepted = cx_dqmc::update::spinful_delayed_continuous_flip(obs, aux_spins, state, rep, 
									      rep_slice,
									      gs_1, gs_0_1, rands_1, alpha_0,
									      avg_sign_1, prob_log_ratios,
									      weight_log_ratios, dummy_log_ratios);
	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_1->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_0_1->log_det += *i; }
	}
	
	obs["Sign 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	
	obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));
	obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));

    }

    // Sum up log ratios for this slice
    std::vector<double> sum_log_ratios, slice_switch_probs;
    sum_log_ratios.resize(prob_log_ratios.size());    
    std::partial_sum(prob_log_ratios.begin(), prob_log_ratios.end(), sum_log_ratios.begin());
    log_det_ratio = sum_log_ratios[sum_log_ratios.size() - 1];

    for (int i = 0; i < sum_log_ratios.size(); ++i) {
	// cout << "Sum log ratios: " << std::min(1., exp(-sum_log_ratios[i])) << endl;
	slice_switch_probs.push_back(std::min(1., exp(-sum_log_ratios[i])));
    }

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
    det_up_1 = 2 * gs_1->get_log_det();	

    if (state == 0) {
	log_det_ratio = det_up_0 - det_up_1;
    } else {
	log_det_ratio = det_up_1 - det_up_0;
    }


    // cout << gs_0_0->get_slice() << " | " << gs_0_1->get_slice() << " - "
    // 	 << gs_1->get_slice() << endl;
    // cout << gs_0_0->get_log_det() << " | " << gs_0_1->get_log_det() << " - "
    // 	 << gs_1->get_log_det() << endl;

    // gs_0_0->build_stack();
    // gs_0_1->build_stack();
    // copy_to_gs_1();
    // gs_1->log_weight();

    // cout << gs_0_0->get_log_det() << " | " << gs_0_1->get_log_det() << " - "
    // 	 << gs_1->get_log_det() << endl;
    
    double sum = std::accumulate(slice_switch_probs.begin(), slice_switch_probs.end(), 0.0);
    switch_probs.push_back(sum / double(slice_switch_probs.size()));

    // cout << "Latest ratio " << std::min(1., exp(-log_det_ratio)) << " vs "
    // 	 << std::min(1., exp(-sum_log_ratios[sum_log_ratios.size() -1])) << endl;

    return flips_accepted;
}

double cx_dqmc_replica_gs::gs_simple_spinless_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1;
    
    rands_0.resize(10 * std::max(100, ws->num_update_bonds));
    rands_1.resize(10 * std::max(100, ws->num_update_bonds));
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0) {
	flips_accepted = cx_dqmc::update::spinless_simple_flip(obs, aux_spins, state, 
							       0, rep_slice,
							       gs_0_0, rands_0, alpha_0,
							       avg_sign_0);

	flips_accepted += cx_dqmc::update::spinless_simple_flip(obs, aux_spins, state, 
								1, rep_slice,
								gs_0_1, rands_1, alpha_0,
								avg_sign_1);       

	sign_0 = avg_sign_0 * avg_sign_1;

	obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_0));
	
	obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));		
	
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));	
	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));
	
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(sign_0));

    }
    else if (state == 1) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	flips_accepted = cx_dqmc::update::spinless_simple_flip(obs, aux_spins, state, rep, 
								rep_slice,
								gs_1, rands_1, alpha_0,
								avg_sign_1);
	
	obs["Sign 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	
	obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));
	obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));

    }
    return flips_accepted;
}


double cx_dqmc_replica_gs::gs_attractive_spinless_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1;
    
    rands_0.resize(10 * std::max(100, ws->num_update_bonds));
    rands_1.resize(10 * std::max(100, ws->num_update_bonds));
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0 && cx_dqmc_params.partial_cuts == false) {
	flips_accepted = cx_dqmc::update::spinless_attractive_flip(obs, aux_spins, state, 
								   0, rep_slice,
								   gs_0_0, rands_0, alpha_0,
								   avg_sign_0);

	flips_accepted += cx_dqmc::update::spinless_attractive_flip(obs, aux_spins, state, 
								    1, rep_slice,
								    gs_0_1, rands_1, alpha_0,
								    avg_sign_1);       

	sign_0 = avg_sign_0 * avg_sign_1;

	obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_0));
	
	obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));		
	
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));	
	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));
	
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(sign_0));

    }
    else  if (state == 0 && cx_dqmc_params.partial_cuts == true) {
	flips_accepted = cx_dqmc::update::spinless_attractive_flip(obs, aux_spins, 1, rep, 
								   rep_slice,
								   gs_0, rands_0, alpha_0,
								   avg_sign_0);
	
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_0));

    } else if (state == 1) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	flips_accepted = cx_dqmc::update::spinless_attractive_flip(obs, aux_spins, state, rep, 
								   rep_slice,
								   gs_1, rands_1, alpha_0,
								   avg_sign_1);
	
	obs["Sign 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	
	obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));
	obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));

    }
    return flips_accepted;
}


double cx_dqmc_replica_gs::gs_delayed_attractive_spinless_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1;
    
    rands_0.resize(10 * std::max(100, ws->num_update_bonds));
    rands_1.resize(10 * std::max(100, ws->num_update_bonds));
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0 && cx_dqmc_params.partial_cuts == false) {
	flips_accepted = cx_dqmc::update::spinless_attractive_delayed_flip(obs, aux_spins, state, 
								0, rep_slice,
								gs_0_0, rands_0, alpha_0,
								avg_sign_0);

	flips_accepted += cx_dqmc::update::spinless_attractive_delayed_flip(obs, aux_spins, state, 
								 1, rep_slice,
								 gs_0_1, rands_1, alpha_0,
								 avg_sign_1);       

	sign_0 = avg_sign_0 * avg_sign_1;

	obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_0));
	
	obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));		
	
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));	
	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));
	
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(sign_0));

    }
    else if (state == 1 || cx_dqmc_params.partial_cuts == true) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	if (state == 0) {
	    flips_accepted = cx_dqmc::update::spinless_attractive_delayed_flip(obs, aux_spins, 1, rep, 
								    rep_slice,
								    gs_0, rands_1, alpha_0,
								    avg_sign_1);

	    obs["Sign 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	} else {
	    flips_accepted = cx_dqmc::update::spinless_attractive_delayed_flip(obs, aux_spins, state, rep, 
								    rep_slice,
								    gs_1, rands_1, alpha_0,
								    avg_sign_1);

	    obs["Sign 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	}
    }
    return flips_accepted;
}


double cx_dqmc_replica_gs::gs_delayed_spinless_flip(alps::ObservableSet& obs, int s) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1;
    
    rands_0.resize(10 * std::max(100, ws->num_update_bonds));
    rands_1.resize(10 * std::max(100, ws->num_update_bonds));
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    if (state == 0 && cx_dqmc_params.partial_cuts == false) {
	flips_accepted = cx_dqmc::update::spinless_delayed_flip(obs, aux_spins, state, 
								0, rep_slice,
								gs_0_0, rands_0, alpha_0,
								avg_sign_0);

	flips_accepted += cx_dqmc::update::spinless_delayed_flip(obs, aux_spins, state, 
								 1, rep_slice,
								 gs_0_1, rands_1, alpha_0,
								 avg_sign_1);       

	sign_0 = avg_sign_0 * avg_sign_1;

	obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_0));	    
	obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_0));
	
	obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	    
	obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));		
	
	obs["Sign 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));	
	
	obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::real(sign_0));
	
	obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(sign_0));

    }
    else if (state == 1 || cx_dqmc_params.partial_cuts == true) {
	// cout << "Sweeping on " << rep << " " << rep_slice << endl;
	if (state == 0) {
	    flips_accepted = cx_dqmc::update::spinless_delayed_flip(obs, aux_spins, 1, rep, 
								    rep_slice,
								    gs_0, rands_1, alpha_0,
								    avg_sign_1);

	    obs["Sign 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	} else {
	    flips_accepted = cx_dqmc::update::spinless_delayed_flip(obs, aux_spins, state, rep, 
								    rep_slice,
								    gs_1, rands_1, alpha_0,
								    avg_sign_1);

	    obs["Sign 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	
	    obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));
	    obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));
	}
    }
    return flips_accepted;
}



double cx_dqmc_replica_gs::gs_delayed_continuous_spinless_flip(alps::ObservableSet& obs, int s, vector<double>& switch_probs) {
    int rep = s / cx_dqmc_params.slices;
    int rep_slice = s % cx_dqmc_params.slices;
    double flips_accepted = 0;
    cx_double avg_sign_0 = 0;
    cx_double avg_sign_1 = 0;


    vector<double> rands_0, rands_1, prob_log_ratios,
	weight_log_ratios, dummy_log_ratios;
    
    rands_0.resize(10 * std::max(100, ws->num_update_bonds));
    rands_1.resize(10 * std::max(100, ws->num_update_bonds));
    
    for (int i = 0; i < rands_0.size(); ++i) {
	rands_0[i] = uniform_01();
	rands_1[i] = uniform_01();
    }

    prob_log_ratios.reserve(2 * ws->num_sites + 1);

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = det_up_0_0 + det_up_0_1;
    det_up_1 = gs_1->get_log_det();	

    if (state == 0) {
	log_det_ratio = det_up_0 - det_up_1;
    } else {
	log_det_ratio = det_up_1 - det_up_0;
    }

    prob_log_ratios.push_back(log_det_ratio);
    weight_log_ratios.reserve(2 * ws->num_sites + 1);
    dummy_log_ratios.reserve(2 * ws->num_sites + 1);

    if (state == 0) {
	if (rep == 0) {
	
	    flips_accepted = cx_dqmc::update::spinless_delayed_continuous_flip(obs, aux_spins, state, 
									       0, rep_slice,
									       gs_0_0, gs_1, rands_0, alpha_0,
									       avg_sign_0,
									       prob_log_ratios, weight_log_ratios,
									       dummy_log_ratios);

	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_0_0->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_1->log_det += *i; }

	    obs["Sign 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Re Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_0));	    
	    obs["Im Phase 0 0" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_0));

	} else if (rep == 1) {
	    flips_accepted += cx_dqmc::update::spinless_delayed_continuous_flip(obs, aux_spins, state, 
										1, rep_slice,
										gs_0_1, gs_1, rands_1, alpha_0,
										avg_sign_1,
									       prob_log_ratios, weight_log_ratios,
									       dummy_log_ratios);       
	
	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_0_0->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_1->log_det += *i; }

	
	    obs["Sign 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Re Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::real(avg_sign_1));	    
	    obs["Im Phase 0 1" + cx_dqmc_params.obs_suffix]
		<< double(std::imag(avg_sign_1));	
    }	else {
	    dqmc::tools::abort("No replica higher than 1 allowed");
	}
	
    }	
    else if (state == 1) {
	if (rep == 0) {
	    flips_accepted = cx_dqmc::update::spinless_delayed_continuous_flip(obs, aux_spins, state, rep, 
									      rep_slice,
									      gs_1, gs_0_0, rands_1, alpha_0,
									      avg_sign_1, prob_log_ratios,
									      weight_log_ratios, dummy_log_ratios);

	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_1->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_0_0->log_det += *i; }
	    
	} else if (rep == 1) {
	    flips_accepted = cx_dqmc::update::spinless_delayed_continuous_flip(obs, aux_spins, state, rep, 
									      rep_slice,
									      gs_1, gs_0_1, rands_1, alpha_0,
									      avg_sign_1, prob_log_ratios,
									      weight_log_ratios, dummy_log_ratios);
	    for (auto i = weight_log_ratios.begin(); i != weight_log_ratios.end(); ++i) { gs_1->log_det += *i; }
	    for (auto i = dummy_log_ratios.begin(); i != dummy_log_ratios.end(); ++i) {	gs_0_1->log_det += *i; }
	}
	
	obs["Sign 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));	
	obs["Re Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::real(avg_sign_1));
	obs["Im Phase 1" + cx_dqmc_params.obs_suffix]
	    << double(std::imag(avg_sign_1));
    }

    std::vector<double> sum_log_ratios, slice_switch_probs;
    sum_log_ratios.resize(prob_log_ratios.size());    
    std::partial_sum(prob_log_ratios.begin(), prob_log_ratios.end(), sum_log_ratios.begin());
    log_det_ratio = sum_log_ratios[sum_log_ratios.size() - 1];

    for (int i = 0; i < sum_log_ratios.size(); ++i) {
	// cout << "Sum log ratios: " << std::min(1., exp(-sum_log_ratios[i])) << endl;
	slice_switch_probs.push_back(std::min(1., exp(-sum_log_ratios[i])));
    }

    det_up_0_0 = gs_0_0->get_log_det();
    det_up_0_1 = gs_0_1->get_log_det();
    det_up_0 = 2 * (det_up_0_0 + det_up_0_1);
    det_up_1 = 2 * gs_1->get_log_det();	

    if (state == 0) {
	log_det_ratio = det_up_0 - det_up_1;
    } else {
	log_det_ratio = det_up_1 - det_up_0;
    }
    double sum = std::accumulate(slice_switch_probs.begin(), slice_switch_probs.end(), 0.0);
    switch_probs.push_back(sum / double(slice_switch_probs.size()));

    return flips_accepted;
}


int cx_dqmc_replica_gs::gs_spinful_naive_flip(alps::ObservableSet& obs, int s) {
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
