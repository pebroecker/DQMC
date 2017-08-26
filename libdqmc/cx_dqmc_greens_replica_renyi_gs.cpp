#include "cx_dqmc_greens_replica_renyi_gs.hpp"

using namespace std;

void cx_dqmc::greens_replica_renyi_gs::initialize() {    
    n_A = p->n_A;
    n_B = p->n_B;
    N = n_A + n_B;
    vol = n_A + 2 * n_B;
    replica = p->replica;
    sites = ws->sites;
    eff_particles = ws->eff_particles;
    particles = ws->particles;

    if (p->rank < 1)
	cout << p->outp << "initializing renyi " << vol << " = " << n_A << " + 2 * " << n_B
	     << "\t" << particles << " " << eff_particles << endl;
    
    if (p->update_type == p->CX_SPINFUL_UPDATE) {
	delayed_buffer_size = min(200, sites/2);
    }
    else if (p->update_type == p->CX_SPINLESS_UPDATE) {
	delayed_buffer_size = min(50, ws->num_update_bonds);
    }

    if (delayed_buffer_size % 2 != 0) {
	++delayed_buffer_size; }
    
    X.resize(vol, delayed_buffer_size);
    Y.resize(vol, delayed_buffer_size);

    X_sls.resize(vol, 2);
    Y_sls.resize(vol, 2);
    Y_sls_row_1.resize(2, vol);
    Y_sls_row_2.resize(2, vol);

    u_temp.resize(N, N);
    d_temp.resize(N);
    t_temp.resize(N, N);	    	    

    slices = p->slices;
    safe_mult = p->safe_mult;

    greens.resize(vol, vol);
    prop_greens.resize(vol, vol);
    temp_greens.resize(vol, vol);
    first_initialization = true;    
}


void cx_dqmc::greens_replica_renyi_gs::build_stack() {
    direction = 1;

    // reg_ws->mat_1.setIdentity();
    // // slice_matrix_left(0, reg_ws->mat_1, false);
    // slice_matrix_right(0, reg_ws->mat_1, true);
    // cout << reg_ws->mat_1 << endl << endl;
    // // cout << "Test" << endl << (reg_ws->mat_1  - reg_ws->identity).cwiseAbs().maxCoeff()<< endl << endl;

    // reg_ws->mat_1.setIdentity();
    // slice_matrix_left(0, reg_ws->mat_1, true);
    // // cout << reg_ws->mat_1 << endl << endl;
    // // slice_matrix_right(0, reg_ws->mat_1, false);
    // // cout << "Test" << endl << (reg_ws->mat_1 - reg_ws->identity).cwiseAbs().maxCoeff() << endl << endl;

    // ws->mat_1.setIdentity();
    // // slice_matrix_left_renyi(0, ws->mat_1, false);
    // slice_matrix_right_renyi(0, ws->mat_1, true);
    // cout << ws->mat_1 << endl << endl;
    // // cout << "Test" << endl << (ws->mat_1  - ws->identity).cwiseAbs().maxCoeff()<< endl << endl;

    // ws->mat_1.setIdentity();
    // slice_matrix_left_renyi(0, ws->mat_1, true);
    // // cout << ws->mat_1 << endl << endl;

    // // slice_matrix_right_renyi(0, ws->mat_1, false);
    // // cout << "Test" << endl << (ws->mat_1 - ws->identity).cwiseAbs().maxCoeff() << endl << endl;
		      
    if (first_initialization == true) {
	// cout << p->outp << "Density before " << endl << ws->den_U << endl << endl;

	vol = ws->vol;
	sites = ws->sites;
	
	n_elements = (2 * slices) / p->safe_mult;
	chunks = n_elements;

	svds.resize(vol);
	svd_example.resize(vol);
	stability_checks.resize(n_elements);
	stability.resize(n_elements);

	for (int i = 0; i < n_elements; ++i) {	    
	    stability[i] = 0;
	    u_stack.push_back(cx_mat_t::Zero(N, p->particles));
	    d_stack.push_back(vec_t::Zero(p->particles));
	    t_stack.push_back(cx_mat_t::Zero(p->particles, p->particles));
	}

	try {
	    Ul.resize(vol, vol);
	    Um.resize(vol, vol);
	    Ur.resize(vol, vol);
	    Tl.resize(vol, vol);
	    Tm.resize(vol, vol);
	    Tr.resize(vol, vol);
	    Dl.resize(vol);
	    Dm.resize(vol);
	    Dr.resize(vol);
	    
	    Us.push_back(&Ur);
	    Us.push_back(&Ul);
	    Ds.push_back(&Dr);
	    Ds.push_back(&Dl);
	    Ts.push_back(&Tr);
	    Ts.push_back(&Tl);

	    large_mats.push_back(&ws->large_mat_1);
	    large_mats.push_back(&ws->large_mat_2);
	    large_mats.push_back(&ws->large_mat_3);
	    large_mats.push_back(&ws->large_mat_4);
	    large_mats.push_back(&ws->large_mat_5);

	    large_vecs.push_back(&ws->large_vec_1);
	    large_vecs.push_back(&ws->large_vec_2);
	    large_vecs.push_back(&ws->large_vec_3);

	} catch (const std::bad_alloc&) {
	    cout << "bad alloc when initializing additional items" << endl;
	    throw std::bad_alloc();
	}
    }

    direction = 1;
    current_slice = 0;
    first_initialization = false;
    dir_switch = true;
    u_temp.setIdentity();
    d_temp.setOnes();
    t_temp.setIdentity();
    return;

    {    
	idx = chunks / 4 ;
	for (int s = slices / 2; s > 0; s -= safe_mult) {
	    bra_update(s - safe_mult, s, idx, idx - 1);
	    --idx;
	}

	idx = chunks / 4 - 1;
	for (int s = slices/2; s < slices; s += safe_mult) {
	    ket_update(s, s + safe_mult, idx, idx + 1);
	    ++idx;
	}

	idx = 3 * (chunks / 4);
	for (int s = 3 * (slices / 2); s > slices; s -= safe_mult) {
	    bra_update(s - safe_mult, s, idx, idx - 1);
	    --idx;
	}

	idx = 3 * (chunks / 4) - 1;
	for (int s = 3 * (slices / 2); s < 4 * (slices / 2); s += safe_mult) {
	    ket_update(s, s + safe_mult, idx, idx + 1);
	    ++idx;
	}

	direction = 1;
	current_slice = 0;
    
	log_weight();
    
	dir_switch = true;

	u_temp.setIdentity();
	d_temp.setOnes();
	t_temp.setIdentity();

	alps::RealObservable dummy("Dummy");    
	fresh_det = true;
    
	// cout << "Basic" << endl;
	calculate_greens_basic();
	// cout << "basic" << endl << endl << greens << endl << endl;
	// calculate_greens_general();
	// cout << "general" << endl << endl << greens << endl << endl;

	// calculate_greens_general();
	propagate(dummy);

	if (first_initialization) {
	    // cout << p->outp << "Density" << endl << ws->den_U << endl << endl;
	    // cout << p->outp << d_stack[0].transpose() << endl;
	    // cout << p->outp << d_stack[chunks / 2 - 1].transpose() << endl;
	    // cout << p->outp << d_stack[chunks / 2].transpose() << endl;
	    // cout << p->outp << d_stack[chunks- 1].transpose() << endl;
	    
	
	    if (greens.rows() < 10) {
		cout << greens.trace() << " is renyi trace" << endl;
		cout << greens << endl;
	    }
	}

	first_initialization = false;
    }
}


int cx_dqmc::greens_replica_renyi_gs::propagate(alps::Observable& stability) {
    // if (n_A + 2*n_B != vol) {
    // 	cout << "Current system sizes " << n_A << " - " << n_B
    // 	     << " vs " << vol << endl;	
    // 	dqmc::tools::abort("Incorrect system sizes");
    // }
    vol = n_A + 2*n_B;
    
    if (direction == -1) {
	if ((current_slice == slices || current_slice == 2 * slices)
	    && dir_switch == true) {
	    dir_switch = false;
	    fresh_sign = true;
	    
	    u_temp.setIdentity();
	    d_temp.setOnes();
	    t_temp.setIdentity();
	    
	    calculate_greens();
	    
	    --current_slice;

	    slice_matrix_left_renyi(current_slice, greens, true);
	    slice_matrix_right_renyi(current_slice, greens, false);
	    current_bond = 0;
	    return 0;
	}	

	else if (current_slice == slices / 2) {
	    dir_switch = true;
	    direction = 1;
	    current_slice = slices;
	    idx = chunks / 4 - 1;
	    for (int s = slices/2; s < slices; s += safe_mult) {
		ket_update(s, s + safe_mult, idx, idx + 1);
		++idx;
	    }
	    propagate(stability);
	    return 2;
	}
	
	else if ((current_slice) == (3 * slices) / 2) {
	    dir_switch = true;
	    direction = 1;
	    current_slice = 0;
	    idx = 3 * (chunks / 4) - 1;
	    for (int s = 3 * (slices/2); s < 2 * slices; s += safe_mult) {
		ket_update(s, s + safe_mult, idx, idx + 1);
		++idx;
	    }
	    propagate(stability);
	    return 2;
	    
	} else if (current_slice % safe_mult == 0) {
	    start = current_slice;
	    stop = current_slice + safe_mult;	    
	    idx = current_slice / safe_mult;
	    
	    slice_sequence_left_t(start, stop, u_temp);
	    reg_ws->mat_1 = u_temp * d_temp.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->mat_1,
					    u_temp, d_temp,
					    reg_ws->mat_2);
	    reg_ws->mat_1 = reg_ws->mat_2 * t_temp;
	    t_temp = reg_ws->mat_1;
	    
	    temp_greens = greens;
	    calculate_greens();
	    
	    --current_slice;		
	    slice_matrix_left_renyi(current_slice, greens, true);
	    slice_matrix_right_renyi(current_slice, greens, false);

	} else {
	    --current_slice;		
	    slice_matrix_left_renyi(current_slice, greens, true);
	    slice_matrix_right_renyi(current_slice, greens, false);
	} 
    }
    
    else if (direction == 1) {
	if ((current_slice == slices || current_slice == 0)
	    && dir_switch == true) {
	    dir_switch = false;
	    fresh_sign = true;
	    
	    u_temp.setIdentity();
	    d_temp.setOnes();
	    t_temp.setIdentity();
	    
	    calculate_greens();
	    return 0;
	}
	else if (current_slice + 1 == slices / 2) {
	    dir_switch = true;
	    current_slice = slices;
	    idx = chunks / 4 ;
	    direction = -1;
	    
	    for (int s = slices / 2; s > 0; s -= safe_mult) {
		bra_update(s - safe_mult, s, idx, idx -1 );
		--idx;
	    }
	    propagate(stability);
	    return 2;
	}
	else if (current_slice + 1 == 3 * (slices / 2)) {
	    current_slice = 2 * slices;
	    dir_switch = true;
	    idx = 3 * (chunks / 4) ;
	    direction = -1;
	    for (int s = 3 * (slices / 2); s > slices; s -= safe_mult) {
		bra_update(s - safe_mult, s, idx, idx - 1);
		--idx;
	    }
	    propagate(stability);
	    return 2;
	    
	} 
	else if ((current_slice + 1) % safe_mult == 0) {
	    idx = (current_slice + 1) / safe_mult;
	    start = (current_slice + 1 - safe_mult);
	    stop = current_slice + 1;

	    slice_sequence_left(start, stop, u_temp);
	    reg_ws->mat_1 = u_temp * d_temp.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->mat_1,
					    u_temp, d_temp, reg_ws->mat_2);
	    reg_ws->mat_1 = reg_ws->mat_2 * t_temp;		
	    t_temp = reg_ws->mat_1;
	    slice_matrix_right_renyi(current_slice, greens, true);
	    slice_matrix_left_renyi(current_slice, greens, false);
	    temp_greens = greens;
	    ++current_slice;
	    calculate_greens();
	}
	else {
	    slice_matrix_right_renyi(current_slice, greens, true);
	    slice_matrix_left_renyi(current_slice, greens, false);
	    ++current_slice;
	}
    }
    return 1;
}


double cx_dqmc::greens_replica_renyi_gs::check_stability() {
    static int warnings = 0;
    static pdouble_t diff;
    static int max_warnings = 8 * slices/safe_mult;
    ws->re_mat_1 = (temp_greens - greens).cwiseAbs().real();

    if (p->update_type == p->CX_SPINFUL_UPDATE) {
	diff = ws->re_mat_1.diagonal().maxCoeff();
    } else {
	diff = ws->re_mat_1.maxCoeff();
    }

    if (warnings < max_warnings) {
    	if (diff > 1e-7) {
    	    ++warnings;
	    cout << p->outp << ws->step << " renyi unstable: "
    		 << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
    		 << current_slice << " / " << slices
		 << " (" << n_A << ", " << n_B << ") "
		 << "\t";
    	    cout << setw(15) << std::right << diff << endl;
    	}
    } else if (diff > 1e-1) {
	++warnings;
	cout << p->outp << ws->step << " renyi really unstable: "
	     << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
	     << current_slice << " / " << slices << "\t";
	cout << setw(15) << std::right << diff << " vs. exact " << endl;

    if (n_B == 0) {
	cout << temp_greens << endl << endl;
	cout << "vs" << endl;
	cout << endl << greens << endl << endl;
    }
    

    }
    return diff;
}


void cx_dqmc::greens_replica_renyi_gs::ket_update(int start, int stop,
						  int i, int j) {
    bool add_den = false;

    if ( i < n_elements ) {
	if (start == slices / 2  || start == 3 * (slices/2)) add_den = true;
	
	if (add_den) 
	    reg_ws->col_mat_1 = reg_ws->den_U;
	else
	    reg_ws->col_mat_1 = u_stack[i];

	slice_sequence_left(start, stop, reg_ws->col_mat_1);
	
	if (add_den == true) {
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_1,
					    u_stack[j],
					    d_stack[j], reg_ws->tiny_mat_2);
	    t_stack[j] = reg_ws->tiny_mat_2;
	} 
	else {
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * d_stack[i].asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2, 
					    u_stack[j], d_stack[j],
					    reg_ws->tiny_mat_2);
	    t_stack[j] = reg_ws->tiny_mat_2 * t_stack[i];
	}
    }
}

void cx_dqmc::greens_replica_renyi_gs::bra_update(int start, int stop,
						  int i, int j) {
    bool add_den = false;

    if (i < n_elements) {
	if (stop == slices / 2
	    || stop == 3 * (slices / 2)) add_den = true;

	if (add_den) reg_ws->col_mat_1 = reg_ws->den_U.conjugate();
	else reg_ws->col_mat_1 = u_stack[i];
	slice_sequence_left_t(start, stop, reg_ws->col_mat_1);

	if (add_den == true) {
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_1, u_stack[j],
					    d_stack[j], reg_ws->tiny_mat_2);
	    t_stack[j] = reg_ws->tiny_mat_2;
	} 
	else {
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * d_stack[i].asDiagonal(); 
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2,
					    u_stack[j], d_stack[j],
					    reg_ws->tiny_mat_2);	    
	    t_stack[j] = reg_ws->tiny_mat_2 * t_stack[i];
	} 
    }
    else {
	cout << i << " " << n_elements << endl;
	dqmc::tools::abort("Renyi - bra_update - Not implemented yet");
    }
}

void cx_dqmc::greens_replica_renyi_gs::slice_sequence_left(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left(i, M); }
}


void cx_dqmc::greens_replica_renyi_gs::slice_sequence_left_renyi(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left_renyi(i, M); }
}


void cx_dqmc::greens_replica_renyi_gs::slice_sequence_left_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }   
}


void cx_dqmc::greens_replica_renyi_gs::slice_sequence_left_renyi_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_renyi_t(i, M); }
}


void cx_dqmc::greens_replica_renyi_gs::slice_sequence_right(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;

    if (inv == false) {
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], 1., 0);
	cx_dqmc::checkerboard::hop_left(reg_ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, -1.);
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], -1., 0);
    }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_left_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_left_t(reg_ws, M, 1.);
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], 1., 0);
    } else {
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_left_t(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_left_renyi(int slice, cx_mat_t& M, bool inv) {
    int replica = slice / slices;
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
					       ((*aux_spins))[replica][s],
					       1, replica);
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, 1.,
					      (2 * slice) / slices);
    } else {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, -1.,
					      (2 * slice) / slices);
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
					       ((*aux_spins))[replica][s],
					       -1., replica);
    }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_left_renyi_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_left_renyi_t(ws, M, 1., (2 * slice) / slices);
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
					       (*aux_spins)[replica][s], 1., replica);
    } else {
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
					       (*aux_spins)[replica][s], -1., replica);
	cx_dqmc::checkerboard::hop_left_renyi_t(ws, M, -1., (2 * slice) / slices);
    }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(reg_ws, M, 1.);
	cx_dqmc::interaction::interaction_right(p, reg_ws, M, reg_ws->vec_1,
						(*aux_spins)[replica][s], 1., 0);	
    } else {
	cx_dqmc::interaction::interaction_right(p, reg_ws, M, reg_ws->vec_1,
						(*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_right(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_renyi_gs::slice_matrix_right_renyi(int slice, cx_mat_t& M, bool inv) {
    int replica = slice / slices;
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, 1.,
					       (2 * slice) / slices);
	cx_dqmc::interaction::interaction_right(p, ws, M, ws->vec_1,
						(*aux_spins)[replica][s],
						1, replica);
    } else {
	cx_dqmc::interaction::interaction_right(p, ws, M, ws->vec_1,
						(*aux_spins)[replica][s],
						-1., replica);
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, -1.,
					       (2 * slice) / slices);
    }
}


void cx_dqmc::greens_replica_renyi_gs::hopping_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_renyi_gs::hopping_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(reg_ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_right(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_renyi_gs::hopping_matrix_left_renyi(int slice, cx_mat_t& M, bool inv) {
    int s = slice / slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, 1., (2 * slice) / slices);
    } else {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, -1., (2 * slice) / slices);
    }
}


void cx_dqmc::greens_replica_renyi_gs::hopping_matrix_right_renyi(int slice, cx_mat_t& M, bool inv) {
    int s = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, 1., (2 * slice) / slices);
    } else {
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, -1., (2 * slice) / slices);
    }
}


void cx_dqmc::greens_replica_renyi_gs::regularize_svd(vec_t& in, vec_t& out)  {
    double length = vol - particles - 1;
    double min_log = -32;
    if (n_B != 0) {
	min_log = log10(in.minCoeff()) - 32.;
    } else {
	min_log = log10(in.minCoeff()) - 48.;
    }
    double max_log = min_log;// min(150., min_log + 2 * length);

    for (int i = 0; i < eff_particles; ++i) {
	out(i) = in(i);
    }

    for (int i = eff_particles; i < vol; ++i) {
	out(i) = pow(10, min_log);
    }
}


void cx_dqmc::greens_replica_renyi_gs::log_weight() {
    if (p->full_piv_stable == true
	|| p->full_piv_det == true) {
	fresh_det = true;
	log_weight_full_piv();
	return;
    }

    fresh_det = true;
    
    if (current_slice % (slices / 2) != 0) {
	dqmc::tools::abort("Not at the correct time "
			   "slice to calculate the log weight");
    }

    int is[4], rs[4];
    
    is[0] = 0;
    is[1] = chunks / 2 - 1;
    is[2] = chunks / 2;
    is[3] = chunks - 1;
	    
    rs[0] = 0;
    rs[1] = 0;
    rs[2] = 1;
    rs[3] = 1;

    try {
	dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);
	dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
	enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);	
	enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
	enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

	enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

	enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

	for (int i = 0; i < vol; ++i) {
	    svds[i] = (ws->re_vec_4[i] + ws->re_vec_3[i] + ws->re_vec_2[i] + ws->re_vec_1[i]) * 0.25;
	    svd_example[i] = ws->re_vec_1[i];
	}

    } catch (const std::bad_alloc&) {
	dqmc::tools::abort("Enlaring matrices failed with bad_alloc");
    }

    static cx_mat large_mat_1;
    static cx_mat large_mat_2;
    static cx_mat large_mat_3;
    static vec large_vec_1;

    try {
	if (large_mat_1.rows() != 4 * vol) {
	    large_mat_1.resize(4 * vol, 4 * vol);
	    large_mat_2.resize(4 * vol, 4 * vol);
	    large_mat_3.resize(4 * vol, 4 * vol);
	    large_vec_1.resize(4 * vol);
	}
    } catch (const std::bad_alloc&) {
	dqmc::tools::abort("Resizing matrices failed with bad_alloc");
    }

    static Eigen::PartialPivLU<cx_mat_t> lu1(ws->mat_1);
    static Eigen::PartialPivLU<cx_mat_t> lu2(ws->mat_1);
    static Eigen::PartialPivLU<cx_mat_t> large_lu(large_mat_1);
        
    large_mat_1.setZero();
    
    det_sign = cx_double(1.0, 0);
    
    lu1.compute(ws->mat_8); lu2.compute(ws->mat_2.transpose());
    large_mat_1.block(0, 0, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= lu1.determinant() * (lu2.determinant());

    if (fabs(fabs(det_sign) - 1) > 1e-9) {
	cout << u_stack[is[3]] << endl << endl;
	cout << reg_ws->mat_8 << endl << endl;
	cout << ws->mat_8 << endl << endl;

	cout << p->outp << "log_weight(): abs det is not one " << det_sign << endl;
    }
    
    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_1.transpose()); lu2.compute(ws->mat_3);    
    large_mat_1.block(vol, vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;

    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_4); lu2.compute(ws->mat_6.transpose());
    large_mat_1.block(2*vol, 2*vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;

    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_5.transpose()); lu2.compute(ws->mat_7);
    large_mat_1.block(3*vol, 3*vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;
    // cout << "current det sign\t" << det_sign << endl;
 
    large_mat_1.block(0, 3*vol, vol, vol).diagonal().real()
	= ws->re_vec_4;
    large_mat_1.block(vol, 0, vol, vol).diagonal().real()
	= -ws->re_vec_1;
    large_mat_1.block(2 * vol, vol, vol, vol).diagonal().real()
	= -ws->re_vec_2;
    large_mat_1.block(3 * vol, 2 * vol, vol, vol).diagonal().real()
	= -ws->re_vec_3;

    dqmc::la::decompose_udt_col_piv(large_mat_1, large_mat_2, large_vec_1, large_mat_3);

    large_lu.compute(large_mat_2);
    det_sign *= (large_lu.determinant());
    // cout << "current det sign\t" << det_sign << endl;
    
    if (fabs(fabs(det_sign) - 1) > 1e-9) cout << p->outp << "log_weight(): abs det is not one" << endl;

    large_lu.compute(large_mat_3);
    det_sign *= (large_lu.determinant());
    // cout << "current det sign\t" << det_sign << endl;
    
    if (fabs(fabs(det_sign) - 1) > 1e-9) cout << p->outp << "log_weight(): abs det is not one" << endl;
    
    dqmc::la::log_sum(large_vec_1, log_det);
    // cout << "and the logdet\t" << log_det << endl;
    double prefactor_spin_sum = 0.;
    
    for(auto i = aux_spins->origin(); i < (aux_spins->origin() + aux_spins->num_elements()); ++i) {
    	prefactor_spin_sum += double(*i);	
    }

    // prefactor_spin_sum = 0.;
    // for (uint s = 0; s < p->slices; s++) {
    // 	for (int i = 0; i < num_aux_spins; i++) {
    // 	    prefactor_spin_sum += (*aux_spins)[0][s][i];
    // 	    prefactor_spin_sum += (*aux_spins)[1][s][i];
    // 	}
    // }

    phase = std::exp(-1. * prefactor_spin_sum * p->cx_osi_lambda);
}


void cx_dqmc::greens_replica_renyi_gs::log_weight_full_piv() {
    fresh_det = true;
    if (current_slice % (slices / 2) != 0) {
	dqmc::tools::abort("Not at the correct time "
			   "slice to calculate the log weight");
    }

    int is[4], rs[4];
    
    is[0] = 0;
    is[1] = chunks / 2 - 1;
    is[2] = chunks / 2;
    is[3] = chunks - 1;
	    
    rs[0] = 0;
    rs[1] = 0;
    rs[2] = 1;
    rs[3] = 1;

    try {
	dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);
	dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
	enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
	enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

	enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

	enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

	for (int i = 0; i < vol; ++i) {
	    svds[i] = (ws->re_vec_4[i] + ws->re_vec_3[i] + ws->re_vec_2[i] + ws->re_vec_1[i]) * 0.25;
	    svd_example[i] = ws->re_vec_1[i];
	}

    } catch (const std::bad_alloc&) {
	dqmc::tools::abort("Enlaring matrices failed with bad_alloc");
    }

    static cx_mat large_mat_1;
    static cx_mat large_mat_2;
    static cx_mat large_mat_3;
    static vec large_vec_1;

    try {
	if (large_mat_1.rows() != 4 * vol) {
	    large_mat_1.resize(4 * vol, 4 * vol);
	    large_mat_2.resize(4 * vol, 4 * vol);
	    large_mat_3.resize(4 * vol, 4 * vol);
	    large_vec_1.resize(4 * vol);
	}
    } catch (const std::bad_alloc&) {
	dqmc::tools::abort("Resizing matrices failed with bad_alloc");
    }

    static Eigen::FullPivLU<cx_mat_t> lu1(vol, vol);
    static Eigen::FullPivLU<cx_mat_t> lu2(vol, vol);
    static Eigen::FullPivLU<cx_mat_t> large_lu(4 * vol, 4 * vol);
        
    large_mat_1.setZero();
    
    det_sign = cx_double(1.0, 0);
    
    lu1.compute(ws->mat_8); lu2.compute(ws->mat_2.transpose());
    large_mat_1.block(0, 0, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= lu1.determinant() * (lu2.determinant());

    if (fabs(fabs(det_sign) - 1) > 1e-9) {
	cout << u_stack[is[3]] << endl << endl;
	cout << reg_ws->mat_8 << endl << endl;
	cout << ws->mat_8 << endl << endl;

	cout << p->outp << "log_weight(): abs det is not one " << det_sign << endl;
    }
    
    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_1.transpose()); lu2.compute(ws->mat_3);    
    large_mat_1.block(vol, vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;

    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_4); lu2.compute(ws->mat_6.transpose());
    large_mat_1.block(2*vol, 2*vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;

    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_5.transpose()); lu2.compute(ws->mat_7);
    large_mat_1.block(3*vol, 3*vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;
    // cout << "current det sign\t" << det_sign << endl;
 
    large_mat_1.block(0, 3*vol, vol, vol).diagonal().real()
	= ws->re_vec_4;
    large_mat_1.block(vol, 0, vol, vol).diagonal().real()
	= -ws->re_vec_1;
    large_mat_1.block(2 * vol, vol, vol, vol).diagonal().real()
	= -ws->re_vec_2;
    large_mat_1.block(3 * vol, 2 * vol, vol, vol).diagonal().real()
	= -ws->re_vec_3;

    dqmc::la::decompose_udt_col_piv(large_mat_1, large_mat_2, large_vec_1, large_mat_3);

    large_lu.compute(large_mat_2);
    det_sign *= (large_lu.determinant());
    // cout << "current det sign\t" << det_sign << endl;
    
    if (fabs(fabs(det_sign) - 1) > 1e-9) cout << p->outp << "log_weight(): abs det is not one" << endl;

    large_lu.compute(large_mat_3);
    det_sign *= (large_lu.determinant());
    // cout << "current det sign\t" << det_sign << endl;
    
    if (fabs(fabs(det_sign) - 1) > 1e-9) cout << p->outp << "log_weight(): abs det is not one" << endl;
    
    dqmc::la::log_sum(large_vec_1, log_det);
    // cout << "and the logdet\t" << log_det << endl;
    double prefactor_spin_sum = 0.;
    
    for(auto i = aux_spins->origin(); i < (aux_spins->origin() + aux_spins->num_elements()); ++i) {
    	prefactor_spin_sum += double(*i);	
    }

    // prefactor_spin_sum = 0.;
    // for (uint s = 0; s < p->slices; s++) {
    // 	for (int i = 0; i < num_aux_spins; i++) {
    // 	    prefactor_spin_sum += (*aux_spins)[0][s][i];
    // 	    prefactor_spin_sum += (*aux_spins)[1][s][i];
    // 	}
    // }

    phase = std::exp(-1. * prefactor_spin_sum * p->cx_osi_lambda);
}


void cx_dqmc::greens_replica_renyi_gs::calculate_greens_basic() {
    // boost::timer::auto_cpu_timer t;
    // cout << "greens_replica_renyi_gs::calculate_greens_basic()\t";
    
    int is[5], rs[5];

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();
    U_is_unitary.clear();
    T_is_unitary.clear();
    
    if (current_slice % slices == 0) {
	if ((direction == 1 && current_slice == 0) ||
	    (direction == -1 && current_slice == 2 * slices)) {
	    is[0] = 0;
	    is[1] = chunks / 2 - 1;
	    is[2] = chunks / 2;
	    is[3] = chunks - 1;
	    
	    rs[0] = 0;
	    rs[1] = 0;
	    rs[2] = 1;
	    rs[3] = 1;
	}
	else {
	    is[0] = chunks / 2;
	    is[1] = chunks - 1;
	    is[2] = 0;
	    is[3] = chunks / 2 - 1;

	    rs[0] = 1;
	    rs[1] = 1;
	    rs[2] = 0;
	    rs[3] = 0;
	}

	
	reg_ws->tiny_mat_4 = d_stack[is[3]].asDiagonal()
	    * (t_stack[is[3]] * t_stack[is[2]].transpose())
	    * d_stack[is[2]].asDiagonal();

	dqmc::la::decompose_udt_col_piv(reg_ws->tiny_mat_4,
					reg_ws->tiny_mat_5,
					reg_ws->re_tiny_vec_5,
					reg_ws->tiny_mat_6);
	reg_ws->col_mat_4 = u_stack[is[3]] * reg_ws->tiny_mat_5;
	reg_ws->col_mat_3 = (reg_ws->tiny_mat_6 * u_stack[is[2]].transpose()).transpose();
	
	reg_ws->tiny_mat_1 = d_stack[is[1]].asDiagonal()
	    * (t_stack[is[1]] * t_stack[is[0]].transpose())
	    * d_stack[is[0]].asDiagonal();

	dqmc::la::decompose_udt_col_piv(reg_ws->tiny_mat_1,
					reg_ws->tiny_mat_2,
					reg_ws->re_tiny_vec_2,
					reg_ws->tiny_mat_3);
	reg_ws->col_mat_2 = u_stack[is[1]] * reg_ws->tiny_mat_2;
	reg_ws->col_mat_1 = (reg_ws->tiny_mat_3 * u_stack[is[0]].transpose()).transpose();

	enlarge_thin(rs[3], reg_ws->col_mat_4, ws->col_mat_4);
	enlarge_thin(rs[2], reg_ws->col_mat_3, ws->col_mat_3);
	enlarge_thin(rs[2], reg_ws->re_tiny_vec_5, ws->re_tiny_vec_3);	
	enlarge_thin(rs[1], reg_ws->col_mat_2, ws->col_mat_2);
	enlarge_thin(rs[1], reg_ws->re_tiny_vec_2, ws->re_tiny_vec_2);

	enlarge_thin(rs[0], reg_ws->col_mat_1, ws->col_mat_1);	

	ws->tiny_mat_1 = ws->re_tiny_vec_3.asDiagonal()
	    * ws->col_mat_3.transpose()
	    * ws->col_mat_2
	    * ws->re_tiny_vec_2.asDiagonal();

	dqmc::la::decompose_udt_col_piv(ws->tiny_mat_1,
					ws->tiny_mat_2, 
					ws->re_tiny_vec_2,
					ws->tiny_mat_3);

	ws->col_mat_3 = ws->col_mat_4 * ws->tiny_mat_2;
	ws->col_mat_2 = (ws->tiny_mat_3 * ws->col_mat_1.transpose()).transpose();

	dqmc::la::thin_col_to_invertible(ws->col_mat_3, Ul);
	dqmc::la::thin_col_plus_random(ws->col_mat_2, ws->mat_1);
	Tl = ws->mat_1.transpose();
	regularize_svd(ws->re_tiny_vec_2, Dl);


	dqmc::calculate_greens::basic_udt_col_piv_qr_partial_piv_lu(Ul, Dl, Tl, *ws, greens);
	// dqmc::calculate_greens::basic_udt_col_piv_qr_col_piv_qr(Ul, Dl, Tl, *ws, greens);
	
	// cx_mat_t Ul_backup = Ul;
	// vec_t Dl_backup = Dl;
	// cx_mat_t Tl_backup = Tl;
	// dqmc::calculate_greens::basic_udt_col_piv_qr_partial_piv_lu(Ul, Dl, Tl, *ws, greens);
	// cout << "Greens" << endl << greens << endl << endl;
	// dqmc::calculate_greens::basic_udt_col_piv_qr_col_piv_qr(Ul_backup, Dl_backup, Tl_backup, *ws, greens);
	// cout << "Greens" << endl << greens << endl << endl;

    }

    // 2 * vol - up
    else if (direction != 4) {
    	if ((direction == 1 && (current_slice + 1) < slices / 2
    	     && current_slice > 0)
    	    || (direction == 1 && (current_slice + 1) < 3 * (slices / 2)
    		&& current_slice > slices)) {

    	    // cout << "Combining my way to the top" << endl;
    	    if (current_slice < slices / 2) {
    		is[0] = current_slice / safe_mult;
    		is[1] = chunks / 2 - 1;
    		is[2] = chunks / 2;
    		is[3] = chunks - 1;

    		rs[0] = 0;
    		rs[1] = 0;
    		rs[2] = 1;
    		rs[3] = 1;
    		rs[4] = 0;	    
    	    }
    	    else {
    		is[0] = current_slice / safe_mult;
    		is[1] = chunks - 1;
    		is[2] = 0;
    		is[3] = chunks / 2 - 1;

    		rs[0] = 1;
    		rs[1] = 1;
    		rs[2] = 0;
    		rs[3] = 0;
    		rs[4] = 1;
    	    }

    	    {
		if (n_B != 0) {
		    enlarge_thin(rs[0], u_stack[is[0]], ws->col_mat_1);
		    enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
		    enlarge_thin(rs[0], t_stack[is[0]], ws->tiny_mat_1);

		    enlarge_thin(rs[1], u_stack[is[1]], ws->col_mat_2);
		    enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
		    enlarge_thin(rs[1], t_stack[is[1]], ws->tiny_mat_2);
		    
		    enlarge_thin(rs[2], u_stack[is[2]], ws->col_mat_3);
		    enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
		    enlarge_thin(rs[2], t_stack[is[2]], ws->tiny_mat_3);
	
		    enlarge_thin(rs[3], u_stack[is[3]], ws->col_mat_4);
		    enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
		    enlarge_thin(rs[3], t_stack[is[3]], ws->tiny_mat_4);

		    enlarge(rs[4], u_temp, ws->mat_1);
		    enlarge(rs[4], d_temp, ws->re_vec_1);
		    enlarge(rs[4], t_temp, ws->mat_2);
		} else {
		    ws->col_mat_1 = u_stack[is[0]];
		    ws->re_tiny_vec_1 = d_stack[is[0]];
		    ws->tiny_mat_1 = t_stack[is[0]];

		    ws->col_mat_2 = u_stack[is[1]];
		    ws->re_tiny_vec_2 = d_stack[is[1]];
		    ws->tiny_mat_2 = t_stack[is[1]];

		    ws->col_mat_3 = u_stack[is[2]];
		    ws->re_tiny_vec_3 = d_stack[is[2]];
		    ws->tiny_mat_3 = t_stack[is[2]];

		    ws->col_mat_4 = u_stack[is[3]];
		    ws->re_tiny_vec_4 = d_stack[is[3]];
		    ws->tiny_mat_4 = t_stack[is[3]];

		    ws->mat_1 = u_temp;
		    ws->re_vec_1 = d_temp;
		    ws->mat_2 = t_temp;
		}

    		//============================================================

    		ws->col_mat_5 = ws->mat_2 * ws->col_mat_4;
    		ws->col_mat_4 = ws->re_vec_1.asDiagonal() * ws->col_mat_5;
    		ws->col_mat_5 = ws->col_mat_4 * ws->re_tiny_vec_4.asDiagonal();

    		dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
						ws->re_tiny_vec_4, ws->tiny_mat_5);

    		ws->col_mat_4 = ws->mat_1 * ws->col_mat_6;
    		ws->tiny_mat_6 = ws->tiny_mat_5 * ws->tiny_mat_4;
	    
    		ws->tiny_mat_5 = ws->tiny_mat_6 * ws->tiny_mat_3.transpose();
    		ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal() * ws->tiny_mat_5;
    		ws->tiny_mat_5 = ws->tiny_mat_4 * ws->re_tiny_vec_3.asDiagonal();

    		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5, ws->tiny_mat_6,
						ws->re_tiny_vec_6, ws->tiny_mat_7);
	    
    		ws->col_mat_5 = ws->col_mat_4 * ws->tiny_mat_6;
		// ws->re_tiny_vec_6
    		ws->col_mat_4 = (ws->tiny_mat_7 * ws->col_mat_3.transpose()).transpose();

    		//============================================================

    		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->tiny_mat_1.transpose();
    		ws->tiny_mat_2 = ws->re_tiny_vec_2.asDiagonal() * ws->tiny_mat_5;
    		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->re_tiny_vec_1.asDiagonal();

    		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
						ws->tiny_mat_6, ws->re_tiny_vec_2,
						ws->tiny_mat_7);
	
    		ws->col_mat_3 = ws->col_mat_2 * ws->tiny_mat_6;
		// ws->re_tiny_vec_2	
    		ws->col_mat_2 = ws->col_mat_1 * ws->tiny_mat_7.transpose();

		// -----------

		ws->tiny_mat_1 = ws->re_tiny_vec_6.asDiagonal()
		    * ws->col_mat_4.transpose()
		    * ws->col_mat_3
		    * ws->re_tiny_vec_2.asDiagonal();
		
		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_1,
						ws->tiny_mat_2, 
						ws->re_tiny_vec_2,
						ws->tiny_mat_3);

		ws->col_mat_3 = ws->col_mat_5 * ws->tiny_mat_2;
		ws->col_mat_4 = (ws->tiny_mat_3 * ws->col_mat_2.transpose()).transpose();
		
		dqmc::la::thin_col_to_invertible(ws->col_mat_3, Ul);
		
		dqmc::la::thin_col_plus_random(ws->col_mat_4, ws->mat_1);
		Tl = ws->mat_1.transpose();
		regularize_svd(ws->re_tiny_vec_2, Dl);
		dqmc::calculate_greens::basic_udt_col_piv_qr_partial_piv_lu(Ul, Dl, Tl, *ws, greens);
		// dqmc::calculate_greens::basic_udt_col_piv_qr_col_piv_qr(Ul, Dl, Tl, *ws, greens);

    		return;
    	    }
    	}

    	// 2 * vol - down
    	else if ((direction == -1 && (current_slice) < slices
    		  && current_slice > slices / 2)
    		 || (direction == -1 && (current_slice) < 2 * slices
    		     && current_slice > 3 * (slices / 2))) {

    	    // cout << "Combining my way to the bottom" << endl;
    	    if (current_slice < slices) {
    		is[0] = chunks / 2;
    		is[1] = chunks - 1;
    		is[2] = 0;
    		is[3] = chunks / 2 - 1 - (slices - current_slice) / safe_mult;

    		rs[0] = 1;
    		rs[1] = 1;
    		rs[2] = 0;
    		rs[3] = 0;


    		enlarge_thin(1, u_stack[is[0]], ws->col_mat_1);
    		enlarge_thin(1, d_stack[is[0]], ws->re_tiny_vec_1);
    		enlarge_thin(1, t_stack[is[0]], ws->tiny_mat_1);
	
    		enlarge_thin(1, u_stack[is[1]], ws->col_mat_2);
    		enlarge_thin(1, d_stack[is[1]], ws->re_tiny_vec_2);
    		enlarge_thin(1, t_stack[is[1]], ws->tiny_mat_2);
	
    		enlarge_thin(0, u_stack[is[2]], ws->col_mat_3);
    		enlarge_thin(0, d_stack[is[2]], ws->re_tiny_vec_3);
    		enlarge_thin(0, t_stack[is[2]], ws->tiny_mat_3);

    		enlarge_thin(0, u_stack[is[3]], ws->col_mat_4);
    		enlarge_thin(0, d_stack[is[3]], ws->re_tiny_vec_4);
    		enlarge_thin(0, t_stack[is[3]], ws->tiny_mat_4);

    		enlarge(0, u_temp, ws->mat_1);
    		enlarge(0, d_temp, ws->re_vec_1);
    		enlarge(0, t_temp, ws->mat_2);
    	    }
    	    else {
    		is[0] = 0;
    		is[1] = chunks / 2 - 1;
    		is[2] = chunks / 2;
    		is[3] = chunks - 1 - (2*slices - current_slice) / safe_mult; //chunks - 1;
	    
    		rs[0] = 0;
    		rs[1] = 0;
    		rs[2] = 1;
    		rs[3] = 1;
	    
    		// cout << "Chunk " << is[3] << " " << chunks << endl;

    		enlarge_thin(0, u_stack[is[0]], ws->col_mat_1);
    		enlarge_thin(0, d_stack[is[0]], ws->re_tiny_vec_1);
    		enlarge_thin(0, t_stack[is[0]], ws->tiny_mat_1);
	
    		enlarge_thin(0, u_stack[is[1]], ws->col_mat_2);
    		enlarge_thin(0, d_stack[is[1]], ws->re_tiny_vec_2);
    		enlarge_thin(0, t_stack[is[1]], ws->tiny_mat_2);
	
    		enlarge_thin(1, u_stack[is[2]], ws->col_mat_3);
    		enlarge_thin(1, d_stack[is[2]], ws->re_tiny_vec_3);
    		enlarge_thin(1, t_stack[is[2]], ws->tiny_mat_3);

    		enlarge_thin(1, u_stack[is[3]], ws->col_mat_4);
    		enlarge_thin(1, d_stack[is[3]], ws->re_tiny_vec_4);
    		enlarge_thin(1, t_stack[is[3]], ws->tiny_mat_4);

    		enlarge(1, u_temp, ws->mat_1);
    		enlarge(1, d_temp, ws->re_vec_1);
    		enlarge(1, t_temp, ws->mat_2);
	    
    		// cout << "Other direction" << endl;
    	    }

    	    //============================================================

    	    ws->tiny_mat_5 = ws->tiny_mat_4 * ws->tiny_mat_3.transpose();
    	    ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal() * ws->tiny_mat_5;
    	    ws->tiny_mat_5 = ws->tiny_mat_4 * ws->re_tiny_vec_3.asDiagonal();
	
    	    dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
					    ws->tiny_mat_6, ws->re_tiny_vec_6,
					    ws->tiny_mat_7);
	
    	    ws->col_mat_5 = ws->col_mat_4 * ws->tiny_mat_6;
	    // ws->re_tiny_vec_6
	    ws->col_mat_4 = (ws->tiny_mat_7 * ws->col_mat_3.transpose()).transpose();
    	    //============================================================

    	    ws->col_mat_3 = ws->re_vec_1.asDiagonal()
		* (ws->mat_2
		   * ws->col_mat_1)
		*ws->re_tiny_vec_1.asDiagonal();
	    
    	    dqmc::la::decompose_udt_col_piv(ws->col_mat_3, ws->col_mat_6,
					    ws->re_tiny_vec_4, ws->tiny_mat_5);

    	    ws->col_mat_1 = ws->mat_1 * ws->col_mat_6;
	    
	    ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal()
		* (ws->tiny_mat_5 * ws->tiny_mat_1 * ws->tiny_mat_2.transpose())
		* ws->re_tiny_vec_2.asDiagonal();

    	    dqmc::la::decompose_udt_col_piv(ws->tiny_mat_4, ws->tiny_mat_6,
					    ws->re_tiny_vec_2, ws->tiny_mat_7);

    	    ws->col_mat_3 = (ws->tiny_mat_7 * ws->col_mat_2.transpose()).transpose();
   	    ws->col_mat_2 = ws->col_mat_1 * ws->tiny_mat_6;

	    // ============================================================


	    ws->tiny_mat_1 = ws->re_tiny_vec_6.asDiagonal()
		* ws->col_mat_4.transpose()
		* ws->col_mat_3
		* ws->re_tiny_vec_2.asDiagonal();
		
	    dqmc::la::decompose_udt_col_piv(ws->tiny_mat_1,
					    ws->tiny_mat_2, 
					    ws->re_tiny_vec_2,
					    ws->tiny_mat_3);

	    ws->col_mat_3 = ws->col_mat_5 * ws->tiny_mat_2;
	    ws->col_mat_4 = (ws->tiny_mat_3 * ws->col_mat_2.transpose()).transpose();
		
	    dqmc::la::thin_col_to_invertible(ws->col_mat_3, Ul);
		
	    dqmc::la::thin_col_plus_random(ws->col_mat_4, ws->mat_1);
	    Tl = ws->mat_1.transpose();
	    regularize_svd(ws->re_tiny_vec_2, Dl);

	    dqmc::calculate_greens::basic_udt_col_piv_qr_partial_piv_lu(Ul, Dl, Tl, *ws, greens);
	    // dqmc::calculate_greens::basic_udt_col_piv_qr_col_piv_qr(Ul, Dl, Tl, *ws, greens);
	    // ws->mat_1 = Tl.adjoint().partialPivLu().solve(Ul);
	    // ws->mat_5 = ws->mat_1.adjoint();
	    // ws->mat_5.diagonal().real() += Dl;
		
	    // dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
		
	    // ws->mat_4 = ws->mat_3 * Tl;
	    // ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Ul.adjoint();
	    // greens = ws->mat_4.partialPivLu().solve(ws->mat_5);
    	    return;
    	}
    }
}

// from alternative
void cx_dqmc::greens_replica_renyi_gs::calculate_greens_half_compressed() {
    // boost::timer::auto_cpu_timer t;
    // cout << "greens_replica_renyi_gs::calculate_greens_half_compressed() " << current_slice << "\t";
    
    int is[5], rs[5];

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();
    U_is_unitary.clear();
    T_is_unitary.clear();
    
    if (current_slice % slices == 0) {
	if ((direction == 1 && current_slice == 0) ||
	    (direction == -1 && current_slice == 2 * slices)) {
	    is[0] = 0;
	    is[1] = chunks / 2 - 1;
	    is[2] = chunks / 2;
	    is[3] = chunks - 1;
	    
	    rs[0] = 0;
	    rs[1] = 0;
	    rs[2] = 1;
	    rs[3] = 1;
	}
	else {
	    is[0] = chunks / 2;
	    is[1] = chunks - 1;
	    is[2] = 0;
	    is[3] = chunks / 2 - 1;

	    rs[0] = 1;
	    rs[1] = 1;
	    rs[2] = 0;
	    rs[3] = 0;
	}
	dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);    
	dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
	enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
	enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

	enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

	enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();

	Us.push_back(&ws->mat_2); U_is_unitary.push_back(0);
	Us.push_back(&ws->mat_4); U_is_unitary.push_back(1);
	Us.push_back(&ws->mat_6); U_is_unitary.push_back(0);
	Us.push_back(&ws->mat_8); U_is_unitary.push_back(1);

	Ts.push_back(&ws->mat_1); T_is_unitary.push_back(1);
	Ts.push_back(&ws->mat_3); T_is_unitary.push_back(0);
	Ts.push_back(&ws->mat_5); T_is_unitary.push_back(1);
	Ts.push_back(&ws->mat_7); T_is_unitary.push_back(0);
	
	Ds.push_back(&ws->re_vec_1);
	Ds.push_back(&ws->re_vec_2);
	Ds.push_back(&ws->re_vec_3);
	Ds.push_back(&ws->re_vec_4);

	cx_mat_t large_mat_1 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_2 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_3 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_4 = cx_mat::Zero(4 * vol, 4 * vol);
    
	large_mats.push_back(&large_mat_1);
	large_mats.push_back(&large_mat_2);
	large_mats.push_back(&large_mat_3);
	large_mats.push_back(&large_mat_4);

	cx_mat_t large_U = cx_mat_t::Zero(4 * vol, 4 * vol);
	cx_mat_t large_T = cx_mat_t::Zero(4 * vol, 4 * vol);
    
	vec_t large_vec_1 = vec_t::Zero(4 * vol);
	vec_t large_vec_2 = vec_t::Zero(4 * vol);
	vec_t large_vec_3 = vec_t::Zero(4 * vol);
	vec_t large_vec_4 = vec_t::Zero(4 * vol);
    
	large_vecs.push_back(&large_vec_1);
	large_vecs.push_back(&large_vec_2);
	large_vecs.push_back(&large_vec_3);
	large_vecs.push_back(&large_vec_4);

	dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
							  Ds,
							  Ts, T_is_unitary,
							  large_mats, 
							  large_U, large_T,
							  large_vecs, *ws,
							  greens);    
    }

    // 2 * vol - up
    else if (direction == 1) {
	if (((current_slice + 1) < slices / 2 && current_slice > 0) ||
	    ((current_slice + 1) < 3 * (slices / 2) && current_slice > slices)) {

	    // cout << "Combining my way to the top" << endl;
	    if (current_slice < slices / 2) {
		is[0] = current_slice / safe_mult;
		is[1] = chunks / 2 - 1;
		is[2] = chunks / 2;
		is[3] = chunks - 1;

		rs[0] = 0;
		rs[1] = 0;
		rs[2] = 1;
		rs[3] = 1;
		rs[4] = 0;	    
	    }
	    else {
		is[0] = current_slice / safe_mult;
		is[1] = chunks - 1;
		is[2] = 0;
		is[3] = chunks / 2 - 1;

		rs[0] = 1;
		rs[1] = 1;
		rs[2] = 0;
		rs[3] = 0;
		rs[4] = 1;
	    }

	    {
		enlarge_thin(rs[0], u_stack[is[0]], ws->col_mat_1);
		enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
		enlarge_thin(rs[0], t_stack[is[0]], ws->tiny_mat_1);

		enlarge_thin(rs[1], u_stack[is[1]], ws->col_mat_2);
		enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
		enlarge_thin(rs[1], t_stack[is[1]], ws->tiny_mat_2);
		    
		enlarge_thin(rs[2], u_stack[is[2]], ws->col_mat_3);
		enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
		enlarge_thin(rs[2], t_stack[is[2]], ws->tiny_mat_3);
	
		enlarge_thin(rs[3], u_stack[is[3]], ws->col_mat_4);
		enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
		enlarge_thin(rs[3], t_stack[is[3]], ws->tiny_mat_4);

		enlarge(rs[4], u_temp, ws->mat_1);
		enlarge(rs[4], d_temp, ws->re_vec_1);
		enlarge(rs[4], t_temp, ws->mat_2);

		//============================================================

		ws->col_mat_5 = ws->mat_2 * ws->col_mat_4;
		ws->col_mat_4 = ws->re_vec_1.asDiagonal() * ws->col_mat_5;
		ws->col_mat_5 = ws->col_mat_4 * ws->re_tiny_vec_4.asDiagonal();

		dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
						ws->re_tiny_vec_4, ws->tiny_mat_5);

		ws->col_mat_4 = ws->mat_1 * ws->col_mat_6;
		ws->tiny_mat_6 = ws->tiny_mat_5 * ws->tiny_mat_4;	    
		ws->tiny_mat_5 = ws->tiny_mat_6 * ws->tiny_mat_3.transpose();
		ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal() * ws->tiny_mat_5;
		ws->tiny_mat_5 = ws->tiny_mat_4 * ws->re_tiny_vec_3.asDiagonal();

		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5, ws->tiny_mat_6,
						ws->re_tiny_vec_6, ws->tiny_mat_7);
	    
		ws->col_mat_5 = ws->col_mat_4 * ws->tiny_mat_6;
		// dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		dqmc::la::thin_col_plus_random(ws->col_mat_5, ws->mat_3);
		Ul = ws->mat_3;
		regularize_svd(ws->re_tiny_vec_6, Dl);

		ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_3.transpose();
	    
		// dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
		dqmc::la::thin_row_plus_random(ws->row_mat_1, ws->mat_3);
		Tl = ws->mat_3;

		//============================================================

		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->tiny_mat_1.transpose();
		ws->tiny_mat_2 = ws->re_tiny_vec_2.asDiagonal() * ws->tiny_mat_5;
		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->re_tiny_vec_1.asDiagonal();

		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
						ws->tiny_mat_6, ws->re_tiny_vec_6,
						ws->tiny_mat_7);
	
		ws->col_mat_5 = ws->col_mat_2 * ws->tiny_mat_6;
		// dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		dqmc::la::thin_col_plus_random(ws->col_mat_5, ws->mat_3);
		Ur = ws->mat_3;
	    
		regularize_svd(ws->re_tiny_vec_6, Dr);
	
		ws->col_mat_2 = ws->col_mat_1 * ws->tiny_mat_7.transpose();
		ws->row_mat_1 = ws->col_mat_2.transpose();
		// dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
		dqmc::la::thin_row_plus_random(ws->row_mat_1, ws->mat_3);
		Tr = ws->mat_3;

		Us.push_back(&Ur); U_is_unitary.push_back(0);
		Us.push_back(&Ul); U_is_unitary.push_back(0);
		Ds.push_back(&Dr);
		Ds.push_back(&Dl);
		Ts.push_back(&Tr); T_is_unitary.push_back(0);
		Ts.push_back(&Tl); T_is_unitary.push_back(0);
	
		large_mats.push_back(&ws->large_mat_1);
		large_mats.push_back(&ws->large_mat_2);
		large_mats.push_back(&ws->large_mat_3);
	
		large_vecs.push_back(&ws->large_vec_1);
		large_vecs.push_back(&ws->large_vec_2);
		large_vecs.push_back(&ws->large_vec_3);

		dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us,
								  U_is_unitary,
								  Ds, Ts, T_is_unitary,
								  large_mats,
								  ws->large_mat_4,
								  ws->large_mat_5,
								  large_vecs, *ws,
								  greens);
		return;
	    }
	}
    }

    else if (direction == -1) {
	// 2 * vol - down
	if ((current_slice < slices && current_slice > slices / 2) ||
	    (current_slice < 2 * slices && current_slice > 3 * (slices / 2))) {

	    // cout << "Combining my way to the bottom" << endl;
	    if (current_slice < slices) {
		is[0] = chunks / 2;
		is[1] = chunks - 1;
		is[2] = 0;
		is[3] = chunks / 2 - 1 - (slices - current_slice) / safe_mult;

		rs[0] = 1;
		rs[1] = 1;
		rs[2] = 0;
		rs[3] = 0;
		rs[4] = 0;
	    }
	    else {
		is[0] = 0;
		is[1] = chunks / 2 - 1;
		is[2] = chunks / 2;
		is[3] = chunks - 1 - (2*slices - current_slice) / safe_mult; //chunks - 1;
	    
		rs[0] = 0;
		rs[1] = 0;
		rs[2] = 1;
		rs[3] = 1;
		rs[4] = 1;
	    
		// cout << "Chunk " << is[3] << " " << chunks << endl;
	    }

	    enlarge_thin(rs[0], u_stack[is[0]], ws->col_mat_1);
	    enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	    enlarge_thin(rs[0], t_stack[is[0]], ws->tiny_mat_1);
	
	    enlarge_thin(rs[1], u_stack[is[1]], ws->col_mat_2);
	    enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	    enlarge_thin(rs[1], t_stack[is[1]], ws->tiny_mat_2);
	
	    enlarge_thin(rs[2], u_stack[is[2]], ws->col_mat_3);
	    enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	    enlarge_thin(rs[2], t_stack[is[2]], ws->tiny_mat_3);

	    enlarge_thin(rs[3], u_stack[is[3]], ws->col_mat_4);
	    enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	    enlarge_thin(rs[3], t_stack[is[3]], ws->tiny_mat_4);

	    enlarge(rs[4], u_temp, ws->mat_1);
	    enlarge(rs[4], d_temp, ws->re_vec_1);
	    enlarge(rs[4], t_temp, ws->mat_2);

	    //============================================================

	    ws->tiny_mat_5 = ws->tiny_mat_4 * ws->tiny_mat_3.transpose();
	    ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal() * ws->tiny_mat_5;
	    ws->tiny_mat_5 = ws->tiny_mat_4 * ws->re_tiny_vec_3.asDiagonal();
	
	    dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
					    ws->tiny_mat_6, ws->re_tiny_vec_6,
					    ws->tiny_mat_7);
	
	    ws->col_mat_5 = ws->col_mat_4 * ws->tiny_mat_6;
	    dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
	    Ul = ws->mat_3;
		    
	    // enlarge_thin(rs[2], reg_ws->re_tiny_vec_6, ws->re_tiny_vec_6); //
	    regularize_svd(ws->re_tiny_vec_6, Dl);
	
	    ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_3.transpose();
	    dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
	    Tl = ws->mat_3;

	    // cout << "Trivial part done" << endl;
	    //============================================================

	    ws->col_mat_5 = ws->mat_2 * ws->col_mat_1;
	    ws->col_mat_4 = ws->re_vec_1.asDiagonal() * ws->col_mat_5;
	    ws->col_mat_5 = ws->col_mat_4 * ws->re_tiny_vec_1.asDiagonal();
	    
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
					    ws->re_tiny_vec_4, ws->tiny_mat_5);

	    ws->col_mat_4 = ws->mat_1 * ws->col_mat_6;
	    ws->tiny_mat_6 = ws->tiny_mat_5 * ws->tiny_mat_1;
	    ws->tiny_mat_5 = ws->tiny_mat_6 * ws->tiny_mat_2.transpose();
	    ws->tiny_mat_4 = ws->re_tiny_vec_4.asDiagonal() * ws->tiny_mat_5;
	    ws->tiny_mat_5 = ws->tiny_mat_4 * ws->re_tiny_vec_2.asDiagonal();

	    dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5, ws->tiny_mat_6,
					    ws->re_tiny_vec_6, ws->tiny_mat_7);
	    ws->col_mat_5 = ws->col_mat_4 * ws->tiny_mat_6;
	    // dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
	    dqmc::la::thin_col_plus_random(ws->col_mat_5, ws->mat_3);
	    Tr = ws->mat_3.transpose();
	    regularize_svd(ws->re_tiny_vec_6, Dr);

	    ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_2.transpose();
	    // dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
	    dqmc::la::thin_row_plus_random(ws->row_mat_1, ws->mat_3);
	    Ur = ws->mat_3.transpose();

	    Us.push_back(&Ur); U_is_unitary.push_back(0);
	    Us.push_back(&Ul); U_is_unitary.push_back(0);
	    Ds.push_back(&Dr);
	    Ds.push_back(&Dl);
	    Ts.push_back(&Tr); T_is_unitary.push_back(0);
	    Ts.push_back(&Tl); T_is_unitary.push_back(0);
	
	    large_mats.push_back(&ws->large_mat_1);
	    large_mats.push_back(&ws->large_mat_2);
	    large_mats.push_back(&ws->large_mat_3);
	
	    large_vecs.push_back(&ws->large_vec_1);
	    large_vecs.push_back(&ws->large_vec_2);
	    large_vecs.push_back(&ws->large_vec_3);

	    dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
							      Ds, Ts, T_is_unitary,
							      large_mats,
							      ws->large_mat_4,
							      ws->large_mat_5,
							      large_vecs, *ws,
							      greens);	
	    return;
	}
    }
}


void cx_dqmc::greens_replica_renyi_gs::calculate_greens_general() {
    // boost::timer::auto_cpu_timer t;
    // cout << "greens_replica_renyi_gs::calculate_greens_general() " << current_slice << "\t";
    
    int is[5], rs[5];

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();
    U_is_unitary.clear();
    T_is_unitary.clear();
    
    if (current_slice % slices == 0) {
	if ((direction == 1 && current_slice == 0) ||
	    (direction == -1 && current_slice == 2 * slices)) {
	    is[0] = 0;
	    is[1] = chunks / 2 - 1;
	    is[2] = chunks / 2;
	    is[3] = chunks - 1;
	    
	    rs[0] = 0;
	    rs[1] = 0;
	    rs[2] = 1;
	    rs[3] = 1;
	}
	else {
	    is[0] = chunks / 2;
	    is[1] = chunks - 1;
	    is[2] = 0;
	    is[3] = chunks / 2 - 1;

	    rs[0] = 1;
	    rs[1] = 1;
	    rs[2] = 0;
	    rs[3] = 0;
	}
	dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);    
	dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
	enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
	enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

	enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

	enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();

	Us.push_back(&ws->mat_2); U_is_unitary.push_back(0);
	Us.push_back(&ws->mat_4); U_is_unitary.push_back(1);
	Us.push_back(&ws->mat_6); U_is_unitary.push_back(0);
	Us.push_back(&ws->mat_8); U_is_unitary.push_back(1);

	Ts.push_back(&ws->mat_1); T_is_unitary.push_back(1);
	Ts.push_back(&ws->mat_3); T_is_unitary.push_back(0);
	Ts.push_back(&ws->mat_5); T_is_unitary.push_back(1);
	Ts.push_back(&ws->mat_7); T_is_unitary.push_back(0);
	
	Ds.push_back(&ws->re_vec_1);
	Ds.push_back(&ws->re_vec_2);
	Ds.push_back(&ws->re_vec_3);
	Ds.push_back(&ws->re_vec_4);

	cx_mat_t large_mat_1 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_2 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_3 = cx_mat::Zero(4 * vol, 4 * vol);
	cx_mat_t large_mat_4 = cx_mat::Zero(4 * vol, 4 * vol);
    
	large_mats.push_back(&large_mat_1);
	large_mats.push_back(&large_mat_2);
	large_mats.push_back(&large_mat_3);
	large_mats.push_back(&large_mat_4);

	cx_mat_t large_U = cx_mat_t::Zero(4 * vol, 4 * vol);
	cx_mat_t large_T = cx_mat_t::Zero(4 * vol, 4 * vol);
    
	vec_t large_vec_1 = vec_t::Zero(4 * vol);
	vec_t large_vec_2 = vec_t::Zero(4 * vol);
	vec_t large_vec_3 = vec_t::Zero(4 * vol);
	vec_t large_vec_4 = vec_t::Zero(4 * vol);
    
	large_vecs.push_back(&large_vec_1);
	large_vecs.push_back(&large_vec_2);
	large_vecs.push_back(&large_vec_3);
	large_vecs.push_back(&large_vec_4);

	dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
							  Ds,
							  Ts, T_is_unitary,
							  large_mats, 
							  large_U, large_T,
							  large_vecs, *ws,
							  greens);    
    }

    // 2 * vol - up
    else if (direction == 1) {
	if (((current_slice + 1) < slices / 2 && current_slice > 0) ||
	    ((current_slice + 1) < 3 * (slices / 2) && current_slice > slices)) {

	    // cout << "Combining my way to the top" << endl;
	    if (current_slice < slices / 2) {
		is[0] = current_slice / safe_mult;
		is[1] = chunks / 2 - 1;
		is[2] = chunks / 2;
		is[3] = chunks - 1;

		rs[0] = 0;
		rs[1] = 0;
		rs[2] = 1;
		rs[3] = 1;
		rs[4] = 0;	    
	    }
	    else {
		is[0] = current_slice / safe_mult;
		is[1] = chunks - 1;
		is[2] = 0;
		is[3] = chunks / 2 - 1;

		rs[0] = 1;
		rs[1] = 1;
		rs[2] = 0;
		rs[3] = 0;
		rs[4] = 1;
	    }

	    {
		dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
		enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);    
		dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
		enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
		dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
		enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
		dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
		enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

		enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
		regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
		enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
		enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
		regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
		enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

		enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
		regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
		enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

		enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
		regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
		enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

		enlarge(rs[4], u_temp, ws->mat_10);
		enlarge(rs[4], d_temp, ws->re_vec_5);
		enlarge(rs[4], t_temp, ws->mat_9);

		ws->la_mat_1 = ws->mat_1;
		ws->mat_1 = ws->mat_2.transpose();
		ws->mat_2 = ws->la_mat_1.transpose();
		ws->la_mat_1 = ws->mat_5;
		ws->mat_5 = ws->mat_6.transpose();
		ws->mat_6 = ws->la_mat_1.transpose();

		Us.push_back(&ws->mat_2); U_is_unitary.push_back(0);
		Us.push_back(&ws->mat_4); U_is_unitary.push_back(1);
		Us.push_back(&ws->mat_6); U_is_unitary.push_back(0);
		Us.push_back(&ws->mat_8); U_is_unitary.push_back(1);
		Us.push_back(&ws->mat_10); U_is_unitary.push_back(1);

		Ts.push_back(&ws->mat_1); T_is_unitary.push_back(1);
		Ts.push_back(&ws->mat_3); T_is_unitary.push_back(0);
		Ts.push_back(&ws->mat_5); T_is_unitary.push_back(1);
		Ts.push_back(&ws->mat_7); T_is_unitary.push_back(0);
		Ts.push_back(&ws->mat_9); T_is_unitary.push_back(0);

		Ds.push_back(&ws->re_vec_1);
		Ds.push_back(&ws->re_vec_2);
		Ds.push_back(&ws->re_vec_3);
		Ds.push_back(&ws->re_vec_4);
		Ds.push_back(&ws->re_vec_5);

		cx_mat_t large_mat_1 = cx_mat::Zero(5 * vol, 5 * vol);
		cx_mat_t large_mat_2 = cx_mat::Zero(5 * vol, 5 * vol);
		cx_mat_t large_mat_3 = cx_mat::Zero(5 * vol, 5 * vol);
		cx_mat_t large_mat_4 = cx_mat::Zero(5 * vol, 5 * vol);
    
		large_mats.push_back(&large_mat_1);
		large_mats.push_back(&large_mat_2);
		large_mats.push_back(&large_mat_3);
		large_mats.push_back(&large_mat_4);

		cx_mat_t large_U = cx_mat_t::Zero(5 * vol, 5 * vol);
		cx_mat_t large_T = cx_mat_t::Zero(5 * vol, 5 * vol);
    
		vec_t large_vec_1 = vec_t::Zero(5 * vol);
		vec_t large_vec_2 = vec_t::Zero(5 * vol);
		vec_t large_vec_3 = vec_t::Zero(5 * vol);
		vec_t large_vec_4 = vec_t::Zero(5 * vol);
    
		large_vecs.push_back(&large_vec_1);
		large_vecs.push_back(&large_vec_2);
		large_vecs.push_back(&large_vec_3);
		large_vecs.push_back(&large_vec_4);

		dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
								  Ds,
								  Ts, T_is_unitary,
								  large_mats, 
								  large_U, large_T,
								  large_vecs, *ws,
								  greens);    		
		return;
	    }
	}
    }

    else if (direction == -1) {
	// 2 * vol - down
	if ((current_slice < slices && current_slice > slices / 2) ||
	    (current_slice < 2 * slices && current_slice > 3 * (slices / 2))) {

	    // cout << "Combining my way to the bottom" << endl;
	    if (current_slice < slices) {
		is[0] = chunks / 2;
		is[1] = chunks - 1;
		is[2] = 0;
		is[3] = chunks / 2 - 1 - (slices - current_slice) / safe_mult;

		rs[0] = 1;
		rs[1] = 1;
		rs[2] = 0;
		rs[3] = 0;
		rs[4] = 0;
	    }
	    else {
		is[0] = 0;
		is[1] = chunks / 2 - 1;
		is[2] = chunks / 2;
		is[3] = chunks - 1 - (2*slices - current_slice) / safe_mult; //chunks - 1;
	    
		rs[0] = 0;
		rs[1] = 0;
		rs[2] = 1;
		rs[3] = 1;
		rs[4] = 1;
	    
		// cout << "Chunk " << is[3] << " " << chunks << endl;
	    }

	    dqmc::la::thin_col_to_invertible(u_stack[is[3]], reg_ws->mat_8);
	    enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);    
	    dqmc::la::thin_col_to_invertible(u_stack[is[2]], reg_ws->mat_6);
	    enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	    dqmc::la::thin_col_to_invertible(u_stack[is[1]], reg_ws->mat_4);
	    enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	    dqmc::la::thin_col_to_invertible(u_stack[is[0]], reg_ws->mat_2);
	    enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	    enlarge_thin(rs[3], d_stack[is[3]], ws->re_tiny_vec_4);
	    regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	    enlarge_thin(rs[3], t_stack[is[3]], ws->mat_7);
    
	    enlarge_thin(rs[2], d_stack[is[2]], ws->re_tiny_vec_3);
	    regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	    enlarge_thin(rs[2], t_stack[is[2]], ws->mat_5);

	    enlarge_thin(rs[1], d_stack[is[1]], ws->re_tiny_vec_2);
	    regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	    enlarge_thin(rs[1], t_stack[is[1]], ws->mat_3);

	    enlarge_thin(rs[0], d_stack[is[0]], ws->re_tiny_vec_1);
	    regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	    enlarge_thin(rs[0], t_stack[is[0]], ws->mat_1);

	    enlarge(rs[4], u_temp, ws->mat_10);
	    enlarge(rs[4], d_temp, ws->re_vec_5);
	    enlarge(rs[4], t_temp, ws->mat_9);

	    ws->la_mat_1 = ws->mat_1;
	    ws->mat_1 = ws->mat_2.transpose();
	    ws->mat_2 = ws->la_mat_1.transpose();
	    ws->la_mat_1 = ws->mat_5;
	    ws->mat_5 = ws->mat_6.transpose();
	    ws->mat_6 = ws->la_mat_1.transpose();
	    ws->la_mat_1 = ws->mat_10;
	    ws->mat_10 = ws->mat_9.transpose();
	    ws->mat_9 = ws->la_mat_1.transpose();

	    Us.push_back(&ws->mat_10); U_is_unitary.push_back(0);
	    Us.push_back(&ws->mat_2); U_is_unitary.push_back(0);
	    Us.push_back(&ws->mat_4); U_is_unitary.push_back(1);
	    Us.push_back(&ws->mat_6); U_is_unitary.push_back(0);
	    Us.push_back(&ws->mat_8); U_is_unitary.push_back(1);

	    Ts.push_back(&ws->mat_9); T_is_unitary.push_back(1);
	    Ts.push_back(&ws->mat_1); T_is_unitary.push_back(1);
	    Ts.push_back(&ws->mat_3); T_is_unitary.push_back(0);
	    Ts.push_back(&ws->mat_5); T_is_unitary.push_back(1);
	    Ts.push_back(&ws->mat_7); T_is_unitary.push_back(0);

	    Ds.push_back(&ws->re_vec_5);
	    Ds.push_back(&ws->re_vec_1);
	    Ds.push_back(&ws->re_vec_2);
	    Ds.push_back(&ws->re_vec_3);
	    Ds.push_back(&ws->re_vec_4);

	    cx_mat_t large_mat_1 = cx_mat::Zero(5 * vol, 5 * vol);
	    cx_mat_t large_mat_2 = cx_mat::Zero(5 * vol, 5 * vol);
	    cx_mat_t large_mat_3 = cx_mat::Zero(5 * vol, 5 * vol);
	    cx_mat_t large_mat_4 = cx_mat::Zero(5 * vol, 5 * vol);
    
	    large_mats.push_back(&large_mat_1);
	    large_mats.push_back(&large_mat_2);
	    large_mats.push_back(&large_mat_3);
	    large_mats.push_back(&large_mat_4);

	    cx_mat_t large_U = cx_mat_t::Zero(5 * vol, 5 * vol);
	    cx_mat_t large_T = cx_mat_t::Zero(5 * vol, 5 * vol);
    
	    vec_t large_vec_1 = vec_t::Zero(5 * vol);
	    vec_t large_vec_2 = vec_t::Zero(5 * vol);
	    vec_t large_vec_3 = vec_t::Zero(5 * vol);
	    vec_t large_vec_4 = vec_t::Zero(5 * vol);
    
	    large_vecs.push_back(&large_vec_1);
	    large_vecs.push_back(&large_vec_2);
	    large_vecs.push_back(&large_vec_3);
	    large_vecs.push_back(&large_vec_4);

	    dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
							      Ds,
							      Ts, T_is_unitary,
							      large_mats, 
							      large_U, large_T,
							      large_vecs, *ws,
								  greens);    		
	    return;
	}
    }
}

//===============================================================
// enlarging
//===============================================================

void cx_dqmc::greens_replica_renyi_gs::enlarge_thin(int replica,
						    vec_t& in,
						    vec_t& out) {
    out.setOnes();
    out.segment(0, in.size()) = in;
}
 
void cx_dqmc::greens_replica_renyi_gs::enlarge_thin(int replica,
						    cx_mat_t& in,
						    cx_mat_t& out) {
    using namespace dqmc::la;
    using namespace std;

    out.setZero();

    if (out.rows() > out.cols()) {
	if (replica == 0) {
	    out.block(0, 0, in.rows(), in.cols())
		= in;
	    out.block(in.rows(), in.cols(),
		      n_B, n_B).setIdentity();	
	} else if (replica == 1) {
	    out.block(0, 0, n_A, particles)
		= in.block(0, 0, n_A, particles);
	    out.block(n_A, particles, n_B, n_B).setIdentity();
	    out.block(N, 0, n_B, particles)
		= in.block(n_A, 0, n_B, particles);	
	}
    } else if (out.rows() < out.cols()) {

	if (replica == 0) {
	    out.block(0, 0, in.rows(), in.cols())
		= in;

	    {
		out.block(in.rows(), in.cols(),
			  n_B, n_B).setIdentity();	
	    }
	} else if (replica == 1) {
	    out.block(0, 0, particles, n_A)
		= in.block(0, 0, particles, n_A);

	    {
		out.block(particles, n_A, n_B, n_B).setIdentity();
		out.block(0, N, particles, n_B)
		    = in.block(0, n_A, particles, n_B);	
	    }
	}	
    }
    else {
        out.setIdentity();
	out.block(0, 0, p->particles, p->particles)
            = in;
    }
}

void cx_dqmc::greens_replica_renyi_gs::enlarge_thin_ized_col(int replica,
							     cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    using namespace std;

    out.setZero();

    if (replica == 0) {
	out.block(0, 0, N, particles)
	    = in.block(0, 0, N, particles);

	out.block(N, particles,
		  n_B, n_B).setIdentity();
	out.block(0, eff_particles, N, in.cols() - particles)
	    = in.block(0, particles, N, in.cols() - particles);
	    
    } else if (replica == 1) {
	out.block(0, 0, n_A, particles)
	    = in.block(0, 0, n_A, particles);

	out.block(n_A, particles, n_B, n_B).setIdentity();
	out.block(N, 0, n_B, particles)
	    = in.block(n_A, 0, n_B, particles);	
	out.block(0, eff_particles, n_A, in.cols() - particles)
	    = in.block(0, particles, n_A, in.cols() - particles);	
	out.block(N, eff_particles, n_B, in.cols() - particles)
	    = in.block(n_A, particles, n_B, in.cols() - particles);
    }
}

void cx_dqmc::greens_replica_renyi_gs::enlarge_thin_ized_row(int replica,
							     cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    using namespace std;

    out.setZero();

    if (replica == 0) {
	out.block(0, 0, particles, N)
	    = in.block(0, 0, particles, N);

	out.block(particles, N,
		  n_B, n_B).setIdentity();	    
	out.block(eff_particles, 0, in.rows() - particles, N)
	    = in.block(particles,  0, in.rows() - particles, N);
    } else if (replica == 1) {
	out.block(0, 0, particles, n_A)
	    = in.block(0, 0, particles, n_A);

	out.block(particles, n_A, n_B, n_B).setIdentity();
	out.block(0, N, particles, n_B)
	    = in.block(0, n_A, particles, n_B);
	
	out.block(eff_particles, 0, in.rows() - particles, n_A)
	    = in.block(particles, 0, in.rows() - particles, n_A);
	
	out.block(eff_particles, N, in.rows() - particles, n_B)
	    = in.block( particles, n_A, in.rows() - particles, n_B);
    }
}

void cx_dqmc::greens_replica_renyi_gs::enlarge(int replica, cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    out.setIdentity();

    if (replica == 0) {
	out.block(0, 0, N, N) = in;
    }
    else if (replica == 1) {
	out.block(0, 0, n_A, n_A)
	    = in.block(0, 0, n_A, n_A);

	out.block(N, 0, n_B, n_A)
	    = in.block(n_A, 0, n_B, n_A);
	out.block(0, N, n_A, n_B)
	    = in.block(0, n_A, n_A, n_B);
	out.block(N, N, n_B, n_B)
	    = in.block(n_A, n_A, n_B, n_B);	
    }
}

    
void cx_dqmc::greens_replica_renyi_gs::enlarge(int replica, vec_t& in, vec_t& out) {
    out.setOnes();
    if (replica == 0) {
	out.segment(0, N) = in;
    }
    else if (replica == 1) {
	out.segment(0, n_A) = in.segment(0, n_A);
	out.segment(N, n_B) = in.segment(n_A, n_B);
    }
}


void cx_dqmc::greens_replica_renyi_gs::calculate_greens_exact(int slice) {
    // cout << "Calculating greens exactly @ " << slice << endl;

    int is[5], rs[5];
    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();

	
    if (slice < slices / 2) {
	// cout << "Part 1" << endl;
	reg_ws->col_mat_1 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_1.setOnes();
	reg_ws->tiny_mat_1.setIdentity();

	for (int s = slices / 2; s > slice; s -= safe_mult) {
	    start =  max(s - safe_mult, slice);
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_1);
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * reg_ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2, reg_ws->col_mat_1,
					    reg_ws->re_tiny_vec_1, reg_ws->tiny_mat_2);
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_2 * reg_ws->tiny_mat_1;
	    reg_ws->tiny_mat_1 = reg_ws->tiny_mat_3;	    
	}

	reg_ws->col_mat_2 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_2.setOnes();
	reg_ws->tiny_mat_2.setIdentity();	
	for (int s = slices/2; s < slices; s += safe_mult) {
	    start = s;
	    stop = s + safe_mult;
	    slice_sequence_left(start, stop, reg_ws->col_mat_2);
	    reg_ws->col_mat_3 = reg_ws->col_mat_2 * reg_ws->re_tiny_vec_2.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_3, reg_ws->col_mat_2,
					    reg_ws->re_tiny_vec_2, reg_ws->tiny_mat_3);
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_3 * reg_ws->tiny_mat_2;
	    reg_ws->tiny_mat_2 = reg_ws->tiny_mat_4;	    
	}

	reg_ws->col_mat_3 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_3.setOnes();
	reg_ws->tiny_mat_3.setIdentity();
	for (int s = 3 * (slices / 2); s > slices; s -= safe_mult) {
	    start =  s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_3);
	    reg_ws->col_mat_4 = reg_ws->col_mat_3 * reg_ws->re_tiny_vec_3.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_4, reg_ws->col_mat_3,
					    reg_ws->re_tiny_vec_3, reg_ws->tiny_mat_4);
	    reg_ws->tiny_mat_5 = reg_ws->tiny_mat_4 * reg_ws->tiny_mat_3;
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_5;	    
	}
	// cout << "col_mat_3" << endl;
	// cout << reg_ws->col_mat_3 << endl;
	
	reg_ws->col_mat_4 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_4.setOnes();
	reg_ws->tiny_mat_4.setIdentity();	
	for (int s = 3 * (slices / 2); s < 4 * (slices / 2); s += safe_mult) {
	    // cout << "tiny mat 4 " << endl <<reg_ws->tiny_mat_4 << endl << endl;
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_4);
	    reg_ws->col_mat_5 = reg_ws->col_mat_4 * reg_ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_5, reg_ws->col_mat_4,
					    reg_ws->re_tiny_vec_4, reg_ws->tiny_mat_5);
	    reg_ws->tiny_mat_6 = reg_ws->tiny_mat_5 * reg_ws->tiny_mat_4;
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_6;
	}

	enlarge_thin(1, reg_ws->col_mat_4, ws->col_mat_4);
	enlarge_thin(1, reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	enlarge_thin(1, reg_ws->tiny_mat_4, ws->tiny_mat_4);
	// cout << "And then adding the rest" << endl;

	// cout << "Enlarged is this" << endl;
	// cout << ws->col_mat_4 << endl << endl;;
	// cout << ws->re_tiny_vec_4 << endl << endl;
	// cout << ws->tiny_mat_4 << endl << endl;

	// cout << ws->col_mat_4 << endl << endl;

	// ws->mat_1.setIdentity();
	// slice_sequence_left_renyi(0, slice, ws->mat_1);
	// cout << "Here's the sequence\t" << 0 << " " << slice
	//      << endl << ws->mat_1 << endl << endl;
	
	for (int s = 0; s < slice; s += safe_mult) {
	    start = s;
	    stop = min(s + safe_mult, slice);
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_renyi(start, stop, ws->col_mat_4);
	    ws->col_mat_5 = ws->col_mat_4 * ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_4,
					    ws->re_tiny_vec_4, ws->tiny_mat_5);
	    ws->tiny_mat_6 = ws->tiny_mat_5 * ws->tiny_mat_4;
	    ws->tiny_mat_4 = ws->tiny_mat_6;	    
	}

	// cout << ws->col_mat_4 << endl << endl;
	
	rs[3] = 1;
	rs[2] = 1;
	rs[1] = 0;
	rs[0] = 0;
	
	dqmc::la::thin_col_to_invertible(ws->col_mat_4, ws->mat_8);

	// cout << "From exact" << endl;
	// cout << ws->col_mat_4 << endl << endl;
	// cout << ws->re_tiny_vec_4.transpose() << endl << endl;
	// cout << ws->tiny_mat_4 << endl << endl;
	
	// cout << "It should be this" << endl << ws->mat_8 << endl << endl;
	// dqmc::la::thin_col_to_invertible(reg_ws->col_mat_4, reg_ws->mat_8);
	// enlarge_thin_ized_col(1, reg_ws->mat_8, ws->mat_8);
	// cout << "But it is actually this " << endl << ws->mat_8 << endl;
	
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_3, reg_ws->mat_6);
	enlarge_thin_ized_col(1, reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_2, reg_ws->mat_4);
	enlarge_thin_ized_col(0, reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_1, reg_ws->mat_2);
	enlarge_thin_ized_col(0, reg_ws->mat_2, ws->mat_2);

	// enlarge_thin(rs[3], ws->tiny_mat_4, ws->mat_7);
	ws->mat_7.setIdentity();
	ws->mat_7.block(0, 0, eff_particles, eff_particles) = ws->tiny_mat_4;
	
	// cout << "I thought it was " << endl << ws->mat_7 << endl << endl;
	// enlarge_thin(rs[3], reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	// enlarge_thin(rs[3], reg_ws->tiny_mat_4, ws->mat_7);
	// cout << "But then " << endl << ws->mat_7 << endl << endl;
    
	enlarge_thin(rs[2], reg_ws->re_tiny_vec_3, ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], reg_ws->tiny_mat_3, ws->mat_5);

	enlarge_thin(rs[1], reg_ws->re_tiny_vec_2, ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], reg_ws->tiny_mat_2, ws->mat_3);

	enlarge_thin(rs[0], reg_ws->re_tiny_vec_1, ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], reg_ws->tiny_mat_1, ws->mat_1);
	
	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();

    }
    else if (slice >= slices / 2 &&
	     slice < slices) {
	
	reg_ws->col_mat_1 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_1.setOnes();
	reg_ws->tiny_mat_1.setIdentity();

	for (int s = 3 * (slices / 2); s > slices; s -= safe_mult) {
	    start = s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_1);
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * reg_ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2, reg_ws->col_mat_1,
					    reg_ws->re_tiny_vec_1, reg_ws->tiny_mat_2);
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_2 * reg_ws->tiny_mat_1;
	    reg_ws->tiny_mat_1 = reg_ws->tiny_mat_3;	    
	}

	enlarge_thin(1, reg_ws->col_mat_1, ws->col_mat_1);
	enlarge_thin(1, reg_ws->re_tiny_vec_1, ws->re_tiny_vec_1);
	enlarge_thin(1, reg_ws->tiny_mat_1, ws->tiny_mat_1);

	// cout << reg_ws->col_mat_1 << endl << endl;
	// ws->mat_2.setIdentity();
	for (int s = slices; s > slice; s -= safe_mult) {	    
	    start = max(s - safe_mult, slice);
	    stop = s;
	    cout << start << " to " << stop << endl;
	    // slice_sequence_left_renyi_t(start, stop, ws->mat_2);
	    // cout << ws->mat_2 << endl << endl;
	    slice_sequence_left_renyi_t(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					    ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	dqmc::la::decompose_udt_col_piv(ws->mat_2, ws->mat_1,
					ws->re_vec_1, ws->mat_3);

	// cout << ws->mat_1 << endl << endl;
	// cout << ws->re_vec_1 << endl << endl;
	// cout << ws->mat_3 << endl << endl;
 
	reg_ws->col_mat_2 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_2.setOnes();
	reg_ws->tiny_mat_2.setIdentity();	
	for (int s = 3 * (slices / 2); s < 2 * slices; s += safe_mult) {
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_2);
	    reg_ws->col_mat_3 = reg_ws->col_mat_2 * reg_ws->re_tiny_vec_2.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_3, reg_ws->col_mat_2,
					    reg_ws->re_tiny_vec_2, reg_ws->tiny_mat_3);
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_3 * reg_ws->tiny_mat_2;
	    reg_ws->tiny_mat_2 = reg_ws->tiny_mat_4;	    
	}

	reg_ws->col_mat_3 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_3.setOnes();
	reg_ws->tiny_mat_3.setIdentity();
	for (int s = slices / 2; s > 0; s -= safe_mult) {
	    start =  s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_3);
	    reg_ws->col_mat_4 = reg_ws->col_mat_3 * reg_ws->re_tiny_vec_3.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_4, reg_ws->col_mat_3,
					    reg_ws->re_tiny_vec_3, reg_ws->tiny_mat_4);
	    reg_ws->tiny_mat_5 = reg_ws->tiny_mat_4 * reg_ws->tiny_mat_3;
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_5;	    
	}


	reg_ws->col_mat_4 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_4.setOnes();
	reg_ws->tiny_mat_4.setIdentity();	
	for (int s = slices / 2; s < slice; s += safe_mult) {
	    start = s;
	    stop = min (s + safe_mult, slice);
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_4);
	    reg_ws->col_mat_5 = reg_ws->col_mat_4 * reg_ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_5, reg_ws->col_mat_4,
					    reg_ws->re_tiny_vec_4, reg_ws->tiny_mat_5);
	    reg_ws->tiny_mat_6 = reg_ws->tiny_mat_5 * reg_ws->tiny_mat_4;
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_6;
	}

	rs[3] = 0;
	rs[2] = 0;
	rs[1] = 1;
	rs[0] = 1;

	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_4, reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_3, reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_2, reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);

	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_2);

	enlarge_thin(rs[3], reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	enlarge_thin(rs[3], reg_ws->tiny_mat_4, ws->mat_7);
    
	enlarge_thin(rs[2], reg_ws->re_tiny_vec_3, ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], reg_ws->tiny_mat_3, ws->mat_5);

	enlarge_thin(rs[1], reg_ws->re_tiny_vec_2, ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], reg_ws->tiny_mat_2, ws->mat_3);


	// enlarge_thin(rs[0], reg_ws->re_tiny_vec_1, ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	// enlarge_thin(rs[0], reg_ws->tiny_mat_1, ws->mat_1);
	ws->mat_1.setIdentity();
	ws->mat_1.block(0, 0, eff_particles, eff_particles) = ws->tiny_mat_1;

	
	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();
    }
    else if (slice >= slices &&
	     slice < 3 * (slices / 2)) {
	// cout << "Part 1" << endl;
	reg_ws->col_mat_1 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_1.setOnes();
	reg_ws->tiny_mat_1.setIdentity();

	for (int s = 3 * (slices / 2); s > slice; s -= safe_mult) {
	    start =  max(s - safe_mult, slice);
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_1);
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * reg_ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2, reg_ws->col_mat_1,
					    reg_ws->re_tiny_vec_1, reg_ws->tiny_mat_2);
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_2 * reg_ws->tiny_mat_1;
	    reg_ws->tiny_mat_1 = reg_ws->tiny_mat_3;	    
	}

	reg_ws->col_mat_2 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_2.setOnes();
	reg_ws->tiny_mat_2.setIdentity();	
	for (int s = 3 * (slices / 2); s < 2 * slices; s += safe_mult) {
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_2);
	    reg_ws->col_mat_3 = reg_ws->col_mat_2 * reg_ws->re_tiny_vec_2.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_3, reg_ws->col_mat_2,
					    reg_ws->re_tiny_vec_2, reg_ws->tiny_mat_3);
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_3 * reg_ws->tiny_mat_2;
	    reg_ws->tiny_mat_2 = reg_ws->tiny_mat_4;	    
	}

	reg_ws->col_mat_3 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_3.setOnes();
	reg_ws->tiny_mat_3.setIdentity();
	
	for (int s = (slices / 2); s > 0; s -= safe_mult) {
	    start =  s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_3);
	    reg_ws->col_mat_4 = reg_ws->col_mat_3 * reg_ws->re_tiny_vec_3.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_4, reg_ws->col_mat_3,
					    reg_ws->re_tiny_vec_3, reg_ws->tiny_mat_4);
	    reg_ws->tiny_mat_5 = reg_ws->tiny_mat_4 * reg_ws->tiny_mat_3;
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_5;	    
	}
	// cout << "col_mat_3" << endl;
	// cout << reg_ws->col_mat_3 << endl;

	reg_ws->col_mat_4 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_4.setOnes();
	reg_ws->tiny_mat_4.setIdentity();	
	for (int s = (slices / 2); s < slices; s += safe_mult) {
	    // cout << "tiny mat 4 " << endl <<reg_ws->tiny_mat_4 << endl << endl;
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_4);
	    reg_ws->col_mat_5 = reg_ws->col_mat_4 * reg_ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_5, reg_ws->col_mat_4,
					    reg_ws->re_tiny_vec_4, reg_ws->tiny_mat_5);
	    reg_ws->tiny_mat_6 = reg_ws->tiny_mat_5 * reg_ws->tiny_mat_4;
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_6;
	}

	enlarge_thin(0, reg_ws->col_mat_4, ws->col_mat_4);
	enlarge_thin(0, reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	enlarge_thin(0, reg_ws->tiny_mat_4, ws->tiny_mat_4);
	// cout << "And then adding the rest" << endl;

	// cout << ws->col_mat_4 << endl << endl;
       
	for (int s = slices; s < slice; s += safe_mult) {
	    start = s;
	    stop = min(s + safe_mult, slice);
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_renyi(start, stop, ws->col_mat_4);
	    ws->col_mat_5 = ws->col_mat_4 * ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_4,
					    ws->re_tiny_vec_4, ws->tiny_mat_5);
	    ws->tiny_mat_6 = ws->tiny_mat_5 * ws->tiny_mat_4;
	    ws->tiny_mat_4 = ws->tiny_mat_6;	    
	}
	
	// cout << ws->col_mat_4 << endl << endl;
	
	rs[3] = 0;
	rs[2] = 0;
	rs[1] = 1;
	rs[0] = 1;
	
	dqmc::la::thin_col_to_invertible(ws->col_mat_4, ws->mat_8);

	// cout << "It should be this" << endl << ws->mat_8 << endl << endl;
	// dqmc::la::thin_col_to_invertible(reg_ws->col_mat_4, reg_ws->mat_8);
	// enlarge_thin_ized_col(1, reg_ws->mat_8, ws->mat_8);
	// cout << "But it is actually this " << endl << ws->mat_8 << endl;
	
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_3, reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_2, reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_1, reg_ws->mat_2);
	enlarge_thin_ized_col(rs[0], reg_ws->mat_2, ws->mat_2);

	// enlarge_thin(rs[3], ws->tiny_mat_4, ws->mat_7);
	ws->mat_7.setIdentity();
	ws->mat_7.block(0, 0, eff_particles, eff_particles) = ws->tiny_mat_4;
	
	// cout << "I thought it was " << endl << ws->mat_7 << endl << endl;
	// enlarge_thin(rs[3], reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	// enlarge_thin(rs[3], reg_ws->tiny_mat_4, ws->mat_7);
	// cout << "But then " << endl << ws->mat_7 << endl << endl;
    
	enlarge_thin(rs[2], reg_ws->re_tiny_vec_3, ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], reg_ws->tiny_mat_3, ws->mat_5);

	enlarge_thin(rs[1], reg_ws->re_tiny_vec_2, ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], reg_ws->tiny_mat_2, ws->mat_3);

	enlarge_thin(rs[0], reg_ws->re_tiny_vec_1, ws->re_tiny_vec_1);
	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);
	enlarge_thin(rs[0], reg_ws->tiny_mat_1, ws->mat_1);
	
	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();

    }
    else if (slice >= 3 * (slices / 2) &&
	     slice <  2 * slices) {
	
	// cout << "Part 1" << endl;
	reg_ws->col_mat_1 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_1.setOnes();
	reg_ws->tiny_mat_1.setIdentity();

	for (int s = slices / 2; s > 0; s -= safe_mult) {
	    start = s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_1);
	    reg_ws->col_mat_2 = reg_ws->col_mat_1 * reg_ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_2, reg_ws->col_mat_1,
					    reg_ws->re_tiny_vec_1, reg_ws->tiny_mat_2);
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_2 * reg_ws->tiny_mat_1;
	    reg_ws->tiny_mat_1 = reg_ws->tiny_mat_3;	    
	}

	enlarge_thin(0, reg_ws->col_mat_1, ws->col_mat_1);
	enlarge_thin(0, reg_ws->re_tiny_vec_1, ws->re_tiny_vec_1);
	enlarge_thin(0, reg_ws->tiny_mat_1, ws->tiny_mat_1);


	for (int s = 2 * slices; s >= slice; s -= safe_mult) {
	    start = max(s - safe_mult, slice);
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_renyi_t(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					    ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	// cout << ws->tiny_mat_1 << endl << endl;
	
	reg_ws->col_mat_2 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_2.setOnes();
	reg_ws->tiny_mat_2.setIdentity();	
	for (int s = slices / 2; s < slices; s += safe_mult) {
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_2);
	    reg_ws->col_mat_3 = reg_ws->col_mat_2 * reg_ws->re_tiny_vec_2.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_3, reg_ws->col_mat_2,
					    reg_ws->re_tiny_vec_2, reg_ws->tiny_mat_3);
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_3 * reg_ws->tiny_mat_2;
	    reg_ws->tiny_mat_2 = reg_ws->tiny_mat_4;	    
	}

	reg_ws->col_mat_3 = reg_ws->den_U.conjugate();	
	reg_ws->re_tiny_vec_3.setOnes();
	reg_ws->tiny_mat_3.setIdentity();
	for (int s = 3 * (slices / 2); s > slices; s -= safe_mult) {
	    start =  s - safe_mult;
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, reg_ws->col_mat_3);
	    reg_ws->col_mat_4 = reg_ws->col_mat_3 * reg_ws->re_tiny_vec_3.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_4, reg_ws->col_mat_3,
					    reg_ws->re_tiny_vec_3, reg_ws->tiny_mat_4);
	    reg_ws->tiny_mat_5 = reg_ws->tiny_mat_4 * reg_ws->tiny_mat_3;
	    reg_ws->tiny_mat_3 = reg_ws->tiny_mat_5;	    
	}


	reg_ws->col_mat_4 = reg_ws->den_U;	
	reg_ws->re_tiny_vec_4.setOnes();
	reg_ws->tiny_mat_4.setIdentity();	
	for (int s = 3 * (slices / 2); s < slice; s += safe_mult) {
	    start = s;
	    stop = min(s + safe_mult, slice);
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, reg_ws->col_mat_4);
	    reg_ws->col_mat_5 = reg_ws->col_mat_4 * reg_ws->re_tiny_vec_4.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(reg_ws->col_mat_5, reg_ws->col_mat_4,
					    reg_ws->re_tiny_vec_4, reg_ws->tiny_mat_5);
	    reg_ws->tiny_mat_6 = reg_ws->tiny_mat_5 * reg_ws->tiny_mat_4;
	    reg_ws->tiny_mat_4 = reg_ws->tiny_mat_6;
	}

	rs[3] = 1;
	rs[2] = 1;
	rs[1] = 0;
	rs[0] = 0;

	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_4, reg_ws->mat_8);
	enlarge_thin_ized_col(rs[3], reg_ws->mat_8, ws->mat_8);	
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_3, reg_ws->mat_6);
	enlarge_thin_ized_col(rs[2], reg_ws->mat_6, ws->mat_6);
	dqmc::la::thin_col_to_invertible(reg_ws->col_mat_2, reg_ws->mat_4);
	enlarge_thin_ized_col(rs[1], reg_ws->mat_4, ws->mat_4);

	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_2);

	enlarge_thin(rs[3], reg_ws->re_tiny_vec_4, ws->re_tiny_vec_4);
	regularize_svd(ws->re_tiny_vec_4, ws->re_vec_4);
	enlarge_thin(rs[3], reg_ws->tiny_mat_4, ws->mat_7);
    
	enlarge_thin(rs[2], reg_ws->re_tiny_vec_3, ws->re_tiny_vec_3);
	regularize_svd(ws->re_tiny_vec_3, ws->re_vec_3);
	enlarge_thin(rs[2], reg_ws->tiny_mat_3, ws->mat_5);

	enlarge_thin(rs[1], reg_ws->re_tiny_vec_2, ws->re_tiny_vec_2);
	regularize_svd(ws->re_tiny_vec_2, ws->re_vec_2);
	enlarge_thin(rs[1], reg_ws->tiny_mat_2, ws->mat_3);

	regularize_svd(ws->re_tiny_vec_1, ws->re_vec_1);

	ws->mat_1.setIdentity();
	ws->mat_1.block(0, 0, eff_particles, eff_particles) = ws->tiny_mat_1;

	
	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();
	ws->la_mat_1 = ws->mat_5;
	ws->mat_5 = ws->mat_6.transpose();
	ws->mat_6 = ws->la_mat_1.transpose();
    }

    Us.push_back(&ws->mat_2);
    Us.push_back(&ws->mat_4);
    Us.push_back(&ws->mat_6);
    Us.push_back(&ws->mat_8);

    Ts.push_back(&ws->mat_1);
    Ts.push_back(&ws->mat_3);
    Ts.push_back(&ws->mat_5);
    Ts.push_back(&ws->mat_7);

    Ds.push_back(&ws->re_vec_1);
    Ds.push_back(&ws->re_vec_2);
    Ds.push_back(&ws->re_vec_3);
    Ds.push_back(&ws->re_vec_4);

    cx_mat_t large_mat_1 = cx_mat::Zero(4 * vol, 4 * vol);
    cx_mat_t large_mat_2 = cx_mat::Zero(4 * vol, 4 * vol);
    cx_mat_t large_mat_3 = cx_mat::Zero(4 * vol, 4 * vol);
    cx_mat_t large_mat_4 = cx_mat::Zero(4 * vol, 4 * vol);
    
    large_mats.push_back(&large_mat_1);
    large_mats.push_back(&large_mat_2);
    large_mats.push_back(&large_mat_3);
    large_mats.push_back(&large_mat_4);

    cx_mat_t large_U = cx_mat_t::Zero(4 * vol, 4 * vol);
    cx_mat_t large_T = cx_mat_t::Zero(4 * vol, 4 * vol);
    
    vec_t large_vec_1 = vec_t::Zero(4 * vol);
    vec_t large_vec_2 = vec_t::Zero(4 * vol);
    vec_t large_vec_3 = vec_t::Zero(4 * vol);
    vec_t large_vec_4 = vec_t::Zero(4 * vol);
    
    large_vecs.push_back(&large_vec_1);
    large_vecs.push_back(&large_vec_2);
    large_vecs.push_back(&large_vec_3);
    large_vecs.push_back(&large_vec_4);

    // cout << "Full piv from four" << endl << endl;
    // cout << "Ts[0]" << endl;
    // cout << *Ts[0] << endl << endl;
    // cout << *Ts[1] << endl << endl;
    // cout << *Ts[2] << endl << endl;
    // cout << *Ts[3] << endl << endl;

    dqmc::calculate_greens::col_piv_qr_full_piv_lu(Us, Ds, Ts,
						   large_mats, 
						   large_U, large_T,
						   large_vecs, *ws,
						   greens);    	
}


void cx_dqmc::greens_replica_renyi_gs::update_remove_interaction() {
    prop_greens.setIdentity();
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens, ws->vec_1,
					    (*aux_spins)[current_slice / slices][current_slice], -1., 0);
}


void cx_dqmc::greens_replica_renyi_gs::update_add_interaction() {
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens, ws->vec_1,
					    (*aux_spins)[current_slice / slices][current_slice], 1., 0);
}
