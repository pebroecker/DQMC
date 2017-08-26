#include "cx_dqmc_greens_replica_ft.hpp"

using namespace std;

void cx_dqmc::greens_replica_ft::initialize() {
    vol = p->N;

    replica = p->replica;
    
    sites = ws->sites;    
    X.resize(vol, vol);
    Y.resize(vol, vol);

    X_sls.resize(vol, 2);
    Y_sls.resize(vol, 2);
    Y_sls_row_1.resize(2, vol);
    Y_sls_row_2.resize(2, vol);

    slices = p->slices;
    safe_mult = p->safe_mult;

    u_temp.resize(vol, vol);
    d_temp.resize(vol);
    t_temp.resize(vol, vol);

    prop_greens.resize(vol, vol);
    greens.resize(vol, vol);
    temp_greens.resize(vol, vol);
    first_initialization = true;    
}


void cx_dqmc::greens_replica_ft::build_stack() {
    direction = 1;
    
    if (first_initialization == true) {
	vol = ws->vol;
	sites = ws->sites;	
	n_elements = slices / p->safe_mult;
	chunks = n_elements;

	for (int i = 0; i < n_elements; ++i) {	    
	    u_stack.push_back(cx_mat_t::Zero(p->N, p->N));
	    d_stack.push_back(vec_t::Zero(p->N));
	    t_stack.push_back(cx_mat_t::Zero(p->N, p->N));
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
    
    idx = chunks / 2;
    for (int s = slices / 2; s > 0; s -= safe_mult) {
	bra_update(s - safe_mult, s, idx, idx - 1);
	--idx;
    }

    idx = chunks/2 - 1;
    for (int s = slices/2; s < slices; s += safe_mult) {
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
    
    alps::RealObservable dummy("dummy");
    propagate(dummy);

    
    if (first_initialization == true) {
	cout << p->outp << d_stack[0][0]
	     << " to onefold col SVD gap\t " << d_stack[0][p->particles - 1] << endl;
	if (greens.cols() < 11) {
	    cout << p->outp << " " << d_stack[0].transpose() << endl;
	    cout << greens.trace() << " is regular trace" << endl;
	    cout << greens << endl;
	}
	first_initialization = false;
    }

    if(p->adapt_mu == true && p->adapted_mu == false) {
	p->adapted_mu = true;
	cout << p->outp << "Smallest SV " << d_stack[idx][p->particles - 1] << endl;
	cout << "Spectrum " << endl << d_stack[idx].transpose() << endl;
	cout << "Target " << p->mu_target << endl;

	if (p->model_id == p->SPINLESS_HUBBARD) {
	    p->mu_site = 1./p->beta * log(p->mu_target / d_stack[idx][p->particles - 1]);
	} else {
	    p->mu_site = 2 / p->beta * log(p->mu_target / d_stack[idx][p->particles - 1]);
	}
	cout << exp( p->mu_site * p->beta/2 ) << endl;
	cout << p->outp << "New mu " << p->mu_site << endl;
	
	p->mu_site_factor[0] = exp(-p->delta_tau * p->mu_site);
	p->mu_site_factor[1] = NAN;
	p->mu_site_factor[2] = exp(p->delta_tau * p->mu_site);
	build_stack();
	cout << p->outp << "New Smallest SV " << d_stack[idx][p->particles - 1] << endl;
	cout << "New Spectrum " << endl << d_stack[idx].transpose() << endl;
    }
}


int cx_dqmc::greens_replica_ft::propagate(alps::Observable& stability) {
    // cout << "propagate\t" << direction << "\t-\t" << current_slice << endl;
    
    if (direction == -1) {
	if (current_slice == slices && dir_switch == true) {
	    // cout << "At the top " << current_slice << endl;
	    dir_switch = false;
	    
	    u_temp.setIdentity();
	    d_temp.setOnes();
	    t_temp.setIdentity();
	    
	    calculate_greens_general();
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	    current_bond = 0;
	    return 0;		
	}
	
	else if (current_slice == slices / 2) {
	    // cout << "Switching directions " << current_slice << endl;
	    dir_switch = true;
	    direction = 1;
	    current_slice = 0;
	    idx = chunks / 2 - 1;
	    current_bond = p->ts.size();
	    for (int s = slices / 2; s < slices; s+= safe_mult) {
		ket_update(s, s + safe_mult, idx, idx + 1);
		++idx;
	    }
	    propagate(stability);
	    return 2;
	}

	else if (current_slice % safe_mult == 0) {
	    start = current_slice;
	    stop = current_slice + safe_mult;
	    idx = current_slice / safe_mult;
	    
	    // cout << "Updating " << start << " " << stop << endl;
	    
	    slice_sequence_left_t(start, stop, u_temp);
	    ws->mat_1 = u_temp * d_temp.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->mat_1,
					    u_temp, d_temp,
					    ws->mat_2);
	    ws->mat_1 = ws->mat_2 * t_temp;
	    t_temp = ws->mat_1;
	    
	    temp_greens = greens;
	    calculate_greens_general();
	    check_stability();
	    
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	}
	
	else {
	    // cout << "Hopping from " << current_slice << endl;
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);		   
	}
    }
    
    else if (direction == 1) {
	if (current_slice == 0 && dir_switch == true) {
	    // cout << "At the top " << idx << " " << chunks << " + psi " << endl;
	    dir_switch = false;

	    u_temp.setIdentity();
	    d_temp.setOnes();
	    t_temp.setIdentity();

	    calculate_greens_general();	    
	    return 0;
		
	} else if ((current_slice + 1) == slices / 2) {
	    // cout << "Switching directions " << endl;
	    dir_switch = true;
	    current_slice = slices;
	    idx = chunks / 2;
	    direction = -1;

	    for (int s = slices / 2; s > 0; s -= safe_mult) {
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

	    // cout << "Updating " << start << " " << stop << endl;

	    slice_sequence_left(start, stop, u_temp);
	    ws->mat_1 = u_temp * d_temp.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->mat_1,
					    u_temp, d_temp, 
					    ws->mat_2);
	    ws->mat_1 = ws->mat_2 * t_temp;		
	    t_temp = ws->mat_1;
	    
	    slice_matrix_right(current_slice, greens, true);
	    slice_matrix_left(current_slice, greens, false);
	    ++current_slice;
	    temp_greens = greens;

	    calculate_greens_general();
	    check_stability();

	}
	else {
	    // cout << "Next slice " << current_slice << endl;
	    //      << current_bond << endl;
	    slice_matrix_right(current_slice, greens, true);
	    slice_matrix_left(current_slice, greens, false);
	    ++current_slice;
	}
    }
    return 1;
}



void cx_dqmc::greens_replica_ft::update_remove_interaction() {
    // throw std::runtime_error("Isn't the test phase over?");
    prop_greens.setIdentity();
    cx_dqmc::interaction::interaction_right(p, prop_greens,
				(*aux_spins)[replica][current_slice], -1., 0);
}

void cx_dqmc::greens_replica_ft::update_add_interaction() {
    // throw std::runtime_error("Isn't the test phase over?");
    cx_dqmc::interaction::interaction_right(p, prop_greens,
				(*aux_spins)[replica][current_slice], 1., 0);
}


double cx_dqmc::greens_replica_ft::check_stability() {
    static int warnings = 0;
    static pdouble_t diff;
    static int max_warnings = 8 * slices/safe_mult;
    ws->re_mat_1 = (temp_greens - greens).cwiseAbs().real();
    diff = ws->re_mat_1.maxCoeff();
    // cout << p->outp << diff << endl;
        
    if (warnings < max_warnings) {
    	if (diff > 1e-7) {
    	    ++warnings;
	    cout << "greens::propagate " << int(spin) << " " << p->osi << " ";
    	    cout << "Diagonal entries of G are unstable: "
    		 << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
    		 << current_slice << " / " << slices << "\t";
    	    cout << setw(15) << std::right << diff << endl;
    	}
    } else if (diff > 1e-1) {
    	    ++warnings;
	    cout << "greens::propagate " << int(spin) << " " << p->osi << " ";
    	    cout << "Diagonal entries of G are really unstable: "
    		 << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
    		 << current_slice << " / " << slices << "\t";
    	    cout << setw(15) << std::right << diff << " vs. exact " << endl;
    }
    return diff;
}


void cx_dqmc::greens_replica_ft::ket_update(int start, int stop,
				 int i, int j) {

    bool add_den = false;

    if (start == slices / 2) add_den = true;
	
    if (add_den) 
	ws->mat_1.setIdentity();
    else
	ws->mat_1 = u_stack[i];

    slice_sequence_left(start, stop, ws->mat_1);
	
    if (add_den == true) {
	dqmc::la::decompose_udt_col_piv(ws->mat_1, u_stack[j],
					d_stack[j], ws->mat_2);
	t_stack[j] = ws->mat_2;
    } 
    else {
	ws->mat_2 = ws->mat_1 * d_stack[i].asDiagonal();
	dqmc::la::decompose_udt_col_piv(ws->mat_2, 
					u_stack[j], d_stack[j],
					ws->mat_3);
	t_stack[j] = ws->mat_3 * t_stack[i];
    }    
}

void cx_dqmc::greens_replica_ft::bra_update(int start, int stop,
					 int i, int j) {
    bool add_den = false;
    
    if (stop == slices / 2) add_den = true;

    if (add_den) ws->mat_1.setIdentity();
    else ws->mat_1 = u_stack[i];

    slice_sequence_left_t(start, stop, ws->mat_1);
    
    if (add_den == true) {
	dqmc::la::decompose_udt_col_piv(ws->mat_1, u_stack[j],
					d_stack[j], ws->mat_2);
	t_stack[j] = ws->mat_2;
    } 
    else {
	ws->mat_2 = ws->mat_1 * d_stack[i].asDiagonal(); 
	dqmc::la::decompose_udt_col_piv(ws->mat_2,
					u_stack[j], d_stack[j],
					ws->mat_3);	    
	t_stack[j] = ws->mat_3 * t_stack[i];
    } 
}

void cx_dqmc::greens_replica_ft::slice_sequence_left(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left(i, M); }
}


void cx_dqmc::greens_replica_ft::slice_sequence_left_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) 
	{
	    slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_replica_ft::slice_sequence_right(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_replica_ft::slice_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::interaction::interaction_left(p, M,
				   (*aux_spins)[replica][s], 1., 0);
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
	cx_dqmc::interaction::interaction_left(p, M,
				   (*aux_spins)[replica][s], -1., 0);
    }
}


void cx_dqmc::greens_replica_ft::slice_matrix_left_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
	// cout << "Checkerboard" << endl << M << endl << endl;
	cx_dqmc::interaction::interaction_left(p, M,
				     (*aux_spins)[replica][s], 1., 0);
	// cout << "Onsite" << endl << M << endl << endl;
    } else {
	cx_dqmc::interaction::interaction_left(p, M,
				    (*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
    }
}

void cx_dqmc::greens_replica_ft::slice_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(ws, M, 1.);
	cx_dqmc::interaction::interaction_right(p, M,
				    (*aux_spins)[replica][s], 1., 0);	
    } else {
	cx_dqmc::interaction::interaction_right(p, M,
				    (*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_right(ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_ft::hopping_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
    }
}


void cx_dqmc::greens_replica_ft::hopping_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_right(ws, M, -1.);
    }
}

void cx_dqmc::greens_replica_ft::regularize_svd(vec_t& in, vec_t& out)  {
    double length = vol - ws->particles - 1;
    double min_log = log10(in.minCoeff()) - 32.;
    double max_log = min_log;// min(150., min_log + 2 * length);

    for (int i = 0; i < ws->particles; ++i) {
	out(i) = in(i);
    }

    for (int i = ws->particles; i < vol; ++i) {
	out(i) = pow(10, min_log);
    }
}


void cx_dqmc::greens_replica_ft::log_weight() {    
    if (current_slice % (slices / 2) != 0) {
	throw std::runtime_error("Not at the correct time "
				 "slice to calculate the log weight");
    }

    fresh_det = true;
    
    int is[4], rs[4];
    
    is[0] = 0;
    is[1] = chunks - 1;

    try {
	ws->mat_4 = u_stack[is[1]];
	ws->mat_2 = u_stack[is[0]];
	ws->re_vec_2 = d_stack[is[1]];
	ws->re_vec_1 = d_stack[is[0]];
	ws->mat_3 = t_stack[is[1]];
	ws->mat_1 = t_stack[is[0]];
	    
	// dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_4);
	// dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_2);

	// regularize_svd(d_stack[is[1]], ws->re_vec_2);
	// ws->mat_3.setIdentity();
	// ws->mat_3.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];
	// regularize_svd(d_stack[is[0]], ws->re_vec_1);
	// ws->mat_1.setIdentity();
	// ws->mat_1.block(0, 0, ws->particles, ws->particles) = t_stack[is[0]];
    } catch (const std::bad_alloc&) {
	throw std::runtime_error("Enlaring matrices failed with bad_alloc");
    }

    static cx_mat large_mat_1;
    static cx_mat large_mat_2;
    static cx_mat large_mat_3;
    static vec large_vec_1;

    try {
	if (large_mat_1.rows() != 2 * vol) {
	    large_mat_1.resize(2 * vol, 2 * vol);
	    large_mat_2.resize(2 * vol, 2 * vol);
	    large_mat_3.resize(2 * vol, 2 * vol);
	    large_vec_1.resize(2 * vol);
	}
    } catch (const std::bad_alloc&) {
	throw std::runtime_error("Resizing matrices failed with bad_alloc");
    }

    static Eigen::FullPivLU<cx_mat_t> lu1(vol, vol);
    static Eigen::FullPivLU<cx_mat_t> lu2(vol, vol);
    static Eigen::FullPivLU<cx_mat_t> large_lu(2 * vol, 2 * vol);
        
    large_mat_1.setZero();
    
    det_sign = cx_double(1.0, 0);
    
    lu1.compute(ws->mat_4); lu2.compute(ws->mat_2.transpose());
    large_mat_1.block(0, 0, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= lu1.determinant() * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9) {
	cout << u_stack[is[1]] << endl << endl;
	cout << ws->mat_4 << endl << endl;

	cout << p->outp << "log_weight(): abs det is not one " << det_sign << endl;
    }
    
    // cout << "current det sign\t" << det_sign << endl;
    
    lu1.compute(ws->mat_1.transpose()); lu2.compute(ws->mat_3);    
    large_mat_1.block(vol, vol, vol, vol) = lu1.inverse() * lu2.inverse();
    det_sign *= (lu1.determinant()) * (lu2.determinant());
    if (fabs(fabs(det_sign) - 1) > 1e-9)
	cout << p->outp << "log_weight(): abs det is not one" << endl;
 
    large_mat_1.block(0, vol, vol, vol).diagonal().real()
	= ws->re_vec_2;
    large_mat_1.block(vol, 0, vol, vol).diagonal().real()
	= -ws->re_vec_1;

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
    
    // for(auto i = aux_spins[replica].origin(); i < (aux_spins[replica].origin() + aux_spins[replica].num_elements()); ++i) {
    // 	prefactor_spin_sum += double(*i);	
    // }

    prefactor_spin_sum = 0.;
    for (uint s = 0; s < p->slices; s++) {
	for (int i = 0; i < p->num_aux_spins; i++) {
	    prefactor_spin_sum += (*aux_spins)[replica][s][i];
	}
    }

    phase = std::exp(-1. * prefactor_spin_sum * p->cx_osi_lambda);
}


void cx_dqmc::greens_replica_ft::calculate_greens_general() {
    int is[5], rs[5];

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();

    if (current_slice % slices == 0) {
	// cout << "calculating greens here" << endl;
	is[0] = 0;
	is[1] = chunks - 1;

	ws->mat_4 = u_stack[is[1]];
	ws->mat_2 = u_stack[is[0]];
	ws->re_vec_2 = d_stack[is[1]];
	ws->re_vec_1 = d_stack[is[0]];
	ws->mat_3 = t_stack[is[1]];
	ws->mat_1 = t_stack[is[0]];
	
	// dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_4);
	// dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_2);

	// regularize_svd(d_stack[is[1]], ws->re_vec_2);
	// ws->mat_3.setIdentity();
	// ws->mat_3.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];

	// regularize_svd(d_stack[is[0]], ws->re_vec_1);
	// ws->mat_1.setIdentity();
	// ws->mat_1.block(0, 0, ws->particles, ws->particles) = t_stack[is[0]];

	ws->la_mat_1 = ws->mat_1;
	ws->mat_1 = ws->mat_2.transpose();
	ws->mat_2 = ws->la_mat_1.transpose();

	Us.push_back(&ws->mat_2);
	Us.push_back(&ws->mat_4);

	Ts.push_back(&ws->mat_1);
	Ts.push_back(&ws->mat_3);

	Ds.push_back(&ws->re_vec_1);
	Ds.push_back(&ws->re_vec_2);

	cx_mat_t large_mat_1 = cx_mat::Zero(2 * vol, 2 * vol);
	cx_mat_t large_mat_2 = cx_mat::Zero(2 * vol, 2 * vol);
	cx_mat_t large_mat_3 = cx_mat::Zero(2 * vol, 2 * vol);
	cx_mat_t large_mat_4 = cx_mat::Zero(2 * vol, 2 * vol);
    
	large_mats.push_back(&large_mat_1);
	large_mats.push_back(&large_mat_2);
	large_mats.push_back(&large_mat_3);
	large_mats.push_back(&large_mat_4);

	cx_mat_t large_U = cx_mat_t::Zero(2 * vol, 2 * vol);
	cx_mat_t large_T = cx_mat_t::Zero(2 * vol, 2 * vol);
    
	vec_t large_vec_1 = vec_t::Zero(2 * vol);
	vec_t large_vec_2 = vec_t::Zero(2 * vol);
	vec_t large_vec_3 = vec_t::Zero(2 * vol);
	vec_t large_vec_4 = vec_t::Zero(2 * vol);
    
	large_vecs.push_back(&large_vec_1);
	large_vecs.push_back(&large_vec_2);
	large_vecs.push_back(&large_vec_3);
	large_vecs.push_back(&large_vec_4);

	dqmc::calculate_greens::full_piv_qr_full_piv_lu(Us, Ds, Ts,
							large_mats, 
							large_U, large_T,
							large_vecs, *ws,
							greens);    
    }

    // 2 * vol - up
    else if (direction != 4) {
	if (direction == 1 && (current_slice + 1) < slices / 2
	    && current_slice > 0) {
	    
	    is[0] = current_slice / safe_mult;
	    is[1] = chunks - 1;

	    ws->mat_4 = u_stack[is[1]];
	    ws->mat_2 = u_stack[is[0]];
	    ws->re_vec_2 = d_stack[is[1]];
	    ws->re_vec_1 = d_stack[is[0]];
	    ws->mat_3 = t_stack[is[1]];
	    ws->mat_1 = t_stack[is[0]];

	    // dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_1);
	    Tr = ws->mat_2.transpose();
	    Dr = ws->re_vec_1;
	    
	    // regularize_svd(d_stack[is[0]], Dr);

	    // Ur.setIdentity();
	    // Ur.block(0, 0, ws->particles, ws->particles) =		
	    // 	t_stack[is[0]].transpose();
	    Ur = ws->mat_1.transpose();
	
	    //============================================================

	    ws->mat_5 = t_temp * u_stack[is[1]];
	    ws->mat_4 = d_temp.asDiagonal() * ws->mat_5;
	    ws->mat_5 = ws->mat_4 * d_stack[is[1]].asDiagonal();

	    dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_6,
					     Dl,
					    ws->mat_7);

	    Ul  = u_temp * ws->mat_6;
	    Tl = ws->mat_7 * t_stack[is[1]];
	    
	    // dqmc::la::thin_col_to_invertible(ws->col_mat_4, Ul);
	    // regularize_svd(ws->re_tiny_vec_4, Dl);
	    // Tl.setIdentity();
	    // Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_6;
	    
	    Us.push_back(&Ur);
	    Us.push_back(&Ul);
	    Ds.push_back(&Dr);
	    Ds.push_back(&Dl);
	    Ts.push_back(&Tr);
	    Ts.push_back(&Tl);
	
	    large_mats.push_back(&ws->large_mat_1);
	    large_mats.push_back(&ws->large_mat_2);
	    large_mats.push_back(&ws->large_mat_3);
	
	    large_vecs.push_back(&ws->large_vec_1);
	    large_vecs.push_back(&ws->large_vec_2);
	    large_vecs.push_back(&ws->large_vec_3);

	    dqmc::calculate_greens::full_piv_qr_full_piv_lu(Us, Ds, Ts,
							    large_mats,
							    ws->large_mat_4,
							    ws->large_mat_5,
							    large_vecs, *ws,
							    greens);
	    return;
	}

	// 2 * vol - down
	else if (direction == -1 && (current_slice) < slices
		 && current_slice > slices / 2) {
	    // cout << "Calculating down" << endl;
	    is[0] = 0;
	    is[1] = chunks - 1 - (slices - current_slice) / safe_mult;

	    // cout << "Indices are " << is[0] << " - " << is[1] << endl;
	    //============================================================
	    Ul = u_stack[is[1]];
	    Dl = d_stack[is[1]];
	    Tl = t_stack[is[1]];

	    // dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_1);
	    // Ul = ws->mat_1;	    
	    // regularize_svd(d_stack[is[1]], Dl);
	    // Tl.setIdentity();
	    // Tl.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];

	    //============================================================	    

	    ws->mat_5 = t_temp * u_stack[is[0]];
	    ws->mat_4 = d_temp.asDiagonal() * ws->mat_5;
	    ws->mat_5 = ws->mat_4 * d_stack[is[0]].asDiagonal();

	    dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_6,
					     Dr, ws->mat_7);

	    ws->mat_4 = u_temp * ws->mat_6;
	    ws->mat_6 = ws->mat_7 * t_stack[is[0]];

	    Tr = ws->mat_4.transpose();
	    Ur = ws->mat_6.transpose();

	    Us.push_back(&Ur);
	    Us.push_back(&Ul);
	    Ds.push_back(&Dr);
	    Ds.push_back(&Dl);
	    Ts.push_back(&Tr);
	    Ts.push_back(&Tl);
	
	    large_mats.push_back(&ws->large_mat_1);
	    large_mats.push_back(&ws->large_mat_2);
	    large_mats.push_back(&ws->large_mat_3);
	
	    large_vecs.push_back(&ws->large_vec_1);
	    large_vecs.push_back(&ws->large_vec_2);
	    large_vecs.push_back(&ws->large_vec_3);

	    dqmc::calculate_greens::full_piv_qr_full_piv_lu(Us, Ds, Ts,
							    large_mats,
							    ws->large_mat_4,
							    ws->large_mat_5,
							    large_vecs, *ws,
							    greens);	
	    return;
	}
    }
}


void cx_dqmc::greens_replica_ft::calculate_greens_exact(int slice) {
    // cout << "current_slice\t" << slice << endl;

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();

    
    if (slice < slices / 2) {
	ws->col_mat_1 = ws->den_U.conjugate();
	ws->re_tiny_vec_1.setOnes();
	ws->tiny_mat_1.setIdentity();
    
	for (int s = slices / 2; s > slice; s -= safe_mult) {
	    start =  max(s - safe_mult, slice);
	    stop = s;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left_t(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	// prep_bra(ws->col_mat_1, ws->re_tiny_vec_1, ws->tiny_mat_1);
	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_1);
	Tr = ws->mat_1.transpose();
	// Dr = ws->re_tiny_vec_1;
	regularize_svd(ws->re_tiny_vec_1, Dr);
	Ur.setIdentity();
	Ur.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1.transpose();
	
	ws->col_mat_1 = ws->den_U;	
	ws->re_tiny_vec_1.setOnes();
	ws->tiny_mat_1.setIdentity();	

	// cout << "add" << endl;
	
	for (int s = slices/2; s < slices; s += safe_mult) {
	    start = s;
	    stop = s + safe_mult;
	    // cout << start << " to " << stop << endl;
	    slice_sequence_left(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	// cout << "Next" << endl;
	
	for (int s = 0; s < slice; s += safe_mult) {
	    start = s;
	    stop = min(s + safe_mult, slice);
	    // cout << start << " to " << stop << endl;
	    
	    slice_sequence_left(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	dqmc::la::thin_col_to_invertible(ws->col_mat_1, Ul);
	// Dl = ws->re_tiny_vec_1;
	regularize_svd(ws->re_tiny_vec_1, Dl);
	Tl.setIdentity();
	Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1;
	
	// prep_ket(ws->col_mat_1, ws->re_tiny_vec_1, ws->tiny_mat_1);
    }
    else {
	ws->col_mat_1 = ws->den_U.conjugate();
	ws->re_tiny_vec_1.setOnes();
	ws->tiny_mat_1.setIdentity();
    
	for (int s = slices / 2; s > 0; s -= safe_mult) {
	    start = s - safe_mult;
	    stop = s;
	    slice_sequence_left_t(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	for (int s = slices; s > slice; s -= safe_mult) {
	    start =  max(s - safe_mult, slice);
	    stop = s;
	    slice_sequence_left_t(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}

	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_1);
	Tr = ws->mat_1.transpose();
	regularize_svd(ws->re_tiny_vec_1, Dr);
	Ur.setIdentity();
	Ur.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1.transpose();	

	ws->col_mat_1 = ws->den_U;	
	ws->re_tiny_vec_1.setOnes();
	ws->tiny_mat_1.setIdentity();	

	for (int s = slices/2; s < slice; s += safe_mult) {
	    start = s;
	    stop = min(s + safe_mult, slice);
	    slice_sequence_left(start, stop, ws->col_mat_1);
	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
					     ws->re_tiny_vec_1, ws->tiny_mat_2);
	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
	}


	dqmc::la::thin_col_to_invertible(ws->col_mat_1, Ul);
	regularize_svd(ws->re_tiny_vec_1, Dl);
	Tl.setIdentity();
	Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1;
    }

    Us.push_back(&Ur);
    Us.push_back(&Ul);
    Ds.push_back(&Dr);
    Ds.push_back(&Dl);
    Ts.push_back(&Tr);
    Ts.push_back(&Tl);
	
    large_mats.push_back(&ws->large_mat_1);
    large_mats.push_back(&ws->large_mat_2);
    large_mats.push_back(&ws->large_mat_3);
	
    large_vecs.push_back(&ws->large_vec_1);
    large_vecs.push_back(&ws->large_vec_2);
    large_vecs.push_back(&ws->large_vec_3);

    // cout << "calculating greens" << endl;
    
    dqmc::calculate_greens::full_piv_qr_full_piv_lu(Us, Ds, Ts,
						    large_mats,
						    ws->large_mat_4,
						    ws->large_mat_5,
						    large_vecs, *ws,
						    greens);    
    // cout << "endl" << endl;
}
