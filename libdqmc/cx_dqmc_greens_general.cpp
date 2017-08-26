#include "cx_dqmc_greens_general.hpp"

using namespace std;

void cx_dqmc::greens_general::initialize() {
    vol = p->N;

    replica = p->replica;    
    sites = ws->sites;    
    X.resize(vol, 100);
    Y.resize(vol, 100);

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

    ws->density = ws->den_U * ws->den_U.adjoint();
    active = false;
    swept_on = true;
}



void cx_dqmc::greens_general::build_stack() {
    
    if (first_initialization == true) {
	vol = ws->vol;
	sites = ws->sites;	
	n_elements = slices / safe_mult + 1;
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

    direction = -1;
    idx = 0;
    u_stack[idx].setIdentity();
    d_stack[idx].setOnes();
    t_stack[idx].setIdentity();

    for (int i = 0; i < chunks - 1; ++i) {
	// cout << "before " << i << " " << chunks << endl;
	update_on_left(safe_mult * i, safe_mult * (i + 1), i, i + 1);
	// cout << "after " << i << " " << chunks << endl;
	++idx;
    }
    // cout << "Done ?" << endl;
    
    Ul = u_stack[idx];
    Dl = d_stack[idx];
    Tl = t_stack[idx];

    Ur.setIdentity();
    Dr.setOnes();
    Tr.setIdentity();
    // cout << "corresponding greens" << endl << greens.diagonal().transpose() << endl << endl;    
    
    if (first_initialization == true) {
	cout << p->outp << d_stack[idx][0]
	     << " general: to onefold col SVD gap\t "
	     << d_stack[idx][p->particles - 1] << endl;
	if (greens.cols() < 10) {
	    cout << greens.trace() << " is regular trace" << endl;
	    cout << greens << endl;
	}
	first_initialization = false;
    }

    if(p->adapt_mu == true && p->adapted_mu == false) {
	p->adapted_mu = true;
	cout << p->outp << "Smallest SV "
	     << d_stack[idx][p->particles - 1] << endl;
	cout << p->outp << d_stack[idx].transpose() << endl;
	cout << p->outp << "Spectrum " << d_stack[idx].maxCoeff() << " - "
	     <<  d_stack[idx].minCoeff() << endl;

	cout << p->outp << "Target " << p->mu_target << endl;

	if (p->model_id == p->SPINLESS_HUBBARD) {
	    p->mu_site = 1./p->beta * log(p->mu_target / d_stack[idx][p->particles - 1]);
	} else {
	    p->mu_site = 2 / p->beta * log(p->mu_target / d_stack[idx][p->particles - 1]);
	}
	cout << p->outp << "New mu guess " << p->mu_site << endl;

	if (p->simple_adapt_mu == false) {
	    p->mu_site = 0.;
	}
	
	p->mu_site_factor[0] = exp(-p->delta_tau * p->mu_site);
	p->mu_site_factor[1] = NAN;
	p->mu_site_factor[2] = exp(p->delta_tau * p->mu_site);
	build_stack();
	cout << p->outp << "New Smallest SV " << d_stack[idx][p->particles - 1] << endl;
	cout << p->outp << d_stack[idx].transpose() << endl;
	cout << p->outp << "New Spectrum " << d_stack[idx].maxCoeff() << " - "
	     <<  d_stack[idx].minCoeff() << endl;	
    }


    idx = chunks - 1;
    u_stack[idx].setIdentity();
    d_stack[idx].setOnes();
    t_stack[idx].setIdentity();
    
    Tr = u_stack[idx].transpose();
    Dr = d_stack[idx];
    Ur = t_stack[idx].transpose();
    // cout << "Calculating greens" << endl;
    calculate_greens_basic();
    cout << p->outp << "New Greens diag "
	 << greens.diagonal()(0) << " - "
	 <<  greens.diagonal()(1) << endl;


    alps::RealObservable dummy("dummy");
    cout << "Propagating" << endl;
    for (int i = 0; i < 3 * slices; ++i) {
    	propagate(dummy);
    }
}


int cx_dqmc::greens_general::propagate(alps::Observable& stability) {
    if (direction == -1) {
	if (current_slice == slices) {
	    Ul = u_stack[chunks - 1];
	    Dl = d_stack[chunks - 1];
	    Tl = t_stack[chunks - 1];
	    
	    Ur.setIdentity();
	    Dr.setOnes();
	    Tr.setIdentity();

	    swept_on = false;
	    if (active == true) { calculate_greens_basic(); }
	    idx = chunks - 1;
	    u_stack[idx].setIdentity();
	    d_stack[idx].setOnes();
	    t_stack[idx].setIdentity();
	    
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	    return 0;		
	}

	else if (current_slice == slices / 2) {
	    start = current_slice;
	    stop = current_slice + safe_mult;
	    idx = current_slice / safe_mult;
	    
	    update_on_right(start, stop,
			    idx + 1, idx);
	    Ul = u_stack[idx - 1];
	    Dl = d_stack[idx - 1];
	    Tl = t_stack[idx - 1];
	    update_on_right(start - safe_mult, stop - safe_mult,
			    idx, idx - 1);
	    current_slice -= safe_mult;    
	    Ur = t_stack[idx - 1].transpose();
	    Dr = d_stack[idx - 1];
	    Tr = u_stack[idx - 1].transpose();

	    // cout << "skipping" << endl;
	    if (active == true) {
		calculate_greens_basic();
	    }

	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	    return 2;
	}
	
	else if (current_slice == 0) {
	    start = current_slice;
	    stop = current_slice + safe_mult;
	    idx = current_slice / safe_mult;

	    // cout << "Last update " << idx + 1 << " - " << idx << endl;
	    update_on_right(start, stop, idx + 1, idx);
	    --current_slice;
	    direction = 1;
	    lowest_sv = d_stack[idx][p->particles - 1];	    
	    propagate(stability);
	    return 3;
	}
	
	else if (current_slice % safe_mult == 0) {
	    start = current_slice;
	    stop = current_slice + safe_mult;
	    idx = current_slice / safe_mult;

	    // --current_slice;
	    // slice_matrix_left(current_slice, greens, true);
	    // slice_matrix_right(current_slice, greens, false);
	    temp_greens = greens;

	    // cout << "Saving " << idx << " / " << chunks << endl;
	    Ul = u_stack[idx];
	    Dl = d_stack[idx];
	    Tl = t_stack[idx];

	    update_on_right(start, stop, idx + 1, idx);

	    Ur = t_stack[idx].transpose();
	    Dr = d_stack[idx];
	    Tr = u_stack[idx].transpose();

	    if (active == true) { calculate_greens_basic();
		check_stability(); }
	    
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	    return 2;
	}
	
	else {
	    --current_slice;
	    slice_matrix_left(current_slice, greens, true);
	    slice_matrix_right(current_slice, greens, false);
	    return 1;
	}
    }
    
    else if (direction == 1) {
	if (current_slice == -1) {
	    Ul.setIdentity();
	    Dl.setOnes();
	    Tl.setIdentity();
	    
	    Ur = t_stack[0].transpose();
	    Dr = d_stack[0];
	    Tr = u_stack[0].transpose();

	    ++current_slice;

	    if (active == true) { calculate_greens_basic(); }

	    u_stack[0].setIdentity();
	    d_stack[0].setOnes();
	    t_stack[0].setIdentity();

	    return 0;

	} else if (current_slice + 1 == slices) {
	    idx = (current_slice + 1) / safe_mult;
	    start = (current_slice + 1 - safe_mult);
	    stop = current_slice + 1;

	    update_on_left(start, stop, idx - 1, idx );
	    
	    ++current_slice;
	    direction = -1;
	    propagate(stability);
	    return 3;
	    
	} else if ((current_slice + 1) == slices / 2) {
	    idx = (current_slice + 1) / safe_mult;
	    start = (current_slice + 1 - safe_mult);
	    stop = current_slice + 1;
	    update_on_left(start, stop, idx - 1, idx);
	    ++current_slice;
	    
	    start += safe_mult;
	    stop += safe_mult;

	    Ur = t_stack[idx + 1].transpose();
	    Dr = d_stack[idx + 1];
	    Tr = u_stack[idx + 1].transpose();
	    update_on_left(start, stop, idx, idx + 1);
	    Ul = u_stack[idx + 1];
	    Dl = d_stack[idx + 1];
	    Tl = t_stack[idx + 1];

	    current_slice += safe_mult;
	    if (active == true) { calculate_greens_basic(); }
	    lowest_sv = d_stack[idx][p->particles - 1];
	    return 2;
	}

	else if ((current_slice + 1) % safe_mult == 0) {
	    idx = (current_slice + 1) / safe_mult;
	    start = (current_slice + 1 - safe_mult);
	    stop = current_slice + 1;

	    Ur = t_stack[idx].transpose();
	    Dr = d_stack[idx];
	    Tr = u_stack[idx].transpose();
	    // cout << "Saving in " << idx << endl;
	    update_on_left(start, stop, idx - 1, idx);
	    Ul = u_stack[idx];
	    Dl = d_stack[idx];
	    Tl = t_stack[idx];

	    
	    slice_matrix_right(current_slice, greens, true);
	    slice_matrix_left(current_slice, greens, false);
	    ++current_slice;
	    temp_greens = greens;

	    if (active == true) { calculate_greens_basic();
		check_stability(); }
	    return 2;
	}
	else {
	    // cout << "Next slice " << current_slice << endl;
	    //      << current_bond << endl;
	    slice_matrix_right(current_slice, greens, true);
	    slice_matrix_left(current_slice, greens, false);
	    ++current_slice;
	    return 1;
	}
    }
    return 1;
}



void cx_dqmc::greens_general::update_remove_interaction() {
    // throw std::runtime_error("Isn't the test phase over?");
    prop_greens.setIdentity();
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens,  ws->vec_1,
				(*aux_spins)[replica][current_slice], -1., 0);
}

void cx_dqmc::greens_general::update_add_interaction() {
    // throw std::runtime_error("Isn't the test phase over?");
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens, ws->vec_1,
				(*aux_spins)[replica][current_slice], 1., 0);
}


double cx_dqmc::greens_general::check_stability() {
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


void cx_dqmc::greens_general::update_on_left(int start, int stop,
					     int i, int j) {
    bool add_den = false;
    bool regularize = false;

    if (start == slices / 2) add_den = true;
    if (start >= slices / 2) regularize = true;

    // cout << "on left\t" << setw(4) << add_den << setw(6) << start << " " << setw(6) << stop << endl;
    
    if (add_den) 
	ws->mat_1 = ws->density * u_stack[i];
    else
	ws->mat_1 = u_stack[i];

    slice_sequence_left(start, stop, ws->mat_1);
    ws->mat_2 = ws->mat_1 * d_stack[i].asDiagonal();

    dqmc::la::decompose_udt_col_piv(ws->mat_2,
				    u_stack[j],
				    d_stack[j], 
				    ws->mat_3);	
    t_stack[j] = ws->mat_3 * t_stack[i];
    if (regularize == true) regularize_svd(d_stack[j]);
}


void cx_dqmc::greens_general::update_on_right(int start, int stop,
					      int i, int j) {
    bool add_den = false;
    bool regularize = false;
    
    if (stop == slices / 2) add_den = true;
    if (stop <= slices / 2) regularize = true;

    // cout << "on right\t" << setw(4) << add_den << setw(6) << start << " " << setw(6) << stop << endl;

    if (add_den == true) {
	ws->mat_1 = ws->density.transpose() * u_stack[i];
    } else {
	ws->mat_1 = u_stack[i];
    }

    slice_sequence_left_t(start, stop, ws->mat_1);
    ws->mat_2 = ws->mat_1 * d_stack[i].asDiagonal();

    dqmc::la::decompose_udt_col_piv(ws->mat_2, u_stack[j],
				    d_stack[j], ws->mat_3);
    t_stack[j] = ws->mat_3 * t_stack[i];
    if (regularize == true) regularize_svd(d_stack[j]);
}


void cx_dqmc::greens_general::slice_sequence_left(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left(i, M); }
}


void cx_dqmc::greens_general::slice_sequence_left_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) 
	{
	    slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_general::slice_sequence_right(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_general::slice_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
				   (*aux_spins)[replica][s], 1., 0);
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
				   (*aux_spins)[replica][s], -1., 0);
    }
}


void cx_dqmc::greens_general::slice_matrix_left_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
	// cout << "Checkerboard" << endl << M << endl << endl;
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
				     (*aux_spins)[replica][s], 1., 0);
	// cout << "Onsite" << endl << M << endl << endl;
    } else {
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
				    (*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
    }
}

void cx_dqmc::greens_general::slice_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(ws, M, 1.);
	cx_dqmc::interaction::interaction_right(p, ws, M, ws->vec_1,
						(*aux_spins)[replica][s], 1., 0);	
    } else {
	cx_dqmc::interaction::interaction_right(p, ws, M, ws->vec_1,
						(*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_right(ws, M, -1.);
    }
}


void cx_dqmc::greens_general::hopping_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(ws, M, -1.);
    }
}


void cx_dqmc::greens_general::hopping_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_right(ws, M, -1.);
    }
}

void cx_dqmc::greens_general::regularize_svd(vec_t& in)  {
    // double length = vol - ws->particles - 1;
    // double min_log = log10(in.minCoeff()) - 32.;
    // double max_log = min_log;// min(150., min_log + 2 * length);

    // for (int i = 0; i < ws->particles; ++i) {
    // 	out(i) = in(i);
    // }

    for (int i = ws->particles; i < vol; ++i) {
	in(i) = 1e-10;
    }
}


void cx_dqmc::greens_general::log_weight_full_piv() {
    log_weight();
}


void cx_dqmc::greens_general::log_weight() {
    // if (current_slice % (slices / 2) != 0) {
    // 	throw std::runtime_error("Not at the correct time "
    // 				 "slice to calculate the log weight");
    // }

    // fresh_det = true;
    
    // int is[4], rs[4];
    
    // is[0] = 0;
    // is[1] = chunks - 1;

    // try {
    // 	dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_4);
    // 	dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_2);

    // 	regularize_svd(d_stack[is[1]], ws->re_vec_2);
    // 	ws->mat_3.setIdentity();
    // 	ws->mat_3.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];

    // 	regularize_svd(d_stack[is[0]], ws->re_vec_1);
    // 	ws->mat_1.setIdentity();
    // 	ws->mat_1.block(0, 0, ws->particles, ws->particles) = t_stack[is[0]];
    // } catch (const std::bad_alloc&) {
    // 	throw std::runtime_error("Enlaring matrices failed with bad_alloc");
    // }

    // static cx_mat large_mat_1;
    // static cx_mat large_mat_2;
    // static cx_mat large_mat_3;
    // static vec large_vec_1;

    // try {
    // 	if (large_mat_1.rows() != 2 * vol) {
    // 	    large_mat_1.resize(2 * vol, 2 * vol);
    // 	    large_mat_2.resize(2 * vol, 2 * vol);
    // 	    large_mat_3.resize(2 * vol, 2 * vol);
    // 	    large_vec_1.resize(2 * vol);
    // 	}
    // } catch (const std::bad_alloc&) {
    // 	throw std::runtime_error("Resizing matrices failed with bad_alloc");
    // }

    // static Eigen::FullPivLU<cx_mat_t> lu1(vol, vol);
    // static Eigen::FullPivLU<cx_mat_t> lu2(vol, vol);
    // static Eigen::FullPivLU<cx_mat_t> large_lu(2 * vol, 2 * vol);
        
    // large_mat_1.setZero();
    
    // det_sign = cx_double(1.0, 0);
    
    // lu1.compute(ws->mat_4); lu2.compute(ws->mat_2.transpose());
    // large_mat_1.block(0, 0, vol, vol) = lu1.inverse() * lu2.inverse();
    // det_sign *= lu1.determinant() * (lu2.determinant());
    // if (fabs(fabs(det_sign) - 1) > 1e-9) {
    // 	cout << u_stack[is[1]] << endl << endl;
    // 	cout << ws->mat_4 << endl << endl;
    // 	cout << p->outp << "log_weight(): abs det is not one " << det_sign << endl;
    // }
    
    // // cout << "current det sign\t" << det_sign << endl;
    
    // lu1.compute(ws->mat_1.transpose()); lu2.compute(ws->mat_3);    
    // large_mat_1.block(vol, vol, vol, vol) = lu1.inverse() * lu2.inverse();
    // det_sign *= (lu1.determinant()) * (lu2.determinant());
    // if (fabs(fabs(det_sign) - 1) > 1e-9) {
    // 	cout << p->outp << "log_weight(): abs det is not one" << endl;
    // }
    
    // large_mat_1.block(0, vol, vol, vol).diagonal().real()
    // 	= ws->re_vec_2;
    // large_mat_1.block(vol, 0, vol, vol).diagonal().real()
    // 	= -ws->re_vec_1;

    // dqmc::la::decompose_udt_col_piv(large_mat_1, large_mat_2, large_vec_1, large_mat_3);

    // large_lu.compute(large_mat_2);
    // det_sign *= (large_lu.determinant());
    // // cout << "current det sign\t" << det_sign << endl;
    
    // if (fabs(fabs(det_sign) - 1) > 1e-9) cout << p->outp << "log_weight(): abs det is not one" << endl;

    // large_lu.compute(large_mat_3);
    // det_sign *= (large_lu.determinant());
    // // cout << "current det sign\t" << det_sign << endl;
    
    // if (fabs(fabs(det_sign) - 1) > 1e-9) {
    // 	cout << p->outp << "log_weight(): abs det is not one" << endl;
    // }
    
    // dqmc::la::log_sum(large_vec_1, log_det);

    // double prefactor_spin_sum = 0.;
    
    // // for(auto i = aux_spins[replica].origin(); i < (aux_spins[replica].origin()
    // // + aux_spins[replica].num_elements()); ++i) {
    // // 	prefactor_spin_sum += double(*i);	
    // // }

    // prefactor_spin_sum = 0.;
    // for (uint s = 0; s < p->slices; s++) {
    // 	for (int i = 0; i < p->num_aux_spins; i++) {
    // 	    prefactor_spin_sum += (*aux_spins)[replica][s][i];
    // 	}
    // }

    // phase = std::exp(-1. * prefactor_spin_sum * p->cx_osi_lambda);
}


void cx_dqmc::greens_general::calculate_greens_general() {
    // // cout << "General " << current_slice << " - " << direction << endl;
    // int is[5], rs[5];

    // Us.clear();
    // Ds.clear();
    // Ts.clear();
    // large_mats.clear();
    // large_vecs.clear();

    // if (current_slice % slices == 0) {
    // 	// cout << "calculating greens here" << endl;
    // 	is[0] = 0;
    // 	is[1] = chunks - 1;

    // 	dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_4);
    // 	dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_2);

    // 	regularize_svd(d_stack[is[1]], ws->re_vec_2);
    // 	ws->mat_3.setIdentity();
    // 	ws->mat_3.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];

    // 	regularize_svd(d_stack[is[0]], ws->re_vec_1);
    // 	ws->mat_1.setIdentity();
    // 	ws->mat_1.block(0, 0, ws->particles, ws->particles) = t_stack[is[0]];

    // 	ws->la_mat_1 = ws->mat_1;
    // 	ws->mat_1 = ws->mat_2.transpose();
    // 	ws->mat_2 = ws->la_mat_1.transpose();

    //     Us.push_back(&ws->mat_2); U_is_unitary.push_back(0);
    //     Us.push_back(&ws->mat_4); U_is_unitary.push_back(1);

    //     Ts.push_back(&ws->mat_1); T_is_unitary.push_back(1);
    //     Ts.push_back(&ws->mat_3); T_is_unitary.push_back(0);

    // 	// Us.push_back(&ws->mat_2);
    // 	// Us.push_back(&ws->mat_4);

    // 	// Ts.push_back(&ws->mat_1);
    // 	// Ts.push_back(&ws->mat_3);

    // 	Ds.push_back(&ws->re_vec_1);
    // 	Ds.push_back(&ws->re_vec_2);

    // 	cx_mat_t large_mat_1 = cx_mat::Zero(2 * vol, 2 * vol);
    // 	cx_mat_t large_mat_2 = cx_mat::Zero(2 * vol, 2 * vol);
    // 	cx_mat_t large_mat_3 = cx_mat::Zero(2 * vol, 2 * vol);
    // 	cx_mat_t large_mat_4 = cx_mat::Zero(2 * vol, 2 * vol);
    
    // 	large_mats.push_back(&large_mat_1);
    // 	large_mats.push_back(&large_mat_2);
    // 	large_mats.push_back(&large_mat_3);
    // 	large_mats.push_back(&large_mat_4);

    // 	cx_mat_t large_U = cx_mat_t::Zero(2 * vol, 2 * vol);
    // 	cx_mat_t large_T = cx_mat_t::Zero(2 * vol, 2 * vol);
    
    // 	vec_t large_vec_1 = vec_t::Zero(2 * vol);
    // 	vec_t large_vec_2 = vec_t::Zero(2 * vol);
    // 	vec_t large_vec_3 = vec_t::Zero(2 * vol);
    // 	vec_t large_vec_4 = vec_t::Zero(2 * vol);
    
    // 	large_vecs.push_back(&large_vec_1);
    // 	large_vecs.push_back(&large_vec_2);
    // 	large_vecs.push_back(&large_vec_3);
    // 	large_vecs.push_back(&large_vec_4);

    // 	dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
    // 							  Ds, Ts, T_is_unitary,
    // 							  large_mats, 
    // 							  large_U, large_T,
    // 							  large_vecs, *ws,
    // 							  greens);    
    // }

    // // 2 * vol - up
    // else if (direction != 4) {
    // 	if (direction == 1 && (current_slice + 1) < slices / 2
    // 	    && current_slice > 0) {

    //         is[0] = 0;
    //         is[1] = chunks - 1 - (slices - current_slice) / safe_mult;

    //         // cout << "Indices are " << is[0] << " - " << is[1] << endl;                                                                                                                                                                     
    //         //============================================================                                                                                                                                                                    

    // 	    dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_1);
    //         Tr = ws->mat_1.transpose();
    //         regularize_svd(d_stack[is[1]], Dr);
    //         Ur.setIdentity();
    //         Ur.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]].transpose();

    //         //============================================================                                                                                                                                                                    

    //         ws->col_mat_5 = t_temp * u_stack[is[0]];
    //         ws->col_mat_4 = d_temp.asDiagonal() * ws->col_mat_5;
    //         ws->col_mat_5 = ws->col_mat_4 * d_stack[is[0]].asDiagonal();

    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
    // 					    ws->re_tiny_vec_4, ws->tiny_mat_5);

    //         ws->col_mat_4 = u_temp * ws->col_mat_6;
    //         ws->tiny_mat_6 = ws->tiny_mat_5 * t_stack[is[0]];

    // 	    dqmc::la::thin_col_to_invertible(ws->col_mat_4, ws->mat_1);
    //         Ul = ws->mat_1;

    //         regularize_svd(ws->re_tiny_vec_4, Dl);
    //         Tl.setIdentity();
    //         Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_6;

    //         Us.push_back(&Ur); U_is_unitary.push_back(0);
    //         Us.push_back(&Ul); U_is_unitary.push_back(1);
    //         Ds.push_back(&Dr);
    //         Ds.push_back(&Dl);
    //         Ts.push_back(&Tr); T_is_unitary.push_back(1);
    //         Ts.push_back(&Tl); T_is_unitary.push_back(0);

    //         large_mats.push_back(&ws->large_mat_1);
    //         large_mats.push_back(&ws->large_mat_2);
    //         large_mats.push_back(&ws->large_mat_3);

    //         large_vecs.push_back(&ws->large_vec_1);
    //         large_vecs.push_back(&ws->large_vec_2);
    //         large_vecs.push_back(&ws->large_vec_3);

    // 	    dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
    // 							      Ds, Ts, T_is_unitary,
    // 							      large_mats,
    // 							      ws->large_mat_4,
    // 							      ws->large_mat_5,
    // 							      large_vecs, *ws,
    // 							      ws->mat_1);
    //         greens = ws->mat_1.transpose();

	    
    // 	    // is[0] = current_slice / safe_mult;
    // 	    // is[1] = chunks - 1;

    // 	    // dqmc::la::thin_col_to_invertible(u_stack[is[0]], ws->mat_1);
    // 	    // Tr = ws->mat_1.transpose();
	    
    // 	    // regularize_svd(d_stack[is[0]], Dr);

    // 	    // Ur.setIdentity();
    // 	    // Ur.block(0, 0, ws->particles, ws->particles) =
    // 	    // 	t_stack[is[0]].transpose();
	
    // 	    // //============================================================

    // 	    // ws->col_mat_5 = t_temp * u_stack[is[1]];
    // 	    // ws->col_mat_4 = d_temp.asDiagonal() * ws->col_mat_5;
    // 	    // ws->col_mat_5 = ws->col_mat_4 * d_stack[is[1]].asDiagonal();

    // 	    // dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
    // 	    // 				     ws->re_tiny_vec_4,
    // 	    // 				    ws->tiny_mat_5);

    // 	    // ws->col_mat_4 = u_temp * ws->col_mat_6;
    // 	    // ws->tiny_mat_6 = ws->tiny_mat_5 * t_stack[is[1]];
	    
    // 	    // dqmc::la::thin_col_to_invertible(ws->col_mat_4, Ul);
    // 	    // regularize_svd(ws->re_tiny_vec_4, Dl);
    // 	    // Tl.setIdentity();
    // 	    // Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_6;


    //         // Us.push_back(&Ur); U_is_unitary.push_back(0);
    //         // Us.push_back(&Ul); U_is_unitary.push_back(1);
    //         // Ds.push_back(&Dr);
    //         // Ds.push_back(&Dl);
    //         // Ts.push_back(&Tr); T_is_unitary.push_back(1);
    //         // Ts.push_back(&Tl); T_is_unitary.push_back(0);
	    
    // 	    // // Us.push_back(&Ur);
    // 	    // // Us.push_back(&Ul);
    // 	    // // Ds.push_back(&Dr);
    // 	    // // Ds.push_back(&Dl);
    // 	    // // Ts.push_back(&Tr);
    // 	    // // Ts.push_back(&Tl);
	
    // 	    // large_mats.push_back(&ws->large_mat_1);
    // 	    // large_mats.push_back(&ws->large_mat_2);
    // 	    // large_mats.push_back(&ws->large_mat_3);
	
    // 	    // large_vecs.push_back(&ws->large_vec_1);
    // 	    // large_vecs.push_back(&ws->large_vec_2);
    // 	    // large_vecs.push_back(&ws->large_vec_3);

    // 	    // dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, Ds, Ts,
    // 	    // 						      large_mats,
    // 	    // 						      ws->large_mat_4,
    // 	    // 						      ws->large_mat_5,
    // 	    // 						      large_vecs, *ws,
    // 	    // 						      greens);
    // 	    return;
    // 	}

    // 	// 2 * vol - down
    // 	else if (direction == -1 && (current_slice) < slices
    // 		 && current_slice > slices / 2) {
    // 	    // cout << "Calculating down" << endl;
    // 	    is[0] = 0;
    // 	    is[1] = chunks - 1 - (slices - current_slice) / safe_mult;

    // 	    // cout << "Indices are " << is[0] << " - " << is[1] << endl;
    // 	    //============================================================

    // 	    dqmc::la::thin_col_to_invertible(u_stack[is[1]], ws->mat_1);
    // 	    Ul = ws->mat_1;	    
    // 	    regularize_svd(d_stack[is[1]], Dl);
    // 	    Tl.setIdentity();
    // 	    Tl.block(0, 0, ws->particles, ws->particles) = t_stack[is[1]];

    // 	    //============================================================	    

    // 	    ws->col_mat_5 = t_temp * u_stack[is[0]];
    // 	    ws->col_mat_4 = d_temp.asDiagonal() * ws->col_mat_5;
    // 	    ws->col_mat_5 = ws->col_mat_4 * d_stack[is[0]].asDiagonal();

    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_5, ws->col_mat_6,
    // 					     ws->re_tiny_vec_4, ws->tiny_mat_5);

    // 	    ws->col_mat_4 = u_temp * ws->col_mat_6;
    // 	    ws->tiny_mat_6 = ws->tiny_mat_5 * t_stack[is[0]];
	    
    // 	    dqmc::la::thin_col_to_invertible(ws->col_mat_4, ws->mat_1);
    // 	    Tr = ws->mat_1.transpose();

    // 	    regularize_svd(ws->re_tiny_vec_4, Dr);
    // 	    Ur.setIdentity();
    // 	    Ur.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_6.transpose();


    // 	    Us.push_back(&Ur); U_is_unitary.push_back(0);
    // 	    Us.push_back(&Ul); U_is_unitary.push_back(1);
    // 	    Ds.push_back(&Dr);
    // 	    Ds.push_back(&Dl);
    // 	    Ts.push_back(&Tr); T_is_unitary.push_back(1);
    // 	    Ts.push_back(&Tl); T_is_unitary.push_back(0);
	
    // 	    large_mats.push_back(&ws->large_mat_1);
    // 	    large_mats.push_back(&ws->large_mat_2);
    // 	    large_mats.push_back(&ws->large_mat_3);
	
    // 	    large_vecs.push_back(&ws->large_vec_1);
    // 	    large_vecs.push_back(&ws->large_vec_2);
    // 	    large_vecs.push_back(&ws->large_vec_3);

    // 	    dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
    // 							      Ds, Ts, T_is_unitary,
    // 							      large_mats,
    // 							      ws->large_mat_4,
    // 							      ws->large_mat_5,
    // 							      large_vecs, *ws,
    // 							      greens);	
    // 	    return;
    // 	}
    // }
}


void cx_dqmc::greens_general::calculate_greens_basic() {
    // cout << "basic "  << current_slice << endl;
    int is[5], rs[5];

    Us.clear();
    Ds.clear();
    Ts.clear();
    large_mats.clear();
    large_vecs.clear();

    if (current_slice == 3 * slices) {
	if (current_slice == slices) {
	    ws->mat_1 = Tl.adjoint().partialPivLu().solve(Ul);
	    ws->mat_5 = ws->mat_1.adjoint();
	    ws->mat_5.diagonal().real() += Dl;
	
	    dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
	
	    ws->mat_4 = ws->mat_3 * Tl;
	    ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Ul.adjoint();
	    greens = ws->mat_4.partialPivLu().solve(ws->mat_5);
	} else {
	    ws->mat_1 = Ur.adjoint().partialPivLu().solve(Tr);
	    ws->mat_5 = ws->mat_1.adjoint();
	    ws->mat_5.diagonal().real() += Dr;
	
	    dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
	
	    ws->mat_4 = ws->mat_3 * Ur;
	    ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Tr.adjoint();
	    greens = ws->mat_4.partialPivLu().solve(ws->mat_5);
	}
    }

    else {

	// greens = (ws->identity + Ul * Dl.asDiagonal() * Tl * Tr * Dr.asDiagonal() * Ur).fullPivLu().inverse();
	// cout << "Diagonal" << greens.diagonal().transpose() << endl;
	// return;
	
	ws->mat_1 = Dl.asDiagonal() * Tl * Ur * Dr.asDiagonal();
	dqmc::la::decompose_udt_col_piv(ws->mat_1, ws->mat_2, ws->re_vec_1, ws->mat_3);
	ws->mat_1 = Ul * ws->mat_2;
	ws->mat_6 = ws->mat_3 * Tr;
		
	ws->mat_2 = ws->mat_6.adjoint().fullPivLu().solve(ws->mat_1);
	// ws->mat_2 = ws->mat_6.adjoint().colPivHouseholverQr().solve(ws->mat_1);
	ws->mat_3 = ws->mat_2.adjoint();
	ws->mat_3.diagonal().real() += ws->re_vec_1;
	
	dqmc::la::decompose_udt_col_piv(ws->mat_3, ws->mat_4, ws->re_vec_2, ws->mat_5);
	
	ws->mat_2 = ws->mat_5 * ws->mat_6;
	ws->mat_3 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_4.adjoint() * ws->mat_1.adjoint();
	// greens = ws->mat_2.partialPivLu().solve(ws->mat_3);
	greens = ws->mat_2.colPivHouseholderQr().solve(ws->mat_3);

	// cout << greens.diagonal().transpose() << endl;
	return;
    }
}


void cx_dqmc::greens_general::calculate_greens_exact(int slice) {
    // // cout << "current_slice\t" << slice << endl;

    // Us.clear();
    // Ds.clear();
    // Ts.clear();
    // large_mats.clear();
    // large_vecs.clear();

    
    // if (slice < slices / 2) {
    // 	ws->col_mat_1 = ws->den_U.conjugate();
    // 	ws->re_tiny_vec_1.setOnes();
    // 	ws->tiny_mat_1.setIdentity();
    
    // 	for (int s = slices / 2; s > slice; s -= safe_mult) {
    // 	    start =  max(s - safe_mult, slice);
    // 	    stop = s;
    // 	    // cout << start << " to " << stop << endl;
    // 	    slice_sequence_left_t(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}

    // 	// prep_bra(ws->col_mat_1, ws->re_tiny_vec_1, ws->tiny_mat_1);
    // 	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_1);
    // 	Tr = ws->mat_1.transpose();
    // 	// Dr = ws->re_tiny_vec_1;
    // 	regularize_svd(ws->re_tiny_vec_1, Dr);
    // 	Ur.setIdentity();
    // 	Ur.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1.transpose();
	
    // 	ws->col_mat_1 = ws->den_U;	
    // 	ws->re_tiny_vec_1.setOnes();
    // 	ws->tiny_mat_1.setIdentity();	

    // 	// cout << "add" << endl;
	
    // 	for (int s = slices/2; s < slices; s += safe_mult) {
    // 	    start = s;
    // 	    stop = s + safe_mult;
    // 	    // cout << start << " to " << stop << endl;
    // 	    slice_sequence_left(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}

    // 	// cout << "Next" << endl;
	
    // 	for (int s = 0; s < slice; s += safe_mult) {
    // 	    start = s;
    // 	    stop = min(s + safe_mult, slice);
    // 	    // cout << start << " to " << stop << endl;
	    
    // 	    slice_sequence_left(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}

    // 	dqmc::la::thin_col_to_invertible(ws->col_mat_1, Ul);
    // 	// Dl = ws->re_tiny_vec_1;
    // 	regularize_svd(ws->re_tiny_vec_1, Dl);
    // 	Tl.setIdentity();
    // 	Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1;
	
    // 	// prep_ket(ws->col_mat_1, ws->re_tiny_vec_1, ws->tiny_mat_1);
    // }
    // else {
    // 	ws->col_mat_1 = ws->den_U.conjugate();
    // 	ws->re_tiny_vec_1.setOnes();
    // 	ws->tiny_mat_1.setIdentity();
    
    // 	for (int s = slices / 2; s > 0; s -= safe_mult) {
    // 	    start = s - safe_mult;
    // 	    stop = s;
    // 	    slice_sequence_left_t(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}

    // 	for (int s = slices; s > slice; s -= safe_mult) {
    // 	    start =  max(s - safe_mult, slice);
    // 	    stop = s;
    // 	    slice_sequence_left_t(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}

    // 	dqmc::la::thin_col_to_invertible(ws->col_mat_1, ws->mat_1);
    // 	Tr = ws->mat_1.transpose();
    // 	regularize_svd(ws->re_tiny_vec_1, Dr);
    // 	Ur.setIdentity();
    // 	Ur.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1.transpose();	

    // 	ws->col_mat_1 = ws->den_U;	
    // 	ws->re_tiny_vec_1.setOnes();
    // 	ws->tiny_mat_1.setIdentity();	

    // 	for (int s = slices/2; s < slice; s += safe_mult) {
    // 	    start = s;
    // 	    stop = min(s + safe_mult, slice);
    // 	    slice_sequence_left(start, stop, ws->col_mat_1);
    // 	    ws->col_mat_2 = ws->col_mat_1 * ws->re_tiny_vec_1.asDiagonal();
    // 	    dqmc::la::decompose_udt_col_piv(ws->col_mat_2, ws->col_mat_1,
    // 					     ws->re_tiny_vec_1, ws->tiny_mat_2);
    // 	    ws->tiny_mat_3 = ws->tiny_mat_2 * ws->tiny_mat_1;
    // 	    ws->tiny_mat_1 = ws->tiny_mat_3;	    
    // 	}


    // 	dqmc::la::thin_col_to_invertible(ws->col_mat_1, Ul);
    // 	regularize_svd(ws->re_tiny_vec_1, Dl);
    // 	Tl.setIdentity();
    // 	Tl.block(0, 0, ws->particles, ws->particles) = ws->tiny_mat_1;
    // }

    // Us.push_back(&Ur);
    // Us.push_back(&Ul);
    // Ds.push_back(&Dr);
    // Ds.push_back(&Dl);
    // Ts.push_back(&Tr);
    // Ts.push_back(&Tl);
	
    // large_mats.push_back(&ws->large_mat_1);
    // large_mats.push_back(&ws->large_mat_2);
    // large_mats.push_back(&ws->large_mat_3);
	
    // large_vecs.push_back(&ws->large_vec_1);
    // large_vecs.push_back(&ws->large_vec_2);
    // large_vecs.push_back(&ws->large_vec_3);

    // // cout << "calculating greens" << endl;
    
    // dqmc::calculate_greens::col_piv_qr_full_piv_lu(Us, Ds, Ts,
    // 						    large_mats,
    // 						    ws->large_mat_4,
    // 						    ws->large_mat_5,
    // 						    large_vecs, *ws,
    // 						    greens);    
    // // cout << "endl" << endl;
}
