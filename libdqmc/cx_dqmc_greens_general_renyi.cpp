#include "cx_dqmc_greens_general_renyi.hpp"

using namespace std;

void cx_dqmc::greens_general_renyi::initialize() {
    vol = p->n_A + 2 * p->n_B;
   
    sites = ws->sites;
    
    if (p->update_type == p->CX_SPINFUL_UPDATE) {
	delayed_buffer_size = min(200, sites/2);
    }
    else if (p->update_type == p->CX_SPINLESS_UPDATE) {
	delayed_buffer_size = min(50, ws->num_bonds);
    }

    if (delayed_buffer_size % 2 != 0) {
	++delayed_buffer_size; }
    
    X.resize(vol, delayed_buffer_size);
    Y.resize(vol, delayed_buffer_size);

    X_sls.resize(vol, 2);
    Y_sls.resize(vol, 2);
    Y_sls_row_1.resize(2, vol);
    Y_sls_row_2.resize(2, vol);

    u_temp.resize(p->N, p->N);
    d_temp.resize(p->N);
    t_temp.resize(p->N, p->N);	    	    

    slices = p->slices;
    safe_mult = p->safe_mult;

    greens.resize(vol, vol);
    prop_greens.resize(vol, vol);
    temp_greens.resize(vol, vol);
    first_initialization = true;    
}

void cx_dqmc::greens_general_renyi::set_greens(cx_dqmc::greens_general * gs_0_,
					       cx_dqmc::greens_general * gs_1_) {
   gs_0 = gs_0_;
   gs_1 = gs_1_;
    
}


void cx_dqmc::greens_general_renyi::build_stack() {
    direction = 1;
		      
    if (first_initialization == true) {
	vol = ws->vol;
	sites = ws->sites;	
	n_elements = (2 * slices) / p->safe_mult;
	chunks = n_elements;

	for (int i = 0; i < n_elements; ++i) {
	    stability.push_back(0);
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

    gs_0->build_stack();
    gs_1->build_stack();

    if (gs_0->current_slice != 0 || gs_0->direction != 1
	|| gs_1->current_slice != 0 || gs_1->direction != 1) {
	throw std::runtime_error("general_renyi: slice and directions incorrect");
    }

    alps::RealObservable dummy("dummy");
    for (int i = 0; i < 3 * slices; ++i) {
    	propagate(dummy);
    }
    // cout << p->outp << "Log weight" << endl;
    // log_weight();
    
    // propagate(dummy);

    if (first_initialization) {
	if (greens.rows() < 10) {
	    cout << greens.trace() << " is renyi trace" << endl;
	    cout << greens << endl;
	}
    }

    first_initialization = false;
}


int cx_dqmc::greens_general_renyi::propagate(alps::Observable& stability) {
    greens_general * gs;
    int offset;
    
    if (current_slice < slices) {
	gs = gs_0;
	offset = 0;
    } else {
	gs = gs_1;
	offset = slices;
    }
    
    int ret = gs->propagate(stability);
    if ((current_slice + direction) % slices != gs->current_slice) {
	cout << p->outp << direction << " - " << current_slice << "\tvs\t"
	     << gs->direction << " - " << gs->current_slice << endl;
	throw std::runtime_error("propagate: invalid slice");
    }

    if (direction != gs->direction) {
	cout << p->outp << direction << " - " << current_slice << "\tvs\t"
	     << gs->direction << " - " << gs->current_slice << endl;
	throw std::runtime_error("propagate: invalid direction");
    }
    if (ret == 1) {
	if (direction == -1) {
	    current_slice = gs->current_slice + offset;
	    slice_matrix_left_renyi(current_slice, greens, true);
	    slice_matrix_right_renyi(current_slice, greens, false);
	    return 1;
	} else {
	    current_slice = gs->current_slice + offset;
	    slice_matrix_right_renyi(current_slice - 1, greens, true);
	    slice_matrix_left_renyi(current_slice - 1, greens, false);
	}	   	
    }
    else if (ret == 2) {
	if (direction == -1) {
	    current_slice = gs_0->current_slice + 1;
	    calculate_greens();
	    check_stability();
	    --current_slice;
	    slice_matrix_left_renyi(current_slice, greens, true);
	    slice_matrix_right_renyi(current_slice, greens, false);
	}
	else {
	    current_slice = gs_0->current_slice - 1;
	    slice_matrix_right_renyi(current_slice, greens, true);
	    slice_matrix_left_renyi(current_slice, greens, false);
	    ++current_slice;
	    temp_greens = greens;
	    calculate_greens();
	    check_stability();
	}
    } else if (ret == 3) {
	
    }
    return -1;
}


double cx_dqmc::greens_general_renyi::check_stability() {
    static int warnings = 0;
    static pdouble_t diff;
    static int max_warnings = 8 * slices/safe_mult;
    ws->re_mat_1 = (temp_greens - greens).cwiseAbs().real();
    diff = ws->re_mat_1.maxCoeff();

    if (warnings < max_warnings) {
    	if (diff > 1e-7) {
    	    ++warnings;
	    cout << p->outp << ws->step << " renyi unstable: "
    		 << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
    		 << current_slice << " / " << slices << "\t";
    	    cout << setw(15) << std::right << diff << endl;
    	}
    } else if (diff > 1e-1) {
    	    ++warnings;
	    cout << p->outp << ws->step << " renyi really unstable: "
    		 << "(" << warnings << ")\t" << direction << "[" << idx << "]\t"
    		 << current_slice << " / " << slices << "\t";
    	    cout << setw(15) << std::right << diff << " vs. exact " << endl;
    }
    return diff;
}


void cx_dqmc::greens_general_renyi::slice_sequence_left(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left(i, M); }
}


void cx_dqmc::greens_general_renyi::slice_sequence_left_renyi(int start, int stop, cx_mat_t& M) {
    for (int i = start; i < stop; ++i) { slice_matrix_left_renyi(i, M); }
}


void cx_dqmc::greens_general_renyi::slice_sequence_left_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }   
}


void cx_dqmc::greens_general_renyi::slice_sequence_left_renyi_t(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_renyi_t(i, M); }
}


void cx_dqmc::greens_general_renyi::slice_sequence_right(int start, int stop, cx_mat_t& M) {
    for (int i = stop - 1; i >= start; --i) { slice_matrix_left_t(i, M); }
}


void cx_dqmc::greens_general_renyi::slice_matrix_left(int slice, cx_mat_t& M, bool inv) {
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


void cx_dqmc::greens_general_renyi::slice_matrix_left_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, 1.);
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], 1., 0);
    } else {
	cx_dqmc::interaction::interaction_left(p, reg_ws, M, reg_ws->vec_1,
					       (*aux_spins)[replica][s], -1., 0);
	cx_dqmc::checkerboard::hop_left(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_general_renyi::slice_matrix_left_renyi(int slice, cx_mat_t& M, bool inv) {
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


void cx_dqmc::greens_general_renyi::slice_matrix_left_renyi_t(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    int replica = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, 1., (2 * slice) / slices);
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
				     (*aux_spins)[replica][s], 1., replica);
    } else {
	cx_dqmc::interaction::interaction_left(p, ws, M, ws->vec_1,
					       (*aux_spins)[replica][s], -1., replica);
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, -1., (2 * slice) / slices);
    }
}


void cx_dqmc::greens_general_renyi::slice_matrix_right(int slice, cx_mat_t& M, bool inv) {
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


void cx_dqmc::greens_general_renyi::slice_matrix_right_renyi(int slice, cx_mat_t& M, bool inv) {
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

void cx_dqmc::greens_general_renyi::hopping_matrix_left(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_left(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_general_renyi::hopping_matrix_right(int slice, cx_mat_t& M, bool inv) {
    int s = slice % slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right(reg_ws, M, 1.);
    } else {
	cx_dqmc::checkerboard::hop_right(reg_ws, M, -1.);
    }
}


void cx_dqmc::greens_general_renyi::hopping_matrix_left_renyi(int slice, cx_mat_t& M, bool inv) {
    int s = slice / slices;

    if (inv == false) {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, 1., (2 * slice) / slices);
    } else {
	cx_dqmc::checkerboard::hop_left_renyi(ws, M, -1., (2 * slice) / slices);
    }
}


void cx_dqmc::greens_general_renyi::hopping_matrix_right_renyi(int slice, cx_mat_t& M, bool inv) {
    int s = slice / slices;
    
    if (inv == false) {
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, 1., (2 * slice) / slices);
    } else {
	cx_dqmc::checkerboard::hop_right_renyi(ws, M, -1., (2 * slice) / slices);
    }
}

void cx_dqmc::greens_general_renyi::regularize_svd(vec_t& in, vec_t& out)  {
    double length = vol - ws->particles - 1;
    double min_log = log10(in.minCoeff()) - 25.;
    double max_log = min_log;// min(150., min_log + 2 * length);

    for (int i = 0; i < ws->eff_particles; ++i) {
	out(i) = in(i);
    }

    for (int i = ws->eff_particles; i < vol; ++i) {
	out(i) = pow(10, min_log);
    }
}


void cx_dqmc::greens_general_renyi::log_weight() {
    fresh_det = true;
    if (current_slice % (slices / 2) != 0) {
	throw std::runtime_error("Not at the correct time "
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
    } catch (const std::bad_alloc&) {
	throw std::runtime_error("Enlaring matrices failed with bad_alloc");
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
	throw std::runtime_error("Resizing matrices failed with bad_alloc");
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
    // 	for (int i = 0; i < p->num_aux_spins; i++) {
    // 	    prefactor_spin_sum += (*aux_spins)[0][s][i];
    // 	    prefactor_spin_sum += (*aux_spins)[1][s][i];
    // 	}
    // }

    phase = std::exp(-1. * prefactor_spin_sum * p->cx_osi_lambda);
}


void cx_dqmc::greens_general_renyi::calculate_greens_general() {
    // boost::timer::auto_cpu_timer t;
    // cout << "greens_general_renyi::calculate_greens_general()\t";
    
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

	// cout << "Full piv from four" << endl;;

	// cout << *Ts[0] << endl << endl;
	// cout << *Ts[1] << endl << endl;
	// cout << *Ts[2] << endl << endl;
	// cout << *Ts[3] << endl << endl;

	dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us, U_is_unitary,
							Ds,
							Ts, T_is_unitary,
							large_mats, 
							large_U, large_T,
							large_vecs, *ws,
							greens);    
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
		dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		Ul = ws->mat_3;
		regularize_svd(ws->re_tiny_vec_6, Dl);

		ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_3.transpose();
	    
		dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
		Tl = ws->mat_3;

		//============================================================

		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->tiny_mat_1.transpose();
		ws->tiny_mat_2 = ws->re_tiny_vec_2.asDiagonal() * ws->tiny_mat_5;
		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->re_tiny_vec_1.asDiagonal();

		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
						 ws->tiny_mat_6, ws->re_tiny_vec_6,
						 ws->tiny_mat_7);
	
		ws->col_mat_5 = ws->col_mat_2 * ws->tiny_mat_6;
		dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		Ur = ws->mat_3;
	    
		regularize_svd(ws->re_tiny_vec_6, Dr);
	
		ws->col_mat_1 = ws->col_mat_1 * ws->tiny_mat_7.transpose();
		ws->row_mat_1 = ws->col_mat_1.transpose();
		dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
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

		// cout << "greens up" << endl;
		// cout << Dr.transpose() << endl;
		// cout << Dl.transpose() << endl;

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
	else if ((direction == -2 && current_slice < slices
		  && current_slice > slices / 2)
		 || (direction == -2 && current_slice < 2 * slices
		     && current_slice > 3 * (slices / 2))) {

	    // cout << "Combining my way to the top" << endl;
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
	    
	    {

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
		dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		Ul = ws->mat_3;
		regularize_svd(ws->re_tiny_vec_6, Dl);

		ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_3.transpose();
	    
		dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
		Tl = ws->mat_3;

		//============================================================

		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->tiny_mat_1.transpose();
		ws->tiny_mat_2 = ws->re_tiny_vec_2.asDiagonal() * ws->tiny_mat_5;
		ws->tiny_mat_5 = ws->tiny_mat_2 * ws->re_tiny_vec_1.asDiagonal();

		dqmc::la::decompose_udt_col_piv(ws->tiny_mat_5,
						 ws->tiny_mat_6, ws->re_tiny_vec_6,
						 ws->tiny_mat_7);
	
		ws->col_mat_5 = ws->col_mat_2 * ws->tiny_mat_6;
		dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
		Ur = ws->mat_3;
	    
		regularize_svd(ws->re_tiny_vec_6, Dr);
	
		ws->col_mat_1 = ws->col_mat_1 * ws->tiny_mat_7.transpose();
		ws->row_mat_1 = ws->col_mat_1.transpose();
		dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
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

		// cout << "greens up" << endl;
		// cout << Dr.transpose() << endl;
		// cout << Dl.transpose() << endl;

		dqmc::calculate_greens::col_piv_qr_partial_piv_lu(Us,
								  U_is_unitary,
								  Ds, Ts, T_is_unitary,
								  large_mats,
								  ws->large_mat_4,
								  ws->large_mat_5,
								  large_vecs, *ws,
								  ws->mat_1);
		greens = ws->mat_1.transpose();
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
	    dqmc::la::thin_col_to_invertible(ws->col_mat_5, ws->mat_3);
	    Tr = ws->mat_3.transpose();
	    regularize_svd(ws->re_tiny_vec_6, Dr);

	    ws->row_mat_1 = ws->tiny_mat_7 * ws->col_mat_2.transpose();
	    dqmc::la::thin_row_to_invertible(ws->row_mat_1, ws->mat_3);
	    Ur = ws->mat_3.transpose();

	    // Us.push_back(&Ur);
	    // Us.push_back(&Ul);
	    // Ds.push_back(&Dr);
	    // Ds.push_back(&Dl);
	    // Ts.push_back(&Tr);
	    // Ts.push_back(&Tl);

            Us.push_back(&Ur); U_is_unitary.push_back(0);
            Us.push_back(&Ul); U_is_unitary.push_back(1);
            Ds.push_back(&Dr);
            Ds.push_back(&Dl);
            Ts.push_back(&Tr); T_is_unitary.push_back(1);
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


void cx_dqmc::greens_general_renyi::calculate_greens_basic() {
    // boost::timer::auto_cpu_timer t;
    // cout << "greens_general_renyi::calculate_greens_general()\t";
    
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

	ws->mat_1 = Tl.adjoint().partialPivLu().solve(Ul);
	ws->mat_5 = ws->mat_1.adjoint();
	ws->mat_5.diagonal().real() += Dl;

	dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
	
	ws->mat_4 = ws->mat_3 * Tl;
	ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Ul.adjoint();
	greens = ws->mat_4.partialPivLu().solve(ws->mat_5);
	
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

		ws->mat_1 = Tl.adjoint().partialPivLu().solve(Ul);
		ws->mat_5 = ws->mat_1.adjoint();
		ws->mat_5.diagonal().real() += Dl;
		
		dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
		
		ws->mat_4 = ws->mat_3 * Tl;
		ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Ul.adjoint();
		greens = ws->mat_4.partialPivLu().solve(ws->mat_5);

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

	    ws->mat_1 = Tl.adjoint().partialPivLu().solve(Ul);
	    ws->mat_5 = ws->mat_1.adjoint();
	    ws->mat_5.diagonal().real() += Dl;
		
	    dqmc::la::decompose_udt_col_piv(ws->mat_5, ws->mat_2, ws->re_vec_2, ws->mat_3);
		
	    ws->mat_4 = ws->mat_3 * Tl;
	    ws->mat_5 = ws->re_vec_2.asDiagonal().inverse() * ws->mat_2.adjoint() * Ul.adjoint();
	    greens = ws->mat_4.partialPivLu().solve(ws->mat_5);


    	    return;
    	}
    }
}


//===============================================================
// enlarging
//===============================================================

void cx_dqmc::greens_general_renyi::enlarge_thin(int replica,
						 vec_t& in,
						 vec_t& out) {
    out.setOnes();
    out.segment(0, in.size()) = in;
}
 
void cx_dqmc::greens_general_renyi::enlarge_thin(int replica,
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
		      p->n_B, p->n_B).setIdentity();	
	} else if (replica == 1) {
	    out.block(0, 0, p->n_A, ws->particles)
		= in.block(0, 0, p->n_A, ws->particles);
	    out.block(p->n_A, ws->particles, p->n_B, p->n_B).setIdentity();
	    out.block(p->N, 0, p->n_B, ws->particles)
		= in.block(p->n_A, 0, p->n_B, ws->particles);	
	}
    } else if (out.rows() < out.cols()) {

	if (replica == 0) {
	    out.block(0, 0, in.rows(), in.cols())
		= in;
	    out.block(in.rows(), in.cols(),
		      p->n_B, p->n_B).setIdentity();	
	} else if (replica == 1) {
	    out.block(0, 0, ws->particles, p->n_A)
		= in.block(0, 0, ws->particles, p->n_A);
	    out.block(ws->particles, p->n_A, p->n_B, p->n_B).setIdentity();
	    out.block(0, p->N, ws->particles, p->n_B)
		= in.block(0, p->n_A, ws->particles, p->n_B);	
	}	
    }
    else {
        out.setIdentity();
	out.block(0, 0, p->particles, p->particles)
            = in;
    }
}

void cx_dqmc::greens_general_renyi::enlarge_thin_ized_col(int replica,
							  cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    using namespace std;

    out.setZero();

    if (replica == 0) {
	out.block(0, 0, p->N, ws->particles)
	    = in.block(0, 0, p->N, ws->particles);
	out.block(p->N, ws->particles,
		  p->n_B, p->n_B).setIdentity();
	out.block(0, ws->eff_particles, p->N, in.cols() - ws->particles)
	    = in.block(0, ws->particles, p->N, in.cols() - ws->particles);
	    
    } else if (replica == 1) {
	out.block(0, 0, p->n_A, ws->particles)
	    = in.block(0, 0, p->n_A, ws->particles);
	out.block(p->n_A, ws->particles, p->n_B, p->n_B).setIdentity();
	out.block(p->N, 0, p->n_B, ws->particles)
	    = in.block(p->n_A, 0, p->n_B, ws->particles);	
	out.block(0, ws->eff_particles, p->n_A, in.cols() - ws->particles)
	    = in.block(0, ws->particles, p->n_A, in.cols() - ws->particles);	
	out.block(p->N, ws->eff_particles, p->n_B, in.cols() - ws->particles)
	    = in.block(p->n_A, ws->particles, p->n_B, in.cols() - ws->particles);

    }
}

void cx_dqmc::greens_general_renyi::enlarge_thin_ized_row(int replica,
							  cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    using namespace std;

    out.setZero();

    if (replica == 0) {
	out.block(0, 0, ws->particles, p->N)
	    = in.block(0, 0, ws->particles, p->N);
	out.block(ws->particles, p->N,
		  p->n_B, p->n_B).setIdentity();

	out.block(ws->eff_particles, 0, in.rows() - ws->particles, p->N)
	    = in.block(ws->particles,  0, in.rows() - ws->particles, p->N);
	
    } else if (replica == 1) {
	out.block(0, 0, ws->particles, p->n_A)
	    = in.block(0, 0, ws->particles, p->n_A);
	out.block(ws->particles, p->n_A, p->n_B, p->n_B).setIdentity();
	out.block(0, p->N, ws->particles, p->n_B)
	    = in.block(0, p->n_A, ws->particles, p->n_B);

	out.block(ws->eff_particles, 0, in.rows() - ws->particles, p->n_A)
	    = in.block(ws->particles, 0, in.rows() - ws->particles, p->n_A);
	
	out.block(ws->eff_particles, p->N, in.rows() - ws->particles, p->n_B)
	    = in.block( ws->particles, p->n_A, in.rows() - ws->particles, p->n_B);

    }
}

void cx_dqmc::greens_general_renyi::enlarge(int replica, cx_mat_t& in, cx_mat_t& out) {
    using namespace dqmc::la;
    out.setIdentity();

    if (replica == 0) {
	out.block(0, 0, p->N, p->N) = in;
    }
    else if (replica == 1) {
	out.block(0, 0, p->n_A, p->n_A)
	    = in.block(0, 0, p->n_A, p->n_A);
	out.block(p->N, 0, p->n_B, p->n_A)
	    = in.block(p->n_A, 0, p->n_B, p->n_A);
	out.block(0, p->N, p->n_A, p->n_B)
	    = in.block(0, p->n_A, p->n_A, p->n_B);
	out.block(p->N, p->N, p->n_B, p->n_B)
	    = in.block(p->n_A, p->n_A, p->n_B, p->n_B);	
    }
}
    
void cx_dqmc::greens_general_renyi::enlarge(int replica, vec_t& in, vec_t& out) {
    out.setOnes();
    if (replica == 0) {
	out.segment(0, p->N) = in;
    }
    else if (replica == 1) {
	out.segment(0, p->n_A) = in.segment(0, p->n_A);
	out.segment(p->N, p->n_B) = in.segment(p->n_A, p->n_B);
    }
}


void cx_dqmc::greens_general_renyi::calculate_greens_exact(int slice) {
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
	ws->mat_7.block(0, 0, ws->eff_particles, ws->eff_particles) = ws->tiny_mat_4;
	
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
	ws->mat_1.block(0, 0, ws->eff_particles, ws->eff_particles) = ws->tiny_mat_1;

	
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
	ws->mat_7.block(0, 0, ws->eff_particles, ws->eff_particles) = ws->tiny_mat_4;
	
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
	ws->mat_1.block(0, 0, ws->eff_particles, ws->eff_particles) = ws->tiny_mat_1;

	
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

void cx_dqmc::greens_general_renyi::update_remove_interaction() {
    prop_greens.setIdentity();
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens, ws->vec_1,
					    (*aux_spins)[current_slice / slices][current_slice], -1., 0);
}

void cx_dqmc::greens_general_renyi::update_add_interaction() {
    cx_dqmc::interaction::interaction_right(p, ws, prop_greens, ws->vec_1,
				(*aux_spins)[current_slice / slices][current_slice], 1., 0);
}
