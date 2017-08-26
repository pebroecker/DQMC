#ifndef CX_UPDATES_HPP
#define CX_UPDATES_HPP
#include "parameters.hpp"
#include "cx_workspace.hpp"
#include "cx_dqmc_abstract_greens.hpp"
#include "la.hpp"

#include <alps/alea.h>

#include <vector>


namespace cx_dqmc {
    namespace update {

	inline bool fixed_rate_updater(alps::Observable& alpha_obs, double alpha,	
				       double rate, double r, double random) {
	    using namespace std;
	    
	    bool result = false;
	    
	    if (r > 1) {
		if (random < r / (alpha + r)) {
		    result = true;
		}
	    	alpha = r/rate - r;
	    } else {
	    	if (random < r / (1. + alpha * r)) {
	    	    result = true;
	    	}
		alpha = 1/rate - 1/r;
	    }
	    // alpha = std::min(1., std::max(0., alpha));
	    alpha = std::min(1., std::max(0., alpha));
	    if (alpha != alpha) {
		// cout << "Alpha is NaN - rate: " << rate << " - ratio: " << r << endl;
		alpha_obs.reset();
		alpha_obs << 1.;
	    } else {
		alpha_obs << alpha;
	    }
	    return result;
	}


	//============================================================
	// onsite delayed flip routines
	//============================================================

	template <typename G>
	inline cx_double get_delayed_diag(G * greens,
					  int site, int& current_flip) {
	    using namespace std;
	    if (current_flip == 0) {
		return greens->greens(site, site);
	    }
	    else {
		cx_double corr = greens->X.row(site) * greens->Y.row(site).transpose();
		// cx_double g = greens->greens(site, site);		
		return greens->greens(site, site) + corr;// + greens->X.row(site) * greens->Y.row(site).transpose();;
	    }
	}


	template <typename G>
	inline cx_double get_delayed_ij(G * greens,
				     int i, int j,
				     int& current_flip) {
	    using namespace std;
	    if (current_flip == 0) {
		return greens->greens(i, j);
	    }
	    else {
		cx_double corr = greens->X.row(i) * greens->Y.row(j).transpose();
		return greens->greens(i, j)
		    + corr;
	    }
	}

	template <typename G>
	inline void update_delayed_greens(G * greens,
					  cx_double gamma,
					  int site, int& current_flip) {
	    using namespace std;
	    static vec x,y;
	    greens->X.col(current_flip) = greens->greens.col(site);
	    greens->Y.col(current_flip) = greens->greens.row(site).transpose();

	    if (current_flip != 0) {
		greens->X.col(current_flip).noalias() += greens->X.block(0, 0, greens->vol, current_flip) 
		    * greens->Y.row(site).segment(0, current_flip).transpose();
				
		greens->Y.col(current_flip).noalias() += greens->Y.block(0, 0, greens->vol, current_flip)
		    * greens->X.row(site).segment(0, current_flip).transpose();
	    }
	    greens->X(site, current_flip) -= cx_double(1., 0.);
	    greens->X.col(current_flip) *= gamma;
	}
	
	template <typename G>
	inline bool flush_delayed_greens(G * greens,
					 int& current_flip,
					 bool force = false) {
	    using namespace std;
	    
	    if (current_flip == 0)
		return false;
	    
	    if (current_flip == greens->delayed_buffer_size || force == true) {

		// cout << "Correction" << endl
		//      << greens->X.block(0, 0, greens->vol,
		// 			current_flip) *
		//     greens->Y.block(0, 0, greens->vol,
		// 		    current_flip).transpose()
		//      << endl << endl;
		
		greens->greens.noalias() +=
		    greens->X.block(0, 0, greens->vol,
				    current_flip) *
		    greens->Y.block(0, 0, greens->vol,
				    current_flip).transpose();
		greens->X.setZero();
		greens->Y.setZero();
		return true;
	    }
	    return false;
	}			

	
	template <typename G>
	inline double spinful_delayed_flip(alps::ObservableSet& obs, 
					   boost::multi_array<double, 3>& spins,
					   int state, int rep, int rep_slice,
					   G * greens,
					   std::vector<double>& random_numbers,
					   double& alpha, 
					   cx_double& avg_phase,
					   bool verbose = false,
					   bool is_thermalized = false) {
	    using namespace std;
	    
	    greens->X.setZero();
	    greens->Y.setZero();
	    
	    dqmc::parameters * p = (greens->p);
	    // cx_dqmc::workspace * ws = (greens->ws);

	    cx_double one(1., 0.);
	    cx_double gamma, gamma_table[2], ratio;	    
	    double prob;
	    int fs;
	    bool decision;

	    avg_phase = cx_double(0, 0);
	    
	    
	    gamma_table[0] = p->cx_osi_gamma;
	    gamma_table[1] = 1. / p->cx_osi_gamma;

	    int current_flush = 0;
	    int rand_counter = 0;
	    int flip_attempt = 0;
	    double new_alpha = 0;
	    int current_flip = 0;
	    int flips_accepted = 0;
	    
	    // run for a maximum of 5 times
	    for (; flip_attempt < std::max(200, greens->N); ++flip_attempt) {
		// cout << "Flipping in run " << run << endl;
		int s = std::floor(double(greens->N) * random_numbers[2 * flip_attempt]);
		fs = s;//int(double(p->N) * random_numbers[2 * s]) ;

		if (state == 1 && rep == 1) {		    
		    fs = (s < greens->n_A) ? s : greens->N + s - greens->n_A;
		}
			
			if (verbose == true) {
			    cout << "Selected " << s << " but really " << fs << endl;
			}
		    
		double& to_flip = spins[rep][rep_slice][s];

		if (to_flip == -1) {	
		    gamma =  gamma_table[0] - cx_double(1, 0);
		} else {
		    gamma = gamma_table[1] - cx_double(1, 0);
		}

		ratio = one
		    + (one - get_delayed_diag(greens, fs, current_flip))
		    * gamma;

		if (verbose) {
		    greens->update_remove_interaction();
		    to_flip *= -1;
		    greens->update_add_interaction();
		    to_flip *= -1;
		    cx_mat_t iden = greens->greens;
		    iden.setIdentity();
		    cout << (iden + (iden - greens->greens)
			     * (greens->prop_greens - iden)).fullPivLu().determinant() << " vs. " << ratio << endl;
		}
		
		prob = std::abs(ratio * ratio);

		if (std::isinf(prob)) {
		    dqmc::tools::abort("prob is inf");
		} else if (std::isnan(prob)) {
		    dqmc::tools::abort("prob is nan");
		}


		//============================================================
		// Decisions are never easy
		//============================================================
		// if (random_numbers[s] < prob) decision = true;
		// else decision = false;

		//============================================================
		
		if (prob > 1) {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (alpha + prob)) decision = true;
		    else decision = false;
		    new_alpha += prob/0.4 - prob;
		} else {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (1. + alpha * prob)) decision = true;
		    else decision = false;
		    new_alpha += 1/0.4 - 1/prob;
		}


		//============================================================

		// if (state == 0) {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 0" + p->obs_suffix],
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);
		// } else {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 1" + p->obs_suffix], 
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);		    
		// }

		// if (ws->renyi == true) {
		//     cout << "Probability\t" << prob << "\t" << decision << endl;
		// }

		if (decision == true) {
		    ++flips_accepted;

		    greens->det_sign *= ratio / std::abs(ratio);
		    
		    if (to_flip == -1) {
			greens->phase /= p->cx_osi_gamma;
		    } else {
			greens->phase *= p->cx_osi_gamma;
		    }

		    update_delayed_greens(greens, gamma/ratio,
					  fs, current_flip);
		    ++current_flip;
		    
		    to_flip *= -1;


		    bool force = (greens->sites < 200 && flip_attempt >= greens->sites);
		    
		    if (verbose == true) {
			force = true;
			cout << "breaking" << endl;
		    }

		    if ( flush_delayed_greens(greens, current_flip, force) || force == true) {
			++current_flush;
			current_flip = 0;

			if (verbose) break;
			
			if (flip_attempt >= greens->sites) {
			    // cout << "Breaking after " << flip_attempt << endl;
			    break;
			}
		    }
		}
		// cout << greens->det_sign * greens->det_sign << " vs " << greens->phase << endl;
		avg_phase += greens->det_sign * greens->det_sign * greens->phase;
	    }
	    
	    if (current_flip != 0) {
		flush_delayed_greens(greens, current_flip, true);
	    }

	    avg_phase /= flip_attempt;
	    if (!is_thermalized) {
		if (state == 0) obs["Alpha 0" + p->obs_suffix] << new_alpha / flip_attempt;
		else obs["Alpha 1" + p->obs_suffix] << new_alpha / flip_attempt;
	    }

	    return double(flips_accepted) / double(flip_attempt);
	}



	template <typename G, typename G_dummy>
	inline double spinful_delayed_continuous_flip(alps::ObservableSet& obs, 
						      boost::multi_array<double, 3>& spins,
						      int state, int rep, int rep_slice,
						      G * greens, G_dummy * dummy,
						      std::vector<double>& random_numbers,
						      double& alpha, 
						      cx_double& avg_phase,
						      std::vector<double>& prob_log_ratios,
						      std::vector<double>& weight_log_ratios,
						      std::vector<double>& dummy_log_ratios,
						      bool is_thermalized = false) {
	    
	    using namespace std;
	    
	    greens->X.setZero();
	    greens->Y.setZero();

	    dummy->X.setZero();
	    dummy->Y.setZero();
	    
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);

	    dqmc::parameters * dummy_p = (dummy->p);
	    cx_dqmc::workspace * dummy_ws = (dummy->ws);

	    cx_double one(1., 0.);
	    cx_double gamma, gamma_table[2], ratio, ratio_dummy;	    
	    double prob;
	    int fs, fs_dummy;
	    bool decision;

	    avg_phase = cx_double(0, 0);
	    
	    
	    gamma_table[0] = p->cx_osi_gamma;
	    gamma_table[1] = 1. / p->cx_osi_gamma;

	    int current_flush = 0;
	    int rand_counter = 0;
	    int flip_attempt = 0;
	    double new_alpha = 0;
	    int current_flip = 0;
	    int flips_accepted = 0;
	    
	    for (; flip_attempt < std::max(200, p->N); ++flip_attempt) {
		// cout << "Flipping in run " << run << endl;
		int s = std::floor(double(p->N) * random_numbers[2 * flip_attempt]);
		fs = s;//int(double(p->N) * random_numbers[2 * s]) ;
		fs_dummy = s;
		
		if ( state == 1 && rep == 1) {
		    fs = (s < p->n_A) ? s : p->N + s - p->n_A;
		}

		if ( state == 0 && rep == 1) {
		    fs_dummy = (s < p->n_A) ? s : p->N + s - p->n_A;
		}
		
		double& to_flip = spins[rep][rep_slice][s];

		if (to_flip == -1) {	
		    gamma =  gamma_table[0] - cx_double(1, 0);
		} else {
		    gamma = gamma_table[1] - cx_double(1, 0);
		}

		// change in determinant per replica is ratio r' / r
		// change in replica determinant ratio is r_1 / r_0
		ratio = one
		    + (one - get_delayed_diag(greens, fs, current_flip))
		    * gamma;

		ratio_dummy = one
		    + (one - get_delayed_diag(dummy, fs_dummy, current_flip))
		    * gamma;
		
		prob = std::abs(ratio * ratio);

		if (std::isinf(prob)) {
		    dqmc::tools::abort("prob is inf");
		} else if (std::isnan(prob)) {
		    dqmc::tools::abort("prob is nan");
		}


		//============================================================
		// Decisions are never easy
		//============================================================
		// if (random_numbers[s] < prob) decision = true;
		// else decision = false;

		//============================================================
		
		if (prob > 1) {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (alpha + prob)) decision = true;
		    else decision = false;
		    new_alpha += prob/0.4 - prob;
		} else {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (1. + alpha * prob)) decision = true;
		    else decision = false;
		    new_alpha += 1/0.4 - 1/prob;
		}

		if (decision == true) {
		    // cout << "Ratios " << ratio << " vs. " << ratio_dummy << endl;
		    prob_log_ratios.push_back(2 * log(std::abs(ratio)) - 2 * log(std::abs(ratio_dummy)));
		    weight_log_ratios.push_back(log(std::abs(ratio)));
		    dummy_log_ratios.push_back(log(std::abs(ratio_dummy)));
		} else {
		    prob_log_ratios.push_back(0.);
		    weight_log_ratios.push_back(0.);
		    dummy_log_ratios.push_back(0.);
		}
		    
		//============================================================

		// if (state == 0) {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 0" + p->obs_suffix],
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);
		// } else {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 1" + p->obs_suffix], 
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);		    
		// }

		// if (ws->renyi == true) {
		//     cout << "Probability\t" << prob << "\t" << decision << endl;
		// }

		if (decision == true) {
		    ++flips_accepted;

		    greens->det_sign *= ratio / std::abs(ratio);
		    
		    if (to_flip == -1) {
			greens->phase /= p->cx_osi_gamma;
		    } else {
			greens->phase *= p->cx_osi_gamma;
		    }

		    update_delayed_greens(greens, gamma/ratio,
					  fs, current_flip);
		    
		    update_delayed_greens(dummy, gamma/ratio_dummy,
					  fs_dummy, current_flip);

		    ++current_flip;
		    
		    to_flip *= -1;


		    bool force = (greens->sites < 200 && flip_attempt >= greens->sites);
		    // force = true;
		    if ( flush_delayed_greens(greens, current_flip, force) ) {
			flush_delayed_greens(dummy, current_flip, true);
			++current_flush;
			current_flip = 0;

			if (flip_attempt >= greens->sites) {
			    // cout << "Breaking after " << flip_attempt << endl;
			    break;
			}
		    }
		    // break;
		}
		// cout << greens->det_sign * greens->det_sign << " vs " << greens->phase << endl;
		avg_phase += greens->det_sign * greens->det_sign * greens->phase;
	    }
	    
	    if (current_flip != 0) {
		flush_delayed_greens(greens, current_flip, true);
		flush_delayed_greens(dummy, current_flip, true);
	    }

	    avg_phase /= flip_attempt;
	    if (!is_thermalized) {
		if (state == 0) obs["Alpha 0" + p->obs_suffix] << new_alpha / flip_attempt;
		else obs["Alpha 1" + p->obs_suffix] << new_alpha / flip_attempt;
	    }

	    return double(flips_accepted) / double(flip_attempt);
	}


	template <typename G>
	inline double spinless_delayed_flip(alps::ObservableSet& obs, 
					    boost::multi_array<double, 3>& spins,
					    int state, int rep, int rep_slice,
					    G * greens,
					    std::vector<double>& random_numbers,
					    double& alpha, 
					    cx_double& avg_phase) {
	    using namespace std;
		
	    greens->X.setZero();
	    greens->Y.setZero();
	    
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);

	    cx_mat_t mat(2, 2);
	    
	    cx_double one(1., 0.);
	    cx_double gamma, gamma_table[2], ratio;	    
	    double prob;
	    int fs;
	    bool decision;

	    avg_phase = cx_double(0, 0);
 	    
	    int current_flush = 0;
	    int rand_counter = 0;
	    int flip_attempt = 0;
	    double new_alpha = 0;
	    int current_flip = 0;
	    int flips_accepted = 0;
	    
	    int b, b_t, s1i, s1, s2i, s2;

	    cx_mat_t imat(2, 2);
	    cx_mat_t cmat(2, 2);
	    cx_double mat_det;
	    
	    // run for a maximum of 5 times
	    for (; flip_attempt < 5 * std::max(greens->delayed_buffer_size / 2, ws->num_update_bonds); ++flip_attempt) {
		// cout << "Flipping in run " << run << endl;
		b = std::floor(double(ws->num_update_bonds) * random_numbers[2 * flip_attempt]);
		b_t = ws->bond_types[b];
		s1i = ws->bonds[b][0];
		s2i = ws->bonds[b][1];
	
		gamma_table[1] = p->cx_nni_gammas[b_t];
		gamma_table[0] = 1. / p->cx_nni_gammas[b_t];
	    
		if ( state == 1 && rep == 1) {
		    s1 = (s1i < greens->n_A) ? s1i : greens->N + s1i - greens->n_A;
		    s2 = (s2i < greens->n_A) ? s2i : greens->N + s2i - greens->n_A;
		} else {
		    s1 = s1i;
		    s2 = s2i;
		}
		// cout << "Flipping sites " << setw(5) << s1 << setw(5) << s2 << endl;
		
		double& to_flip = spins[rep][rep_slice][b];

		if (to_flip == -1) {	
		    gamma =  gamma_table[0] - cx_double(1, 0);
		} else {
		    gamma = gamma_table[1] - cx_double(1, 0);
		}


		mat(0, 0) = one + (one - get_delayed_ij(greens, s1, s1, current_flip)) * gamma;
		mat(0, 1) = ( - get_delayed_ij(greens, s1, s2, current_flip)) * gamma;
		mat(1, 1) = one + (one - get_delayed_ij(greens, s2, s2, current_flip)) * gamma;
		mat(1, 0) = ( - get_delayed_ij(greens, s2, s1, current_flip)) * gamma;

		ratio = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		prob = std::abs(ratio);

		// greens->update_remove_interaction();
		// to_flip *= -1;
		// greens->update_add_interaction();
		// to_flip *= -1;

		// cx_mat_t iden = greens->greens;
		// iden.setIdentity();
		// cout << endl << (iden + (iden - greens->greens)
		// 		 * (greens->prop_greens - iden)).fullPivLu().determinant() << " vs. " << ratio<< endl << endl;
		// cx_mat_t corr = (iden - greens->greens)
		//     * (iden + (greens->prop_greens - iden) * (iden - greens->greens)).fullPivLu().inverse()
		//     * (greens->prop_greens - iden) * greens->greens;
		// return 0.0;

		// ratio = one
		//     + (one - get_delayed_diag(greens, fs, current_flip))
		//     * gamma;
		
		// prob = std::abs(ratio * ratio);

		if (std::isinf(prob)) {
		    dqmc::tools::abort("prob is inf");
		} else if (std::isnan(prob)) {
		    dqmc::tools::abort("prob is nan");
		}


		//============================================================
		// Decisions are never easy
		//============================================================
		// if (random_numbers[s] < prob) decision = true;
		// else decision = false;

		//============================================================
		
		if (prob > 1) {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (alpha + prob)) decision = true;
		    else decision = false;
		    new_alpha += prob/0.4 - prob;
		} else {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (1. + alpha * prob)) decision = true;
		    else decision = false;
		    new_alpha += 1/0.4 - 1/prob;
		}


		//============================================================

		// if (state == 0) {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 0" + p->obs_suffix],
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);
		// } else {
		//     decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 1" + p->obs_suffix], 
		// 						alpha, .4, prob, random_numbers[2 * s + 1]);		    
		// }

		// if (ws->renyi == true) {
		//     cout << "Probability\t" << prob << "\t" << decision << endl;
		// }

		if (decision == true) {
		    ++flips_accepted;

		    greens->det_sign *= ratio / std::abs(ratio);
		    
		    if (to_flip == -1) {
			greens->phase /= p->cx_nni_gammas[b_t];
		    } else {
			greens->phase *= p->cx_nni_gammas[b_t];
		    }

		    // mat(0, 0) = one + gamma * (one - get_delayed_ij(greens, s1, s1, current_flip));
		    // mat(0, 1) = gamma * ( - get_delayed_ij(greens, s1, s2, current_flip));
		    // mat(1, 1) = one + gamma * (one - get_delayed_ij(greens, s2, s2, current_flip));
		    // mat(1, 0) = gamma * ( - get_delayed_ij(greens, s2, s1, current_flip));
		    
		    mat_det = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		    imat(0, 0) = 1./mat_det * mat(1, 1);
		    imat(0, 1) = -1./mat_det * mat(0, 1);
		    imat(1, 0) = -1./mat_det * mat(1, 0);
		    imat(1, 1) = 1./mat_det * mat(0, 0);

		    greens->X_sls.col(0) = greens->greens.col(s1);
		    greens->X_sls.col(1) = greens->greens.col(s2);
		    
		    greens->Y.col(current_flip) = greens->greens.row(s1).transpose();
		    greens->Y.col(current_flip + 1) = greens->greens.row(s2).transpose();

		    if (current_flip != 0) {
			greens->X_sls.col(0).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s1).segment(0, current_flip).transpose(); ;

			greens->X_sls.col(1).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s2).segment(0, current_flip).transpose(); ;
			
			greens->Y.col(current_flip).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s1).segment(0, current_flip).transpose();

			greens->Y.col(current_flip + 1).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s2).segment(0, current_flip).transpose();

		    }
		    greens->X_sls(s1, 0) -= cx_double(1., 0.);
		    greens->X_sls(s2, 1) -= cx_double(1., 0.);
		    greens->X_sls *= gamma;
		    
		    // cout << greens->X_sls << endl << endl;
		    
		    greens->X.col(current_flip) = greens->X_sls.col(0) * imat(0, 0) + greens->X_sls.col(1) * imat(1, 0);
		    greens->X.col(current_flip + 1) = greens->X_sls.col(0) * imat(0, 1) + greens->X_sls.col(1) * imat(1, 1);
		    current_flip += 2;

		    // cout << greens->X.block(0, 0, greens->vol, current_flip) << endl << endl;
		    // cout << greens->Y.block(0, 0, greens->vol, current_flip) << endl << endl;

		    // cout << "correction" << endl << greens->X.block(0, 0, greens->vol, current_flip)
		    // 	* greens->Y.block(0, 0, greens->vol, current_flip).transpose() << endl;
		    // update_delayed_greens(greens, gamma/ratio,
		    // 			  s1, current_flip);
		    // ++current_flip;
		    // update_delayed_greens(greens, gamma/ratio,
		    // 			  s2, current_flip);
		    // ++current_flip;
		    
		    to_flip *= -1;

		    // cout << "Simple correction " << endl <<  - corr << endl << endl;
		    // greens->greens -= corr;
		    // return 0.0;
		    bool force = (ws->num_update_bonds < greens->delayed_buffer_size/2 && flip_attempt >= ws->num_update_bonds);
		    if (flush_delayed_greens(greens, current_flip, force) ) {
			++current_flush;
			current_flip = 0;

			if (flip_attempt >= ws->num_update_bonds) {
			    // cout << "Breaking after " << flip_attempt << endl;
			    break;
			}
		    }
		}
		// cout << greens->det_sign * greens->det_sign << " vs " << greens->phase << endl;
		avg_phase += greens->det_sign * greens->phase;
	    }
	    
	    if (current_flip != 0) {
		flush_delayed_greens(greens, current_flip, true);
	    }

	    avg_phase /= flip_attempt;
	    
	    if (state == 0) obs["Alpha 0" + p->obs_suffix] << new_alpha / flip_attempt;
	    else obs["Alpha 1" + p->obs_suffix] << new_alpha / flip_attempt;
	    
	    return double(flips_accepted) / double(flip_attempt);
	}


	template <typename G, typename G_dummy>
	inline double spinless_delayed_continuous_flip(alps::ObservableSet& obs, 
						       boost::multi_array<double, 3>& spins,
						       int state, int rep, int rep_slice,
						       G * greens, G_dummy * dummy,
						       std::vector<double>& random_numbers,
						       double& alpha, 
						       cx_double& avg_phase,
						       std::vector<double>& prob_log_ratios,
						       std::vector<double>& weight_log_ratios,
						       std::vector<double>& dummy_log_ratios) {
	    using namespace std;
	    
	    greens->X.setZero();
	    greens->Y.setZero();

	    dummy->X.setZero();
	    dummy->Y.setZero();
	    
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);
	    
	    cx_double one(1., 0.);
	    cx_double gamma, gamma_table[2], ratio, ratio_dummy;	    
	    double prob;
	    int fs, fs_dummy;
	    bool decision;

	    avg_phase = cx_double(0, 0);
 	    
	    gamma_table[1] = p->cx_nni_gamma;
	    gamma_table[0] = 1. / p->cx_nni_gamma;

	    int current_flush = 0;
	    int rand_counter = 0;
	    int flip_attempt = 0;
	    double new_alpha = 0;
	    int current_flip = 0;
	    int flips_accepted = 0;
	    
	    int b, s1i, s1, s2i, s2, s1_dummy, s2_dummy;

	    cx_mat_t mat(2, 2), mat_dummy(2, 2);
	    cx_mat_t imat(2, 2), imat_dummy(2, 2);
	    cx_mat_t cmat(2, 2), cmat_dummy(2, 2);
	    cx_double mat_det, mat_det_dummy;
	    
	    // run for a maximum of 5 times
	    for (; flip_attempt < 5 * std::max(greens->delayed_buffer_size / 2, ws->num_update_bonds); ++flip_attempt) {
		// cout << "Flipping in run " << run << endl;
		b = std::floor(double(ws->num_update_bonds) * random_numbers[2 * flip_attempt]);
		s1i = ws->bonds[b][0];
		s2i = ws->bonds[b][1];
		    
		if ( state == 1 && rep == 1) {
		    s1 = (s1i < p->n_A) ? s1i : p->N + s1i - p->n_A;
		    s2 = (s2i < p->n_A) ? s2i : p->N + s2i - p->n_A;
		} else {
		    s1 = s1i;
		    s2 = s2i;
		}

		if ( state == 0 && rep == 1) {
		    s1_dummy = (s1i < p->n_A) ? s1i : p->N + s1i - p->n_A;
		    s2_dummy = (s2i < p->n_A) ? s2i : p->N + s2i - p->n_A;
		} else {
		    s1_dummy = s1i;
		    s2_dummy = s2i;
		}

		// cout << "Flipping sites " << setw(5) << s1 << setw(5) << s2 << endl;
		
		double& to_flip = spins[rep][rep_slice][b];

		if (to_flip == -1) {	
		    gamma =  gamma_table[0] - cx_double(1, 0);
		} else {
		    gamma = gamma_table[1] - cx_double(1, 0);
		}


		mat(0, 0) = one + (one - get_delayed_ij(greens, s1, s1, current_flip)) * gamma;
		mat(0, 1) = ( - get_delayed_ij(greens, s1, s2, current_flip)) * gamma;
		mat(1, 1) = one + (one - get_delayed_ij(greens, s2, s2, current_flip)) * gamma;
		mat(1, 0) = ( - get_delayed_ij(greens, s2, s1, current_flip)) * gamma;

		ratio = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		prob = std::abs(ratio);


		mat_dummy(0, 0) = one + (one - get_delayed_ij(dummy, s1_dummy, s1_dummy, current_flip)) * gamma;
		mat_dummy(0, 1) = ( - get_delayed_ij(dummy, s1_dummy, s2_dummy, current_flip)) * gamma;
		mat_dummy(1, 1) = one + (one - get_delayed_ij(dummy, s2_dummy, s2_dummy, current_flip)) * gamma;
		mat_dummy(1, 0) = ( - get_delayed_ij(dummy, s2_dummy, s1_dummy, current_flip)) * gamma;

		ratio_dummy = mat_dummy(0,0) * mat_dummy(1,1) - mat_dummy(1,0) * mat_dummy(0, 1);


		if (std::isinf(prob)) {
		    dqmc::tools::abort("prob is inf");
		} else if (std::isnan(prob)) {
		    dqmc::tools::abort("prob is nan");
		}

		
		if (prob > 1) {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (alpha + prob)) decision = true;
		    else decision = false;
		    new_alpha += prob/0.4 - prob;
		} else {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (1. + alpha * prob)) decision = true;
		    else decision = false;
		    new_alpha += 1/0.4 - 1/prob;
		}

		if (decision == true) {
		    prob_log_ratios.push_back(log(std::abs(ratio)) - log(std::abs(ratio_dummy)));
		    weight_log_ratios.push_back(log(std::abs(ratio)));
		    dummy_log_ratios.push_back(log(std::abs(ratio_dummy)));
		} else {
		    prob_log_ratios.push_back(0.);
		    weight_log_ratios.push_back(0.);
		    dummy_log_ratios.push_back(0.);
		}

		if (decision == true) {
		    ++flips_accepted;

		    greens->det_sign *= ratio / std::abs(ratio);
		    
		    if (to_flip == -1) {
			greens->phase /= p->cx_osi_gamma;
		    } else {
			greens->phase *= p->cx_osi_gamma;
		    }
		    
		    mat_det = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		    imat(0, 0) = 1./mat_det * mat(1, 1);
		    imat(0, 1) = -1./mat_det * mat(0, 1);
		    imat(1, 0) = -1./mat_det * mat(1, 0);
		    imat(1, 1) = 1./mat_det * mat(0, 0);

		    greens->X_sls.col(0) = greens->greens.col(s1);
		    greens->X_sls.col(1) = greens->greens.col(s2);
		    
		    greens->Y.col(current_flip) = greens->greens.row(s1).transpose();
		    greens->Y.col(current_flip + 1) = greens->greens.row(s2).transpose();

		    if (current_flip != 0) {
			greens->X_sls.col(0).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s1).segment(0, current_flip).transpose(); ;

			greens->X_sls.col(1).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s2).segment(0, current_flip).transpose(); ;
			
			greens->Y.col(current_flip).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s1).segment(0, current_flip).transpose();

			greens->Y.col(current_flip + 1).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s2).segment(0, current_flip).transpose();

		    }
		    greens->X_sls(s1, 0) -= cx_double(1., 0.);
		    greens->X_sls(s2, 1) -= cx_double(1., 0.);
		    greens->X_sls *= gamma;
		    
		    greens->X.col(current_flip) = greens->X_sls.col(0) * imat(0, 0) + greens->X_sls.col(1) * imat(1, 0);
		    greens->X.col(current_flip + 1) = greens->X_sls.col(0) * imat(0, 1) + greens->X_sls.col(1) * imat(1, 1);

		    // and dummy

		    mat_det = mat_dummy(0,0) * mat_dummy(1,1) - mat_dummy(1,0) * mat_dummy(0, 1);
		    imat(0, 0) = 1./mat_det * mat_dummy(1, 1);
		    imat(0, 1) = -1./mat_det * mat_dummy(0, 1);
		    imat(1, 0) = -1./mat_det * mat_dummy(1, 0);
		    imat(1, 1) = 1./mat_det * mat_dummy(0, 0);

		    dummy->X_sls.col(0) = dummy->greens.col(s1_dummy);
		    dummy->X_sls.col(1) = dummy->greens.col(s2_dummy);
		    
		    dummy->Y.col(current_flip) = dummy->greens.row(s1_dummy).transpose();
		    dummy->Y.col(current_flip + 1) = dummy->greens.row(s2_dummy).transpose();
		    
		    if (current_flip != 0) {
			dummy->X_sls.col(0).noalias() +=
			    dummy->X.block(0, 0, dummy->vol, current_flip)
			    * dummy->Y.row(s1_dummy).segment(0, current_flip).transpose(); ;

			dummy->X_sls.col(1).noalias() +=
			    dummy->X.block(0, 0, dummy->vol, current_flip)
			    * dummy->Y.row(s2_dummy).segment(0, current_flip).transpose(); ;
			
			dummy->Y.col(current_flip).noalias() +=
			    dummy->Y.block(0, 0, dummy->vol, current_flip)
			    * dummy->X.row(s1_dummy).segment(0, current_flip).transpose();

			dummy->Y.col(current_flip + 1).noalias() +=
			    dummy->Y.block(0, 0, dummy->vol, current_flip)
			    * dummy->X.row(s2_dummy).segment(0, current_flip).transpose();

		    }
		    dummy->X_sls(s1_dummy, 0) -= cx_double(1., 0.);
		    dummy->X_sls(s2_dummy, 1) -= cx_double(1., 0.);
		    dummy->X_sls *= gamma;
		    
		    dummy->X.col(current_flip) = dummy->X_sls.col(0) * imat(0, 0) + dummy->X_sls.col(1) * imat(1, 0);
		    dummy->X.col(current_flip + 1) = dummy->X_sls.col(0) * imat(0, 1) + dummy->X_sls.col(1) * imat(1, 1);

		    current_flip += 2;
		    to_flip *= -1;

		    bool force = (ws->num_update_bonds < greens->delayed_buffer_size/2 && flip_attempt >= ws->num_update_bonds);
		    if (flush_delayed_greens(greens, current_flip, force) ) {
			flush_delayed_greens(dummy, current_flip, true);
			++current_flush;
			current_flip = 0;

			if (flip_attempt >= ws->num_update_bonds) {
			    break;
			}
		    }
		}
		avg_phase += greens->det_sign * greens->phase;
	    }
	    
	    if (current_flip != 0) {
		flush_delayed_greens(greens, current_flip, true);
		flush_delayed_greens(dummy, current_flip, true);
	    }

	    avg_phase /= flip_attempt;
	    
	    if (state == 0) obs["Alpha 0" + p->obs_suffix] << new_alpha / flip_attempt;
	    else obs["Alpha 1" + p->obs_suffix] << new_alpha / flip_attempt;
	    
	    return double(flips_accepted) / double(flip_attempt);
	}



	template <typename G>
	inline int spinful_naive_flip(alps::ObservableSet& obs, 
					boost::multi_array<double, 3>& spins,
					int state, int rep, int rep_slice,
					G * greens,
					std::vector<double>& random_numbers,
					double& alpha, cx_double& sign,
					cx_double& avg_sign,
					int& flips_accepted) {
	    // using namespace std;
	    
	    // greens->X.setZero();
	    // greens->Y.setZero();
	    
	    // dqmc::parameters * p = (greens->p);
	    // cx_dqmc::workspace * ws = (greens->ws);

	    // cx_mat_t eye = cx_mat_t::Identity(greens->prop_greens.rows(), greens->prop_greens.cols());
	    // cx_double one(1., 0.);
	    // cx_double gamma, gamma_table[2], ratio;	    
	    // double prob;
	    // int fs;
	    // bool decision;
	    // flips_accepted = 0;
	    // int current_flip = 0;
	    
	    // gamma_table[0] = p->cx_osi_gamma;
	    // gamma_table[1] = 1. / p->cx_osi_gamma;

	    // for (int s = 0; s < p->N; ++s) {
	    // 	fs = s;
	    // 	if ( state == 1 && rep == 1) {
	    // 	    fs = (s < p->n_A) ? s : p->N + s -  p->n_A;
	    // 	}
	    // 	double& to_flip = spins[rep][rep_slice][s];

	    // 	greens->update_remove_interaction();
	    // 	to_flip *= -1;
	    // 	greens->update_add_interaction();
	    // 	to_flip *= -1;

	    // 	cout << endl <<  greens->prop_greens - eye << endl << endl;
		
	    // 	if (to_flip == -1) {	
	    // 	    gamma =  gamma_table[0] - cx_double(1, 0);
	    // 	} else {
	    // 	    gamma = gamma_table[1] - cx_double(1, 0);
	    // 	}

	    // 	cout << endl << "vs" << endl << gamma << endl;
		
	    // 	ratio = one + (one - get_delayed_diag(greens, fs, current_flip))
	    // 	    * gamma;

	    // 	greens->X_sls.col(0) = (eye - greens->greens).col(fs);
	    // 	greens->Y_sls.col(0).setZero();
	    // 	greens->Y_sls(fs, 0) = (greens->prop_greens - eye)(fs, fs);
		
	    // 	cout << "Naive ratio is" << endl
	    // 	     << eye + (eye - greens->greens) * (greens->prop_greens - eye) << endl << endl;
	    // 	cout << "Another naive ratio is" << endl
	    // 	     << eye + greens->X_sls.col(0) * greens->Y_sls.col(0).transpose() << endl << endl;
	    // 	cout << "ratio is " << ratio << endl << endl;
	    // 	prob = std::abs(ratio * ratio);
		

	    // 	cx_mat correction = greens->X_sls.col(0)
	    // 	    * (greens->Y_sls.col(0).transpose() * greens->greens) / ratio;
	    // 	cout << "Correction is" << endl << correction << endl << endl;
		
	    // 	if (state == 0) {
	    // 	    decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 0" + p->obs_suffix],
	    // 							alpha, .4, prob, random_numbers[s]);
	    // 	} else {
	    // 	    decision = cx_dqmc::update::fixed_rate_updater(obs["Alpha 1" + p->obs_suffix], 
	    // 							alpha, .4, prob, random_numbers[s]);		    
	    // 	}
	    // }
	    // // 	if (decision == true) {
	    // // 	    ++flips_accepted;

	    // // 	    update_delayed_greens(greens, gamma/ratio,
	    // // 				  fs, current_flip);
	    // // 	    ++current_flip;
		    
	    // // 	    to_flip *= -1;

	    // // 	    if (flush_delayed_greens(greens, current_flip)) {
	    // // 		current_flip = 0;
	    // // 	    }
	    // // 	}
	    // // }
	    
	    // // if (current_flip != 0) {
	    // // 	flush_delayed_greens(greens, current_flip, true);
	    // // }

	    // return flips_accepted;
	}


	template <typename G>
	inline int spinless_simple_flip(alps::ObservableSet& obs, 
					boost::multi_array<double, 3>& spins,
					int state, int rep, int rep_slice,
					G& greens,
					std::vector<double>& random_numbers,
					double& alpha, cx_double& avg_phase) {
	    using namespace std;
	    using namespace dqmc::la;

	    // cout << "==================================================" << endl;
		
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);
	    int flips_accepted = 0;
	    

	    cx_mat iden = greens->greens;
	    iden.setIdentity();
		
	    cx_double one(1, 0);
	    static cx_mat_t mat(2, 2);    
	    static cx_double prob;
	    static cx_double gamma_table[2], gamma;
	    
	    static alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    static int b, s1, s2;
    
	    gamma_table[1] = p->cx_nni_gamma;
	    gamma_table[0] = 1./p->cx_nni_gamma;

	    for (boost::tie(itr1, itr1_end) = p->graph.bonds(); itr1 != itr1_end; ++itr1) {
		// greens->print_diag();
		
		b = p->graph.index(*itr1);
		s1 = p->graph.source(*itr1);
		s2 = p->graph.target(*itr1);

		if ( state == 1 && rep == 1) {
		    s1 = (s1 < p->n_A) ? s1 : p->N + s1 -  p->n_A;
		    s2 = (s2 < p->n_A) ? s2 : p->N + s2 -  p->n_A;
		}
         
		double& to_flip = spins[rep][rep_slice][b];

		//==================================================
		// setup
		//==================================================
	
		if (to_flip == -1) {	
		    gamma = gamma_table[0] - one;
		} else {
		    gamma = gamma_table[1] - one;
		}

		// greens->update_remove_interaction();
		// to_flip *= -1;
		// greens->update_add_interaction();
		// to_flip *= -1;

		// if (state == 1)
		//     cout << endl <<
		// 	(iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().determinant() << endl << endl;

		mat(0, 0) = one + (one - greens->greens(s1, s1)) * gamma;
		mat(0, 1) = ( - greens->greens(s1, s2)) * gamma;
		mat(1, 1) = one + (one - greens->greens(s2, s2)) * gamma;
		mat(1, 0) = ( - greens->greens(s2, s1)) * gamma;

		// if (state == 1)
		//     cout << "prob mat new" << endl << mat.fullPivLu().determinant() << endl << endl;


		prob = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);		
		
		bool decision = random_numbers[b] < std::abs(prob);

		if (decision == true) {
		    ++flips_accepted;
		    to_flip *= -1;
			    
		    static cx_mat_t imat(2, 2);
		    static cx_mat_t cmat(2, 2);
		    static cx_double mat_det;
		    
		    greens->X_sls.col(0).noalias() = -one * greens->greens.col(s1);
		    greens->X_sls.col(1).noalias() = -one * greens->greens.col(s2);
		    greens->X_sls(s1, 0) += one;
		    greens->X_sls(s2, 1) += one;

		    mat(0, 0) = one + gamma * (one - greens->greens(s1, s1));
		    mat(0, 1) = gamma * ( - greens->greens(s1, s2));
		    mat(1, 1) = one + gamma * (one - greens->greens(s2, s2));
		    mat(1, 0) = gamma * ( - greens->greens(s2, s1));
		    
		    mat_det = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		    imat(0, 0) = 1./mat_det * mat(1, 1);
		    imat(0, 1) = -1./mat_det * mat(0, 1);
		    imat(1, 0) = -1./mat_det * mat(1, 0);
		    imat(1, 1) = 1./mat_det * mat(0, 0);

		    greens->Y_sls.setZero();
		    greens->Y_sls.col(0) = (gamma * imat(0, 0) * greens->greens.row(s1)
					    + gamma * imat(0, 1) * greens->greens.row(s2)).transpose();
		    greens->Y_sls.col(1) = (gamma * imat(1, 0) * greens->greens.row(s1)
					    + gamma * imat(1, 1) * greens->greens.row(s2)).transpose();

		    // cx_mat corr = (iden - (iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().inverse()) * greens->greens;
		    // corr = (iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().inverse();
		    // greens->greens = corr * greens->greens;
		    // cout << "Correction exact " << endl << corr << endl << endl;
		    // cout << "Correction to greens " << endl << greens->X_sls * greens->Y_sls.transpose() << endl;
		    greens->greens.noalias() -= greens->X_sls * greens->Y_sls.transpose();
		}
	    }
	    return double(flips_accepted) / p->num_update_bonds;
	}	    



	template <typename G>
	inline int spinless_attractive_flip(alps::ObservableSet& obs, 
					    boost::multi_array<double, 3>& spins,
					    int state, int rep, int rep_slice,
					    G& greens,
					    std::vector<double>& random_numbers,
					    double& alpha, cx_double& avg_phase) {
	    using namespace std;
	    using namespace dqmc::la;

	    // cout << "==================================================" << endl;
		
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);
	    int flips_accepted = 0;
	    

	    cx_mat iden = greens->greens;
	    iden.setIdentity();
		
	    cx_double one(1, 0);
	    static cx_mat_t mat(2, 2);    
	    static cx_double prob;
	    static cx_double gamma_table[2], gamma_vec[2];
	    
	    static alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    static int b, s1, s2;
    
	    gamma_table[1] = p->cx_nni_gamma;
	    gamma_table[0] = 1./p->cx_nni_gamma;

	    for (boost::tie(itr1, itr1_end) = p->graph.bonds(); itr1 != itr1_end; ++itr1) {
		// greens->print_diag();
		
		b = p->graph.index(*itr1);
		s1 = p->graph.source(*itr1);
		s2 = p->graph.target(*itr1);

		if ( state == 1 && rep == 1) {
		    s1 = (s1 < greens->n_A) ? s1 : greens->N + s1 -  greens->n_A;
		    s2 = (s2 < greens->n_A) ? s2 : greens->N + s2 -  greens->n_A;
		}
         
		double& to_flip = spins[rep][rep_slice][b];

		//==================================================
		// setup
		//==================================================
	
		if (to_flip == -1) {	
		    gamma_vec[0] = gamma_table[0] - one;
		    gamma_vec[1] = gamma_table[1] - one;
		} else {
		    gamma_vec[0] = gamma_table[1] - one;
		    gamma_vec[1] = gamma_table[0] - one;
		}
		// greens->update_remove_interaction();
		// to_flip *= -1;
		// greens->update_add_interaction();
		// to_flip *= -1;

		// if (state == 1)
		//     cout << endl <<
		// 	(iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().determinant() << endl << endl;

		mat(0, 0) = one + (one - greens->greens(s1, s1)) * gamma_vec[0];
		mat(0, 1) = ( - greens->greens(s1, s2)) * gamma_vec[1];
		mat(1, 1) = one + (one - greens->greens(s2, s2)) * gamma_vec[1];
		mat(1, 0) = ( - greens->greens(s2, s1)) * gamma_vec[0];

		// if (state == 1)
		//     cout << "prob mat new" << endl << mat.fullPivLu().determinant() << endl << endl;


		prob = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);		
		
		bool decision = random_numbers[b] < std::abs(prob);
		
		if (decision == true) {
		    ++flips_accepted;
		    to_flip *= -1;
			    
		    static cx_mat_t imat(2, 2);
		    static cx_mat_t cmat(2, 2);
		    static cx_double mat_det;
		    
		    greens->X_sls.col(0).noalias() = -one * greens->greens.col(s1);
		    greens->X_sls.col(1).noalias() = -one * greens->greens.col(s2);
		    greens->X_sls(s1, 0) += one;
		    greens->X_sls(s2, 1) += one;

		    mat(0, 0) = one + gamma_vec[0] * (one - greens->greens(s1, s1));
		    mat(0, 1) = gamma_vec[0] * ( - greens->greens(s1, s2));
		    mat(1, 1) = one + gamma_vec[1] * (one - greens->greens(s2, s2));
		    mat(1, 0) = gamma_vec[1] * ( - greens->greens(s2, s1));
		    
		    mat_det = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		    imat(0, 0) = 1./mat_det * mat(1, 1);
		    imat(0, 1) = -1./mat_det * mat(0, 1);
		    imat(1, 0) = -1./mat_det * mat(1, 0);
		    imat(1, 1) = 1./mat_det * mat(0, 0);

		    greens->Y_sls.setZero();
		    greens->Y_sls.col(0) = (gamma_vec[0] * imat(0, 0) * greens->greens.row(s1)
					    + gamma_vec[1] * imat(0, 1) * greens->greens.row(s2)).transpose();
		    greens->Y_sls.col(1) = (gamma_vec[0] * imat(1, 0) * greens->greens.row(s1)
					    + gamma_vec[1] * imat(1, 1) * greens->greens.row(s2)).transpose();

		    // cx_mat corr = (iden - (iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().inverse()) * greens->greens;
		    // corr = (iden + (iden - greens->greens) * (greens->prop_greens - iden)).fullPivLu().inverse();
		    // greens->greens = corr * greens->greens;
		    // cout << "Correction exact " << endl << corr << endl << endl;
		    // cout << "Correction to greens " << endl << greens->X_sls * greens->Y_sls.transpose() << endl;
		    greens->greens.noalias() -= greens->X_sls * greens->Y_sls.transpose();
		}
	    }
	    return double(flips_accepted) / p->num_update_bonds;
	}	    


	template <typename G>
	inline double spinless_attractive_delayed_flip(alps::ObservableSet& obs, 
						       boost::multi_array<double, 3>& spins,
						       int state, int rep, int rep_slice,
						       G * greens,
						       std::vector<double>& random_numbers,
						       double& alpha, 
						       cx_double& avg_phase) {
	    using namespace std;
		
	    greens->X.setZero();
	    greens->Y.setZero();
	    
	    dqmc::parameters * p = (greens->p);
	    cx_dqmc::workspace * ws = (greens->ws);

	    cx_mat_t mat(2, 2);
	    
	    cx_double one(1., 0.);
	    cx_double gamma_vec[2], gamma_table[2], ratio;	    
	    double prob;
	    int fs;
	    bool decision;

	    avg_phase = cx_double(0, 0);
 	    
	    gamma_table[1] = p->cx_nni_gamma;
	    gamma_table[0] = 1. / p->cx_nni_gamma;

	    int current_flush = 0;
	    int rand_counter = 0;
	    int flip_attempt = 0;
	    double new_alpha = 0;
	    int current_flip = 0;
	    int flips_accepted = 0;
	    
	    int b, s1i, s1, s2i, s2;

	    cx_mat_t imat(2, 2);
	    cx_mat_t cmat(2, 2);
	    cx_double mat_det;
	    
	    // run for a maximum of 5 times
	    for (; flip_attempt < 5 * std::max(greens->delayed_buffer_size / 2, ws->num_update_bonds); ++flip_attempt) {
		// cout << "Flipping in run " << run << endl;
		b = std::floor(double(ws->num_update_bonds) * random_numbers[2 * flip_attempt]);
		s1i = ws->bonds[b][0];
		s2i = ws->bonds[b][1];
		    
		if ( state == 1 && rep == 1) {
		    s1 = (s1i < greens->n_A) ? s1i : greens->N + s1i - greens->n_A;
		    s2 = (s2i < greens->n_A) ? s2i : greens->N + s2i - greens->n_A;
		} else {
		    s1 = s1i;
		    s2 = s2i;
		}
		// cout << "Flipping sites " << setw(5) << s1 << setw(5) << s2 << endl;
		
		double& to_flip = spins[rep][rep_slice][b];


		if (to_flip == -1) {	
		    gamma_vec[0] = gamma_table[0] - one;
		    gamma_vec[1] = gamma_table[1] - one;
		} else {
		    gamma_vec[0] = gamma_table[1] - one;
		    gamma_vec[1] = gamma_table[0] - one;
		}

		mat(0, 0) = one + (one - get_delayed_ij(greens, s1, s1, current_flip)) * gamma_vec[0];
		mat(0, 1) = ( - get_delayed_ij(greens, s1, s2, current_flip)) * gamma_vec[1];
		mat(1, 1) = one + (one - get_delayed_ij(greens, s2, s2, current_flip)) * gamma_vec[1];
		mat(1, 0) = ( - get_delayed_ij(greens, s2, s1, current_flip)) * gamma_vec[0];

		ratio = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		prob = std::abs(ratio);

		if (std::isinf(prob)) {
		    dqmc::tools::abort("prob is inf");
		} else if (std::isnan(prob)) {
		    dqmc::tools::abort("prob is nan");
		}


		//============================================================
		// Decisions are never easy
		//============================================================
		// if (random_numbers[s] < prob) decision = true;
		// else decision = false;

		//============================================================
		
		if (prob > 1) {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (alpha + prob)) decision = true;
		    else decision = false;
		    new_alpha += prob/0.4 - prob;
		} else {
		    if (random_numbers[2 * flip_attempt + 1]
			< prob / (1. + alpha * prob)) decision = true;
		    else decision = false;
		    new_alpha += 1/0.4 - 1/prob;
		}


		if (decision == true) {
		    ++flips_accepted;

		    greens->det_sign *= ratio / std::abs(ratio);
		    
		    if (to_flip == -1) {
			greens->phase /= p->cx_osi_gamma;
		    } else {
			greens->phase *= p->cx_osi_gamma;
		    }
		    
		    mat_det = mat(0,0) * mat(1,1) - mat(1,0) * mat(0, 1);
		    imat(0, 0) = 1./mat_det * mat(1, 1);
		    imat(0, 1) = -1./mat_det * mat(0, 1);
		    imat(1, 0) = -1./mat_det * mat(1, 0);
		    imat(1, 1) = 1./mat_det * mat(0, 0);

		    greens->X_sls.col(0) = greens->greens.col(s1);
		    greens->X_sls.col(1) = greens->greens.col(s2);
		    
		    greens->Y.col(current_flip) = greens->greens.row(s1).transpose();
		    greens->Y.col(current_flip + 1) = greens->greens.row(s2).transpose();

		    if (current_flip != 0) {
			greens->X_sls.col(0).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s1).segment(0, current_flip).transpose(); ;

			greens->X_sls.col(1).noalias() +=
			    greens->X.block(0, 0, greens->vol, current_flip)
			    * greens->Y.row(s2).segment(0, current_flip).transpose(); ;
			
			greens->Y.col(current_flip).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s1).segment(0, current_flip).transpose();

			greens->Y.col(current_flip + 1).noalias() +=
			    greens->Y.block(0, 0, greens->vol, current_flip)
			    * greens->X.row(s2).segment(0, current_flip).transpose();

		    }
		    greens->X_sls(s1, 0) -= cx_double(1., 0.);
		    greens->X_sls(s2, 1) -= cx_double(1., 0.);

		    greens->X_sls.col(0) *= gamma_vec[0];
		    greens->X_sls.col(1) *= gamma_vec[1];
		    
		    // cout << greens->X_sls << endl << endl;
		    
		    greens->X.col(current_flip) = greens->X_sls.col(0) * imat(0, 0) + greens->X_sls.col(1) * imat(1, 0);
		    greens->X.col(current_flip + 1) = greens->X_sls.col(0) * imat(0, 1) + greens->X_sls.col(1) * imat(1, 1);
		    current_flip += 2;

		    // cout << greens->X.block(0, 0, greens->vol, current_flip) << endl << endl;
		    // cout << greens->Y.block(0, 0, greens->vol, current_flip) << endl << endl;

		    // cout << "correction" << endl << greens->X.block(0, 0, greens->vol, current_flip)
		    // 	* greens->Y.block(0, 0, greens->vol, current_flip).transpose() << endl;
		    // update_delayed_greens(greens, gamma/ratio,
		    // 			  s1, current_flip);
		    // ++current_flip;
		    // update_delayed_greens(greens, gamma/ratio,
		    // 			  s2, current_flip);
		    // ++current_flip;
		    
		    to_flip *= -1;

		    // cout << "Simple correction " << endl <<  - corr << endl << endl;
		    // greens->greens -= corr;
		    // return 0.0;
		    bool force = (ws->num_update_bonds < greens->delayed_buffer_size/2 && flip_attempt >= ws->num_update_bonds);
		    if (flush_delayed_greens(greens, current_flip, force) ) {
			++current_flush;
			current_flip = 0;

			if (flip_attempt >= ws->num_update_bonds) {
			    // cout << "Breaking after " << flip_attempt << endl;
			    break;
			}
		    }
		}
		// cout << greens->det_sign * greens->det_sign << " vs " << greens->phase << endl;
		avg_phase += greens->det_sign * greens->phase;
	    }
	    
	    if (current_flip != 0) {
		flush_delayed_greens(greens, current_flip, true);
	    }

	    avg_phase /= flip_attempt;
	    
	    if (state == 0) obs["Alpha 0" + p->obs_suffix] << new_alpha / flip_attempt;
	    else obs["Alpha 1" + p->obs_suffix] << new_alpha / flip_attempt;
	    
	    return double(flips_accepted) / double(flip_attempt);
	}

    }
}
#endif
