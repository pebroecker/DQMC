#ifndef CX_INTERACTIONS_HPP
#define CX_INTERACTIONS_HPP

#include "parameters.hpp"
#include "cx_workspace.hpp"
#include "la.hpp"

#include <boost/multi_array.hpp>

#include <cmath>
#include <iomanip>

typedef boost::multi_array<double, 1> spin_t;

namespace cx_dqmc {
    namespace interaction {
	
	inline void onsite_left(dqmc::parameters * p, cx_mat_t& M, 
				cx_vec_t& int_vec,
				const spin_t& spins,
				double inv, int replica) {
	    using namespace std;
	    if (M.rows() == p->N) replica = 0;

	    int_vec.setOnes();	    

	    for (int i = 0; i < p->N; ++i) {
		for (int j = 0; j < M.cols(); ++j) {
		    if (i < p->n_A) {
			int_vec(i) = std::exp(p->mu_factor * inv)
			    * std::exp(p->cx_osi_lambda * spins[i] * inv);
		    } else {
			int_vec(i + replica * p->n_B) =
			    std::exp(p->mu_factor * inv)
			    * std::exp(p->cx_osi_lambda * spins[i] * inv);
		    }
		}
	    }
	    M = int_vec.asDiagonal() * M;
	}

	inline void onsite_right(dqmc::parameters * p, cx_mat_t& M,
				 cx_vec_t& int_vec,
				 const spin_t& spins,
				 double inv, int replica) {
	    if (M.cols() == p->N) replica = 0;
	    int_vec.setOnes();
	    
	    for (int i = 0; i < p->N; ++i) {
		for (int j = 0; j < M.rows(); ++j) {
		    if (i < p->n_A) {
			int_vec(i) =
			    std::exp(p->mu_factor * inv)
			    * std::exp(p->cx_osi_lambda * spins[i] * inv);

		    } else {
			int_vec(i + replica * p->n_B) =
			    std::exp(p->mu_factor * inv)
			    * std::exp(p->cx_osi_lambda * spins[i] * inv);
		    }
		}
	    }
	    M = M * int_vec.asDiagonal();
	}



	inline void onsite_alternative_left(dqmc::parameters * p, cx_mat_t& M,
					    cx_vec_t& int_vec,
					    const spin_t& spins,
					    double inv, int replica) {
	    using namespace std;
	    if (M.rows() == p->N) replica = 0;

	    int_vec.setOnes();

	    // cout << "Onsite left\t" 
	    // 	 << p->cx_interaction_vals[0] << " "
	    // 	 << p->cx_interaction_vals[1] << " "
	    // 	 << p->cx_interaction_vals[2] << " " << endl;

	    for (int i = 0; i < p->n_A; ++i) {
		for (int j = 0; j < M.cols(); ++j) {
		    int_vec(i) = 
			p->mu_site_factor[int(inv + 1)]
			* p->cx_interaction_vals[int(-1. * spins[i] * inv + 1)];
		    // * std::exp(p->cx_osi_lambda * spins[i] * inv);
		}
	    }

	    for (int i = p->n_A; i < p->N; ++i) {
		for (int j = 0; j < M.cols(); ++j) {
		    int_vec(i + replica * p->n_B) =
			p->mu_site_factor[int(inv + 1)]
			* p->cx_interaction_vals[int(-1. * spins[i] * inv + 1)];
		    // * std::exp(p->cx_osi_lambda * spins[i] * inv);
		}
	    }
	    M = int_vec.asDiagonal() * M;
	}

	inline void onsite_alternative_right(dqmc::parameters * p, cx_mat& M,
					     cx_vec_t& int_vec,
					     const spin_t& spins,
					     double inv, int replica) {
	    using namespace std;
	    if (M.cols() == p->N) replica = 0;
	    int_vec.setOnes();
	    
	    // cout << "Onsite right\t" 
	    // 	 << p->cx_interaction_vals[0] << " "
	    // 	 << p->cx_interaction_vals[1] << " "
	    // 	 << p->cx_interaction_vals[2] << " " << endl;

	    for (int i = 0; i < p->n_A; ++i) {
		for (int j = 0; j < M.rows(); ++j) {
		    int_vec(i) =
			p->mu_site_factor[int(inv + 1)]
			* p->cx_interaction_vals[int(-1 * spins[i] * inv + 1)];
		    // * std::exp(p->cx_osi_lambda * spins[i] * inv);
		}
	    }

	    for (int i = p->n_A; i < p->N; ++i) {
		for (int j = 0; j < M.rows(); ++j) {
		    int_vec(i + replica * p->n_B) =
			p->mu_site_factor[int(inv + 1)]
			* p->cx_interaction_vals[int(-1. * spins[i] * inv + 1)];
		    // * std::exp(p->cx_osi_lambda * spins[i] * inv);
		}
	    }
	    M = M * int_vec.asDiagonal();
	}

	//============================================================
	// alternative next-nearest neighbor interaction for spinless
	// fermions
	//============================================================

	inline void nn_left_alternative(dqmc::parameters * p, cx_dqmc::workspace * ws,
					cx_mat_t& M, const spin_t& spins,
					double inv, int replica) {
	    using namespace std;
	    if (M.rows() == p->N) replica = 0;
	    
	    dqmc::la::ones(ws->la_vec_1);
	    for (int s = 0; s < p->N; ++s) {
		if (s < p->n_A) {
		    ws->la_vec_1(s) = p->mu_site_factor[int(inv + 1)];
		} else {
		    ws->la_vec_1(s + replica * p->n_B) = p->mu_site_factor[int(inv + 1)];
		}
		
	    }
	    
	    for (int s = 0; s < p->N; ++s) {
		// cout << "Site "  << s << endl;
		for (int b = 0; b < 40; ++b) {
		    // cout << "Bond: " << b << endl;
		    if (ws->site_signs[s][b] == 0) break;
		    if (s < p->n_A) {
			ws->la_vec_1(s) *=
			    p->cx_bond_vals[ws->bond_types[ ws->site_bonds[s][b] ] ][ws->site_signs[s][b] *
											 spins[ws->site_bonds[s][b]] * inv + 1];
			// * p->mu_bond_factor[int(inv + 1)];			    
		    } else {
			ws->la_vec_1(s + replica * p->n_B) *=
			    p->cx_bond_vals[ws->bond_types[ws->site_bonds[s][b]]][ws->site_signs[s][b] *
						   spins[ws->site_bonds[s][b]] * inv + 1];
			    // * p->mu_bond_factor[int(inv + 1)];
		    }
	    	}
	    }
	    M = ws->la_vec_1.asDiagonal() * M;
	}

	

	inline void nn_right_alternative(dqmc::parameters * p, cx_dqmc::workspace * ws,
					 cx_mat_t& M, const spin_t& spins,
					 double inv, int replica) {
	    // std::cout << "NNNNNN" << std::endl;
	    if (M.cols() == p->N) replica = 0;
	    
	    dqmc::la::ones(ws->la_vec_1);
	    for (int s = 0; s < p->N; ++s) {
		if (s < p->n_A) {
		    ws->la_vec_1(s) = p->mu_site_factor[int(inv + 1)];
		} else {
		    ws->la_vec_1(s + replica * p->n_B) = p->mu_site_factor[int(inv + 1)];
		}
		
	    }
	    for (int s = 0; s < p->N; ++s) {
		for (int b = 0; b < 40; ++b) {
		    if (ws->site_signs[s][b] == 0) break;
		    if (s < p->n_A) {
			ws->la_vec_1(s) *=
			    p->cx_bond_vals[ws->bond_types[ws->site_bonds[s][b]]][ws->site_signs[s][b] *
										  spins[ws->site_bonds[s][b]] * inv + 1];
			    // * p->mu_bond_factor[int(inv + 1)];			    
		    } else {
			ws->la_vec_1(s + replica * p->n_B) *=
			    p->cx_bond_vals[ws->bond_types[ws->site_bonds[s][b]]][ws->site_signs[s][b] *
										  spins[ws->site_bonds[s][b]] * inv + 1];
			    // * p->mu_bond_factor[int(inv + 1)];
		    }
	    	}
	    }

	    M = M * ws->la_vec_1.asDiagonal();
	    return;
	}


	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////
	// Next-nearest neighbor interaction for spinless fermions
	////////////////////////////////////////////////////////////// 
 	inline void nn_left(dqmc::parameters * p, cx_dqmc::workspace * ws,
			    cx_mat_t& M, const spin_t& spins,
			    double inv, int replica) {
	    dqmc::tools::abort ("nn left is too simple");
	    if (M.rows() == p->N) replica = 0;

	    dqmc::la::ones(ws->la_vec_1);
	    for (int s = 0; s < p->N; ++s) {
		for (int b = 0; b < 4; ++b) {
		    if (ws->site_signs[s][b] == 0) break;
		    if (s < p->n_A) {
			ws->la_vec_1(s) *=
			    p->cx_interaction_vals[ws->site_signs[s][b] *
						   int(spins[ws->site_bonds[s][b]] * inv) + 1]
			    * cx_double(p->mu_vals[int(inv + 1)], 0);
		    } else {
			ws->la_vec_1(s + replica * p->n_B) *=
			    p->cx_interaction_vals[ws->site_signs[s][b] *
						   int(spins[ws->site_bonds[s][b]] * inv) + 1]
			    * p->mu_vals[int(inv + 1)];
		    }			
	    	}
	    }
	    // M = (ws->la_vec_1.asDiagonal()) * M;
	    M = (ws->la_vec_1.asDiagonal() * p->mu_site_factor[int(inv + 1)]) * M;
	    return;
	}

	
	
	inline void nn_right(dqmc::parameters * p, cx_dqmc::workspace * ws,
			     cx_mat_t& M, const spin_t& spins,
			     double inv, int replica) {
	    dqmc::tools::abort ("nn right is too simple");
	    // std::cout << "NNNNNN" << std::endl;
	    if (M.cols() == p->N) replica = 0;
	    
	    dqmc::la::ones(ws->la_vec_1);
	    for (int s = 0; s < p->N; ++s) {
		for (int b = 0; b < 4; ++b) {
		    if (ws->site_signs[s][b] == 0) break;
		    if (s < p->n_A) {
			ws->la_vec_1(s) *=
			    p->cx_interaction_vals[ws->site_signs[s][b] *
						   spins[ws->site_bonds[s][b]] * inv + 1]
			* p->mu_vals[int(inv + 1)];
		    } else {
			ws->la_vec_1(s + replica * p->n_B) *=
			    p->cx_interaction_vals[ws->site_signs[s][b] *
						   spins[ws->site_bonds[s][b]] * inv + 1]
			    * p->mu_vals[int(inv + 1)];
		    }
	    	}
	    }
	    //M = M * ws->la_vec_1.asDiagonal();
	    M = M * (ws->la_vec_1.asDiagonal() * p->mu_site_factor[int(inv + 1)]);
	}	
	 


	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////


	inline void interaction_left(dqmc::parameters * p, cx_dqmc::workspace * ws,
				     cx_mat_t& M, cx_vec_t& int_vec,
				     const spin_t& spins,
				     double inv, int replica) {
	    if (p->model_id == p->CX_SPINFUL_HUBBARD) 
		cx_dqmc::interaction::onsite_left(p, M, int_vec,
						  spins, inv, replica);
	    else if (p->model_id == p->CX_SPINFUL_HUBBARD_ALTERNATIVE) 
		cx_dqmc::interaction::onsite_alternative_left(p, M, int_vec,
							      spins, inv, replica);
	    else if (p->model_id == p->CX_SPINLESS_HUBBARD) 
		cx_dqmc::interaction::nn_left(p, ws, M,
					      spins, inv, replica);
	    else if (p->model_id == p->CX_SPINLESS_HUBBARD_ALTERNATIVE) 
		cx_dqmc::interaction::nn_left_alternative(p, ws, M,
							  spins, inv, replica);

	}

	inline void interaction_right(dqmc::parameters * p, cx_dqmc::workspace * ws,
				      cx_mat_t& M, cx_vec_t& int_vec,
				      const spin_t& spins,
				      double inv, int replica) {
	    if (p->model_id == p->CX_SPINFUL_HUBBARD) 
		cx_dqmc::interaction::onsite_right(p, M, int_vec,
						   spins, inv, replica);
	    else if (p->model_id == p->CX_SPINFUL_HUBBARD_ALTERNATIVE) 
		cx_dqmc::interaction::onsite_alternative_right(p, M, int_vec,
							       spins, inv, replica);
	    else if (p->model_id == p->CX_SPINLESS_HUBBARD) 
		cx_dqmc::interaction::nn_right(p, ws, M,
						   spins, inv, replica);
	    else if (p->model_id == p->CX_SPINLESS_HUBBARD_ALTERNATIVE) 
		cx_dqmc::interaction::nn_right_alternative(p, ws, M,
							   spins, inv, replica);

	}
    }
}
#endif
