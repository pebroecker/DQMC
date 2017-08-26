#ifndef CX_DQMC_CHECKERBOARD_HPP
#define CX_DQMC_CHECKERBOARD_HPP

#include "la.hpp"
#include "parameters.hpp"

#include <boost/multi_array.hpp>

#include <exception>
#include <cmath>
#include <memory>

namespace cx_dqmc {
    namespace checkerboard {

	inline void graph_to_checkerboard(dqmc::parameters& p, cx_dqmc::workspace& ws) {
	    for (int i = 0; i < p.graph.num_bonds(); ++i) {
		ws.bond_used[i] = -1;
	    }
	    
	    int bonds_used = 0;
	    int sites_used = 0;
	    int first_unused_bond = 0;

	    alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    int b, s1, s2;
	    double t, im_t;
	    while (bonds_used < p.graph.num_bonds()) {
		sp_mat hopping(p.N, p.N);
		sp_mat hopping_inv(p.N, p.N);

		cx_sp_mat cx_hopping(p.N, p.N);
		cx_sp_mat cx_hopping_inv(p.N, p.N);

		int elements = 0;
		    
		for (int i = 0; i < p.N; ++i) { ws.site_used[i] = -1; }
		
		for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end; ++itr1) {	   
		    b = p.graph.index(*itr1);
		    s1 = p.graph.source(*itr1);
		    s2 = p.graph.target(*itr1);
		    
		    if (ws.bond_used[b] == 1) continue;
		    if (ws.site_used[s1] == 1 || ws.site_used[s2] == 1) continue;
		    
		    ws.bond_used[b] = 1;
		    ws.site_used[s1] = 1;
		    ws.site_used[s2] = 1;
		    t = p.ts[p.graph.bond_type(*itr1)];

		    if (p.complex_hoppings == false) {
			hopping.insert(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping.insert(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping.insert(s1, s2) = sinh(t * p.delta_tau/2.);
			hopping.insert(s2, s1) = sinh(t * p.delta_tau/2.);
		    
			hopping_inv.insert(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping_inv.insert(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping_inv.insert(s1, s2) = -sinh(t * p.delta_tau/2.);
			hopping_inv.insert(s2, s1) = -sinh(t * p.delta_tau/2.);
		    } else {
			im_t = p.im_ts[p.graph.bond_type(*itr1)];			
			std::complex<double> cx_t(t, im_t);
			std::complex<double> abs_t(sqrt(t*t + im_t*im_t), 0);
			cx_hopping.insert(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping.insert(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping.insert(s1, s2) = sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t;
			cx_hopping.insert(s2, s1) = sinh(abs_t * p.delta_tau/2.) * std::conj(cx_t / abs_t);
		    
			cx_hopping_inv.insert(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_inv.insert(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_inv.insert(s1, s2) = -(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
			cx_hopping_inv.insert(s2, s1) = -sinh(abs_t * p.delta_tau/2.) * std::conj(cx_t / abs_t);
		    }
		    // std::cout << "Using " << s1 << " and " << s2 << std::endl;
		    ++bonds_used;
		    ++elements;
		}

		if (elements != 0) {
		    for (int i = 0; i < p.N; ++i) {
			if(ws.site_used[i] == -1) {
			    if (p.complex_hoppings == false) {
				hopping.insert(i, i) = 1.;
				hopping_inv.insert(i, i) = 1.;
			    } else {
				cx_hopping.insert(i, i) = std::complex<double>(1., 0.);
				cx_hopping_inv.insert(i, i) = std::complex<double>(1., 0.);
			    }				
			}
		    }
		    // arma::mat(hopping).print("A hopping matrix");

		    if (p.complex_hoppings == false) {
			ws.sparse_hoppings.push_back(hopping);
			ws.sparse_hoppings_inv.push_back(hopping_inv);
		    } else {
			ws.cx_sparse_hoppings.push_back(cx_hopping);
			ws.cx_sparse_hoppings_inv.push_back(cx_hopping_inv);
		    }
		}
	    }	    
	}

	inline void graph_to_checkerboard_renyi(dqmc::parameters& p, cx_dqmc::workspace& ws) {
	    for (int i = 0; i < p.graph.num_bonds(); ++i) { ws.bond_used[i] = -1; }
	    
	    int bonds_used = 0;
	    int sites_used = 0;
	    int first_unused_bond = 0;

	    alps::graph_helper<>::bond_iterator itr1, itr1_end;
	    int b, s1, s2, s1a, s2a;
	    double t;
	    std::complex<double> one(1., 0.);
	    
	    while (bonds_used < p.graph.num_bonds()) {
		sp_mat hopping_0(ws.vol, ws.vol), hopping_0_inv(ws.vol, ws.vol),
		    hopping_1(ws.vol, ws.vol), hopping_1_inv(ws.vol, ws.vol),
		    hopping_2(ws.vol, ws.vol), hopping_2_inv(ws.vol, ws.vol),
		    hopping_3(ws.vol, ws.vol), hopping_3_inv(ws.vol, ws.vol);

		cx_sp_mat cx_hopping_0(ws.vol, ws.vol), cx_hopping_0_inv(ws.vol, ws.vol),
		    cx_hopping_1(ws.vol, ws.vol), cx_hopping_1_inv(ws.vol, ws.vol),
		    cx_hopping_2(ws.vol, ws.vol), cx_hopping_2_inv(ws.vol, ws.vol),
		    cx_hopping_3(ws.vol, ws.vol), cx_hopping_3_inv(ws.vol, ws.vol);
		
		int elements = 0;
		
		for (int i = 0; i < p.N; ++i) ws.site_used[i] = -1;
		for (int i = 0; i < p.N + p.n_B; ++i) {
		    if (p.complex_hoppings == false) {
			hopping_0.insert(i, i) = 1.;
			hopping_0_inv.insert(i, i) = 1.;
			hopping_1.insert(i, i) = 1.;
			hopping_1_inv.insert(i, i) = 1.;			
			hopping_2.insert(i, i) = 1.;
			hopping_2_inv.insert(i, i) = 1.;
			hopping_3.insert(i, i) = 1.;
			hopping_3_inv.insert(i, i) = 1.;
		    } else {
			cx_hopping_0.insert(i, i) = one;
			cx_hopping_0_inv.insert(i, i) = one;
			cx_hopping_1.insert(i, i) = one;
			cx_hopping_1_inv.insert(i, i) = one;
			cx_hopping_2.insert(i, i) = one;
			cx_hopping_2_inv.insert(i, i) = one;
			cx_hopping_3.insert(i, i) = one;
			cx_hopping_3_inv.insert(i, i) = one;
		    }
		}
		
		for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end; ++itr1) {	   
		    b = p.graph.index(*itr1);
		    s1 = p.graph.source(*itr1);
		    s2 = p.graph.target(*itr1);
		    s1a = s1;
		    s2a = s2;
		    
		    if (ws.bond_used[b] == 1) continue;
		    if (ws.site_used[s1] == 1 || ws.site_used[s2] == 1) continue;
		    
		    ws.bond_used[b] = 1;
		    ws.site_used[s1] = 1;
		    ws.site_used[s2] = 1;

		    t = p.ts[p.graph.bond_type(*itr1)];	
		    
		    if (s1 >= p.n_A) s1a = s1 + p.n_B;
		    if (s2 >= p.n_A) s2a = s2 + p.n_B;

		    if (p.complex_hoppings == false) {
			hopping_0.coeffRef(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping_0.coeffRef(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping_0.coeffRef(s1, s2) = sinh(t * p.delta_tau/2.);
			hopping_0.coeffRef(s2, s1) = sinh(t * p.delta_tau/2.);
		    
			hopping_0_inv.coeffRef(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping_0_inv.coeffRef(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping_0_inv.coeffRef(s1, s2) = -sinh(t * p.delta_tau/2.);
			hopping_0_inv.coeffRef(s2, s1) = -sinh(t * p.delta_tau/2.);
		    
			hopping_1.coeffRef(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping_1.coeffRef(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping_1.coeffRef(s1, s2) = sinh(t * p.delta_tau/2.);
			hopping_1.coeffRef(s2, s1) = sinh(t * p.delta_tau/2.);
		    
			hopping_1_inv.coeffRef(s1, s1) = cosh(t * p.delta_tau/2.);
			hopping_1_inv.coeffRef(s2, s2) = cosh(t * p.delta_tau/2.);
			hopping_1_inv.coeffRef(s1, s2) = -sinh(t * p.delta_tau/2.);
			hopping_1_inv.coeffRef(s2, s1) = -sinh(t * p.delta_tau/2.);
			
			hopping_2.coeffRef(s1a, s1a) = cosh(t * p.delta_tau/2.);
			hopping_2.coeffRef(s2a, s2a) = cosh(t * p.delta_tau/2.);
			hopping_2.coeffRef(s1a, s2a) = sinh(t * p.delta_tau/2.);
			hopping_2.coeffRef(s2a, s1a) = sinh(t * p.delta_tau/2.);

			hopping_2_inv.coeffRef(s1a, s1a) = cosh(t * p.delta_tau/2.);
			hopping_2_inv.coeffRef(s2a, s2a) = cosh(t * p.delta_tau/2.);
			hopping_2_inv.coeffRef(s1a, s2a) = -sinh(t * p.delta_tau/2.);
			hopping_2_inv.coeffRef(s2a, s1a) = -sinh(t * p.delta_tau/2.);

			hopping_3.coeffRef(s1a, s1a) = cosh(t * p.delta_tau/2.);
			hopping_3.coeffRef(s2a, s2a) = cosh(t * p.delta_tau/2.);
			hopping_3.coeffRef(s1a, s2a) = sinh(t * p.delta_tau/2.);
			hopping_3.coeffRef(s2a, s1a) = sinh(t * p.delta_tau/2.);

			hopping_3_inv.coeffRef(s1a, s1a) = cosh(t * p.delta_tau/2.);
			hopping_3_inv.coeffRef(s2a, s2a) = cosh(t * p.delta_tau/2.);
			hopping_3_inv.coeffRef(s1a, s2a) = -sinh(t * p.delta_tau/2.);
			hopping_3_inv.coeffRef(s2a, s1a) = -sinh(t * p.delta_tau/2.);
		    } else {
			double im_t = p.im_ts[p.graph.bond_type(*itr1)];
			std::complex<double> cx_t(t, im_t);
			std::complex<double> abs_t(sqrt(t*t + im_t*im_t), 0);
			cx_hopping_0.coeffRef(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_0.coeffRef(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_0.coeffRef(s1, s2) = sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t;
			cx_hopping_0.coeffRef(s2, s1) = sinh(abs_t * p.delta_tau/2.) * std::conj(cx_t / abs_t);
		    
			cx_hopping_0_inv.coeffRef(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_0_inv.coeffRef(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_0_inv.coeffRef(s1, s2) = -(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t ) ;
			cx_hopping_0_inv.coeffRef(s2, s1) = -sinh(abs_t * p.delta_tau/2.) * std::conj(cx_t / abs_t );
		    
			cx_hopping_1.coeffRef(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_1.coeffRef(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_1.coeffRef(s1, s2) = sinh(abs_t * p.delta_tau/2.)  * cx_t / abs_t;
			cx_hopping_1.coeffRef(s2, s1) = std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
		    
			cx_hopping_1_inv.coeffRef(s1, s1) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_1_inv.coeffRef(s2, s2) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_1_inv.coeffRef(s1, s2) = -(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
			cx_hopping_1_inv.coeffRef(s2, s1) = -std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
			
			cx_hopping_2.coeffRef(s1a, s1a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_2.coeffRef(s2a, s2a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_2.coeffRef(s1a, s2a) = sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t;
			cx_hopping_2.coeffRef(s2a, s1a) = std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);

			cx_hopping_2_inv.coeffRef(s1a, s1a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_2_inv.coeffRef(s2a, s2a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_2_inv.coeffRef(s1a, s2a) = -(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
			cx_hopping_2_inv.coeffRef(s2a, s1a) = -std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);

			cx_hopping_3.coeffRef(s1a, s1a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_3.coeffRef(s2a, s2a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_3.coeffRef(s1a, s2a) = sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t;
			cx_hopping_3.coeffRef(s2a, s1a) = std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);

			cx_hopping_3_inv.coeffRef(s1a, s1a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_3_inv.coeffRef(s2a, s2a) = cosh(abs_t * p.delta_tau/2.);
			cx_hopping_3_inv.coeffRef(s1a, s2a) = -(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
			cx_hopping_3_inv.coeffRef(s2a, s1a) = -std::conj(sinh(abs_t * p.delta_tau/2.) * cx_t / abs_t);
		    }
		    ++bonds_used;
		    ++elements;
		}

		if (elements != 0) {
		    if (p.complex_hoppings == false) {
			ws.sparse_hoppings_0.push_back(hopping_0);
			ws.sparse_hoppings_0_inv.push_back(hopping_0_inv);

			ws.sparse_hoppings_1.push_back(hopping_1);
			ws.sparse_hoppings_1_inv.push_back(hopping_1_inv);

			ws.sparse_hoppings_2.push_back(hopping_2);
			ws.sparse_hoppings_2_inv.push_back(hopping_2_inv);
			
			ws.sparse_hoppings_3.push_back(hopping_3);
			ws.sparse_hoppings_3_inv.push_back(hopping_3_inv);
		    } else {
			ws.cx_sparse_hoppings_0.push_back(cx_hopping_0);
			ws.cx_sparse_hoppings_0_inv.push_back(cx_hopping_0_inv);

			ws.cx_sparse_hoppings_1.push_back(cx_hopping_1);
			ws.cx_sparse_hoppings_1_inv.push_back(cx_hopping_1_inv);

			ws.cx_sparse_hoppings_2.push_back(cx_hopping_2);
			ws.cx_sparse_hoppings_2_inv.push_back(cx_hopping_2_inv);
			
			ws.cx_sparse_hoppings_3.push_back(cx_hopping_3);
			ws.cx_sparse_hoppings_3_inv.push_back(cx_hopping_3_inv);
		    }
		}		    
	    }	    
	}

	
	inline void hop_left(cx_dqmc::workspace * ws, cx_mat_t& M, double pref) {
	    using namespace std;
	    int par = 0;
	    
	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings[i] * M;
			else 
			    M = ws->sparse_hoppings[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings[i] * M;
			else 
			    M = ws->sparse_hoppings[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings_inv[i] * M;
			else 
			    M = ws->sparse_hoppings_inv[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings_inv[i] * M;
			else 
			    M = ws->sparse_hoppings_inv[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }
	    else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings[i] * M;
			else 
			    M = ws->cx_sparse_hoppings[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings[i] * M;
			else 
			    M = ws->cx_sparse_hoppings[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings_inv[i] * M;
			else 
			    M = ws->cx_sparse_hoppings_inv[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings_inv[i] * M;
			else 
			    M = ws->cx_sparse_hoppings_inv[i] * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }

	    
	    if (par == 1)
		M = ws->hop_temp;	    
	}


	inline void hop_left_t(cx_dqmc::workspace * ws, cx_mat_t& M, double pref) {
	    using namespace std;
	    int par = 0;

	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings[i].transpose() * M;
			else 
			    M = ws->sparse_hoppings[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings[i].transpose() * M;
			else 
			    M = ws->sparse_hoppings[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings_inv[i].transpose() * M;
			else 
			    M = ws->sparse_hoppings_inv[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->sparse_hoppings_inv[i].transpose() * M;
			else 
			    M = ws->sparse_hoppings_inv[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
	    } else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings[i].transpose() * M;
			else 
			    M = ws->cx_sparse_hoppings[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings[i].transpose() * M;
			else 
			    M = ws->cx_sparse_hoppings[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings_inv[i].transpose() * M;
			else 
			    M = ws->cx_sparse_hoppings_inv[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = ws->cx_sparse_hoppings_inv[i].transpose() * M;
			else 
			    M = ws->cx_sparse_hoppings_inv[i].transpose() * ws->hop_temp;
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }

	    
	    if (par == 1)
		M = ws->hop_temp;	    
	}


	inline void hop_right(cx_dqmc::workspace * ws, cx_mat_t& M, double pref) {
	    int par = 0;

	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->sparse_hoppings[i];
			else
			    M = ws->hop_temp * ws->sparse_hoppings[i];
			par = (par == 0) ? 1 : 0;		    
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->sparse_hoppings[i];
			else
			    M = ws->hop_temp * ws->sparse_hoppings[i];
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->sparse_hoppings_inv[i];
			else
			    M = ws->hop_temp * ws->sparse_hoppings_inv[i];
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->sparse_hoppings_inv[i];
			else
			    M = ws->hop_temp * ws->sparse_hoppings_inv[i];
			par = (par == 0) ? 1 : 0;
		    }
		}
	    } else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->cx_sparse_hoppings[i];
			else
			    M = ws->hop_temp * ws->cx_sparse_hoppings[i];
			par = (par == 0) ? 1 : 0;		    
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->cx_sparse_hoppings[i];
			else
			    M = ws->hop_temp * ws->cx_sparse_hoppings[i];
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings.size(); ++i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->cx_sparse_hoppings_inv[i];
			else
			    M = ws->hop_temp * ws->cx_sparse_hoppings_inv[i];
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings.size() - 1; i >=0; --i) {
			if (par == 0) 
			    ws->hop_temp = M * ws->cx_sparse_hoppings_inv[i];
			else
			    M = ws->hop_temp * ws->cx_sparse_hoppings_inv[i];
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }
	    if (par == 1)
		M = ws->hop_temp;	    
	}


	inline void hop_left_renyi(cx_dqmc::workspace * ws,
				   cx_mat_t& M,
				   double pref, 
				   int section) {
	    using namespace std;
	    int par = 0;
	    // cout << "section " << section << endl;

	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0[i] * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1[i] * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2[i] * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3[i] * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0[i] * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1[i] * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2[i] * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0[i] * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1[i] * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2[i] * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3[i] * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0[i] * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1[i] * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2[i] * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0_inv[i] * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1_inv[i] * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2_inv[i] * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3_inv[i] * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0_inv[i] * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1_inv[i] * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2_inv[i] * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3_inv[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0_inv[i] * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1_inv[i] * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2_inv[i] * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3_inv[i] * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0_inv[i] * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1_inv[i] * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2_inv[i] * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3_inv[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    } else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0[i] * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1[i] * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2[i] * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3[i] * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0[i] * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1[i] * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2[i] * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0[i] * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1[i] * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2[i] * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3[i] * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0[i] * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1[i] * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2[i] * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0_inv[i] * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1_inv[i] * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2_inv[i] * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3_inv[i] * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0_inv[i] * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1_inv[i] * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2_inv[i] * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3_inv[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0_inv[i] * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1_inv[i] * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2_inv[i] * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3_inv[i] * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0_inv[i] * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1_inv[i] * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2_inv[i] * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3_inv[i] * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }
	    if (par == 1)
		M = ws->hop_temp;
	}



	inline void hop_left_renyi_t(cx_dqmc::workspace * ws,
				   cx_mat_t& M,
				   double pref, 
				   int section) {
	    using namespace std;
	    int par = 0;
	    // cout << "section " << section << endl;

	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0_inv[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1_inv[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2_inv[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3_inv[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0_inv[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1_inv[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2_inv[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3_inv[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->sparse_hoppings_0_inv[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->sparse_hoppings_1_inv[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->sparse_hoppings_2_inv[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->sparse_hoppings_3_inv[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->sparse_hoppings_0_inv[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->sparse_hoppings_1_inv[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->sparse_hoppings_2_inv[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->sparse_hoppings_3_inv[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    } else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0_inv[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1_inv[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2_inv[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3_inv[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0_inv[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1_inv[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2_inv[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3_inv[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }

		    // cx_dqmc::interaction::onsite_left_fake(p, ws, M, spin, pref, section, slice);
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = ws->cx_sparse_hoppings_0_inv[i].transpose() * M;
			    if (section == 1) ws->hop_temp = ws->cx_sparse_hoppings_1_inv[i].transpose() * M;
			    if (section == 2) ws->hop_temp = ws->cx_sparse_hoppings_2_inv[i].transpose() * M;
			    if (section == 3) ws->hop_temp = ws->cx_sparse_hoppings_3_inv[i].transpose() * M;
			} else {
			    if (section == 0) M = ws->cx_sparse_hoppings_0_inv[i].transpose() * ws->hop_temp ;
			    if (section == 1) M = ws->cx_sparse_hoppings_1_inv[i].transpose() * ws->hop_temp ;
			    if (section == 2) M = ws->cx_sparse_hoppings_2_inv[i].transpose() * ws->hop_temp ;
			    if (section == 3) M = ws->cx_sparse_hoppings_3_inv[i].transpose() * ws->hop_temp ;
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }
	    if (par == 1)
		M = ws->hop_temp;
	}


	inline void hop_right_renyi(cx_dqmc::workspace * ws,
				    cx_mat_t&__restrict__ M,
				    double pref, int section) {
	    int par = 0;

	    if (ws->complex_hoppings == false) {
		if (pref > 0) {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->sparse_hoppings_0[i];
			    if (section == 1) ws->hop_temp = M * ws->sparse_hoppings_1[i];
			    if (section == 2) ws->hop_temp = M * ws->sparse_hoppings_2[i];
			    if (section == 3) ws->hop_temp = M * ws->sparse_hoppings_3[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->sparse_hoppings_0[i];
			    if (section == 1) M = ws->hop_temp * ws->sparse_hoppings_1[i];
			    if (section == 2) M = ws->hop_temp * ws->sparse_hoppings_2[i];
			    if (section == 3) M = ws->hop_temp * ws->sparse_hoppings_3[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->sparse_hoppings_0[i];
			    if (section == 1) ws->hop_temp = M * ws->sparse_hoppings_1[i];
			    if (section == 2) ws->hop_temp = M * ws->sparse_hoppings_2[i];
			    if (section == 3) ws->hop_temp = M * ws->sparse_hoppings_3[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->sparse_hoppings_0[i];
			    if (section == 1) M = ws->hop_temp * ws->sparse_hoppings_1[i];
			    if (section == 2) M = ws->hop_temp * ws->sparse_hoppings_2[i];
			    if (section == 3) M = ws->hop_temp * ws->sparse_hoppings_3[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->sparse_hoppings_0_inv[i];
			    if (section == 1) ws->hop_temp = M * ws->sparse_hoppings_1_inv[i];
			    if (section == 2) ws->hop_temp = M * ws->sparse_hoppings_2_inv[i];
			    if (section == 3) ws->hop_temp = M * ws->sparse_hoppings_3_inv[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->sparse_hoppings_0_inv[i];
			    if (section == 1) M = ws->hop_temp * ws->sparse_hoppings_1_inv[i];
			    if (section == 2) M = ws->hop_temp * ws->sparse_hoppings_2_inv[i];
			    if (section == 3) M = ws->hop_temp * ws->sparse_hoppings_3_inv[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->sparse_hoppings_0_inv[i];
			    if (section == 1) ws->hop_temp = M * ws->sparse_hoppings_1_inv[i];
			    if (section == 2) ws->hop_temp = M * ws->sparse_hoppings_2_inv[i];
			    if (section == 3) ws->hop_temp = M * ws->sparse_hoppings_3_inv[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->sparse_hoppings_0_inv[i];
			    if (section == 1) M = ws->hop_temp * ws->sparse_hoppings_1_inv[i];
			    if (section == 2) M = ws->hop_temp * ws->sparse_hoppings_2_inv[i];
			    if (section == 3) M = ws->hop_temp * ws->sparse_hoppings_3_inv[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    } 
	    else {
		if (pref > 0) {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->cx_sparse_hoppings_0[i];
			    if (section == 1) ws->hop_temp = M * ws->cx_sparse_hoppings_1[i];
			    if (section == 2) ws->hop_temp = M * ws->cx_sparse_hoppings_2[i];
			    if (section == 3) ws->hop_temp = M * ws->cx_sparse_hoppings_3[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->cx_sparse_hoppings_0[i];
			    if (section == 1) M = ws->hop_temp * ws->cx_sparse_hoppings_1[i];
			    if (section == 2) M = ws->hop_temp * ws->cx_sparse_hoppings_2[i];
			    if (section == 3) M = ws->hop_temp * ws->cx_sparse_hoppings_3[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->cx_sparse_hoppings_0[i];
			    if (section == 1) ws->hop_temp = M * ws->cx_sparse_hoppings_1[i];
			    if (section == 2) ws->hop_temp = M * ws->cx_sparse_hoppings_2[i];
			    if (section == 3) ws->hop_temp = M * ws->cx_sparse_hoppings_3[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->cx_sparse_hoppings_0[i];
			    if (section == 1) M = ws->hop_temp * ws->cx_sparse_hoppings_1[i];
			    if (section == 2) M = ws->hop_temp * ws->cx_sparse_hoppings_2[i];
			    if (section == 3) M = ws->hop_temp * ws->cx_sparse_hoppings_3[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
		else {
		    for (int i = 0; i < ws->cx_sparse_hoppings_0.size(); ++i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->cx_sparse_hoppings_0_inv[i];
			    if (section == 1) ws->hop_temp = M * ws->cx_sparse_hoppings_1_inv[i];
			    if (section == 2) ws->hop_temp = M * ws->cx_sparse_hoppings_2_inv[i];
			    if (section == 3) ws->hop_temp = M * ws->cx_sparse_hoppings_3_inv[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->cx_sparse_hoppings_0_inv[i];
			    if (section == 1) M = ws->hop_temp * ws->cx_sparse_hoppings_1_inv[i];
			    if (section == 2) M = ws->hop_temp * ws->cx_sparse_hoppings_2_inv[i];
			    if (section == 3) M = ws->hop_temp * ws->cx_sparse_hoppings_3_inv[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		
		    for (int i = ws->cx_sparse_hoppings_0.size() - 1; i >=0; --i) {
			if (par == 0) {
			    if (section == 0) ws->hop_temp = M * ws->cx_sparse_hoppings_0_inv[i];
			    if (section == 1) ws->hop_temp = M * ws->cx_sparse_hoppings_1_inv[i];
			    if (section == 2) ws->hop_temp = M * ws->cx_sparse_hoppings_2_inv[i];
			    if (section == 3) ws->hop_temp = M * ws->cx_sparse_hoppings_3_inv[i];
			}
			else {
			    if (section == 0) M = ws->hop_temp * ws->cx_sparse_hoppings_0_inv[i];
			    if (section == 1) M = ws->hop_temp * ws->cx_sparse_hoppings_1_inv[i];
			    if (section == 2) M = ws->hop_temp * ws->cx_sparse_hoppings_2_inv[i];
			    if (section == 3) M = ws->hop_temp * ws->cx_sparse_hoppings_3_inv[i];
			}
			par = (par == 0) ? 1 : 0;
		    }
		}
	    }		
	    if (par == 1)
		M = ws->hop_temp;
	}	
    }
}
#endif
