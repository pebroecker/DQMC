#ifndef CX_DQMC_WORKSPACE_HPP
#define CX_DQMC_WORKSPACE_HPP

#include "la.hpp"
#include "parameters.hpp"

#include <cmath>

#include <exception>
#include <utility>

typedef std::pair<double, int> max_pair;
typedef boost::array<int, 3> bond;
typedef std::vector<bond>::iterator bond_it;
typedef std::vector<cx_sp_mat>::iterator cx_sp_it;

namespace cx_dqmc {
    struct workspace {
	bool renyi;
	int vol, sites, particles, eff_particles, boundary_conditions,
	    connected, disconnected, num_update_bonds, num_sites,
	    bonds_per_flush, large_vol, step;

	cx_mat_t density, eigvec, den_U;
	cx_mat_t identity;
	cx_mat_t la_mat_1, la_mat_2;
	cx_mat_t mat_1, mat_2, mat_3, mat_4,
	    mat_5, mat_6, mat_7, mat_8, mat_9, mat_10;
	cx_mat_t diff_mat;
	cx_mat_t col_mat_1, col_mat_2, col_mat_3, col_mat_4,
	    col_mat_5, col_mat_6, col_mat_7,
	    row_mat_1;
	cx_mat_t tiny_mat_1, tiny_mat_2, tiny_mat_3, tiny_mat_4,
	    tiny_mat_5, tiny_mat_6, tiny_mat_7;
	cx_mat_t large_mat_1, large_mat_2, large_mat_3, large_mat_4, large_mat_5;
	cx_vec la_vec_1, vec_1, vec_2, vec_3, vec_4;
	cx_vec tiny_vec_1, tiny_vec_2, tiny_vec_3, tiny_vec_4 ;
	vec_t large_vec_1, large_vec_2, large_vec_3,
	    re_vec_1, re_vec_2, re_vec_3, re_vec_4, re_vec_5;

	mat re_mat_1, re_mat_2, re_mat_3, re_mat_4;
	vec re_la_vec_1, re_tiny_vec_1, re_tiny_vec_2,
	    re_tiny_vec_3, re_tiny_vec_4, re_tiny_vec_5, re_tiny_vec_6;
	cx_mat hop_temp;
	
	iarr_t bonds;
	std::vector<int> bond_types;
	std::vector<int> bond_used;
	std::vector<int> site_used;
	
	std::vector<double> hopping_diag;
	std::vector<double> hopping_offd;
	
	boost::multi_array<int, 2> site_bonds;
	boost::multi_array<int, 2> site_signs;	
	boost::multi_array<int, 2> chkr;
	std::vector<std::vector<bond> > chkr_bonds;

	bool complex_hoppings;
	std::vector<sp_mat> sparse_hoppings;
	std::vector<sp_mat> sparse_hoppings_0;
	std::vector<sp_mat> sparse_hoppings_1;
	std::vector<sp_mat> sparse_hoppings_2;
	std::vector<sp_mat> sparse_hoppings_3;
	std::vector<sp_mat> sparse_hoppings_inv;
	std::vector<sp_mat> sparse_hoppings_0_inv;
	std::vector<sp_mat> sparse_hoppings_1_inv;
	std::vector<sp_mat> sparse_hoppings_2_inv;
	std::vector<sp_mat> sparse_hoppings_3_inv;


	std::vector<cx_sp_mat> cx_sparse_hoppings;
	std::vector<cx_sp_mat> cx_sparse_hoppings_0;
	std::vector<cx_sp_mat> cx_sparse_hoppings_1;
	std::vector<cx_sp_mat> cx_sparse_hoppings_2;
	std::vector<cx_sp_mat> cx_sparse_hoppings_3;
	std::vector<cx_sp_mat> cx_sparse_hoppings_inv;
	std::vector<cx_sp_mat> cx_sparse_hoppings_0_inv;
	std::vector<cx_sp_mat> cx_sparse_hoppings_1_inv;
	std::vector<cx_sp_mat> cx_sparse_hoppings_2_inv;
	std::vector<cx_sp_mat> cx_sparse_hoppings_3_inv;
	       	
	workspace() {
	    renyi = false;
	    step = -1;
	}

	~workspace() {	}

	
	void parameters(dqmc::parameters& p) {
	    using namespace std;
	    vol = p.N;
	    large_vol = 2*vol;
	    sites = p.graph.num_sites();
	    particles = p.particles;
	    eff_particles = particles;

	    complex_hoppings = p.complex_hoppings;
	    num_sites = p.graph.num_sites();

	    alps::graph_helper<>::bond_iterator itr1, itr1_end;

	    bond_used.resize(p.graph.num_bonds());
	    site_used.resize(p.graph.num_sites());

	    site_bonds.resize(boost::extents[p.graph.num_sites()][40]);
	    site_signs.resize(boost::extents[p.graph.num_sites()][40]);
	    
	    int site_index[p.graph.num_sites()];
	    std::fill(&site_index[0], &site_index[p.graph.num_sites()], 0);
	    
	    std::fill(site_bonds.origin(),
		      site_bonds.origin() + site_bonds.size(),
		      -1);
	    std::fill(site_signs.origin(),
		      site_signs.origin() + site_signs.size(),
		      0);

	    int b, s1, s2;
	    if (p.nni < 0) {
		cout << p.outp << "Assigning alternating bond "
		     << "signs for attractive interaction" << endl;
	    }

	    num_update_bonds = p.num_update_bonds;
	    bonds.resize(boost::extents[num_update_bonds][2]);
	    bond_types.resize(num_update_bonds);
	    if (p.rank < 1) cout << p.outp << "Resized bonds " << num_update_bonds << endl;
	    int bond_idx = 0;
	    
	    for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end;
		 ++itr1) {
		if (p.model_id == 0 || p.model_id == 1 || p.model_id == 6 || p.model_id == 7) break;

		if (std::find(p.active_bond_types.begin(), p.active_bond_types.end(),
			      p.graph.bond_type(*itr1)) != p.active_bond_types.end()) {
		    // b = p.graph.index(*itr1);

		    s1 = p.graph.source(*itr1);
		    s2 = p.graph.target(*itr1);

		    bonds[bond_idx][0] = s1;
		    bonds[bond_idx][1] = s2;
		    bond_types[bond_idx] = p.graph.bond_type(*itr1);
		    
		    site_bonds[s1][site_index[s1]] = bond_idx;
		    site_signs[s1][site_index[s1]] = 1;
		    ++site_index[s1];
		    site_bonds[s2][site_index[s2]] = bond_idx;
		
		    if (p.nni < 0) {
			site_signs[s2][site_index[s2]] = -1;
		    } else {
			site_signs[s2][site_index[s2]] = 1;
		    }
		    ++site_index[s2];
		    ++bond_idx;
		}
	    }
	    renyi = false;

	    int n_decomps = p.ts.size();
	    // cout << "Initiailizing chkr_bonds " << n_decomps << endl;
	    chkr_bonds.resize(n_decomps);
	    
	    int rows = p.N + 1;
	    int entries = p.N * 2;
	    density.resize(p.N, p.N);
	    density.fill({0, 0});
	    // cout << "Done initializing" << endl;
	}

	void initialize_gs() {	    
	    using namespace dqmc::la;
	    using namespace std;

	    if (renyi == false) {
		eff_particles = particles;
		sites = vol;
	    }

	    cout << eff_particles << " " << particles << "\t"
		 << sites << " " << vol << " " << large_vol << endl;
	    //============================================================
	    // square matrices
	    //============================================================
	    identity = cx_mat_t::Identity(vol, vol);
	    mat_1.resize(vol, vol);	    
	    mat_2.resize(vol, vol);	    
	    mat_3.resize(vol, vol);	    
	    mat_4.resize(vol, vol);	    
	    mat_5.resize(vol, vol);	    
	    mat_6.resize(vol, vol);	    
	    mat_7.resize(vol, vol);	    
	    mat_8.resize(vol, vol);	    
	    mat_9.resize(vol, vol);	    
	    mat_10.resize(vol, vol);	    

	    la_mat_1.resize(vol, vol);	    
	    la_mat_2.resize(vol, vol);
	    la_mat_1.fill({0, 0});
	    diff_mat.resize(vol, vol);
	    hop_temp.resize(vol, vol);
	    re_mat_1 = mat::Zero(vol, vol);
	    re_mat_2 = mat::Zero(vol, vol);
	    re_mat_3 = mat::Zero(vol, vol);
	    re_mat_4 = mat::Zero(vol, vol);

	    
	    //============================================================
	    // column matrices
	    //============================================================
	    den_U.resize(sites, particles);	    
	    col_mat_1.resize(vol, eff_particles);	    
	    col_mat_2.resize(vol, eff_particles);
	    col_mat_3.resize(vol, eff_particles);
	    col_mat_4.resize(vol, eff_particles);
	    col_mat_5.resize(vol, eff_particles);
	    col_mat_6.resize(vol, eff_particles);
	    col_mat_7.resize(vol, eff_particles);

	    row_mat_1.resize(eff_particles, vol);	    


	    //============================================================
	    // small square matrices
	    //============================================================
	    tiny_mat_1.resize(eff_particles, eff_particles);
	    tiny_mat_2.resize(eff_particles, eff_particles);
	    tiny_mat_3.resize(eff_particles, eff_particles);
	    tiny_mat_4.resize(eff_particles, eff_particles);
	    tiny_mat_5.resize(eff_particles, eff_particles);
	    tiny_mat_6.resize(eff_particles, eff_particles);
	    tiny_mat_7.resize(eff_particles, eff_particles);


	    //============================================================
	    // vectors
	    //============================================================
	    la_vec_1.resize(vol);
	    re_la_vec_1.resize(vol);
	    vec_1.resize(vol);
	    vec_2.resize(vol);
	    vec_3.resize(vol);
	    vec_4.resize(vol);
	    re_vec_1.resize(vol);
	    re_vec_2.resize(vol);
	    re_vec_3.resize(vol);
	    re_vec_4.resize(vol);
	    re_vec_5.resize(vol);
	    
	    tiny_vec_1.resize(eff_particles);
	    tiny_vec_2.resize(eff_particles);
	    tiny_vec_3.resize(eff_particles);
	    tiny_vec_4.resize(eff_particles);

	    re_tiny_vec_1 = vec::Zero(eff_particles);
	    re_tiny_vec_2 = vec::Zero(eff_particles);
	    re_tiny_vec_3 = vec::Zero(eff_particles);
	    re_tiny_vec_4 = vec::Zero(eff_particles);
	    re_tiny_vec_5 = vec::Zero(eff_particles);
	    re_tiny_vec_6 = vec::Zero(eff_particles);


	    //============================================================
	    // large matrices and vectors for replica formulation
	    //============================================================
	    large_mat_1.resize(large_vol, large_vol);
	    large_mat_2.resize(large_vol, large_vol);
	    large_mat_3.resize(large_vol, large_vol);
	    large_mat_4.resize(large_vol, large_vol);
	    large_mat_5.resize(large_vol, large_vol);

	    large_vec_1.resize(large_vol);
	    large_vec_2.resize(large_vol);
	    large_vec_3.resize(large_vol);
	}
    };
}
#endif
