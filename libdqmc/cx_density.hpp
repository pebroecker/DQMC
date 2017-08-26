#ifndef CX_DQMC_DENSITY_HPP
#define CX_DQMC_DENSITY_HPP

#include "la.hpp"
#include "parameters.hpp"
#include "cx_workspace.hpp"

#include <Eigen/Eigenvalues>

#include <boost/foreach.hpp>

#include <cmath>
#include <exception>
#include <iomanip>

namespace cx_dqmc {
    namespace density {       
	inline void density_matrix(dqmc::parameters& p,
				   cx_dqmc::workspace& ws) {
	    using namespace std;
	    srand(23);
	    double rand_hop;
	    double rand_eps = 0;
	    
	    pvec_t eigval(p.N);
	    pmat_t lowest_eigenvectors(p.N, p.particles);
	    pmat_t density(p.N, p.N), eigvec(p.N, p.N);

	    density.setZero();
	    
	    if (ws.density.cols() != p.N) {
		throw std::runtime_error("Dimensions of density are incorrect");
	    }
	    
	    for (int attempt = 0; attempt < 4; ++attempt) {
		alps::graph_helper<>::bond_iterator itr1, itr1_end;	    
		for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end; ++itr1) {
		    rand_hop = (1 + rand_eps * ( -0.5 + (std::rand() % RAND_MAX)/double(RAND_MAX)));

		    density(p.graph.source(*itr1), p.graph.target(*itr1)) =
			-p.ts[p.graph.bond_type(*itr1)] * rand_hop;
		    density(p.graph.target(*itr1), p.graph.source(*itr1)) =
			-p.ts[p.graph.bond_type(*itr1)] * rand_hop;
		}		    
		eigvec = density;

		Eigen::SelfAdjointEigenSolver<mat> eig(density);
		eig.compute(density);
		eigval = eig.eigenvalues();
		eigvec = eig.eigenvectors();

		bool error = false;
		for (int i = 0; i < eigval.size() - 1; ++i) {
		    double diff;
		    diff = fabs(eigval(i + 1) - eigval(i));
		    if (diff < 1e-4) {
			if (attempt == 0) {
			    rand_eps = 1e-3;
			}
			else if (attempt == 1) {
			    rand_eps = 1e-2;
			}		
			else if (attempt == 2) {
			    rand_eps = 1e-1;
			}		
			else if (attempt == 3) {
			    rand_eps = 5e-1;
			}		
			else if (attempt == 4) {
			    if (p.rank < 1) {
				cout << attempt << " @ rand_eps = " << rand_eps << " - " << diff << endl;
			    }
			    throw std::runtime_error("Degeneracy not lifted");
			}
			error = true;
			break;
		    }
		}
		if (!error) {
		    break;
		}
	    }
	    
	    for (int i = 0; i < p.N; i++) {
	    	for (int j = 0; j < p.real_particles; j++) {
		    lowest_eigenvectors(i, j) = eigvec(i, j);
	    	}
	    }
	    
	    for (int row = 0; row < p.N; row++) {
		for (int col = 0; col < p.N; col++) {
		    ws.density(row, col) = cx_double(0, 0);

		    for (int i = 0; i < p.real_particles; i++) {
			ws.density(row, col) +=
			    cx_double(lowest_eigenvectors(row, i) * lowest_eigenvectors(col, i), 0);
			ws.den_U(row, i) = cx_double(lowest_eigenvectors(row, i), 0);
		    }
		}
	    }

	    // cout << p.outp << "And den_u" << endl << ws.den_U << endl << endl;
 	}

	inline void cx_density_matrix(dqmc::parameters& p, cx_dqmc::workspace& ws) {
	    using namespace std;

	    srand(23);
	    double rand_hop;
	    double rand_eps = 0;
	    
	    cx_vec eigval(p.N);
	    cx_mat_t lowest_eigenvectors(p.N, p.real_particles);
	    cx_mat_t density(p.N, p.N), eigvec(p.N, p.N);
	    

	    if (ws.density.cols() != p.N) {
		throw std::runtime_error("Dimensions of density are incorrect");
	    }

	    ws.density.setZero();
	    for (int attempt = 0; attempt < 4; ++attempt) {
		alps::graph_helper<>::bond_iterator itr1, itr1_end;	    
		int s, t, b;
		double pref;

		for (boost::tie(itr1, itr1_end) = p.graph.bonds(); itr1 != itr1_end; ++itr1) {
		    rand_hop = (1 + rand_eps * ( -0.5 + (std::rand() % RAND_MAX)/double(RAND_MAX)));
		    s = p.graph.source(*itr1);
		    t = p.graph.target(*itr1);
		    b = p.graph.bond_type(*itr1);
		    		    
		    ws.density(s, t) = -1. * cx_double(p.ts[b] * rand_hop, -p.im_ts[b]* rand_hop);
		    ws.density(t, s) = -1. * cx_double(p.ts[b] * rand_hop, p.im_ts[b] * rand_hop);
		}
		ws.eigvec = ws.density;

		Eigen::SelfAdjointEigenSolver<cx_mat_t> eig(ws.density);
		eig.compute(ws.density);
		vec revals = eig.eigenvalues();
		ws.eigvec = eig.eigenvectors();

		if (attempt == 0) {
		    ofstream mat_file("eigenvectors.mat", ios::out | ios::trunc);
		    mat_file << ws.eigvec;
		    mat_file.close();

		    ofstream vec_file("eigenvalues.vec", ios::out | ios::trunc);
		    vec_file << revals;
		    vec_file.close();
		}

		std::vector<max_pair > sorting;
		sorting.reserve(revals.size());
		for (int i = 0; i < revals.size(); i++) {
		    double val = revals(i);
		    sorting.push_back(max_pair(val,i));
		}
		std::sort(sorting.begin(),sorting.end());
		cx_mat_t sorted_eigvec;
		sorted_eigvec.resizeLike(ws.eigvec);
		for (int i = 0; i < revals.size(); i++) {
		    revals.coeffRef(i,0) = sorting[i].first;
		    sorted_eigvec.col(i) = ws.eigvec.col(sorting[i].second);
		}
		if (attempt == 0) {
		    ofstream sorted_mat_file("sorted_eigenvectors.mat", ios::out | ios::trunc);
		    sorted_mat_file << sorted_eigvec;
		    sorted_mat_file.close();

		    ofstream sorted_file("sorted_eigenvalues.vec", ios::out | ios::trunc);
		    sorted_file << revals;
		    sorted_file.close();
		}

		// eigval = revals;
		ws.eigvec = sorted_eigvec;

		bool error = false;
		for (int i = 0; i < revals.size() - 1; ++i) {
		    // break;
		    double diff;
		    diff = abs(revals(i + 1) - revals(i));
		    if (diff < 1e-5) {

			if (attempt == 0) {
			    rand_eps = 1e-3;
			}
			else if (attempt == 1) {
			    rand_eps = 1e-2;
			}		
			else if (attempt == 2) {
			    rand_eps = 1e-1;
			}		
			else if (attempt == 3) {
			    throw std::runtime_error("Degeneracy not lifted");
			}
			error = true;
			break;
		    }
		}
		if (!error) {
		    break;
		}
	    }
	    ws.den_U = ws.eigvec.block(0, 0, p.N, p.real_particles);
	    ws.density = ws.den_U * ws.den_U.adjoint();
 	}
    }
}
#endif
