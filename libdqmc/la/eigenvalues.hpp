#ifndef EIGENVALUES_HPP
#define EIGENVALUES_HPP

#include <cmath>

namespace dqmc {
    namespace la {
	inline int eigenvalues_sym(int& sites, pmat_t& M, pvec_t& V) {
	    using namespace std;
	    
            char full = 'V';
            char up = 'U';
	    int lwork = M.size();
	    pvec_t work(lwork);
	    int info = 0;
	    // arma::dsyev_(&full, &up, &sites, &M(0, 0), &sites, &V(0), &work(0), &lwork, &info);
	    return info;
	    
        };
    }
}
#endif
