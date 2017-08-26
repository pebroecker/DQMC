#ifndef CX_EIGENVALUES_HPP
#define CX_EIGENVALUES_HPP

#include <cmath>

#include <exception>

namespace dqmc {
    namespace la {
	inline int cx_eigenvalues_sym(int& sites, pmat_t& M, pvec_t& V) {
	    using namespace std;
	    throw std::runtime_error("cx_eigenvalues_sym is not implemented");
	    return -1;	    
        };
    }
}
#endif
