#include "la_types.hpp"
#ifndef DIFFS_HPP
#define DIFFS_HPP

#include <cmath>

namespace dqmc {
    namespace la {
	
	// inline void largest_abs_diff(int size, pmat_t& A, pmat_t& B, pdouble_t& diff) {
	//     diff = 0;

	//     for (int i = 0; i < size; i++) {
	// 	for (int j = 0; j < size; j++) {
	// 	    if (fabs(fabs(A(i, j)) - fabs(B(i, j))) > diff) {
	// 		diff = fabs(fabs(A(i, j)) - fabs(B(i, j)));
	// 	    }
	// 	}
	//     }
	// }	    

	
	// inline void largest_rel_diff(pmat_t& A, pmat_t& B, pdouble_t& diff) {
	//     diff = 0;
	//     pdouble_t curr_diff;

	//     for (int i = 0; i < A.rows(); ++i) {
	// 	for (int j = 0; j < A.cols()s; ++j) {
	// 	    curr_diff = 2*fabs(A(i, j) - B(i, j))
	// 		/(fabs(A(i, j)) + fabs(B(i, j)));
	// 	    if (curr_diff > diff) {
	// 		diff = curr_diff;
	// 	    }
	// 	}
	//     }
	// }	    


	// inline void diag_largest_abs_diff(int size, pmat_t& A, pmat_t& B, pdouble_t& diff) {
	//     diff = 0;
	//     int index = 0;
	//     for (int i = 0; i < size; i++) {
	// 	if (fabs(fabs(A(i, i)) - fabs(B(i, i))) > diff) {
	// 	    diff = fabs(fabs(A(i, i)) - fabs(B(i, i)));
	// 	    index = i;
	// 	}
	//     }
	// }	    

	
	// inline void diag_largest_rel_diff(int size, pmat_t& A, pmat_t& B, pdouble_t& diff) {
	//     diff = 0;
	//     pdouble_t curr_diff;
	//     int index = 0;
	//     for (int i = 0; i < size; i++) {
	// 	curr_diff = 2*fabs(A(i, i) - B(i, i))
	// 	    /(fabs(A(i, i)) + fabs(B(i, i)));
	// 	if (curr_diff > diff) {
	// 	    diff = curr_diff;
	// 	    index = i;
	// 	}
	//     }
	// }	    

	// inline void largest_abs_diff(int size, pvec_t& A, pvec_t& B, pdouble_t& diff) {
	//     diff = 0;

	//     for (int i = 0; i < size; i++) {
	// 	if (fabs(fabs(A(i)) - fabs(B(i))) > diff) {
	// 	    diff = fabs(fabs(A(i)) - fabs(B(i)));
	// 	}
	//     }
	// }	    


	// inline void largest_abs_diff(pvec_t& A, pvec_t& B, pdouble_t& diff) {
	//     diff = 0;

	//     for (int i = 0; i < A.size(); i++) {
	// 	if (fabs(fabs(A(i)) - fabs(B(i))) > diff) {
	// 	    diff = fabs(fabs(A(i)) - fabs(B(i)));
	// 	}
	//     }
	// }	    


	// inline void largest_rel_diff(pvec_t& A, pvec_t& B, pdouble_t& diff) {
	//     diff = 0;

	//     for (int i = 0; i < A.size(); i++) {
	// 	if (2*fabs(fabs(A(i)) - fabs(B(i)))
	// 	    /fabs(fabs(A(i)) + fabs(B(i))) > diff) {
	// 	    diff = 2*fabs(fabs(A(i)) - fabs(B(i)))
	// 		/fabs(fabs(A(i)) + fabs(B(i)));
	// 	}
	//     }
	// }	    
    }
}

#endif
