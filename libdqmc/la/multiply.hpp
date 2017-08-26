#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP

#include "forward.hpp"
#include "print.hpp"
#include "dets.hpp"
#include "diffs.hpp"

#include "svd.hpp"
#include "sub.hpp"
#include "special.hpp"

namespace dqmc {
    namespace la {
	template<typename T, typename U>
        inline void scale_rows(T&__restrict__ M, U&__restrict__ V) {
            for (int col = 0; col < M.cols(); col++) {
                for (int row = 0; row < M.rows(); row++) {
                    M(row, col) *= V(row);
                }
            }
        }


        template<typename T, typename U>
        inline void scale_cols(T&__restrict__ M, U&__restrict__ V) {
            for (int col = 0; col < M.cols(); col++) {
                for (int row = 0; row < M.rows(); row++) {
                    M(row, col) *= V(col);
                }
            }
        }

        template<typename T, typename U>
        inline void scale_cols(T&__restrict__ M, U&__restrict__ V,
			       T&__restrict__ O) {
            for (int col = 0; col < M.cols(); col++) {
                for (int row = 0; row < M.rows(); row++) {
                    O(row, col) = M(row, col) * V(col);
                }
            }
        }

        template<typename T, typename U>
        inline void scale_cols_inv(T&__restrict__ M, U&__restrict__ V) {
            for (int col = 0; col < M.cols(); col++) {
                for (int row = 0; row < M.rows(); row++) {
                    M(row, col) /= V(col);
                }
            }
        }

        template<typename T, typename U>
        inline void scale_cols(const int n, T&__restrict__ M, U&__restrict__ V) {
            for (int col = 0; col < n; col++) {
                for (int row = 0; row < n; row++) {
                    M(row, col) *= V(col);
                }
            }
        }

        template<typename T, typename U>
        inline void scale_cols_inv(const int n, T&__restrict__ M, U&__restrict__ V) {
            for (int col = 0; col < n; col++) {
                for (int row = 0; row < n; row++) {
                    M(row, col) /= V(col);
                }
            }
        }

	inline void dot(pvec_t&__restrict__ v1, pvec_t&__restrict__ v2,
			pdouble_t&__restrict__ result) {
	    result = 0;

	    for (int i = 0; i < v1.size(); ++i) {
		result += v1(i) * v2(i);
	    }
	}
	

	inline void thin_sandwich(pmat_t&__restrict__ A, pvec_t&__restrict__ V, pmat_t&__restrict__ B,
				  pmat_t&__restrict__ out) {
	    
	    for (int r = 0; r < A.rows(); r++) {
		for (int c = 0; c < B.cols(); ++c) {
		    out(r, c) = 0.;
		    for (int k = 0; k < V.size(); ++k) {
			out(r,c) += A(r, k) * B(k, c) * V(k);
		    }
		}
	    }	    
	}

	inline void thin_inv_sandwich(pmat_t&__restrict__ A, pvec_t&__restrict__ V, pmat_t&__restrict__ B,
				      pmat_t&__restrict__ out) {
	    // scale_cols_inv(A, V);
	    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.rows(), A.rows(), double(1), &A(0, 0), A.rows(),  &B(0, 0), B.rows(), double(0), &out(0, 0), out.rows());
	    for (int r = 0; r < A.rows(); r++) {
	    	for (int c = 0; c < B.cols(); ++c) {
	    	    out(r, c) = 0.;
	    	    //for (int k = A.cols() - 1; k >= 0; --k) {
		    for (int k = 0; k < V.size(); ++k) {
	    		out(r,c) += A(r, k) * B(k, c) / V(k);
	    	    }
	    	}
	    }	    
	}


#ifdef USE_DD
	inline void thin_inv_sandwich(dd_mat&__restrict__ A, dd_vec&__restrict__ V, dd_mat&__restrict__ B,
				      dd_mat&__restrict__ out) {

	    dd_real zero(0, 0);
	    dd_real one(1., 0);
	    for (int r = 0; r < A.rows(); r++) {
	    	for (int c = 0; c < B.cols(); ++c) {
	    	    out(r, c) = zero;
		    for (int k = 0; k < V.size(); ++k) {
	    		out(r,c) += A(r, k) * B(k, c) / V(k);
	    	    }
	    	}
	    }	    
	}
#endif

	inline void diag_prod(pvec_t& D, pdouble_t& prod) {
	    prod = 1.;

	    for (int i = 0; i < D.size(); i++) {
		prod *= D(i);
	    }
	}

	inline void sparse_hop_right(pmat_t& A, pdouble_t* hop,
				     int* hop_indices,
				     pmat_t& C) {
	    for (int r = 0; r < A.rows(); ++r) {
		for (int c = 0; c < C.cols(); ++c) {
		    C(r, c) = A(r, hop_indices[2*c]) * hop[2*c]
			+ A(r, hop_indices[2*c + 1]) * hop[2*c + 1];
		}
	    }
	}	

	inline void mm(pmat_t& A, pmat_t& B, pmat_t& C) {
	    C = A*B;
	}

	inline void mtm(pmat_t& A, pmat_t& B, pmat_t& C) {
	    C = A.transpose()*B;
	}

	inline void mmt(pmat_t& A, pmat_t& B, pmat_t& C) {
	    C = A * B.transpose();
	}	

	inline void mtmt(pmat_t& A, pmat_t& B, pmat_t& C) {
	    C = A.transpose() * B.transpose();
	}		    
    }
}
#endif
