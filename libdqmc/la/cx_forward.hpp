#ifndef CX_FORWARD_HPP
#define CX_FORWARD_HPP

namespace dqmc {
    namespace cx_la {
	inline void dot(pvec_t&__restrict__ v1, pvec_t&__restrict__ v2,
			pdouble_t&__restrict__ result);

	template<typename T, typename U>
        inline void scale_rows(T&__restrict__ M, U&__restrict__ V);

        template<typename T, typename U>
        inline void scale_cols(T&__restrict__ M, U&__restrict__ V);

        template<typename T, typename U>
        inline void scale_cols_inv(T&__restrict__ M, U&__restrict__ V);

        template<typename T, typename U>
        inline void scale_cols(const int n, T&__restrict__ M, U&__restrict__ V);

        template<typename T, typename U>
        inline void scale_cols_inv(const int n, T&__restrict__ M, U&__restrict__ V);

	inline void mm(const char& transa, const char& transb,
		       int m, int k, int n, pmat_t& A, pmat_t& B, pmat_t& C,
		       const double& alpha, const double& beta);


	inline void dot(pvec_t&__restrict__ v1, pvec_t&__restrict__ v2,
			pdouble_t&__restrict__ result);

	
	inline void mm(pmat_t&__restrict__ A, pmat_t&__restrict__ B,
		       pmat_t&__restrict__ C);

	inline void mm_t(pmat_t&__restrict__ A, pmat_t&__restrict__ B, pmat_t&__restrict__ C);

	inline void mtmt(pmat_t&__restrict__ A, pmat_t&__restrict__ B, pmat_t&__restrict__ C);

	inline void mmt(pmat_t&__restrict__ A, pmat_t&__restrict__ B, pmat_t&__restrict__ C);

	inline void mtm(pmat_t&__restrict__ A, pmat_t&__restrict__ B,
			pmat_t&__restrict__ C);
       
	inline void mm_transpose(pmat_t&__restrict__ A, pmat_t&__restrict__ B, pmat_t&__restrict__ C);

	inline void thin_sandwich(pmat_t&__restrict__ A, pvec_t&__restrict__ V, pmat_t&__restrict__ B,
				  pmat_t&__restrict__ out);

	inline void thin_inv_sandwich(pmat_t&__restrict__ A, pvec_t&__restrict__ V, pmat_t&__restrict__ B,
				      pmat_t&__restrict__ out);       
    }
}

#endif
