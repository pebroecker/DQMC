#ifndef CX_SPECIAL_HPP
#define CX_SPECIAL_HPP

namespace dqmc {
    namespace la {
	inline void tiny_plus_random(const cx_mat&__restrict__ in, cx_mat& out) {	   
	    // out.zeros();
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), in.cols(), out.rows() - in.rows(), out.cols() - in.cols()).setRandom();
	}


	inline void thin_col_plus_random(const cx_mat&__restrict__ in, cx_mat&__restrict__ out) {
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(0, in.cols(), out.rows(), out.cols() - in.cols()).setRandom();
	}
		

	inline void thin_row_plus_random(const cx_mat&__restrict__ in,
					 cx_mat&__restrict__ out) {
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), 0, out.rows() - in.rows(), in.cols()).setRandom();
	}

	inline void thin_col_to_invertible(const cx_mat_t&__restrict__ in, cx_mat_t&__restrict__ out) {				   
	    using namespace std;
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(0, in.cols(), out.rows(), out.cols() - in.cols()).setRandom();

	    // Gram-Schmidt orthogonalization
	    cx_double_t projection;

	    for (int iter = 0; iter < 2; ++iter) {
		for (int c = in.cols(); c < out.cols(); ++c) {
		    for (int c2 = 0; c2 < c; ++c2) {
			projection = out.col(c).transpose() * out.col(c2).conjugate();
			out.col(c).noalias() -= projection * out.col(c2);
		    }		    
		    out.col(c).normalize();		 
		}
	    }
	}

	inline void thin_row_to_invertible(const cx_mat_t&__restrict__ in,
					   cx_mat_t&__restrict__ out) {
					   
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), 0, out.rows() - in.rows(), in.cols()).setRandom();
	    
	    // Gram-Schmidt orthogonalization
	    cx_double_t projection;

	    for (int iter = 0; iter < 2; ++iter) {
		for (int r = in.rows(); r < out.rows(); ++r) {
		    for (int r2 = 0; r2 < r; ++r2) {
			projection = out.row(r).conjugate() * out.row(r2).transpose();
			out.row(r).noalias() -= projection * out.row(r2);
		    }
		    out.row(r).normalize();
		}
	    }
	}

    }
}
#endif
