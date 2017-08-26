#ifndef SUB_HPP
#define SUB_HPP

namespace dqmc {
    namespace la {
	template<typename T, typename V>
	inline void copy_diag(T& in, V& out) {
            for (int i = 0; i < in.rows(); ++i) {
		out(i) = in(i, i);
	    }
	}

	template<typename T>
	inline void to_submat(int x_0, int y_0, int x_1, int y_1, const T& in, T& out) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
                    out(i, j) = in(i - x_0, j - y_0);
                }
            }	    
	}


	template<typename T>
	inline void submat_from_transpose(int x_0, int y_0, int x_1, int y_1, const T& in, T& out) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
                    out(i - x_0, j - y_0) = in(j, i);
                }
            }	    
	}


	template<typename T>
	inline void from_submat(int x_0, int y_0, int x_1, int y_1, const T& in, T& out) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
                    out(i - x_0, j - y_0) = in(i, j);
                }
            }	    
	}


	template<typename T>
	inline void from_submat(const T& in, int x_0, int y_0, T& out) {
            for (int j = 0; j < in.cols(); j++) {
                for (int i = 0; i < in.rows(); i++) {
                    out(i + x_0, j + y_0) = in(i, j);
                }
            }	    
	}


	template<typename T>
	inline void from_submat(int in_x_0, int in_y_0, int in_x_1, int in_y_1,
				const T& in, 
				int out_x_0, int out_y_0, int out_x_1, int out_y_1,
				T& out) {
            for (int j = 0; j < in_y_1 - in_y_0; j++) {
                for (int i = 0; i < in_x_1 - in_x_0; i++) {
		    // std::cout << i << " " << j << " | " << in_x_1 - in_x_0
		    // 	      << " " << in_y_1 - in_y_0 << " | "
		    // 	      << in.rows() << " " << in.cols() << " / "
		    // 	      << i + in_x_0 << " " << j + in_y_0 << " ||| "
		    // 	      << out.rows() << " " << out.cols() << " / "
		    // 	      << i + out_x_0 << " " << j + out_y_0 << " ||| "
		    // 	      << std::endl;
                    out(i + out_x_0, j + out_y_0) = in(i + in_x_0, j + in_y_0);
                }
            }	    
	}

	template<typename T>
	inline void from_subvec(const T& in, T& out) {
	    for (int i = 0; i < in.size();++i) {
		out(i) = in(i);
	    }
	}

	
	template<typename T>
	inline void from_subvec(int x_0, int x_1, const T& in, T& out) {
	    for (int i = x_0; i < x_1;++i) {
		out(i) = in(i);
	    }
	}


	template<typename T>
	inline void subvec(int x_0, int x_1, const T& in, T& out) {
	    for (int i = x_0; i < x_1; ++i) {
		out(i - x_0) = in(i);
	    }
	}


	template<typename T, typename V>
	inline void row(int r, const T& in, V& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		out(c) = in(r, c);
	    }
	}


	template<typename T, typename V>
	inline void row(int r, double alpha, const T& in, V& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		out(c) = alpha * in(r, c);
	    }
	}


	template<typename T, typename V>
	inline void row_add(int r, const T& in, V& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		out(c) += in(r, c);
	    }
	}


	template<typename T, typename V>
	inline void to_row(int r, T& in, V& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		in(r, c) = out(c);
	    }
	}

	template<typename T, typename V>
	inline void col(int c, double alpha, const T& in, V& out) {
	    for (int r = 0; r < in.rows(); ++r) {
		out(r) = alpha * in(r, c);
	    }
	}


	template<typename T, typename V>
	inline void col(int c, const T& in, V& out) {
	    for (int r = 0; r < in.rows(); ++r) {
		out(r) = in(r, c);
	    }
	}


	template<typename T, typename V>
	inline void col_add(int c, const T& in, V& out) {
	    for (int r = 0; r < in.rows(); ++r) {
		out(r) += in(r, c);
	    }
	}

	template<typename T, typename V>
	inline void to_col(int c, T& mat, V& vec) {
	    for (int r = 0; r < mat.rows(); ++r) {
		mat(r, c) = vec(r);
	    }
	}

	template<typename T, typename U>
	inline void diag_to_submat(int x_0, int y_0, int x_1, int y_1, U& in, T& out) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
		    if (i - x_0 == j - y_0) { 
			out(i , j) = in(i - x_0);
		    } else {
			out(i , j ) = 0.;
		    }
                }
            }	    
	}

	template<typename T, typename U>
	inline void diag_to_submat(int x_0, int y_0, int x_1, int y_1,
				   U& in, T& out, double alpha) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
		    if (i - x_0 == j - y_0) { 
			out(i , j) = alpha *in(i - x_0);
		    } else {
			out(i , j ) = 0.;
		    }
                }
            }	    
	}


        template<typename T>
        inline T one_m_submat(const int x_0, const int y_0,
			    const int x_1, const int y_1, const T& M) {
            T temp(x_1 - x_0, y_1 - y_0);
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
		    if (i == j)
			temp(i - x_0, j - y_0) = 1. - M(i, j);
		    else
			temp(i - x_0, j - y_0) = -M(i, j);
                }

            }
	    return temp;
        }

        template<typename T>
        inline void one_m_submat(const int x_0, const int y_0,
			       const int x_1, const int y_1, const T& M, T& out) {
            for (int j = y_0; j < y_1; j++) {
                for (int i = x_0; i < x_1; i++) {
		    if (i == j)
			out(i - x_0, j - y_0) = 1. - M(i, j);
		    else
			out(i - x_0, j - y_0) = -M(i, j);
                }

            }
        }

    }
}
#endif
