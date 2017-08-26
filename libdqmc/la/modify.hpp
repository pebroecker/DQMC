#ifndef MODIFY_HPP
#define MODIFY_HPP

namespace dqmc {
    namespace la {

	template<typename T>
        inline void resize(const int&__restrict__ rows, const int&__restrict__ cols, T& M) {
            M.resize(rows, cols);
	    // M.zeros();
	    // zeros(rows, cols, M);
        }


	template<typename T>
        inline void resize(const int rows, T&__restrict__ M) {
            M.resize(rows);
	    // M.zeros();
	    // zeros(rows, M);
        }

	
	template<typename T>
        inline void zeros(int rows, int cols, T&__restrict__ M) {
            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < rows; row++) {
                    M(row, col) = 0;
                }
            }
        }

	template<typename T>
        inline void zeros(T&__restrict__ M) {
            for (int col = 0; col < M.cols(); col++) {
                for (int row = 0; row < M.rows(); row++) {
                    M(row, col) = 0;
                }
            }
        }


	template<typename T>
        inline void zeros(int rows, T&__restrict__ M) {
	    for (int row = 0; row < rows; row++) {
		M(row) = 0;
            }
        }


	template<typename T>
        inline void ones(int rows, T&__restrict__ M) {
	    for (int row = 0; row < rows; row++) {
		M(row) = 1.;
            }
        }


	template<typename T>
        inline void ones(T&__restrict__ M) {
	    for (int i = 0; i < M.size(); ++i) {
		M(i) = 1.;
            }
        }


	template<typename T>
        inline void mones(T&__restrict__ M) {
	    for (int i = 0; i < M.rows(); ++i) {
		for (int j = 0; j < M.cols(); ++j) {
		    M(i, j) = 1.;
		}
            }
        }


	template<typename T>
        inline void identity(T&__restrict__ M) {
	    if (M.rows() != M.cols()) { throw std::runtime_error("identity(): matrix is not square"); }
		
	    for (int i = 0; i < M.rows(); ++i) {
		for (int j = 0; j < M.cols(); ++j) {
		    M(i, j) = (i == j) ? 1. : 0;
		}
            }
        }


	template<typename T>
        inline void matrix_transpose(T&__restrict__ in, T&__restrict__ out) {
	    if (in.rows() != out.cols() ||
		in.cols() != out.rows()) {
		throw std::runtime_error("Transpose not possible");
	    }
            for (int j = 0; j < in.cols(); j++) {
                for (int i = 0; i < in.rows(); i++) {
                    out(j, i) = in(i, j);
                }
            }
        }


	template<typename T>
        inline void matrix_transpose(const int rows, const int cols, T&__restrict__ in,
				     T&__restrict__ out) {
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    out(j, i) = in(i, j);
                }
            }
        }

	template<typename T>
        inline void matrix_transpose(const int& sites, T&__restrict__ in, T&__restrict__ out) {
	    matrix_transpose(sites, sites, in, out);
        }

    }
}
#endif
