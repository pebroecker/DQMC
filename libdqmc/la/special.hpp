#ifndef SPECIAL_HPP
#define SPECIAL_HPP

#include <boost/shared_ptr.hpp>
#include "../workspace.hpp"
#include <cmath>
#include <memory>

namespace dqmc {
    namespace la {

#ifdef USE_DD       
	inline void copy_to_dd(pmat_t& in, dd_mat& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    out(r, c) = ddouble(in(r, c));
		}
	    }
	}
	
	inline void copy_to_dd(pvec_t& in, dd_vec& out) {
	    for (int c = 0; c < in.size(); ++c) {
		out(c) = ddouble(in(c));
	    }
	}


	inline void copy_from_dd(dd_mat& in, pmat_t& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {		    
		    out(r, c) = to_double(in(r, c));
		}
	    }
	}


	inline void copy_from_dd(dd_vec& in, pvec_t& out) {
	    for (int c = 0; c < in.rows(); ++c) {
		out(c) = to_double(in(c));
	    }
	}


	inline void dd_normalize(dd_vec&__restrict__ v) {
	    dd_real norm = 0;

	    for (int i = 0; i < v.rows() * v.cols(); ++i) {
		if ((v(i) * v(i)).isnan()) {
		    std::cout << v(i) << " squared is nan" << std::endl;
		    throw std::runtime_error("error");
		}
		norm += v(i) * v(i);
	    }

	    v /= sqrt(norm);
	}

	
	inline void dd_dot(dd_vec&__restrict__ v1, dd_vec&__restrict__ v2,
			   dd_real&__restrict__ result) {
	    result = dd_real(0, 0);

	    for (int i = 0; i < v1.rows() * v1.cols(); ++i) {
		if ((v1(i) * v2(i)).isnan()) {
		    std::cout << v1(i) << " times " << v2(i) << " is nan" << std::endl;
		    throw std::runtime_error("error");
		}
		result += v1(i) * v2(i);
	    }
	}


	inline void dd_from_col(int c, const dd_mat& in, dd_vec& out) {
	    for (int r = 0; r < in.rows(); ++r) {
		out(r) = in(r, c);
	    }
	}


	inline void dd_to_col(int c, const dd_vec& in, dd_mat& out) {
	    for (int r = 0; r < in.rows(); ++r) {
		out(r, c) = in(r);
	    }
	}


	inline void dd_from_row(int r, const dd_mat& in, dd_vec& out) {
	    for (int c = 0; c < in.cols(); ++c) {
		out(c) = in(r, c);
	    }
	}


	inline void dd_to_row(int r, const dd_vec& in, dd_mat& out) {
	    for (int c = 0; c < out.cols(); ++c) {
		out(r, c) = in(c);
	    }
	}


	inline void dd_thin_col_to_invertible(dd_mat& in, dd_mat& temp, dd_mat& out,
					      dd_vec& t_vec_1, dd_vec& t_vec_2) {
	    using namespace std;
	    
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    temp(r, c) = in(r,c);
		    out(r, c) = in(r, c);
		}
	    }
	    
	    for (int c = in.cols(); c < out.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    temp(r, c) = ddrand();// / dd_real::_max;
		}
	    }

	    // cout << "Randomized full col" << endl;
	    // dqmc::la::print_matrix(temp.rows(), temp.cols(), temp);
	    // Gram-Schmidt orthogonalization
	    dd_real projection;

	    for (int iter = 0; iter < 5; ++iter) {
		// cout << "First iteration" << endl;
		
		for (int c = in.cols(); c < out.cols(); ++c) {
		    // cout << "from_col" << endl;
		    dd_from_col(c, temp, t_vec_1);

		    for (int c2 = 0; c2 < c; ++c2) {
			// cout << "c2" << endl;
			dd_from_col(c2, out, t_vec_2);
			// cout << "dd_dot" << endl;
			dd_dot(t_vec_1, t_vec_2, projection);
			// cout << projection << endl;
			t_vec_1 -= projection * t_vec_2;
		    }
		    
		    dd_normalize(t_vec_1);
		    dd_to_col(c, t_vec_1, out);
		}
		temp = out;
	    }
	    // dqmc::la::print_matrix(out.rows(), out.cols(), out);
	}


	inline void dd_thin_row_to_invertible(dd_mat& in, dd_mat& temp, dd_mat& out,
					      dd_vec& t_vec_1, dd_vec& t_vec_2) {
	    using namespace std;
	    
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    temp(r, c) = in(r,c);
		    out(r, c) = in(r, c);
		}
	    }
	    
	    for (int c = in.rows(); c < out.rows(); ++c) {
		for (int r = 0; r < in.cols(); ++r) {
		    temp(r, c) = ddrand();// / dd_real::_max;
		    if (isnan(temp(r, c))) {
			cout << "Random nan???" << endl;
			throw std::runtime_error("no");
		    }
		}
	    }

	    dd_real projection;

	    for (int iter = 0; iter < 5; ++iter) {
		for (int r = in.rows(); r < out.rows(); ++r) {
		    dd_from_row(r, temp, t_vec_1);
		    
		    for (int r2 = 0; r2 < r; ++r2) {
			dd_from_row(r2, out, t_vec_2);
			dd_dot(t_vec_1, t_vec_2, projection);
			t_vec_1 -= projection * t_vec_2;
		    }
		    
		    dd_normalize(t_vec_1);
		    dd_to_row(r, t_vec_1, out);
		}
		temp = out;
	    }
	}
#endif
	
	inline void randomize(pmat_t& M) {
	    for (int r = 0; r < M.rows(); r++) {
		for ( int c = 0; c < M.cols(); c++) {
		    M(r, c) = (std::rand() % RAND_MAX)/double(RAND_MAX);
		}
	    }
	    std::srand (std::rand());
	}


	inline void normalize(pvec_t&__restrict__ v) {
	    pdouble_t norm = 0;

	    for (int i = 0; i < v.size(); ++i) {
		norm += v(i) * v(i);
	    }

	    for (int i = 0; i < v.size(); ++i) {
		v(i) /= sqrt(norm);
	    }
	}

	inline void random_with_inverse(pmat_t&__restrict__ in,
					pmat_t&__restrict__ in_inv) {
	    throw std::runtime_error("Broken");
	    // for (int c = 0; c < in.cols(); ++c) {
	    // 	for (int r = 0; r < in.rows(); ++r) {
	    // 	    in(r, c) = (std::rand() % RAND_MAX)/double(RAND_MAX);
	    // 	}
	    // }

	    // pmat_t U = in;
	    // pmat_t T = in;
	    // pvec_t D(in.rows());
	    // pmat_t U_inv = in;
	    // pmat_t T_inv = in;

	    // decompose_dgejsv_nt(in, U, D, T);
	    // matrix_transpose(U, T_inv);
	    // matrix_transpose(T, U_inv);
	    // thin_inv_sandwich(U_inv, D, T_inv, in_inv);
	}

	
	inline void random_orthogonal(pmat_t&__restrict__ in, boost::shared_ptr<dqmc::workspace> ws) {
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    in(r, c) = (std::rand() % RAND_MAX)/double(RAND_MAX);
		}
	    }
	    pdouble_t projection, inner_prod;

	    pvec_t vec_1(in.rows());
	    pvec_t vec_2(in.rows());
	    pvec_t vec_3(in.rows());

	    for(int c = 0; c < in.cols(); ++c) {
		col(c, in, vec_1);
		normalize(vec_1);
		to_col(c, in, vec_1);
		dot(vec_1, vec_1, projection);
		// std::cout << "Projection of " << c << " - " << projection << std::endl;
	    }
	    // gram-Schmidt orthogonalization

	    for (int iter = 0; iter < 5; ++iter) {
		pmat_t mat_1 = in;
		for(int c = 0; c < in.cols(); ++c) {
		    col(c, in, vec_1);
		    dot(vec_1, vec_1, projection);
		    // std::cout << "Before Projection of " << c << " - " << projection << std::endl;
		}

		for (int i = 1; i < in.rows(); ++i) {
		    col(i, mat_1, vec_2);
		    col(i, mat_1, vec_3);

		    for (int j = 0; j < i; ++j) {
			col(j, mat_1, vec_1);
			
			dot(vec_1, vec_3, projection);
			dot(vec_1, vec_1, inner_prod);
			vec_2 -= projection/inner_prod * vec_1;			
		    }
		    normalize(vec_2);
		    to_col(i, mat_1, vec_2);		    
		}
		in = mat_1;
		// std::cout << "Done with iter " << iter << std::endl;
		// for(int c = 0; c < in.cols(); ++c) {
		//     col(c, in, vec_1);
		//     dot(vec_1, vec_1, projection);
		//     std::cout << "Projection of " << c << " - " << projection << std::endl;
		// }

	    }	    
	    // for(int c = 0; c < in.cols(); ++c) {
	    // 	col(c, in, vec_1);
	    // 	dot(vec_1, vec_1, projection);
	    // 	std::cout << "Projection of " << c << " - " << projection << std::endl;
	    // }
	}
	

	inline void tiny_to_invertible(const pmat_t&__restrict__ in, const pmat_t& trans, pmat_t&__restrict__ out) {	   
	    // out.zeros();
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), in.cols(), trans.rows(), trans.cols()) = trans;
	}


	inline void tiny_plus_random(const pmat_t&__restrict__ in, pmat_t& out) {	   
	    // out.zeros();
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), in.cols(), out.rows() - in.rows(), out.cols() - in.cols()).setRandom();
	}


	inline void thin_col_plus_random(const pmat_t&__restrict__ in, pmat_t&__restrict__ out) {
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(0, in.cols(), out.rows(), out.cols() - in.cols()).setRandom();
	}
	
	inline void thin_col_to_invertible(const pmat_t&__restrict__ in, pmat_t&__restrict__ out) {
	    using namespace std;
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(0, in.cols(), out.rows(), out.cols() - in.cols()).setRandom();

	    // Gram-Schmidt orthogonalization
	    pdouble_t projection;

	    for (int iter = 0; iter < 2; ++iter) {
		for (int c = in.cols(); c < out.cols(); ++c) {
		    for (int c2 = 0; c2 < c; ++c2) {
			projection = out.col(c).transpose() * out.col(c2);
			out.col(c) -= projection * out.col(c2);
		    }		    
		    out.col(c).normalize();		    
		}
	    }
	}

	
	inline void thin_col_to_invertible_old(const pmat_t&__restrict__ in, pmat_t&__restrict__ out, 
					       dqmc::workspace& ws) {
	    using namespace std;
	    // out.zeros();
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    ws.la_mat_1(r, c) = in(r,c);
		    out(r, c) = in(r, c);
		}
	    }
	    // return;

	    for (int c = in.cols(); c < out.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    ws.la_mat_1(r, c) = (std::rand() % RAND_MAX)/double(RAND_MAX);
		    // for (int o = 0; o < in.rows()*in.cols(); o++) {
		    // 	if (!(fabs(in((r * in.rows() + c * in.cols() + o) % in.rows(), 
		    // 		      (r * in.cols() + c * in.rows() + o) % in.cols())) == 0)) {
		    // 	    ws.la_mat_1(r, c) = fabs(in((r * in.rows() + c * in.cols() + o) % in.rows(), 
		    // 					 (r * in.cols() + c * in.rows() + o) % in.cols()));
		    // 	    break;
		    // 	}
		    // }
		    // ws.la_mat_1(r, c) = 0.;

		    if (r == c) {
			if (ws.la_mat_1(r, c) == 0) { ws.la_mat_1(r, c) = 1.;	}
			ws.la_mat_1(r, c) *= -1;
		    }
		}
	    }
	    // Gram-Schmidt orthogonalization
	    pdouble_t projection;

	    for (int iter = 0; iter < 5; ++iter) {
		for (int c = in.cols(); c < out.cols(); ++c) {
		    //col(c, ws.la_mat_1, ws.la_vec_1);
		    ws.la_vec_1 = ws.la_mat_1.col(c);
		    
		    for (int c2 = 0; c2 < c; ++c2) {
			// col(c2, out, ws.la_vec_2);
			ws.la_vec_2 = out.col(c2);
			// dot(ws.la_vec_1, ws.la_vec_2, projection);
			projection = ws.la_vec_1.transpose() * ws.la_vec_2;
			ws.la_vec_1 -= projection * ws.la_vec_2;
		    }		    
		    // normalize(ws.la_vec_1);
		    ws.la_vec_1.normalize();
		    // to_col(c, out, ws.la_vec_1);
		    out.col(c) = ws.la_vec_1;
		}
		ws.la_mat_1 = out;
	    }
	}


	inline void thin_row_plus_random(const pmat_t&__restrict__ in,
					 pmat_t&__restrict__ out) {
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), 0, out.rows() - in.rows(), in.cols()).setRandom();
	}

	inline void thin_row_to_invertible(const pmat_t&__restrict__ in,
					   pmat_t&__restrict__ out) {
	    out.block(0, 0, in.rows(), in.cols()) = in;
	    out.block(in.rows(), 0, out.rows() - in.rows(), in.cols()).setRandom();
	    
	    // Gram-Schmidt orthogonalization
	    pdouble_t projection;

	    for (int iter = 0; iter < 2; ++iter) {
		for (int r = in.rows(); r < out.rows(); ++r) {
		    for (int r2 = 0; r2 < r; ++r2) {
			projection = out.row(r) * out.row(r2).transpose();
			out.row(r) -= projection * out.row(r2);
		    }
		    out.row(r).normalize();
		}
	    }
	}


	inline void thin_row_to_invertible_old(const pmat_t&__restrict__ in,
					       pmat_t&__restrict__ out, 
					       dqmc::workspace& ws) {
	    // out.zeros();
	    for (int c = 0; c < in.cols(); ++c) {
		for (int r = 0; r < in.rows(); ++r) {
		    ws.la_mat_1(r, c) = in(r,c);
		    out(r, c) = in(r, c);
		}
	    }
	    // return;
	    
	    for (int c = 0; c < out.cols(); ++c) {
		for (int r = in.rows(); r < out.rows(); ++r) {
		    ws.la_mat_1(r, c) = (std::rand() % RAND_MAX)/double(RAND_MAX);
		    if (r == c) {
			if (ws.la_mat_1(r, c) == 0) { ws.la_mat_1(r, c) = 1.;	}
			ws.la_mat_1(r, c) *= -1;
		    }
		}
	    }
		

	    // Gram-Schmidt orthogonalization
	    pdouble_t projection;
	    vec temp = ws.la_mat_1.row(0);

	    for (int iter = 0; iter < 5; ++iter) {
		for (int r = in.rows(); r < out.rows(); ++r) {
		    row(r, ws.la_mat_1, ws.la_vec_1);
		    // temp = ws.la_mat_1.row(r);
		    for (int r2 = 0; r2 < r; ++r2) {
			// projection = ws.la_mat_1.row(r) * out.row(r2).transpose();
			row(r2, out, ws.la_vec_2);
			dot(ws.la_vec_1, ws.la_vec_2, projection);
			ws.la_vec_1 -= projection * ws.la_vec_2;
			// ws.la_mat_1.row(r) -= projection * out.row(r2);
		    }
		    
		    // ws.la_mat_1.row(r) /= sqrt(ws.la_mat_1.row(r) * ws.la_mat_1.row(r).transpose());
		    // ws.la_mat_1.row(r).normalize();
		    normalize(ws.la_vec_1);
		    to_row(r, out, ws.la_vec_1);
		}
		ws.la_mat_1 = out;
	    }
	}


	inline pdouble_t trace(pmat_t& M) {
	    pdouble_t tr = 0.;
	    for (int i = 0; i < M.rows(); i++) {
		tr += M(i, i);
	    }
	    return tr;
	}

	inline void split_vec_sqrt(pvec_t&__restrict__ in, pvec_t&__restrict__ out) {
	    for (int i = 0; i < in.size(); ++i) {
		out(i) = sqrt(in(i));
	    }
	}

	inline bool all_below_eps(const pmat_t&__restrict M) {
	    for (int c = 0; c < M.cols(); ++c) {
		for (int r = 0; r < M.rows(); ++r) {
		    if (fabs(M(r, c)) > 1e-14) { return false; }
		}
	    }
	    return true;
	}

	inline void sum(const pvec_t&__restrict__ v, double& sum) {
	    sum = 0;
	    for (int i = 0; i < v.size(); ++i) {
		sum += v(i);
	    }}
	    
    }		    
}
#endif
