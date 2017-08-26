#ifndef CX_DQMC_COMPARE_HOPPING_HPP
#define CX_DQMC_COMPARE_HOPPING_HPP

#include "la.hpp"
#include "cx_checkerboard.hpp"
#include <exception>

namespace cx_dqmc {
    namespace checks {

	inline void compare_hopping_checkerboard(dqmc::parameters& p, cx_dqmc::workspace& ws,
						 mat_t& hopping, mat_t& hopping_inv,
						 cx_mat_t& mat_1, cx_mat_t& mat_2) {
	    using namespace std;
	    double diff_left, diff_right, diff_left_inv, diff_right_inv;

	    mat_1.setIdentity();
	    mat_2 = hopping.cast<std::complex<double> >();
	    cx_dqmc::checkerboard::hop_left(&ws, mat_1, 1);
	    diff_left = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right(&ws, mat_1, 1);
	    diff_right = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    mat_2 = hopping_inv.cast<std::complex<double> >();	   
	    cx_dqmc::checkerboard::hop_left(&ws, mat_1, -1);
	    diff_left_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right(&ws, mat_1, -1);
	    diff_right_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();
	    cout << p.outp << "Maxima: " << diff_left
		 << " - " << diff_right
		 << " - " << diff_left_inv
		 << " - " << diff_right_inv
		 << endl;
	}

	inline void compare_hopping_checkerboard(dqmc::parameters& p,
						 cx_dqmc::workspace& ws,
						 cx_mat_t& hopping,
						 cx_mat_t& hopping_inv,
						 cx_mat_t& mat_1, cx_mat_t& mat_2) {
	    using namespace std;
	    double diff_left, diff_right, diff_left_inv, diff_right_inv;

	    mat_1.setIdentity();
	    mat_2 = hopping;
	    cx_dqmc::checkerboard::hop_left(&ws, mat_1, 1);
	    diff_left = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right(&ws, mat_1, 1);
	    diff_right = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    mat_2 = hopping_inv;
	    cx_dqmc::checkerboard::hop_left(&ws, mat_1, -1);
	    diff_left_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right(&ws, mat_1, -1);
	    diff_right_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();
	    cout << p.outp << "Maxima: " << diff_left
		 << " - " << diff_right
		 << " - " << diff_left_inv
		 << " - " << diff_right_inv
		 << endl;
	}




	inline void compare_hopping_checkerboard_renyi(dqmc::parameters& p, cx_dqmc::workspace& ws,
						       mat_t& hopping, mat_t& hopping_inv,
						       cx_mat_t& mat_1, cx_mat_t& mat_2) {
	    using namespace std;
	    double diff_left, diff_right, diff_left_inv, diff_right_inv;

	    mat_1.setIdentity();
	    mat_2 = hopping.cast<std::complex<double> >();
	    cx_dqmc::checkerboard::hop_left_renyi(&ws, mat_1, 1, 2);
	    diff_left = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right_renyi(&ws, mat_1, 1, 2);
	    diff_right = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    mat_2 = hopping_inv.cast<std::complex<double> >();
	    cx_dqmc::checkerboard::hop_left_renyi(&ws, mat_1, -1, 2);
	    diff_left_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();

	    mat_1.setIdentity();
	    cx_dqmc::checkerboard::hop_right_renyi(&ws, mat_1, -1, 2);
	    diff_right_inv = (mat_1 - mat_2).cwiseAbs().maxCoeff();
	    cout << p.outp << "Maxima Renyi: " << diff_left
		 << " - " << diff_right
		 << " - " << diff_left_inv
		 << " - " << diff_right_inv
		 << endl;
	}
    }
}
#endif
