// #define EIGEN_NO_DEBUG
// #ifdef CHEOPS
// #define EIGEN_NO_DEBUG
// #endif
//#define EIGEN_USE_LAPACKE

#ifndef LA_TYPES_HPP
#define LA_TYPES_HPP
#include <complex>
#include <mkl.h>
#define EIGEN_USE_MKL_BLAS

//#define MKL_Complex16 std::complex<double>

#ifdef USE_DD
#include <qd/dd_real.h>
#include <qd/c_dd.h>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/multi_array.hpp>
#include <utility>

typedef std::pair<double, int> max_pair;
typedef double pdouble_t;

#ifdef USE_DD
typedef dd_real ddouble;
typedef Eigen::Matrix<ddouble, Eigen::Dynamic, Eigen::Dynamic> dd_mat;
typedef Eigen::Matrix<ddouble, Eigen::Dynamic, 1> dd_vec;
#endif

//typedef std::complex<double> cx_double;
typedef MKL_Complex16 cx_double;
typedef MKL_Complex16 cx_double_t;

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> doub_mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> doub_vec;
typedef Eigen::MatrixXd pmat_t;
typedef Eigen::MatrixXd mat;
typedef Eigen::MatrixXd mat_t;
typedef Eigen::Matrix<cx_double, Eigen::Dynamic, Eigen::Dynamic> cx_mat;
typedef Eigen::Matrix<cx_double, Eigen::Dynamic, Eigen::Dynamic> cx_mat_t;
typedef Eigen::MatrixXi imat;
typedef Eigen::VectorXd pvec_t;
typedef Eigen::VectorXd vec;
typedef Eigen::VectorXd vec_t;
typedef Eigen::Matrix<cx_double, Eigen::Dynamic, 1> cx_vec;
typedef Eigen::Matrix<cx_double, Eigen::Dynamic, 1> cx_vec_t;
typedef Eigen::VectorXi ivec;
typedef Eigen::SparseMatrix<double> sp_mat;
typedef Eigen::SparseMatrix<cx_double> cx_sp_mat;

typedef boost::multi_array<int, 2> iarr_t;
typedef boost::multi_array<double, 2> darr_t;
#endif
