#ifndef PRINT_HPP
#define PRINT_HPP

#include <iostream>
#include <iomanip>

namespace dqmc {
    namespace la {

        template<typename T>
        inline void print_matrix(const int rows, const int cols, const T& M,
				 int width = 11, int precision = 3) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    std::cout << std::right << std::setw(width)
                              << std::setprecision(precision) 
			      << std::scientific << M(i, j);
                }
                std::cout << std:: endl;
            }
	    std::cout << std::endl << std::endl;
        }       

        template<typename T>
        inline void print_matrix_threshold(const int rows, const int cols, const T& M,
				 int width = 11, int precision = 3) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
		    if (fabs(M(i, j)) > 1e-14) {
			std::cout << std::right << std::setw(width)
				  << std::setprecision(precision) 
				  << std::scientific << M(i, j);
		    } else {
			std::cout << std::right << std::setw(width) << 0;
		    }
                }
                std::cout << std:: endl;
            }
	    std::cout << std::endl << std::endl;
        }       

        template<typename T>
        inline void print_vector(const int cols, const T& M,
                                int width = 11, int precision = 3) {
	    for (int j = 0; j < cols; j++) {
		std::cout << std::right << std::setw(width)
			  << std::setprecision(precision) 
			  << std::scientific << M(j);
	    }
	    std::cout << std::endl << std::endl; 
        }       

        template<typename T>
        inline void print_matrix(const T& M, int width = 11, int precision = 3) {
            for (int i = 0; i < M.rows(); i++) {
                for (int j = 0; j < M.cols(); j++) {
                    std::cout << std::right << std::setw(width)
                              << std::setprecision(precision) 
			      << std::scientific << M(i, j);
                }
                std::cout << std:: endl;
            }
	    std::cout << std::endl << std::endl;
        }       


        template<typename T>
        inline void print_matrix_threshold(const T& M, double threshold=1e-12, int width = 9, int precision = 1) {
            for (int i = 0; i < M.rows(); i++) {
                for (int j = 0; j < M.cols(); j++) {
		    if (fabs(M(i, j)) > threshold) {
			std::cout << std::right << std::setw(width)
				  << std::setprecision(precision) 
				  << std::scientific << M(i, j);
		    } else {
			std::cout << std::right << std::setw(width)
				  << std::setprecision(precision) 
				  << std::scientific << 0;
		    }
                }
                std::cout << std:: endl;
            }
	    std::cout << std::endl << std::endl;
        }       

        template<typename T>
        inline void print_diag(const T& M, int width = 11, int precision = 3) {
            for (int i = 0; i < M.rows(); i++) {
		std::cout << std::right << std::setw(width)
			  << std::setprecision(precision) 
			  << std::scientific << M(i, i);
	    }
	    std::cout << std::endl;
        }       


        template<typename T>
        inline void print_matrix(const std::string& name,
				 const T& M, int width = 11, int precision = 3) {
	    std::cout << name << std::endl << std::endl;
	    print_matrix(M, width, precision);
        }       


        template<typename T>
        inline void print_matrix_threshold(const std::string& name,
					   const T& M, double threshold=1e-12, int width = 9, int precision = 1) {
	    std::cout << name << std::endl << std::endl;
	    print_matrix_threshold(M, threshold, width, precision);
        }       

        template<typename T>
        inline void print_vector(const T& M, int width = 11, int precision = 3) {
	    for (int j = 0; j < M.size(); j++) {
		std::cout << std::right << std::setw(width)
			  << std::setprecision(precision) 
			  << std::scientific << M(j);
	    }
	    std::cout << std::endl << std::endl; 
        }       

        template<typename T>
        inline void print_vector(const std::string& name,
				 const T& M, int width = 11, int precision = 3) {
	    std::cout << name << std::endl << std::endl;
	    print_vector(M, width, precision);
        }       
    }
}
#endif
