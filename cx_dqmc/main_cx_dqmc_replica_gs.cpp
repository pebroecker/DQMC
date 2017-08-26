#include <boost/mpi.hpp>
#include <libdqmc/global.hpp>
#include "cx_dqmc_replica_gs.hpp"
#include <alps/parapack/parapack.h>
#include <alps/osiris/comm.h>
#include <vector>
#include <iterator> 

PARAPACK_SET_VERSION("SU(2) Invariant Determinantal QMC Simulations for Ground State");
PARAPACK_REGISTER_ALGORITHM(cx_dqmc_replica_gs, "Complex DQMC");
PARAPACK_REGISTER_PARALLEL_ALGORITHM(cx_dqmc_replica_gs, "Complex DQMC");

int main(int argc, char** argv)
{
    global::use_mpi = false;
    global::flip_test = true;
    
    std::vector<std::string> args(argv,  argv + argc);

    boost::filesystem::path in_file(args[1]);    
    global::terminate_worker_file = std::string(in_file.parent_path().c_str());
	
    for (int i = 0; i < argc; i++) {
	if (std::string("--mpi").compare(std::string(argv[i])) == 0)
	    global::use_mpi = true;

	if (std::string("--nofliptest").compare(std::string(argv[i])) == 0) {
	    global::flip_test = false;
	    args.erase(args.begin() + i);
	}
    }

    argc = args.size();
    argv = new char*[args.size()];
    for(size_t i = 0; i < args.size(); i++){
	argv[i] = new char[args[i].size() + 1];
	strcpy(argv[i], args[i].c_str());
    }


#ifndef BOOST_NO_EXCEPTIONS
    try {
#endif
	// return alps::scheduler::start(argc,argv, hcrFactory());
	return alps::parapack::start(argc,argv);

#ifndef BOOST_NO_EXCEPTIONS
    }
    catch (std::exception& exc) {
	std::cerr << exc.what() << "\n";
	alps::comm_exit(true);
	return -1;
    }
    catch (...) {
	std::cerr << "Fatal Error: Unknown Exception!\n";
	return -2;
    }
#endif
}
