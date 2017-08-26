SHELL := /bin/bash

include $(ALPS_ROOT)/share/alps/include.mk

ifeq ($(WHERE), thp)
	CPPFLGS = $(CPPFLAGS) -fno-strict-aliasing # -ggdb -O0 -march=pentium -DDEBUG -lqd 
	LDFLGS = $(LDFLAGS)
endif

ifeq ($(WHERE), cheops)
	CPPFLGS = $(CPPFLAGS) -O3 -restrict  -march=native -DCHEOPS
	LDFLGS = $(LDFLAGS) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
endif

ifeq ($(WHERE), jureca)
	CPPFLGS = $(CPPFLAGS) -O3 -restrict  -march=native -DCHEOPS
	LDFLGS = $(LDFLAGS) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
endif

PRODCPPFLGS = $(CPPFLGS) -O3 -ggdb  -fno-strict-aliasing  -ansi_alias -march=native -std=c++0x -DUSE_COMPLEX -DMKL_Complex8="std::complex<float>" -DMKL_Complex16="std::complex<double>" -DNDEBUG

IDFLGS = $(IDFLAGS) -I$(HOME)/.local/include -I/scratch/broecker/.local/include -I. -I.. -I$(ALPS_ROOT)/include

LDFLGS = $(LDFLAGS) -L$(HOME)/.local/lib -L/scratch/broecker/.local/lib

all:	cx_dqmc_stable

global.o: ../libdqmc/global.cpp
	$(CXX) $(PRODCPPFLGS)  $(IDFLGS) -c ../libdqmc/global.cpp -o global.o

main_cx_dqmc.o: main_cx_dqmc.cpp
	$(CXX) $(PRODCPPFLGS)  $(IDFLGS) -c main_cx_dqmc.cpp -o main_cx_dqmc.o

cx_dqmc_greens_general_renyi.o: ../libdqmc/cx_dqmc_greens_general_renyi.cpp ../libdqmc/cx_dqmc_greens_general_renyi.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c ../libdqmc/cx_dqmc_greens_general_renyi.cpp

cx_dqmc_greens_general.o: ../libdqmc/cx_dqmc_greens_general.cpp ../libdqmc/cx_dqmc_greens_general.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c ../libdqmc/cx_dqmc_greens_general.cpp

cx_dqmc.o: cx_dqmc.cpp cx_dqmc.hpp ../libdqmc/cx_updates.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c cx_dqmc.cpp $(COMFLAG)

cx_dqmc: cx_dqmc.o cx_dqmc_greens_general.o cx_dqmc_greens_general_renyi.o main_cx_dqmc.o global.o
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -o $@ cx_dqmc.o global.o main_cx_dqmc.o cx_dqmc_greens_general_renyi.o cx_dqmc_greens_general.o $(LDFLGS) $(LIBS)

main_cx_dqmc_replica_gs.o: main_cx_dqmc_replica_gs.cpp
	$(CXX) $(PRODCPPFLGS)  $(IDFLGS) -c main_cx_dqmc_replica_gs.cpp -o main_cx_dqmc_replica_gs.o

cx_dqmc_greens_replica_gs.o: ../libdqmc/cx_dqmc_greens_replica_gs.cpp ../libdqmc/cx_dqmc_greens_replica_gs.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c ../libdqmc/cx_dqmc_greens_replica_gs.cpp

cx_dqmc_greens_replica_renyi_gs.o: ../libdqmc/cx_dqmc_greens_replica_renyi_gs.cpp ../libdqmc/cx_dqmc_greens_replica_renyi_gs.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c ../libdqmc/cx_dqmc_greens_replica_renyi_gs.cpp

cx_dqmc_replica_gs.o: cx_dqmc_replica_gs.cpp cx_dqmc_replica_gs.hpp ../libdqmc/cx_updates.hpp
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -c cx_dqmc_replica_gs.cpp $(COMFLAG)

cx_dqmc_stable: cx_dqmc_replica_gs.o cx_dqmc_greens_replica_gs.o cx_dqmc_greens_replica_renyi_gs.o main_cx_dqmc_replica_gs.o global.o 
	$(CXX) $(PRODCPPFLGS) $(IDFLGS) -o $@ cx_dqmc_replica_gs.o global.o main_cx_dqmc_replica_gs.o cx_dqmc_greens_replica_gs.o cx_dqmc_greens_replica_renyi_gs.o $(LDFLGS) $(LIBS)



clean: 
	$(RM) $(RMFLAGS) *.o
