DEBUG = TRUE

USE_MPI  = TRUE
USE_OMP  = FALSE

USE_HYPRE = FALSE
USE_PETSC = FALSE

COMP = gnu

DIM = 3

AMREX_HOME ?= /Users/equon/amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs 	:= Base Boundary AmrCore LinearSolvers/MLMG

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

