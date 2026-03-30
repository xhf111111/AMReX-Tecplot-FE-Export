PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2
#DIM       = 3

COMP       = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_CUDA   = FALSE
USE_PARTICLES = FALSE
USE_BITTREE = FALSE

Bpack   := ./Make.package
Blocs   := .

TEC_HOME = /home/Tecplot/tecplot/360ex_2025r2

DEFINES += -DUSE_TEC_BIN_IO
INCLUDE_LOCATIONS += -I$(TEC_HOME)/include
LIBRARY_LOCATIONS += -L$(TEC_HOME)/bin
LIBRARIES += -ltecio

include Make.Adv
