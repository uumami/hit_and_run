CUDADIR      = /usr/lib/cuda
CC            = gcc
LDFLAGS       = -lm


# ----------------------------------------
# Flags and paths to MAGMA, CUDA, and LAPACK/BLAS
LSOLVER_LIB      	:= ./lpsolve/liblpsolve55.a -ldl
NVCCFLAGS += -Xcompiler -no-pie

# ----------------------------------------
har:
	nvcc  har.c -o har.o  $(LDFLAGS) $(LSOLVER_LIB) $(NVCCFLAGS)

clean:
	rm har.o
