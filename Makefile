CUDADIR      = /usr/lib/cuda
CC            = gcc
LDFLAGS       = -lm


# ----------------------------------------
# Flags and paths to MAGMA, CUDA, and LAPACK/BLAS
LSOLVER_LIB      	:= ./lpsolve/liblpsolve55.a -ldl
PCG_GEN						:= ./direction_creation/pcg-c-0.94/src/libpcg_random.a
NVCCFLAGS += -Xcompiler -no-pie

# ----------------------------------------
har:
	nvcc  har.c -o har.o  $(LDFLAGS) $(LSOLVER_LIB) $(PCG_GEN) $(NVCCFLAGS) -lcublas -lcusolver -lopenblas

clean:
	rm har.o
