# hit_and_run
Hit and Run algorithm for a regular polytope, using GPGPU. (MAGMA, LPSOLVER, PCG-c).

Most Investigation:
- Change the spath.sh in order to reflect the correct paths of the MAGMA libraries in the docker.
- Add a tutorial
- Fix the bug in the Chebyshev Routine (as of now is hardcoded to be able to replicate it)

Possible to do:
+ Modify the algorithm to support sparse matrices.
+ Add Strassen GPU
+ Add prototype functions to the headers to ease exposition

Nice to have:
+ Manage MAGAM threads via CUDA kernels in order to drastically reduce HOST-DEVICE communication.



Instructions, No Docker:
+ Clone the repository
+ Modify the .txt files in the input_matrices directory (columns separated by spaces, and rows by \n)
+ You need CUDA 6.0 or higher in order to run the routines so install it.
+ Modify the Makefile to refrect the correct paths for CUDA binaries and compilers in your computer.
+ Modify paths.sh to refeclec the correct directories of the libraries specified in the same file.
+ RUN:
  $ make
  $ ./har.o
+ Have Fun
