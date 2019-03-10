# hit_and_run
Hit and Run algorithm for a regular polytope, using GPGPU. (MAGMA, MKL, CUBLAS
LINPACK, LPSOLVER).

In this branch we take many direction at the same time ad project them to the
hyperplane. THus, generating multiple random walks while taking advantage of
the matrix-to-matrix multiplication vs matrix-to-vector. Given the thinning
factor of n, this improves the algorithm. 

To do Investigation:
- Compare Strassen multiplication in the gpu against CUBLAS routines, 
and find the sweetspot between them.
- Find the best algorithm to find an interiror point in the polytope, as for now
Chebysehv center is the winner.
- Find a suitable pseudorandom geenrator for C (ask Gary)

Possible to do:
+ Modify the algorithm to support sparse matrices.
+ Use SCALAPACK to make the algorithm distributed.

To do Code: (REVAMP the old code)
+ Pseudocode
+ Structure
+ Read Matrices (probably in row major, instead of column major)
+ Find Inner Point (Chebyshev center)
+ Calculate projection Matrix
+ Generate direction (n2 directions ~Normal iid) Mersenne Twister
+ Project d
+ Find lambdas (FIND LINE SEGMENT)
+ Take random point from the line segment

To do code investigation:
+ It may be better to extract the xs every n iterations, and thus reducing the
device-host comunication. I may need to use some CUDA directives.
