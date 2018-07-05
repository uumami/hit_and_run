# hit_and_run
Hit and Run algorithm for a regular polytope, using GPGPU. (MAGMA, MKL, CUBLAS).
This branch is version 0.1:
Possible mods:
  + Improve Chebyshev center algortihm (gpu, numerically)
  + Eliminate the projection b: So d will sum zero.
  + Leave the projection matrix loaded in the GPU
  + Find the fastest method for generating a Normal Sampling of n (current method Box-Muller)
  + Find positive and negative lambda for the chord inside the polytope
  + Use batch matrices.

Code cleaning:
  + Unfied code
  + Clearer code
  + Look for redundancy
  
