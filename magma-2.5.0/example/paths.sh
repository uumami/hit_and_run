 #!/bin/bash
 export MAGMADIR=/home/mario/Documents/github/hit_and_run/magma-2.5.0
 export CUDADIR=/usr/local/cuda
 export OPENBLASDIR=/usr/lib/x86_64-linux-gnu/openblas
 export LD_LIBRARY_PATH=$MAGMADIR/lib:$CUDADIR/lib64:$OPENBLASDIR/lib
