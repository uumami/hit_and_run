 #!/bin/bash
 export MAGMADIR=/app/magma-2.5.0
 export CUDADIR=/usr/local/cuda
 export OPENBLASDIR=/usr/lib/x86_64-linux-gnu/openblas
 export LD_LIBRARY_PATH=$MAGMADIR/lib:$CUDADIR/lib64:$OPENBLASDIR/lib
