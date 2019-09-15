 #!/bin/bash
cd /app/direction_creation/pcg-c-0.94/
make
cd /app
apt-get update 
apt-get upgrade
apt-get install -y gfortran
apt-get install - y python
cd /app/magma-2.5.0
make
cd /app
 . ./p_paths.sh
 make clean
 make 
 ./har.o
