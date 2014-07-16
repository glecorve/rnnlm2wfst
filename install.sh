#!/bin/bash
#
# Default installation of all components
# -> Read the README.md file for more details
#

cd `readlink -f $0`

# OpenFst
cd openfst-1.2.0
./configure --prefix=`pwd`
make
cd ..

# K-means
cd kmeans
make
cd ..

# RNNLM
cd rnnlm-0.2b
# Do not use USE_BLAS=1 if BLAS is not installed
make USE_BLAS=1
cd ..
