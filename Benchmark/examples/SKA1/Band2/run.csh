#!/bin/csh
#PBS -V
#PBS -l nodes=1:gpus=2,walltime=01:00:00
#PBS -q sstar

cd $HOME/dspsr/Benchmark/examples/SKA1/Band2

../../../bench.csh --gpu=0,1 --hdr=header.dada --nchan=1296 --nbw=1

