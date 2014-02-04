#!/bin/csh
#PBS -V
#PBS -l nodes=1:gpus=2,walltime=01:00:00
#PBS -q sstar

cd $HOME/dspsr/Benchmark/examples/SKA1/Band4

../../../bench.csh --gpu=0,1 --freq=2805 --bw=10 --nchan=16 --nbw=1

