#!/bin/csh -f

source $HOME/.cshrc

cd DIR

./bench.csh --freq=FREQ --bw=BW --gpu=GPU

mkdir -p results/freqFREQ/
mv f*.time f*.dat results/freqFREQ

