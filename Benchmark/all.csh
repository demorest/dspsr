#!/bin/csh -f

rm -f *.time
rm -f *.dat

# Start with a bandwidth of 8 MHz at 375 MHz, 16 at 750, ...
# This starting bandwidth will be doubled at the end of each of the
# four loops in dspsr_bench.csh

set bw=8

foreach freq ( 375 750 1500 3000 )

  ./bench.csh $freq $bw
  ./report.csh $freq $bw

  mkdir -p results/freq$freq/
  mv *.time *.dat results/freq$freq/

  @ bw = $bw * 2

end

cd results
gnuplot ../plot.gnu

echo Benchmark completed.
echo Results in results/

