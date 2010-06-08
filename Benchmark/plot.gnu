
set terminal postscript eps enhanced mono solid "helvetica" 25
set output "dspsr_bench.eps"

set size 1,3.2

set multiplot

set style data linespoints
set style line 1 lw 1 pt 8 ps 1.5
set style line 2 lw 1 pt 2 ps 1.5
set style line 3 lw 1 pt 4 ps 1.5
set style line 4 lw 1 pt 6 ps 1.5

set pointsize 1.5

set log x

set size 1,0.7
set origin 0,0.3
set tmargin 0
set lmargin 10
set bmargin 0

set xlabel "Dispersion Measure (DM)"
set key off

set label 1 "375 MHz" at graph 0.05,0.90 left

plot "freq375/f375_b8.dat" ls 1, "freq375/f375_b16.dat" ls 2, "freq375/f375_b32.dat" ls 3, "freq375/f375_b64.dat" ls 4

set format x ""
set xlabel ""
set ylabel "                      Processing time / Real time"

set origin 0,1.0
set bmargin 0

set label 1 "750 MHz" at graph 0.05,0.90 left

plot "freq750/f750_b16.dat" ls 1, "freq750/f750_b32.dat" ls 2, "freq750/f750_b64.dat" ls 3, "freq750/f750_b128.dat" ls 4

set ylabel ""
set origin 0,1.7

set label 1 "1.5 GHz" at graph 0.05,0.90 left

plot "freq1500/f1500_b32.dat" ls 1, "freq1500/f1500_b64.dat" ls 2, "freq1500/f1500_b128.dat" ls 3, "freq1500/f1500_b256.dat" ls 4

set origin 0,2.4

set label 1 "3 GHz" at graph 0.05,0.90 left

plot "freq3000/f3000_b64.dat" ls 1, "freq3000/f3000_b128.dat" ls 2, "freq3000/f3000_b256.dat" ls 3, "freq3000/f3000_b512.dat" ls 4

set nomultiplot
unset output


