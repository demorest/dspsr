
set style data linespoints
set style line 1 lw 1 pt 1 ps 1.5
set style line 2 lw 1 pt 2 ps 1.5
set style line 3 lw 1 pt 4 ps 1.5
set style line 4 lw 1 pt 6 ps 1.5

set pointsize 1.5

set xlabel "log_2(N_{FFT})"
set ylabel "Gflops"
set key off

set terminal postscript eps enhanced mono solid "helvetica" 25
set output "fft_bench.eps"

set label 1 "CUFFT" at graph 0.95,0.95 right

plot [1:23] '< grep "^4 " filterbank_bench.out' using 5:($6/1000) ls 1, '< grep "^32 " filterbank_bench.out' using 5:($6/1000) ls 2, '< grep "^256 " filterbank_bench.out' using 5:($6/1000) ls 3, '< grep "^2048 " filterbank_bench.out' using 5:($6/1000) ls 4

unset output


