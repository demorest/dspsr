
set terminal postscript eps enhanced mono solid "helvetica" 25

set key off
set xlabel "log_2 (N_{beam})"
set ylabel "Processing time / Real time"

set output "fold_bench.eps"
plot "fold.dat" using (log($1)/log(2)):($4*$1) w l
unset output

