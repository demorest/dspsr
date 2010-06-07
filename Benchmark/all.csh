#!/bin/csh -f

set gpu=""
set queue=""

foreach option ( $* )

  switch ( $option )

  case -*=*:
    set arg=`echo "$option" | awk -F= '{print $1}' | sed -e 's/-//g'`
    set optarg=`echo "$option" | awk -F= '{print $2}'`
    breaksw

  case *:
    set arg=$option
    set optarg=
    breaksw

  endsw

  ##########################################################################
  #
  ##########################################################################

  switch ( $arg )

  case gpu:
    set gpu="$optarg"
    breaksw

  case queue:
    set queue="$optarg"
    breaksw

  case *:
    cat <<EOF

Run the dspsr benchmark presented in van Straten & Bailes (2010)

Usage: all.csh [OPTION]

Known values for OPTION are:

  --gpu=device[s]    comma-separated list of CUDA devices
  --queue=name       name of batch queue on which jobs are run

EOF
    breaksw

  endsw

end

\rm -f f*.time f*.dat

# Start with a bandwidth of 8 MHz at 375 MHz, 16 at 750, ...
# This starting bandwidth will be doubled at the end of each of the
# four loops in dspsr_bench.csh

set bw=8

foreach freq ( 375 750 1500 3000 )

  ./bench.csh --freq=$freq --bw=$bw --gpu$gpu
  ./report.csh $freq $bw

  mkdir -p results/freq$freq/
  mv f*.time f*.dat results/freq$freq/

  @ bw = $bw * 2

end

cd results
gnuplot ../plot.gnu

echo Benchmark completed.
echo Results in results/

