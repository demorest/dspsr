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

    exit 0

  endsw

end

\rm -f f*.time f*.dat

# Start with a bandwidth of 8 MHz at 375 MHz, 16 at 750, ...
# This starting bandwidth will be doubled at the end of each of the
# four loops in dspsr_bench.csh

set bw=8

foreach freq ( 375 750 1500 3000 )

  echo "s/FREQ/$freq/g" > template.sed
  echo "s/BW/$bw/g" >> template.sed
  echo "s/GPU/$gpu/g" >> template.sed
  echo "s|DIR|$PWD|g" >> template.sed

  sed -f template.sed template.csh > run${freq}.csh

  if ( "$queue" != "" ) then
    qsub -q fermi run${freq}.csh
  else
    source run${freq}.csh
  endif
 
  @ bw = $bw * 2

end

cat << EOF

When all of the benchmark scripts have completed, run

cd results
gnuplot ../plot.gnu

to produce dspsr_bench.eps

EOF

