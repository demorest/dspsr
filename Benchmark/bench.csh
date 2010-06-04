#!/bin/csh -f

set nthread=8
set cache=1
set gpu=""

if ( "$1" == "" ) then
  echo "USAGE: bench.csh <freq> <bw> [gpu]"
  echo "where: <freq> is the centre frequency"
  echo "       <bw> is the starting bandwidth"
  echo "   and [gpu] is the optional GPU configuration"
  exit
endif

if ( "$2" == "" ) then
  echo "must set starting bandwidth"
  exit
endif

if ( "$3" != "" ) then
  set gpu=$3
endif

if ( "$gpu" != "" ) then
  echo using GPUs: $gpu
else
  echo using $nthread CPU threads
  echo using at least $cache MB of cache
endif

set freq=$1
set bw=$2

foreach bwtrial ( 1 2 3 4 )

  @ nchan = $bw * 2
  @ time = 1024 / $bw

  foreach DM ( 1 3 10 30 100 300 1000 )

    echo Frequency: $freq Bandwidth: $bw Time: $time DM: $DM
    set file=f${freq}_b${bw}_DM${DM}.time
    rm -f $file

    set psr="-E pulsar.par -P polyco.dat -D $DM -B $bw -f $freq"
    set args="--fft-bench -r -F ${nchan}:D -T $time $psr header.dada"

    foreach trial ( a b c d e f )

      if ( "$gpu" != "" ) then
        set cmd="dspsr --cuda=$gpu $args"
      else
        set cmd="dspsr -t $nthread --minram=$cache $args"
      endif

      echo "trial ${trial}: $cmd"
      ( time $cmd ) >>& $file

      rm -f *.ar

    end

  end

  @ bw = $bw * 2

end

