#!/bin/csh -f

set nthread=8
set cache=1
set gpu=0

if ( "$1" == "" ) then
  echo "USAGE: bench.csh <freq> <bw> [ngpu]"
  echo "where: <freq> is the centre frequency"
  echo "       <bw> is the starting bandwidth"
  echo "   and [ngpu] is the optional number of GPUs to use"
  exit
endif

if ( "$2" == "" ) then
  echo "must set starting bandwidth"
  exit
endif

if ( "$3" != "" ) then
  set gpu=$3
endif

if ( $gpu ) then
  echo using $gpu GPUs
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

    set args="-r -F ${nchan}:D -T $time -B $bw -f $freq -D $DM header.dada"

    foreach trial ( a b c d e f )

      echo trial $trial

      if ( $gpu ) then
        ( time dspsr --cuda=$gpu -t$gpu $args ) >>& $file
      else
        ( time dspsr --fft-bench -t$nthread --minram=$cache $args ) >>& $file
      endif

    end

  end

  @ bw = $bw * 2

end

