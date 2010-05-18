#!/bin/csh -f

if ( "$1" == "" ) then
  echo "must set frequency and starting bandwidth"
  exit
endif

if ( "$2" == "" ) then
  echo "must set starting bandwidth"
  exit
endif

set freq=$1
set bw=$2

foreach bwtrial ( 1 2 3 4 )

  set outfile=f${freq}_b${bw}.dat
  rm -f $outfile

  @ time = 1024 / $bw

  foreach DM ( 1 3 10 30 100 300 1000 )

    echo Frequency: $freq Bandwidth: $bw DM: $DM

    set infile=f${freq}_b${bw}_DM${DM}.time

    set preptime=`grep prepared $infile | tail -5 | awk -vtot=0 'tot+=$4 {print tot/5}' | tail -1`

    set realtime=`grep -F % $infile | grep -v Operation | tail -5 | awk '{print $3}' | awk -F: -vtot=0 'tot+=$1*60+$2 {print tot/5}' | tail -1`

    echo "$DM $realtime $preptime" | awk '{print $1, ($2-$3)/'$time'}' >> $outfile

  end

  @ bw = $bw * 2

end

