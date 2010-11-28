#!/bin/csh -f

set max_nthread=32
set cache=1.5    # MB
set time=60  # seconds

echo Time: $time seconds

set psr="-E vela.par -P vela.polyco"
set args="-r -T $time $psr fold_header.dada"

set nthread=1

set result="fold.dat"
rm -f $result

while ( $nthread <= $max_nthread )

set file="fold_times.$nthread"
rm -f $file

foreach trial ( a b c d e f )

  set cmd="dspsr -t $nthread --minram=$cache $args"

  echo "nthread=$nthread trial ${trial}: $cmd"
  ( time $cmd ) >>& $file

  rm -f *.ar

end

set realtime=`grep -F % $file | grep -v Operation | tail -5 | awk '{print $3}' | awk -F: -vtot=0 'tot+=$1*60+$2 {print tot/5}' | tail -1`

echo $realtime $nthread | awk '{print $2, $1 " ratio " $1/'$time'}' >> $result

@ nthread = $nthread * 2

end

