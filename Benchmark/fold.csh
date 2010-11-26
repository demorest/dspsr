#!/bin/csh -f

set nthread=8
set cache=1    # MB
set time = 60  # seconds

echo Time: $time seconds

set file=fold.time
rm -f $file

set psr="-E pulsar.par -P polyco.dat"
set args="-r -T $time $psr fold_header.dada"

foreach trial ( a b c d e f )

  set cmd="dspsr -t $nthread --minram=$cache $args"

  echo "trial ${trial}: $cmd"
  ( time $cmd ) >>& $file

  rm -f *.ar

end

set realtime=`grep -F % $file | grep -v Operation | tail -5 | awk '{print $3}' | awk -F: -vtot=0 'tot+=$1*60+$2 {print tot/5}' | tail -1`
echo $realtime | awk '{print "ratio " $1/'$time'}'

