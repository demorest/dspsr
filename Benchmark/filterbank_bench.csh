#!/bin/sh

nfft_min=4
nfft_max=16777216

nchan_min=4
nchan_max=4096

rm -f filterbank_bench.out

nchan=$nchan_min

while test $nchan -le $nchan_max; do

  nfft=$nfft_min
  effective_nfft_max=`expr $nfft_max '/' $nchan`

  while test $nfft -le $effective_nfft_max; do

    echo -n "Testing nchan=$nchan nfft=$nfft "
  
    time ../Signal/General/filterbank_speed -c$nchan -n$nfft -cuda >> filterbank_bench.out

    nfft=`expr $nfft '*' 2`

  done

  nchan=`expr $nchan '*' 2`

  echo "" >> filterbank_bench.out

done

