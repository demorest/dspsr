#!/bin/tcsh

set npt = 8
set done = 0

while ( ( $npt < 5000000 ) && ( $done ==  0 ) )

  set npt_pow2 = `echo $npt | awk '{print ($1 * 4)}'`

  set result = `./undersampling_speed -cuda -f $npt_pow2 -b $npt_pow2 -c 1 -p 1 -t 64|& tail -n 1`
  if ( $? == 0) then
    set gflops_pow2  = `echo $result | awk -F= '{print $NF}'`
  else
    set done = 1
  endif

  set npt_fwd = `echo $npt | awk '{print ($1 * 6)}'`
  set npt_bwd = `echo $npt | awk '{print ($1 * 5)}'`
  set result = `./undersampling_speed -cuda -f $npt_fwd -b $npt_bwd -c 1 -p 1 -t 64 |& tail -n 1`
  if ( $? == 0) then
    set gflops_over  = `echo $result | awk -F= '{print $NF}'`
  else
    set done = 1
  endif

  echo "npts pow2=$npt_pow2 -> gflops=$gflops_pow2  npt_fwd=$npt_fwd -> gflops=$gflops_over"

   @ npt = $npt * 2

end
