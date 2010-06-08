#!/bin/csh -f

set nthread=8
set cache=1
set gpu=""

set freq=""
set bw=""

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

  case nthread:
    set nthread="$optarg"
    breaksw

  case cache:
    set cache="$optarg"
    breaksw

  case freq:
    set freq="$optarg"
    breaksw

  case bw:
    set bw="$optarg"
    breaksw

  case *:
    cat <<EOF

Run the dspsr benchmark presented in van Straten & Bailes (2010)

Usage: all.csh [OPTION]

Known values for OPTION are:

  --gpu=device[s]    comma-separated list of CUDA devices
  --nthread=N        number of CPU threads
  --cache=MB         minimum CPU memory to use

  --freq=MHz         centre frequency
  --bw=MHz           bandwidth
  --nchan=N          number of channels in filterbank

EOF
    breaksw

  endsw

end

if ( "$freq" == "" ) then
  echo "please set the centre frequency with --freq"
  exit
endif

if ( "$bw" == "" ) then
  echo "please set the bandwidth with --bw"
  exit
endif

if ( "$gpu" != "" ) then
  echo using GPUs: $gpu
else
  echo using $nthread CPU threads
  echo using at least $cache MB of cache
endif

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
        set cmd="dspsr --cuda=$gpu --minram=256 $args"
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

