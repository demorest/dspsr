#!/bin/csh -f

set nthread=2
set cache=1
set gpu=""

set freq=""
set bw=""

@ nbwtrial = 4
@ nchan = 0

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

  case nbw:
    set nbwtrial="$optarg"
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
  --bw=MHz           starting bandwidth
  --nbw=N            number of bandwidth benchmarks (default N=4)
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

@ bwtrial = 0

while ( $bwtrial < $nbwtrial )

  if ( $nchan == 0 ) then
    # ensure that nchan is a power of two
    @ nchan = 2
    while ( $nchan <= $bw )
      @ nchan = $nchan * 2
    end
  endif

  @ time = 1024 / $bw

  @ bwtrial = $bwtrial + 1

  set outfile=f${freq}_b${bw}.dat
  rm -f $outfile

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

    set preptime=`grep prepared $file | tail -5 | awk 'tot+=$4 {print tot/5}' | tail -1`

    set realtime=`grep -F % $file | grep -v Operation | tail -5 | awk '{print $3}' | awk -F: 'tot+=$1*60+$2 {print tot/5}' | tail -1`

    echo "$DM $realtime $preptime" | awk '{print $1, ($2-$3)/'$time'}' >> $outfile

  end

  @ bw = $bw * 2

end

