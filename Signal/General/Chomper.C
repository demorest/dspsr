#include "Error.h"
#include "environ.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Chomper.h"

dsp::Chomper::Chomper() : Transformation<TimeSeries,TimeSeries>("Chomper",inplace){
  new_ndat = 0;
  rounding = 1;
  use_new_ndat = false;
  multiplier = 0.0;
  dont_multiply = true;
}

void dsp::Chomper::transformation(){
  if( use_new_ndat ){
    if( get_input()->get_ndat() < new_ndat )
      throw Error(InvalidParam,"dsp::Chomper::transformation()",
		  "You wanted to chomp the timeseries to ndat="UI64" but it only had "UI64" points!",
		  new_ndat, get_input()->get_ndat());
    get_input()->set_ndat( new_ndat );
  }

  get_input()->set_ndat( get_input()->get_ndat() - get_input()->get_ndat() % rounding );

  if( !dont_multiply )
    get_input()->operator*=( multiplier );
}
