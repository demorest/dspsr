#include "Error.h"
#include "environ.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Chomper.h"

//! Default constructor- always inplace
dsp::Chomper::Chomper() : Transformation<TimeSeries,TimeSeries>("Chomper",inplace,true){
  new_ndat = 0;
  rounding = 1;
  use_new_ndat = false;
  multiplier = 0.0;
  dont_multiply = true;
}

//! Do stuff
void dsp::Chomper::transformation(){
  if( verbose )
    fprintf(stderr,"In dsp::Chomper::Transformation() with input ndat="UI64"\n",
	    get_input()->get_ndat());

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

  if( verbose )
    fprintf(stderr,"Returning from dsp::Chomper::Transformation() with output ndat="UI64"\n",
	    get_output()->get_ndat());
}
