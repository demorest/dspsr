#include "Error.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Accumulator.h"

//! Default constructor
dsp::Accumulator::Accumulator() : Transformation<TimeSeries,TimeSeries>("Accumulator",outofplace) {
  append = false;
  max_samps = 0;
}

//! Virtual destructor
dsp::Accumulator::~Accumulator(){ }

//! Reset the output
void dsp::Accumulator::reset(){
  append = false;
  if( has_output() )
    output->set_ndat(0);
}

//! Do the work
void dsp::Accumulator::transformation(){
  if( !append ){
    if( max_samps==0 )
      throw Error(InvalidParam,"dsp::Accumulator::transformation()",
		  "max_samps=0.  You forgot to set it properly");
    output->Observation::operator=( *input );
    fprintf(stderr,"accumulator resizing output to have "UI64" samps\n",
	    max_samps);
    output->resize( max_samps );
    output->set_ndat(0);
    append = true;
  }

  output->append( input );
}

