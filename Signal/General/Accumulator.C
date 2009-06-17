/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "Error.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Accumulator.h"

//! Default constructor
dsp::Accumulator::Accumulator() : Transformation<TimeSeries,TimeSeries>("Accumulator",outofplace) {
  append = false;
  max_samps = 0;
  max_ndat = 0;
  never_drop_samples = false;
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
    if( verbose )
      fprintf(stderr,"accumulator resizing output to have "UI64" samps\n",
	      max_samps);
    if( max_ndat > max_samps )
      output->resize( max_ndat );
    else
      output->resize( max_samps );
    output->set_ndat(0);
    append = true;
  }

  if( get_output()->get_ndat() + get_input()->get_ndat() > get_output()->maximum_ndat() ){
    //    fprintf(stderr,"\nARRRGGHH dsp::Accumulator::transformation(): this append of "UI64" samps will bring output ndat from "UI64" to "UI64" samps but it only has room for "UI64" samps!\n\n",
    //    get_input()->get_ndat(),
    //    get_output()->get_ndat(),
    //    get_input()->get_ndat() + get_output()->get_ndat(),
    //    get_output()->maximum_ndat());
    
    if( never_drop_samples ){
      fprintf(stderr,"\nWARNING: dsp::Accumulator::transformation(): never_drop_samples=true so accomodating\n\n");
    
      Reference::To<dsp::TimeSeries> dummy(new dsp::TimeSeries);
      dummy->copy_configuration( get_output() ); 
      dummy->resize( get_output()->get_ndat() + get_input()->get_ndat() );
      dummy->set_ndat( 0 );
      dummy->append( get_output() );
      dummy->append( get_input() );

      get_output()->swap_data( *dummy );
      return;
    }
    
    int64_t new_ndat = int64_t(get_output()->maximum_ndat()) - int64_t(get_input()->get_ndat());
    if( new_ndat < 0 )
      new_ndat = 0;

    const_cast<TimeSeries*>(get_input())->set_ndat( uint64_t(new_ndat) );

    //throw Error(InvalidState,"dsp::Accumulator::transformation()",
    //	"This method throws an Error in this situation.  If this is a problem you may like to recode it.  If you are running 'reducer' you may wish to use --dumpsize_buffer to increase the capacity of your Accumulator, or your dumpsize may be a silly number");
  }

  get_output()->append( input );
}

//! Returns true if its time to write out the buffe
bool dsp::Accumulator::full(){
  if( !has_output() )
    throw Error(InvalidState,"dsp::Accumulator::full()",
		"Your output is null so inquiring whether or not it is full is not really a valid question");

  return get_output()->get_ndat() >= max_samps;
}

//! Returns true if buffer is within 'close_enough' samples of being full
bool dsp::Accumulator::nearly_full(uint64_t close_enough){
  if( !has_output() )
    throw Error(InvalidState,"dsp::Accumulator::nearly_full()",
                "Your output is null so inquiring whether or not it is nearly full is not really a valid question");

  return get_output()->get_ndat() + close_enough >= max_samps;
}
