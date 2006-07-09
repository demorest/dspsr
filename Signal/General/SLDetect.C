/***************************************************************************
 *
 *   Copyright (C) 2002 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>

#include <iostream>

#include "genutil.h"
#include "environ.h"

#include "dsp/TimeSeries.h"
#include "dsp/Operation.h"
#include "dsp/SLDetect.h"

dsp::SLDetect::SLDetect (Behaviour _type) 
  : Transformation <TimeSeries, TimeSeries> ("SLDetect", _type,true)
{ }

void dsp::SLDetect::transformation ()
{
  if( verbose ){
    fprintf(stderr,"\nIn %s::transformation() (%s)\n",
	    get_name().c_str(),(get_input()==get_output())?"inplace":"outofplace");
    fprintf(stderr,"In SLDetect::Operation input has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d\n",
	    input->get_ndim(), input->get_ndat(), input->get_npol(), input->get_nchan(),input->get_nbit());    
  }

  if( input->get_detected() ) {
    fprintf (stderr, "SLDetect::operate data is already detected!\n");
    throw_str ("SLDetect::operate invalid state");
  }

  if( input.get() != output.get() ){
    if(verbose)
      fprintf(stderr,"input.get()!=output.get()\n");

    /* Set output's capacity etc.
       Things that are different from input are state, subsize
       These can be set correctly by simply setting state to detected
       and resizing output */

    get_output()->copy_configuration( input );
    get_output()->set_npol( get_input()->get_npol() );
    get_output()->set_nchan( get_input()->get_nchan() );
    get_output()->set_ndim( 1 );

    if( input->get_npol()==2 )
      output->set_state( Signal::PPQQ );
    else if( input->get_npol()==1 ){
      fprintf(stderr,"dsp::SLDetect input had 1 polarisation- assuming this is total power over all polarisations\n"
	      "ie setting output->state to Signal::Intensity\n");
      output->set_state( Signal::Intensity );
    }
    else throw Error(InvalidState,"dsp::SLDetect::operate()",
		     "Input npol not one or two (It is %d)",
		     get_input()->get_npol());
    
    uint64 new_output_ndat = input->get_ndat();

    /* this will wipe existing data if space has not been preallocated */
    fprintf(stderr,"SLD: going to call output->resize("UI64")\n",
	    new_output_ndat);
    output->resize( new_output_ndat );
    fprintf(stderr,"SLD: out of call to output->resize("UI64")\n",
	    new_output_ndat);
  }

  for( unsigned ichan=0;ichan<input->get_nchan();ichan++){
    for( unsigned ipol=0;ipol<input->get_npol();ipol++){

      // increments values to be squared.
      register const float* in_ptr = input->get_datptr(ichan,ipol);
      register float* out_ptr = output->get_datptr(ichan,ipol);

      // When in_ptr=dend, we are on our last timesample to SLD
      register const float* dend = in_ptr + input->get_ndim()*input->get_ndat(); 

      /* See ~hknight/h_code/test2Aug02.C to see the test I
	 ran to find the quickest way of doing the squaring.
	 Not that this is the fastest though- but it's hopefully
	 less confusing.
      */
      if( input->get_state()==Signal::Nyquist ){
	while( in_ptr != dend ){
	  *out_ptr = *in_ptr * *in_ptr;
	  out_ptr++;
	  in_ptr++;
	} 
      }
      
      else if( input->get_state()==Signal::Analytic ){
	while( in_ptr!=dend ){
	  *out_ptr = *in_ptr * *in_ptr;  // Re*Re
	  in_ptr++;
	  
	  *out_ptr += *in_ptr * *in_ptr; // Add in Im*Im
	  in_ptr++;
	  out_ptr++;
	} 
      }
    }  // for each ipol
  }  // for each ichan

  // Set new state if output==input
  if( input->get_npol()==2 )
    output->set_state( Signal::PPQQ );
  else if( input->get_npol()==1 ){
    fprintf(stderr,"dsp::SLDetect input had 1 polarisation- assuming this is total power over all polarisations\n"
	    "ie setting output->state to Signal::Intensity\n");
    output->set_state( Signal::Intensity );
  }

  if( verbose )
    fprintf(stderr,"\n\nNear end of SLDetection, state is '%s'\n",
	    output->get_state_as_string().c_str());

  if(verbose)
    fprintf(stderr,"after sld output has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d state=%s\n",
	    output->get_ndim(), output->get_ndat(), output->get_npol(), output->get_nchan(),output->get_nbit(),
	    output->get_state_as_string().c_str());

  if( verbose )
    fprintf(stderr,"Exiting from %s::transformation()\n\n",get_name().c_str());
}

