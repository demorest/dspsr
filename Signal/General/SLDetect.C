#include <stdio.h>

#include <iostream>

#include "genutil.h"

#include "dsp/SLDetect.h"

dsp::SLDetect::SLDetect () 
  : Transformation <TimeSeries, TimeSeries> ("SLDetect", anyplace)
{
}

void dsp::SLDetect::transformation ()
{
  if( verbose ){
    fprintf(stderr,"\nIn %s::transformation()\n",get_name().c_str());
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
    output->Observation::operator=( *input );
    if( input->get_npol()==2 )
      output->set_state( Signal::PPQQ );
    else if( input->get_npol()==1 ){
      fprintf(stderr,"dsp::SLDetect input had 1 polarisation- assuming this is total power over all polarisations\n"
	      "ie setting output->state to Signal::Intensity\n");
      output->set_state( Signal::Intensity );
    }
    output->resize( input->get_ndat() );
  }

  // increments values to be squared.
  register float* in_ptr = const_cast<float*>(input->get_datptr(0,0));
  register float* out_ptr = output->get_datptr(0,0);
  // When in_ptr=dend, we are on our last timesample to SLD
  register float* dend = in_ptr + input->get_npol()*input->get_nchan()*input->get_ndim()*input->get_ndat(); 

  if(verbose)
    fprintf(stderr,"For calculating dend npol=%d nchan=%d ndim=%d ndat="I64"\n",
	    input->get_npol(),input->get_nchan(),input->get_ndim(),input->get_ndat());

  if(verbose){
    cerr << "in_ptr="<<in_ptr<<"\n";
    cerr << "out_ptr="<<out_ptr<<"\n";
    cerr << "dend="<<dend<<"\n";
    cerr << "dend-in_ptr="<<dend-in_ptr<<"\n";
  }
  else if(verbose)
    fprintf(stderr,"input.get()==output.get()\n");
  /* See ~hknight/h_code/test2Aug02.C to see the test I
     ran to find the quickest way of doing the squaring.
     Not that this is the fastest though- but it's hopefully
     less confusing.
  */
  
  if( input->get_state()==Signal::Nyquist ){
    if(verbose)
      fprintf(stderr,"SLDetect case is an %s transformation on Nyquist data\n",
	      input.get()==output.get()?"inplace":"outofplace"); 
    
    do{
      *out_ptr = *in_ptr * *in_ptr;
      out_ptr++;
      in_ptr++;
    } while( in_ptr != dend );
  }

  else if( input->get_state()==Signal::Analytic ){
    if(verbose)
      fprintf(stderr,"SLDetect case is an %s transformation on Analytic data\n",
	      input.get()==output.get()?"inplace":"outofplace");

    do{
      *out_ptr = *in_ptr * *in_ptr;  // Re*Re
      in_ptr++;
      
      *out_ptr += *in_ptr * *in_ptr; // Add in Im*Im
      in_ptr++;
      out_ptr++;
    } while( in_ptr!=dend );
  }

  // Set new state if output==input
  if( input->get_npol()==2 )
    output->set_state( Signal::PPQQ );
  else if( input->get_npol()==1 ){
    fprintf(stderr,"dsp::SLDetect input had 1 polarisation- assuming this is total power over all polarisations\n"
	    "ie setting output->state to Signal::Intensity\n");
    output->set_state( Signal::Intensity );
  }

  fprintf(stderr,"\n\nNear end of SLDetection, state is '%s'\n",
	  output->get_state_as_string().c_str());

  // Bad:
  //  output->resize( output->get_ndat() );
  // Good: (ndim has changed so subsize changes:)
  // WvS: subsize is the number of *floats* in a data block and need not
  //      necessarily equal ndat
  // output->set_subsize( (output->get_ndat()*output->get_nbit())/8 );
  
    if(verbose)
    fprintf(stderr,"after sld output has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d state=%s\n",
	    output->get_ndim(), output->get_ndat(), output->get_npol(), output->get_nchan(),output->get_nbit(),
	    output->get_state_as_string().c_str());

  if( verbose )
    fprintf(stderr,"Exiting from %s::transformation()\n\n",get_name().c_str());
}

