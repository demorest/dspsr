#include <stdio.h>

#include "genutil.h"

#include "Timeseries.h"

#include "SLDetect.h"

dsp::SLDetect::SLDetect( Behaviour _type) : Operation ("SLDetect", _type){
}

void dsp::SLDetect::operation ()
{
  if( verbose ){
    fprintf(stderr," In %s::operation()\n",get_name().c_str());
    fprintf(stderr,"after sld input has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d\n",
	    input->get_ndim(), input->get_ndat(), input->get_npol(), input->get_nchan(),input->get_nbit());    
  }

  if( input->get_detected() ) {
    fprintf (stderr, "SLDetect::operate data is already detected!\n");
    throw_str ("SLDetect::operate invalid state");
  }

  if( input.get() != output.get() ){
    /* Set output's capacity etc.
       Things that are different from input are state, size, subsize
       These can be set correctly by simply setting state to detected
       and resizing output */
    output->Observation::operator=( *input );
    output->set_state( Signal::Intensity );
    output->resize( input->get_ndat() );
  }

  // increments values to be squared.
  register float* in_ptr = const_cast<float*>(input->get_datptr(0,0));
  register float* out_ptr = output->get_datptr(0,0);
  // When in_ptr=dend, we are on our last timesample to SLD
  register float* dend = in_ptr + input->get_ndat(); 

  /* See ~hknight/h_code/test2Aug02.C to see the test I
     ran to find the quickest way of doing the squaring.
     Not that this is the fastest though- but it's hopefully
     less confusing.
  */
  
  if( input->get_state()==Signal::Nyquist ){
    if(verbose)
      fprintf(stderr,"SLDetect case is an %s operation on Nyquist data\n",
	      input.get()==output.get()?"inplace":"outofplace"); 

    do{
      *out_ptr = *in_ptr * *in_ptr;
      out_ptr++;
      in_ptr++;
    } while( out_ptr != dend );
  }

  else if( input->get_state()==Signal::Analytic ){
    if(verbose)
      fprintf(stderr,"SLDetect case is an %s operation on Nyquist data\n",
	      input.get()==output.get()?"inplace":"outofplace");

    do{
      *out_ptr = *in_ptr * *in_ptr;  // Re*Re
      in_ptr++;
      
      *out_ptr += *in_ptr * *in_ptr; // Add in Im*Im
      in_ptr++;
      out_ptr++;
    } while( out_ptr!=dend );
  }

  /* just to make sure that output has all the correct values */
  output->Observation::operator=( *input );
  output->set_state( Signal::Intensity );
  output->resize( output->get_ndat() );

  if(verbose)
    fprintf(stderr,"after sld output has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d\n",
	    output->get_ndim(), output->get_ndat(), output->get_npol(), output->get_nchan(),output->get_nbit());

  if( verbose )
    fprintf(stderr,"Exiting from %s::operation()\n",get_name().c_str());
}

