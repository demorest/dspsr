#include <stdio.h>

#include "genutil.h"

#include "Timeseries.h"

#include "SLDetect.h"

dsp::SLDetect::SLDetect( Behaviour _type) : Operation ("SLDetect", _type){
}

void dsp::SLDetect::operation ()
{
  if( verbose )
    fprintf(stderr," In %s::operation()\n",get_name().c_str());
  if( input->get_detected() ) {
    fprintf (stderr, "SLDetect::operate data is already detected!\n");
    throw_str ("SLDetect::operate invalid state");
  }

  if( get_type() != Operation::inplace ){
    /* Set output's capacity etc.
       Things that are different from input are state, size, subsize
       These can be set correctly by simply setting state to detected
       and resizing output */
    output->Observation::operator=( *input );
    output->set_state( Observation::Detected );
    output->resize( input->get_ndat() );
  }

  // increments values to be squared.
  register float* in_ptr = const_cast<float*>(input->get_datptr(0,0));

  if (input->get_state() == Observation::Nyquist) {

    if( get_type() == Operation::inplace ){
      
      register float dat;
      // When in_ptr=dend, we are on our last timesample to SLD
      register float* dend = in_ptr + input->get_ndat();  
      
      /* See ~hknight/h_code/test2Aug02.C to see the test I
	 ran to find the quickest way of doing the squaring.
	 Not that this is the fastest though- but it's hopefully
	 less confusing.
      */
      do{
	dat = *in_ptr;  // not necessary but more portable-looking
	*in_ptr = dat*dat;
      }while( ++in_ptr != dend );
      /* cf this: while( &(*in_ptr *= *in_ptr) != dend ) in_ptr++; */
    
    } // end operations for inplace and Nyquist sampled

    else{   // Nyquist sampled && (type==outofplace || type==anyplace)
      
      // increments result values
      register float* out_ptr = output->get_datptr(0,0);
      // When out_ptr=dend, we are on our last timesample to SLD
      register float* dend = out_ptr + output->get_ndat();  
      
      do{
	*out_ptr = *in_ptr * *in_ptr;
	out_ptr++;
	in_ptr++;
      }while( out_ptr != dend );
    
    } // end operations for outofplace and Nyquist sampled
    
  } // end operations for Nyquist sampled

  else if (input->get_state() == Observation::Analytic) {
    
    // increments stored values
    register float* out_ptr = output->get_datptr(0,0);
    // When out_ptr=dend, we are on our last timesample to SLD
    register float* dend = out_ptr + output->get_ndat();
    
    do{
      *out_ptr = *in_ptr * *in_ptr;  // Re*Re
      in_ptr++;
      
      *out_ptr += *in_ptr * *in_ptr; // Add in Im*Im
      in_ptr++;
      out_ptr++;
    }while( out_ptr!=dend );
  
  } // end operations for Analytically sampled

  output->change_state( Observation::Detected );

  if( verbose )
    fprintf(stderr,"Exiting from %s::operation()\n",get_name().c_str());
}

