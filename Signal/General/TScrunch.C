#include <stdio.h>

#include "genutil.h"

#include "dsp/TScrunch.h"

#if ACTIVATE_IPP
#include <ipps.h>
#endif

dsp::TScrunch::TScrunch(Behaviour place) 
  : Transformation <TimeSeries, TimeSeries> ("TScrunch", place)
{
  ScrunchFactor = -1;
  TimeRes = -1.0;
  use_tres = false;
}

/* returns the ScrunchFactor determined by this tres */
void dsp::TScrunch::set_NewTimeRes( double microseconds ){
  TimeRes = microseconds;
  use_tres = true;
}

unsigned dsp::TScrunch::get_sfactor(){
  if( UsingScrunchFactor() ){
    if( ScrunchFactor < 1 )
      throw_str ("dsp::TScrunch: invalid scrunch factor:%d",
		 ScrunchFactor);
    TimeRes = 1.0e6/(input->get_rate()*double(ScrunchFactor));
  }
  else{
    if( TimeRes < 0.0 )
      throw_str ("dsp::TScrunch: invalid time resolution:%f",
		 TimeRes);
    double in_tsamp = 1.0e6/input->get_rate();  // in microseconds
    ScrunchFactor = int64(TimeRes/in_tsamp + 0.00001);
    
    if( verbose )
      fprintf(stderr,"Setting Scrunchfactor to int64(%f/%f+0.00001) = "I64"\n",
	      TimeRes,in_tsamp,ScrunchFactor);
    
    if( ScrunchFactor<1 )
      ScrunchFactor = 1;
  }
  
  return unsigned(ScrunchFactor);
}

void dsp::TScrunch::transformation ()
{
  if( verbose )
    fprintf(stderr,"\nIn %s::transformation()\n",get_name().c_str());

  if( UsingScrunchFactor() && ScrunchFactor==1 ){
    if( verbose )
      fprintf(stderr,"ScrunchFactor=1 so no need to scrunch!\n");
    if( input.get() != output.get() )
      output->operator=( *input );
    return;
  }

  unsigned sfactor = get_sfactor();

  if( !input->get_detected() )
    throw_str ("dsp::TScrunch: invalid input state: " + input->get_state_as_string());

  const unsigned nscrunchings = input->get_ndat()/sfactor;

  if( input.get() != output.get() ){
    output->Observation::operator=( *input );
    output->resize( input->get_ndat()/sfactor );
  }

  output->rescale( sfactor );
  output->set_rate( input->get_rate()/sfactor );

  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<input->get_npol(); ipol++) {
      float* in  = input->get_datptr(ichan, ipol);
      float* out = output->get_datptr(ichan, ipol);
      
      unsigned j=0;

      /*
#if ACTIVATE_IPP
      //if( sfactor >= 8 ){
	for( unsigned iscrunching=0; iscrunching<nscrunchings; ++iscrunching){
	  ippsSum_32f(in+j,sfactor,out+iscrunching,ippAlgHintFast);
	  j+=sfactor;
	}
	continue;
	//}
#endif
      */      

	      for( unsigned iscrunching=0; iscrunching<nscrunchings; ++iscrunching){
	unsigned stop = j + sfactor;
	
	out[iscrunching] = in[j]; 	++j;
	
	for( ; j<stop; ++j)
	  out[iscrunching] += in[j];
	  }
            
    } // for each ipol
  } // for each ichan

  if( input.get() == output.get() )
    output->set_ndat( input->get_ndat()/sfactor );

  if( verbose )
    fprintf(stderr,"Exiting from %s::transformation()\n\n",get_name().c_str()); 
}




