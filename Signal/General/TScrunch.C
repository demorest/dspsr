#include <stdio.h>

#include "genutil.h"

#include "Timeseries.h"

#include "Operation.h"
#include "TScrunch.h"

dsp::TScrunch::TScrunch(Behaviour _type) : Operation("TScrunch", _type){
  ScrunchFactor = -1;
  TimeRes = -1.0;
  use_tres = false;
}

/* returns the ScrunchFactor determined by this tres */
void dsp::TScrunch::set_NewTimeRes( double microseconds ){
  TimeRes = microseconds;
  use_tres = true;
}

void dsp::TScrunch::operation ()
{
  if( verbose )
    fprintf(stderr," In %s::operation()\n",get_name().c_str());

  if( UsingScrunchFactor() ){
    if( ScrunchFactor <= 1 ){
      throw_str ("dsp::TScrunch: invalid scrunch factor:%d",
		 ScrunchFactor);
      return;
    }
    TimeRes = 1.0e6/(input->get_rate()*double(ScrunchFactor));
  }
  else{
    if( TimeRes < 0.0 ){
      throw_str ("dsp::TScrunch: invalid time resolution:%f",
		 TimeRes);
      return;
    }
    ScrunchFactor = int64(TimeRes/(1.0e6/input->get_rate()));
    if( ScrunchFactor<1 )
      ScrunchFactor = 1;
  }

  if( verbose )
    fprintf(stderr,"dsp::TScrunch::operation() is scrunching by %f->"I64" to a time resolutions of %f->%f\n",
	    TimeRes/(1.0e6/input->get_rate()),ScrunchFactor,TimeRes,double(ScrunchFactor)*1.0e6/input->get_rate());

  if( !input->get_detected() ){
    throw_str ("dsp::TScrunch: invalid input state:%s", 
	       input->state_as_string().c_str());
    return;
  }

  /* eg if input->ndat==101 & ScrunchFactor==3 then this is 33 */
  register int64 normal_scrunches = input->get_ndat()/ScrunchFactor;
  /* eg if input->ndat==101 & ScrunchFactor==3 then last 2 points get scrunched into 1 */
  register int64 nfinal = input->get_ndat()%ScrunchFactor;

  int64 output_ndat = normal_scrunches;
  if( nfinal>0 )
    output_ndat++;

  output->rescale( double(output_ndat) / double(input->get_ndat()) );

  if(get_input() != get_output() ){
    output->Observation::operator=( *input );
    output->resize( output_ndat ); 
    output->set_rate( input->get_rate()/(double(input->get_ndat())/double(output->get_ndat())) );
  }

  register float* scr;
  register float* un_scr;
  register float* onetoomany_scr;
  register float* onetoomany_un;

  register int64 sfactor = ScrunchFactor;

  for (int ichan=0; ichan<input->get_nchan(); ichan++) {
    for (int ipol=0; ipol<input->get_npol(); ipol++) {

      un_scr =  (float*)input->get_datptr(ichan, ipol);
      scr    = output->get_datptr(ichan, ipol);
      onetoomany_un = un_scr + sfactor;
      onetoomany_scr = scr + normal_scrunches;
      
      while( scr != onetoomany_scr ){
	
	*scr = *un_scr;
	un_scr++;
	
	/* a whole while loop is one scrunching */
	while( un_scr != onetoomany_un ){
	  *scr += *un_scr;
	  un_scr++;
	}
	
	onetoomany_un += sfactor;
	scr++;
      }
  
      /* e.g. if input->ndat=101 and ScrunchFactor==3 then this is doing those last 2 points */
      if( nfinal ){
	*scr = *un_scr;
	onetoomany_un = un_scr + nfinal;
	
	while( un_scr != onetoomany_un ){
	  *scr += *un_scr;
	    un_scr++;
	}
      }
      
      
    } // for each ipol
  } // for each npol

  if( verbose )
    fprintf(stderr,"Exiting from %s::operation()\n",get_name().c_str()); 
}




