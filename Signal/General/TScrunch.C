#include <stdio.h>

#include "genutil.h"

#include "dsp/TScrunch.h"

dsp::TScrunch::TScrunch() 
  : Transformation <TimeSeries, TimeSeries> ("TScrunch", anyplace)
{
  ScrunchFactor = -1;
  TimeRes = -1.0;
  use_tres = false;
  do_only_full_scrunches = false;
}

/* returns the ScrunchFactor determined by this tres */
void dsp::TScrunch::set_NewTimeRes( double microseconds ){
  TimeRes = microseconds;
  use_tres = true;
}

void dsp::TScrunch::transformation ()
{
  if( verbose )
    fprintf(stderr,"\nIn %s::transformation()\n",get_name().c_str());

  if( UsingScrunchFactor() ){
    if( ScrunchFactor < 1 ){
      throw_str ("dsp::TScrunch: invalid scrunch factor:%d",
		 ScrunchFactor);
      return;
    }
    else if(ScrunchFactor==1){
      fprintf(stderr,"ScrunchFactor=1 so no need to scrunch!\n");
      if( input.get() != output.get() )
	output->operator=( *input );
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
    fprintf(stderr,"dsp::TScrunch::transformation() is scrunching by %f (ie -> a ScrunchFactor of "I64") to a time resolution of %f (Which will actually come out as %f microsecs)\n",
	    TimeRes/(1.0e6/input->get_rate()),ScrunchFactor,TimeRes,double(ScrunchFactor)*1.0e6/input->get_rate());

  if( !input->get_detected() ){
    throw_str ("dsp::TScrunch: invalid input state: " + input->get_state_as_string());
    return;
  }

  /* eg if input->ndat==101 & ScrunchFactor==3 then this is 33 */
  register int64 normal_scrunches = input->get_ndat()/ScrunchFactor;
  /* eg if input->ndat==101 & ScrunchFactor==3 then last 2 points get scrunched into 1 */
  register int64 nfinal = input->get_ndat()%ScrunchFactor;

  if( verbose )
    fprintf(stderr,"There will be "I64" normal scrunches of ScrunchFactor="I64" and "I64" points in final scrunch\n",
	    normal_scrunches, ScrunchFactor, nfinal);

  uint64 output_ndat = normal_scrunches;
  if( nfinal>0 && !do_only_full_scrunches )
    output_ndat++;

  if( get_input() != get_output() ){
    output->Observation::operator=( *input );
    if( verbose )
      fprintf(stderr,"Going to resize output to "UI64"\n",output_ndat);
    output->resize( output_ndat );
    if( verbose )
      fprintf(stderr,"Resized output to size "UI64"\n",output->get_ndat());
    output->set_rate( input->get_rate()/(double(input->get_ndat())/double(output->get_ndat())) );
  }

  register float* scr;
  register float* un_scr;
  register float* onetoomany_scr;
  register float* onetoomany_un;

  register int64 sfactor = ScrunchFactor;

  if( verbose ){
    fprintf(stderr,"input has ndim=%d ndat="I64" npol=%d nchan=%d nbit=%d\n",
	    input->get_ndim(), input->get_ndat(), input->get_npol(), input->get_nchan(),input->get_nbit());  
    fprintf(stderr,"input data is at %p input->get_ndim()=%d input->get_ndat()="I64" with %d pols and %d chans\n",
	    input->get_datptr(0,0), input->get_ndim(), input->get_ndat(), input->get_npol(),input->get_nchan());
    fprintf(stderr,"output data is at %p output->get_ndim()=%d output->get_ndat()="I64" with %d pols and %d chans\t\n",
	    output->get_datptr(0,0), output->get_ndim(), output->get_ndat(), output->get_npol(),output->get_nchan());
    fprintf(stderr,"un_scr will range from %p to %p ("I64") to %p ("I64") ["I64"]\n",
	    input->get_datptr(0,0), input->get_datptr(0,1),int64((float*)input->get_datptr(0,1)-(float*)input->get_datptr(0,0)),
	    (float*)input->get_datptr(0,1)+input->nbytes()/8,
	    int64((float*)input->get_datptr(0,1)+input->nbytes()/8 - (float*)input->get_datptr(0,1)),
	    int64((float*)input->get_datptr(0,1)+input->nbytes()/8 - (float*)input->get_datptr(0,0)));
    fprintf(stderr,"scr will range from %p to %p ("I64")\n",
	    output->get_datptr(0,0), (float*)output->get_datptr(0,1)+output->nbytes()/8,
	    int64((float*)output->get_datptr(0,1)+output->nbytes()/8 - (float*)output->get_datptr(0,0)));
  }

  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<input->get_npol(); ipol++) {

      un_scr = input->get_datptr(ichan, ipol);
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
      if( nfinal && !do_only_full_scrunches ){
	*scr = *un_scr;
	onetoomany_un = un_scr + nfinal;
	
	while( un_scr != onetoomany_un ){
	  *scr += *un_scr;
	  un_scr++;
	}
	*scr *= float(sfactor)/float(nfinal);
      }
      
      
    } // for each ipol
  } // for each ichan

  /* make sure output has correct parameters */
  output->rescale( double(input->get_ndat()) / double(output_ndat) );

  if( verbose )
    fprintf(stderr,"TScrunch:: input->rate=%f\tinput=>ndat="I64"\toutput_ndat="I64"\n",
	    input->get_rate(), input->get_ndat(), output_ndat);

  output->set_rate( input->get_rate()/(double(input->get_ndat())/double(output_ndat)) );
  
  // This is usually very bad but should work (and is needed) in both the inplace and outofplace situations:
  output->resize(output_ndat);

  if( verbose )
    fprintf(stderr,"Exiting from %s::transformation()\n\n",get_name().c_str()); 
}




