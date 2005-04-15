#include <memory>

#include "dsp/Detection.h"
#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"
#include "dsp/SLDetect.h"
#include "dsp/PScrunch.h"
#include "dsp/Observation.h"

#include "Error.h"
#include "genutil.h"
#include "cross_detect.h"
#include "stokes_detect.h"

//! Constructor
dsp::Detection::Detection () 
  : Transformation <TimeSeries,TimeSeries> ("Detection", anyplace,true)
{
  state = Signal::Intensity;
  ndim = 1;
}

//! Set the state of output data
void dsp::Detection::set_output_state (Signal::State _state)
{
  switch (_state)  {
  case Signal::Intensity:  // Square-law detected total power (1 pol)
  case Signal::PPQQ:       // Square-law detected, two polarizations
  case Signal::NthPower:    // Square-law total power to the nth power
    ndim = 1;
  case Signal::Coherence:  // PP, QQ, Re[PQ], Im[PQ]
  case Signal::Stokes:     // Stokes I,Q,U,V
    break;
  default:
    throw Error (InvalidParam, "dsp::Detection::set_output_state",
		 "invalid state=%s", Signal::state_string (_state));
  }

  state = _state;

  if (verbose)
    cerr << "dsp::Detection::set_output_state to " 
	 << Signal::state_string(state) << endl;

}

//! Detect the input data
void dsp::Detection::transformation ()
{
  MJD st("52644.176409458518541");

  if( 1 )
    fprintf(stderr,"In dsp::Detection::transformation() with input ndat="UI64" (st=%f)\n",
	    get_input()->get_ndat(),
	    (get_input()->get_start_time()-st).in_seconds());

  if (1)
    cerr << "dsp::Detection::transformation output state="
	 << Signal::state_string(state) << " and input state=" 
	 << Signal::state_string(get_input()->get_state())
	 << " and input ndat=" << get_input()->get_ndat() << endl;

  checks();

  bool inplace = (input.get() == output.get());

  if( state==input->get_state() ) {
    if( !inplace ) {
      if (verbose) cerr << "dsp::Detection::transformation inplace and no state change" << endl;
    }
    else {
      if (verbose) cerr << "dsp::Detection::transformation no state change- just copying input" << endl;
      output->operator=( *input );
    }

    return;
  }

  bool understood = true;

  if( get_input()->get_detected() && state==Signal::Intensity )
    redetect ();

  else if( !get_input()->get_detected() ) {

    if (state==Signal::Coherence || state==Signal::Stokes)
      polarimetry();

    else if (state==Signal::PPQQ)
      square_law();

    else if (state==Signal::Intensity)
      form_stokes_I();
    
    else if (state==Signal::NthPower)
      form_nthpower();

    else
      understood = false;

  }
  else
    understood = false;

  if (!understood)
    throw Error (InvalidState, "dsp::Detection::transformation",
		 "dsp::Detection cannot convert from " 
		 + State2string(get_input()->get_state()) + " to "
		 + State2string(state));

  if( verbose )
    fprintf(stderr,"Returning from dsp::Detection::transformation() with output ndat="UI64" (st=%f)\n",
	    get_output()->get_ndat(),
	    (get_output()->get_start_time()-st).in_seconds());
}

void dsp::Detection::resize_output ()
{
  if (verbose)
    cerr << "dsp::Detection::resize_output" << endl;

  unsigned output_ndim = 1;
  unsigned output_npol = input->get_npol();

  if (state == Signal::Stokes || state == Signal::Coherence) {
    output_ndim = ndim;
    output_npol = 4/ndim;
    if (verbose)
      cerr << "dsp::Detection::resize_output state: "
	   << Signal::state_string(state) << " ndim=" << ndim << endl;
  }
  else if(state==Signal::PPQQ)
    output_npol = 2;
  else if(state==Signal::Intensity )
    output_npol = 1;

  get_output()->copy_configuration( get_input() );
  get_output()->set_npol( output_npol );
  get_output()->set_state( state );
  get_output()->set_ndim( output_ndim );

  get_output()->resize( get_input()->get_ndat() );
}

void dsp::Detection::square_law ()
{
  if (verbose)
    cerr << "dsp::Detection::square_law" << endl;
 
  Reference::To<dsp::SLDetect> sld(new dsp::SLDetect);
  sld->set_input( input );
  sld->set_output( output ); 
  sld->operate();  
}

void dsp::Detection::form_stokes_I(){
  if( verbose ) fprintf(stderr,"In dsp::Detection::form_stokes_I()\n");
 
  if( get_input() == get_output() ){
    square_law();
    Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
    pscrunch->set_input( get_output() );
    pscrunch->set_output( get_output() );
    pscrunch->operate();
    return;
  }

  unsigned input_ndim = get_input()->get_ndim();

  get_output()->copy_configuration( get_input() );
  get_output()->set_npol( 1 );
  get_output()->set_ndim( 1 );
  get_output()->set_state( Signal::Intensity );
  
  get_output()->resize( get_input()->get_ndat() );

  if( input_ndim==1 ){ // Signal::Nyquist
    for( unsigned ichan=0;ichan<input->get_nchan();ichan++){
      float* pol0 = get_input()->get_datptr(ichan,0);
      float* pol1 = get_input()->get_datptr(ichan,1);    
      float* out = get_output()->get_datptr(ichan,0);
      
      uint64 ndat = get_input()->get_ndat();
      
      for( uint64 i=0; i<ndat; i++)
	out[i] = SQR(pol0[i]) + SQR(pol1[i]); 
    }
  }
  else{ // Signal::Analytic
    for( unsigned ichan=0;ichan<input->get_nchan();ichan++){
      float* pol0 = get_input()->get_datptr(ichan,0);
      float* pol1 = get_input()->get_datptr(ichan,1);    
      float* out = get_output()->get_datptr(ichan,0);
      
      uint64 ndat = get_input()->get_ndat();
      
      for( uint64 i=0; i<ndat; i++)
	out[i] = SQR(pol0[2*i]) + SQR(pol0[2*i+1]) + SQR(pol1[2*i]) + SQR(pol1[2*i+1]); 
    }
  }
  
}
  

void dsp::Detection::form_nthpower(int _n){
  if( verbose ) fprintf(stderr,"In dsp::Detection::nthpower()\n");
 
  if( (get_input() == get_output()) && verbose ){
    fprintf(stderr,"In dsp::Detection::nthpower but calling square-law\n");
    
    square_law();
    Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
    pscrunch->set_input( get_output() );
    pscrunch->set_output( get_output() );
    pscrunch->operate();
    return;
  }
  
  fprintf(stderr,"In dsp::Detection::nthpower forming 2nd moment\n");
  unsigned input_ndim = get_input()->get_ndim();

  get_output()->copy_configuration( get_input() );
  get_output()->set_npol( 1 );
  get_output()->set_ndim( 1 );
  get_output()->set_state( Signal::NthPower );
  
  get_output()->resize( get_input()->get_ndat() );

  if( input_ndim==1 ){ // Signal::Nyquist
    for( unsigned ichan=0;ichan<input->get_nchan();ichan++){
      float* pol0 = get_input()->get_datptr(ichan,0);
      float* pol1 = get_input()->get_datptr(ichan,1);    
      float* out = get_output()->get_datptr(ichan,0);
      
      uint64 ndat = get_input()->get_ndat();
      double temp; 
      for( uint64 i=0; i<ndat; i++){
        temp = SQR(pol0[i]) + SQR(pol1[i]);
	out[i] = (float) pow(temp,(double) _n);
      }
     
    }
  }
  else{ // Signal::Analytic
    for( unsigned ichan=0;ichan<input->get_nchan();ichan++){
      float* pol0 = get_input()->get_datptr(ichan,0);
      float* pol1 = get_input()->get_datptr(ichan,1);    
      float* out = get_output()->get_datptr(ichan,0);
      
      uint64 ndat = get_input()->get_ndat();
      double temp; 
      for( uint64 i=0; i<ndat; i++){
        temp = SQR(pol0[2*i]) + SQR(pol0[2*i+1]) + SQR(pol1[2*i]) + SQR(pol1[2*i+1]);
	out[i] = (float) pow(temp,(double) _n);
      }
 
    }
  }
  
}
  
void dsp::Detection::polarimetry ()
{
  if (verbose)
    cerr << "dsp::Detection::polarimetry ndim=" << ndim << endl;

  unsigned input_npol = get_input()->get_npol();
  unsigned input_ndim = get_input()->get_ndim();

  if( ndim != 1 && get_input()->get_state()==Signal::Nyquist )
    throw Error(InvalidState,"dsp::Detection::polarimetry ()",
		"This function throws this Error if ndim!=1 and input state is Nyquist as I can't be bothered checking to see if other ndims work HSK 2 November 2004");

  if ( get_input() != get_output() )
    resize_output ();    

  uint64 ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  bool inplace = (input.get() == output.get());
  if( verbose )
    fprintf(stderr,"dsp::Detection::polarimetry () inplace=%d\n",inplace);
  
  uint64 required_space = 0;
  uint64 copy_bytes = 0;

  float* copyp  = NULL;
  float* copyq = NULL;

  if (inplace && ndim != 2) {
    // only when ndim==2 is this transformation really inplace.
    // so when ndim==1or4, a copy of the data must be made
    
    // need to copy both polarizations
    if (ndim == 1)
      required_space = input_ndim * input_npol * ndat;

    // need to copy only the first polarization
    if (ndim == 4)
      required_space = input_ndim * ndat;
    
    copyp = float_workingspace (required_space);

    copy_bytes = input_ndim * ndat * sizeof(float);

    if (ndim == 1)
      copyq = copyp + input_ndim * ndat;
  }

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    const float* p = input->get_datptr (ichan, 0);
    const float* q = input->get_datptr (ichan, 1);
    if (inplace && ndim != 2) {
      memcpy (copyp, p, copy_bytes);
      p = copyp;

      if (ndim == 1) {
	memcpy (copyq, q, copy_bytes);
	q = copyq;
      }
    }
    vector<float*> r = get_result_pointers(ichan);

    if (input_ndim == 2) {

      // ie Analytic
      if (state == Signal::Stokes)
	stokes_detect (ndat, p, q, r[0], r[1], r[2], r[3], ndim);
      else
	cross_detect (ndat, p, q, r[0], r[1], r[2], r[3], ndim);

    }
    else{ // ie Nyquist ndim==1
      float*& pp = r[0];
      float*& qq = r[1];
      float*& Rpq= r[2];
      float*& Ipq= r[3];

      for (unsigned j=0; j<ndat; j++)  {
	float p_r = *p; p++;
	float q_r = *q; q++;
	
	*pp  = SQR(p_r);  pp ++;    /*  p* p      */
	*qq  = SQR(q_r);  qq ++;    /*  q* q      */
	*Rpq = p_r * q_r; Rpq ++;   /*  Re[p* q]  */
	*Ipq = 0.0;       Ipq ++;   /*  Im[p* q]  */
      }
    }
  }

  if ( get_input() == get_output() )
    resize_output ();

  if( verbose )
    fprintf(stderr,"Returning from dsp::Detection::polarimetry()\n");
}

vector<float*> dsp::Detection::get_result_pointers(unsigned ichan){
  vector<float*> r(4);
  uint64 ndat = get_input()->get_ndat();
  bool inplace = (get_input()==get_output());

  r[0] = get_output()->get_datptr(ichan,0);

  switch (ndim) {
  case 1:
    if( inplace ){
      r[1] = r[0] + ndat;
      r[2] = r[1] + ndat;
      r[3] = r[2] + ndat;
    }
    else{
      r[1] = get_output()->get_datptr(ichan,1);
      r[2] = get_output()->get_datptr(ichan,2);
      r[3] = get_output()->get_datptr(ichan,3);
    }
    break;
  case 2:
    r[1] = r[0] + 1;
    if( inplace ) // Insanity
      r[2] = r[0] + ndat * 2;
    else
      r[2] = get_output()->get_datptr(ichan,1);
    r[3] = r[2] + 1;
    break;
  case 4:
    r[1] = r[0] + 1;
    r[2] = r[1] + 1;
    r[3] = r[2] + 1;
    break;
  }
  
  return r;
}

void dsp::Detection::redetect(){
  Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
  pscrunch->set_input( get_input() );
  pscrunch->set_output( get_output() );
  
  pscrunch->operate();
}

void dsp::Detection::checks(){
  if( get_input()->get_detected() && state != Signal::Intensity )
    throw Error(InvalidState,"dsp::Detection::checks()",
		"Sorry, but this class currently can only redetect data to Stokes I (You had input='%s' and output='%s')",
		State2string(get_input()->get_state()).c_str(),
		State2string(state).c_str());

  if( get_input()->get_state()==Signal::Nyquist && get_input()->get_machine()=="CPSR2" )
    fprintf(stderr,"\n\ndsp::Detection::checks(): input state is Nyquist from CPSR2... continuing... but have you forgotten to add an -F option to form a filterbank or to deconvolve?\n\n");
  
  if (state == Signal::Stokes || state == Signal::Coherence) {
    if (get_input()->get_npol() != 2)
      throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid npol=%d for %s formation",
		   input->get_npol(), Signal::state_string(state));
    
    if (get_input()->get_state() != Signal::Analytic && get_input()->get_state() != Signal::Nyquist)
      throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid state=%s for %s formation",
		   get_input()->get_state_as_string().c_str(),
		   Signal::state_string(state));
    
    // Signal::Coherence product and Signal::Stokes parameter
    // formation can be performed in three ways, corresponding to
    // ndim = 1,2,4
    
    if (!(ndim==1 || ndim==2 || ndim==4))
      throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid ndim=%d for %s formation",
		   ndim, Signal::state_string(state));
  }    
  
}
