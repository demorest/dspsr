#include <memory>

#include "dsp/Detection.h"
#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"
#include "dsp/SLDetect.h"
#include "dsp/PScrunch.h"

#include "Error.h"
#include "genutil.h"
#include "cross_detect.h"

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
    ndim = 1;
  case Signal::Coherence:  // PP, QQ, Re[PQ], Im[PQ]
    break;
  case Signal::Stokes:     // Stokes I,Q,U,V
    throw Error (InvalidParam, "dsp::Detection::set_output_state",
		 "Stokes output not implemented");
  default:
    throw Error (InvalidParam, "dsp::Detection::set_output_state",
		 "invalid state=%s", Signal::state_string (_state));
  }

  state = _state;
}

//! Detect the input data
void dsp::Detection::transformation ()
{
  if (verbose)
    cerr << "dsp::Detection::transformation output state="
	 << Signal::state_string(state) << " and input ndat=" 
	 << get_input()->get_ndat() << endl;

  checks();

  if( state==input->get_state() ) {
    if( input.get()==output.get() ) {
      if (verbose)
	cerr << "dsp::Detection::transformation inplace and no state change" << endl;
    }
    else {
      if (verbose)
	cerr << "dsp::Detection::transformation no state change- just copying input" << endl;
      output->operator=( *input );
    }

    return;
  }

  if( get_input()->get_detected() ){
    redetect();
    return;
  }

  bool inplace = (input.get() == output.get());
  if (verbose)
    cerr << "dsp::Detection::transformation inplace" << endl;

  if (!inplace)
    resize_output ();

  if (state == Signal::Intensity || state == Signal::PPQQ)
    square_law ();

  else if (state == Signal::Stokes || state == Signal::Coherence)
    polarimetry ();

  else
    throw Error (InvalidState, "dsp::Detection::transformation",
		 "unknown output state=%s", Signal::state_string (state));
		  
  if (inplace)
    resize_output ();

  if( verbose )
    fprintf(stderr,"Returning from dsp::Detection::transformation() with output ndat="UI64"\n",get_output()->get_ndat());
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

  output->Observation::operator=(*input);

  output->set_state (state);
  output->set_ndim (output_ndim);
  output->set_npol (output_npol);

  uint64 output_ndat = input->get_ndat();
  output->resize (output_ndat);

  /*
  if (state == Signal::Stokes || state == Signal::Coherence) {
    // double-check the basic assumption of the polarimetry() method

    unsigned block_size = output_ndim * output_ndat;
    
    for (unsigned ichan=0; ichan < output->get_nchan(); ichan ++) {
      float* base = output->get_datptr (ichan, 0);
      
      for (unsigned ipol=1; ipol<output_npol; ipol++)
	if (output->get_datptr (ichan, ipol) != base + ipol*block_size)
	  throw Error (InvalidState, "dsp::Detection::resize_output",
		       "pointer mis-match");
    }
  }
  */
}

void dsp::Detection::square_law ()
{
  if (verbose)
    cerr << "dsp::Detection::square_law" << endl;
 
  if( state==Signal::Intensity && input->get_state()==Signal::PPQQ ){
    Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
    pscrunch->set_input( input );
    pscrunch->set_output( output );

    pscrunch->operate();
    return;
  }

  Reference::To<dsp::SLDetect> sld(new dsp::SLDetect);
  sld->set_input( input );
  sld->set_output( output );
  
  sld->operate();
  
  if( state==Signal::Intensity && output->get_state()==Signal::PPQQ ){
    Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
    pscrunch->set_input( output );
    pscrunch->set_output( output );
    
    pscrunch->operate();
  }
  
}

void dsp::Detection::polarimetry ()
{
  if (verbose)
    cerr << "dsp::Detection::polarimetry ndim=" << ndim << endl;

  uint64 ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  // necessary conditions of this form of detection
  unsigned input_npol = 2;
  unsigned input_ndim = 2;

  bool inplace = (input.get() == output.get());

  unsigned required_space = 0;
  unsigned copy_bytes = 0;

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

  // pointers to the results
  float* r[4];

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

    r[0] = output->get_datptr (ichan, 0);

    switch (ndim) {
    case 1:
      r[1] = get_output()->get_datptr(ichan,1);
      r[2] = get_output()->get_datptr(ichan,2);
      r[3] = get_output()->get_datptr(ichan,3);
      break;
    case 2:
      r[1] = r[0] + 1;
      r[2] = get_output()->get_datptr(ichan,1);
      r[3] = r[2] + 1;
      break;
    case 4:
      r[1] = r[0] + 1;
      r[2] = r[1] + 1;
      r[3] = r[2] + 1;
     break;
    }

    cross_detect (ndat, p, q, r[0], r[1], r[2], r[3], ndim);
  }
}

void dsp::Detection::redetect(){
  Reference::To<dsp::PScrunch> pscrunch(new dsp::PScrunch);
  pscrunch->set_input( get_input() );
  pscrunch->set_output( get_output() );
  
  pscrunch->operate();
}

void dsp::Detection::checks(){
  if( get_input()->get_detected() &&  state != Signal::Intensity )
    throw Error(InvalidState,"dsp::Detection::checks()",
		"Sorry, but this class currently can only redetect data to Stokes I (You had input='%s' and output='%s')",
		State2string(get_input()->get_state()).c_str(),
		State2string(state).c_str());
  
  if (state == Signal::Stokes || state == Signal::Coherence) {
    if (input->get_npol() != 2)
      throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid npol=%d for %s formation",
		   input->get_npol(), Signal::state_string(state));
    
    if (input->get_state() != Signal::Analytic){
      if( get_input()->get_state()==Signal::Nyquist )
	throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid state=%s for %s formation- have you forgotten to add a -F option to form a filterbank or to deconvolve?",
		   input->get_state_as_string().c_str(),
		   Signal::state_string(state));
      else
	throw Error (InvalidState, "dsp::Detection::transformation",
		     "invalid state=%s for %s formation",
		     input->get_state_as_string().c_str(),
		     Signal::state_string(state));
    }

    // Signal::Coherence product and Signal::Stokes parameter
    // formation can be performed in three ways, corresponding to
    // ndim = 1,2,4
    
    if (!(ndim==1 || ndim==2 || ndim==4))
      throw Error (InvalidState, "dsp::Detection::transformation",
		   "invalid ndim=%d for %s formation",
		   ndim, Signal::state_string(state));
  }    
  
}
