/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Detection.h"
#include "dsp/SLDetect.h"
#include "dsp/Observation.h"
#include "dsp/Scratch.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"

#include <memory>

using namespace std;

//! Constructor
dsp::Detection::Detection () 
  : Transformation <TimeSeries,TimeSeries> ("Detection", anyplace, true)
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
  case Signal::NthPower:   // Square-law total power to the nth power
  case Signal::PP_State:   // Just PP
  case Signal::QQ_State:   // Just QQ
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
void dsp::Detection::transformation () try {

  static MJD st = get_input()->get_start_time();

  if (verbose)
    cerr << "dsp::Detection::transformation input ndat=" << input->get_ndat()
	 << " state=" << Signal::state_string(get_input()->get_state())
	 << " output state=" << Signal::state_string(state) << endl;

  checks();

  bool inplace = (input.get() == output.get());

  if( state==input->get_state() ) {
    if( !inplace ) {
      if (verbose)
	cerr << "dsp::Detection::transformation inplace and no state change" 
	     << endl;
    }
    else {
      if (verbose) 
	cerr << "dsp::Detection::transformation just copying input" << endl;
      output->operator=( *input );
    }
    return;
  }

  bool understood = true;

  if( !get_input()->get_detected() ) {

    if (state==Signal::Coherence || state==Signal::Stokes)
      polarimetry();

    else if (state==Signal::PPQQ)
      square_law();

    else if( state==Signal::PP_State || state==Signal::QQ_State )
      onepol_detect();

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

  if (verbose)
    cerr << "dsp::Detection::transformation exit" << endl;
}
 catch (Error& error) {
   throw error += "dsp::Detection::transformation";
 }

void dsp::Detection::resize_output ()
{
  if (verbose)
    cerr << "dsp::Detection::resize_output" << endl;

  bool inplace = input.get() == output.get();

  unsigned output_ndim = 1;
  unsigned output_npol = input->get_npol();

  if (state == Signal::Stokes || state == Signal::Coherence) {
    output_ndim = ndim;
    output_npol = 4/ndim;
    if (verbose)
      cerr << "dsp::Detection::resize_output state: "
	   << Signal::state_string(state) << " ndim=" << ndim << endl;
  }
  else if (state==Signal::PPQQ)
    output_npol = 2;
  else if (state==Signal::Intensity)
    output_npol = 1;

  if (!inplace)
    get_output()->copy_configuration( get_input() );

  get_output()->set_state( state );

  if (!inplace) {
    get_output()->set_npol( output_npol );
    get_output()->set_ndim( output_ndim );

    // note that TimeSeries::resize( 0 ) deletes arrays
    if (input->get_ndat())
      get_output()->resize( get_input()->get_ndat() );
    else
      get_output()->set_ndat( 0 );
  }
  else
    get_output()->reshape ( output_npol, output_ndim );


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

//! Quick and dirty method for detecting to PP or QQ
void
dsp::Detection::onepol_detect()
{
  Reference::To<dsp::SLDetect> sld(new dsp::SLDetect);
  sld->set_input( input );
  sld->set_output( new dsp::TimeSeries ); 
  sld->operate();  

  {
    Reference::To<Observation> onepol = new Observation( *sld->get_output() );
    onepol->set_npol( 1 );
    onepol->set_state( state );
    output->Observation::operator=( *onepol );
  }    

  output->resize( output->get_ndat() );

  unsigned goodpol = 0;
  if( state==Signal::QQ_State )
    goodpol = 1;

  for( unsigned ichan=0; ichan<output->get_nchan(); ichan++){
    float* in = sld->get_output()->get_datptr(ichan,goodpol);
    float* out= get_output()->get_datptr(ichan,0);

    memcpy(out,in,size_t(output->get_ndat()*sizeof(float)));
  }

}
  
void dsp::Detection::polarimetry () try {

  if (verbose)
    cerr << "dsp::Detection::polarimetry ndim=" << ndim << endl;

  unsigned input_npol = get_input()->get_npol();
  unsigned input_ndim = get_input()->get_ndim();

  if (ndim != 1 && get_input()->get_state()==Signal::Nyquist)
    throw Error (InvalidState, "dsp::Detection::polarimetry",
		 "Cannot detect Nyquist input when ndim == 1");

  bool inplace = input.get() == output.get();

  if (verbose)
    cerr << "dsp::Detection::polarimetry "
	 << ((inplace) ? "in" : "outof") << "place" << endl;

  if (!inplace)
    resize_output ();    

  uint64 ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

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
    
    copyp = scratch->space<float> (unsigned(required_space));

    copy_bytes = input_ndim * ndat * sizeof(float);

    if (ndim == 1)
      copyq = copyp + input_ndim * ndat;
  }

  float* r[4];

  for (unsigned ichan=0; ichan<nchan; ichan++) {

    const float* p = input->get_datptr (ichan, 0);
    const float* q = input->get_datptr (ichan, 1);

    if (inplace && ndim != 2) {
      memcpy (copyp, p, size_t(copy_bytes));
      p = copyp;

      if (ndim == 1) {
	memcpy (copyq, q, size_t(copy_bytes));
	q = copyq;
      }
    }

    get_result_pointers (ichan, inplace, r);

    if (input_ndim == 2) {

      // ie Analytic
      if (state == Signal::Stokes)
	stokes_detect (unsigned(ndat), p, q, r[0], r[1], r[2], r[3], ndim);
      else
	cross_detect (unsigned(ndat), p, q, r[0], r[1], r[2], r[3], ndim);

    }
    else{ // ie Nyquist ndim==1
      float*& pp = r[0];
      float*& qq = r[1];
      float*& Rpq= r[2];
      float*& Ipq= r[3];

      for (unsigned j=0; j<ndat; j++)  {
	float p_r = *p; p++;
	float q_r = *q; q++;
	
	*pp  = sqr(p_r);  pp ++;    /*  p* p      */
	*qq  = sqr(q_r);  qq ++;    /*  q* q      */
	*Rpq = p_r * q_r; Rpq ++;   /*  Re[p* q]  */
	*Ipq = 0.0;       Ipq ++;   /*  Im[p* q]  */
      }
    }
  }

  if ( get_input() == get_output() )
    resize_output ();

  if (verbose)
    cerr << "dsp::Detection::polarimetry exit" << endl;
}
 catch (Error& error) {
   throw error += "dsp::Detection::polarimetry";
 }

void dsp::Detection::get_result_pointers (unsigned ichan, bool inplace, 
					  float* r[4])
{
  switch (ndim) {

    // Stokes I,Q,U,V in separate arrays
  case 1:
    if( inplace ){
      r[0] = get_output()->get_datptr (ichan,0);
      r[2] = get_output()->get_datptr (ichan,0);
      uint64 diff = uint64(r[2] - r[0])/2;
      r[1] = r[0] + diff;
      r[3] = r[2] + diff;
    }
    else{
      r[0] = get_output()->get_datptr (ichan,0);
      r[1] = get_output()->get_datptr (ichan,1);
      r[2] = get_output()->get_datptr (ichan,2);
      r[3] = get_output()->get_datptr (ichan,3);
    }
    break;

    // Stokes I,Q and Stokes U,V in separate arrays
  case 2:
    r[0] = get_output()->get_datptr (ichan,0);
    r[1] = r[0] + 1;
    r[2] = get_output()->get_datptr (ichan,1);
    r[3] = r[2] + 1;
    break;

    // Stokes I,Q,U,V in one array
  case 4:
    r[0] = get_output()->get_datptr (ichan,0);
    r[1] = r[0] + 1;
    r[2] = r[1] + 1;
    r[3] = r[2] + 1;
    break;
  }
  
}

void dsp::Detection::checks(){
  if( get_input()->get_detected() && state != Signal::Intensity )
    throw Error(InvalidState, "dsp::Detection::checks",
		"Sorry, but this class currently can only redetect data to Stokes I (You had input='%s' and output='%s')",
		State2string(get_input()->get_state()).c_str(),
		State2string(state).c_str());


  if (state == Signal::Stokes || state == Signal::Coherence) {

    if (get_input()->get_npol() != 2)
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid npol=%d for %s formation",
		   input->get_npol(), Signal::state_string(state));
    
    if (get_input()->get_state() != Signal::Analytic && 
	get_input()->get_state() != Signal::Nyquist)
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid state=%s for %s formation",
		   get_input()->get_state_as_string().c_str(),
		   Signal::state_string(state));
    
    // Signal::Coherence product and Signal::Stokes parameter
    // formation can be performed in three ways, corresponding to
    // ndim = 1,2,4
    
    if (!(ndim==1 || ndim==2 || ndim==4))
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid ndim=%d for %s formation",
		   ndim, Signal::state_string(state));
  }    

}
