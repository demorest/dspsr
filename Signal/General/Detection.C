/***************************************************************************
 *
 *   Copyright (C) 2002-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Detection.h"
#include "dsp/Observation.h"
#include "dsp/Scratch.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"

#include <memory>

#include <string.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


using namespace std;

//! Constructor
dsp::Detection::Detection ()
  : Transformation <TimeSeries,TimeSeries> ("Detection", anyplace)
{
  state = Signal::Intensity;
  ndim = 1;
}

void dsp::Detection::set_engine (Engine* _engine)
{
  engine = _engine;
}

//! Set the state of output data
void dsp::Detection::set_output_state (Signal::State _state)
{
  switch (_state)
  {
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

void dsp::Detection::prepare ()
{
  resize_output ();
}

//! Detect the input data
void dsp::Detection::transformation () try
{
  if (verbose)
    cerr << "dsp::Detection::transformation input ndat=" << input->get_ndat()
	 << " state=" << Signal::state_string(get_input()->get_state())
	 << " output state=" << Signal::state_string(state) << endl;

  checks();

  bool inplace = (input.get() == output.get());

#if 0
  // when debugging pointer offsets
  cerr << "IN subsize=" << output->get_subsize() << endl;
  for (unsigned i=0; i<2; i++)
    cerr << "IN " << i << " " << output->get_datptr(i,0) << endl;
#endif

  if (state==input->get_state())
  {
    if (!inplace)
    {
      if (verbose)
	cerr << "dsp::Detection::transformation inplace and no state change" 
	     << endl;
    }
    else
    {
      if (verbose) 
	cerr << "dsp::Detection::transformation just copying input" << endl;
      output->operator=( *input );
    }
    return;
  }

  if (input->get_ndat() == 0)
    return;

  if (!inplace)
    resize_output ();    

  bool understood = true;

  if( !get_input()->get_detected() )
  {
    if (state==Signal::Coherence || state==Signal::Stokes)
      polarimetry();

    else if (state==Signal::PPQQ || state==Signal::Intensity
             || state==Signal::PP_State || state==Signal::QQ_State)
      square_law();

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

  if ( inplace )
    resize_output ();

#if 0
  // when debugging pointer offsets
  cerr << "OUT subsize=" << output->get_subsize() << endl;
  for (unsigned i=0; i<2; i++)
    cerr << "OUT " << i << " " << output->get_datptr(i,0) << endl;
#endif

  if (verbose)
    cerr << "dsp::Detection::transformation exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::Detection::transformation";
}

void dsp::Detection::resize_output ()
{
  if (verbose)
    cerr << "dsp::Detection::resize_output" << endl;

  bool inplace = input.get() == output.get();
  bool reshape = true;

  unsigned output_ndim = 1;
  unsigned output_npol = input->get_npol();

  if (state == Signal::Stokes || state == Signal::Coherence)
  {
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

  if (!inplace)
  {
    if (verbose)
      cerr << "dsp::Detection::resize_output resize npol=" << output_npol
           << " ndim=" << output_ndim << endl;
    get_output()->set_npol( output_npol );
    get_output()->set_ndim( output_ndim );
    get_output()->resize( get_input()->get_ndat() );
  }
  else if (reshape)
  {
    if (verbose)
      cerr << "dsp::Detection::resize_output reshape FROM"
	" npol=" << output->get_npol() <<
	" ndim=" << output->get_ndim() << " TO"
	" npol=" << output_npol <<
	" ndim=" << output_ndim << endl;

    get_output()->reshape ( output_npol, output_ndim );
  }

  get_output()->set_state( state );
}

void dsp::Detection::square_law ()
{
  if (verbose)
    cerr << "dsp::Detection::square_law" << endl;
 
  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const unsigned nfloat = input->get_ndim() * input->get_ndat();
  
  // if nyq is 2, then we are going to do signal analytic, and will need only 
  // half as many threads

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      register const float* in_ptr = input->get_datptr (ichan,ipol);
      register const float* dend = in_ptr + nfloat;
	      
      register float* out_ptr = output->get_datptr (ichan,ipol);
      
      if (input->get_state()==Signal::Nyquist)
	while( in_ptr != dend)
	  {
	    *out_ptr = *in_ptr * *in_ptr;
	    out_ptr++;
	    in_ptr++;
	  } 
      
      else if (input->get_state()==Signal::Analytic)
	while( in_ptr!=dend)
	  {
	    *out_ptr = *in_ptr * *in_ptr;  // Re*Re
	    in_ptr++;
	    
	    *out_ptr += *in_ptr * *in_ptr; // Add in Im*Im
	    in_ptr++;
	    out_ptr++;
	  } 
      
    }  // for each ipol
  }  // for each ichan
  
  if (state == Signal::Intensity && npol == 2)
  {
    // pscrunching
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      register float* p0 = output->get_datptr (ichan, 0);
      register float* p1 = output->get_datptr (ichan, 1);
      const register float* pend = p0 + output->get_ndat();
      
      while (p0!=pend)
      {
	*p0 += *p1;
	p0 ++;
	p1 ++;
      }
    }
  } 
}

void dsp::Detection::polarimetry () try
{
  if (verbose)
    cerr << "dsp::Detection::polarimetry ndim=" << ndim << endl;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::Detection::polarimetry using Engine" << endl;

    engine->polarimetry (ndim, input, output);
    return;
  }

  const unsigned input_npol = get_input()->get_npol();
  const unsigned input_ndim = get_input()->get_ndim();

  if (input_ndim != 2 || get_input()->get_state() != Signal::Analytic)
    throw Error (InvalidState, "dsp::Detection::polarimetry",
          "Cannot detect polarization when ndim != 2 or state != Analytic");

  bool inplace = input.get() == output.get();

  if (verbose)
    cerr << "dsp::Detection::polarimetry "
	 << ((inplace) ? "in" : "outof") << "place" << endl;

  uint64_t ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  uint64_t required_space = 0;
  uint64_t copy_bytes = 0;

  float* copyp  = NULL;
  float* copyq = NULL;

  if (inplace && ndim != 2)
  {
    // only when ndim==2 is this transformation really inplace.
    // so when ndim==1 or 4, a copy of the data must be made
    
    // need to copy both polarizations
    if (ndim == 1)
      required_space = input_ndim * input_npol * ndat;

    // need to copy only the first polarization
    if (ndim == 4)
      required_space = input_ndim * ndat;

    if (verbose)
      cerr << "dsp::Detection::polarimetry require_floats="
	   << required_space << endl;

    copyp = scratch->space<float> (unsigned(required_space));

    copy_bytes = input_ndim * ndat * sizeof(float);

    if (ndim == 1)
      copyq = copyp + input_ndim * ndat;
  }

  float* r[4];

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    const float* p = input->get_datptr (ichan, 0);
    const float* q = input->get_datptr (ichan, 1);
	  
    if (inplace && ndim != 2)
    {
      if (verbose && ichan == 0)
	cerr << "dsp::Detection::polarimetry copy_bytes="
	     << copy_bytes << endl;
	      
      memcpy (copyp, p, size_t(copy_bytes));
      p = copyp;
      
      if (ndim == 1)
      {
	memcpy (copyq, q, size_t(copy_bytes));
	q = copyq;
      }
    }
    
    get_result_pointers (ichan, inplace, r);
    
    if (state == Signal::Stokes)
      stokes_detect (unsigned(ndat), p, q, r[0], r[1], r[2], r[3], ndim);
    else
      cross_detect (ndat, p, q, r[0], r[1], r[2], r[3], ndim);
  }
  
  if (verbose)
    cerr << "dsp::Detection::polarimetry exit" << endl;
  
}
catch (Error& error)
{
  throw error += "dsp::Detection::polarimetry";
}

void dsp::Detection::get_result_pointers (unsigned ichan, bool inplace, 
					  float* r[4])
{
  if (verbose && ichan == 0)
    cerr << "dsp::Detection::get_result_pointers ndim=4" << endl;

  switch (ndim)
  {

    // Stokes I,Q,U,V in separate arrays
  case 1:
    if( inplace )
    {
      r[0] = get_output()->get_datptr (ichan,0);
      r[2] = get_output()->get_datptr (ichan,0);
      uint64_t diff = uint64_t(r[2] - r[0])/2;
      r[1] = r[0] + diff;
      r[3] = r[2] + diff;
    }
    else
    {
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

  default:
    throw Error (InvalidState, "dsp::Detection::get_result_pointers",
		 "invalid ndim=%d", ndim);
  }
}

void dsp::Detection::checks()
{
  if (state == Signal::Stokes || state == Signal::Coherence)
  {
    if (get_input()->get_npol() != 2)
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid npol=%d for %s formation",
		   input->get_npol(), Signal::state_string(state));
    
    if (get_input()->get_state() != Signal::Analytic && 
	get_input()->get_state() != Signal::Nyquist)
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid state=%s for %s formation",
		   tostring(get_input()->get_state()).c_str(),
		   tostring(state).c_str());
    
    // Signal::Coherence product and Signal::Stokes parameter
    // formation can be performed in three ways, corresponding to
    // ndim = 1,2,4
    
    if (!(ndim==1 || ndim==2 || ndim==4))
      throw Error (InvalidState, "dsp::Detection::checks",
		   "invalid ndim=%d for %s formation",
		   ndim, Signal::state_string(state));
  }
}
