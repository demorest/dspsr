#include "Detection.h"
#include "Timeseries.h"

#include "genutil.h"
#include "cross_detect.h"

//! Constructor
dsp::Detection::Detection () 
  : Operation ("Detection", anyplace)
{
  state = Observation::Detected;
  ndim = 1;
}

//! Set the state of output data
void dsp::Detection::set_output_state (Observation::State _state)
{
  switch (_state)  {
  case Observation::Detected:
    ndim = 1;
  case Observation::Stokes:
  case Observation::Coherence:
    break;
  default:
    throw_str ("Detection::set_output_state unknown state");
  }

  state = _state;
}

//! Detect the input data
void dsp::Detection::operation ()
{
  if (verbose)
    cerr << "Detection::operation output state="
	 << Observation::state2string(state) << endl;

  if (input->get_nbit() != sizeof(float) * 8)
    throw_str ("Detection::operation input not floating point");

  if (state == Observation::Stokes || state == Observation::Coherence) {

    if (input->get_npol() != 2)
      throw_str ("Detection::operation invalid npol=%d for %s formation",
		 input->get_npol(), Observation::state2string(state).c_str());

    if (input->get_state() != Observation::Analytic)
      throw_str ("Detection::operation invalid state=%s for %s formation",
		 input->get_state_as_string().c_str(),
		 Observation::state2string(state).c_str());

    // Coherence product and Stokes parameter formation can be performed
    // in three ways, corresponding to ndim = 1,2,4

    if (!(ndim==1 || ndim==2 || ndim==4))
      throw_str ("Detection::operation invalid ndim=%d for %s formation",
		 ndim, Observation::state2string(state).c_str());
    
  }

  bool inplace = (input == output);
  if (verbose)
    cerr << "Detection::operation inplace" << endl;

  if (!inplace)
    resize_output ();

  if (state == Observation::Detected)
    square_law ();

  else
    polarimetry ();

  if (inplace)
    resize_output ();
}

void dsp::Detection::resize_output ()
{
  if (verbose)
    cerr << "Detection::resize_output" << endl;

  int output_ndim = 1;
  int output_npol = input->get_npol();

  if (state == Observation::Stokes || state == Observation::Coherence) {
    output_ndim = ndim;
    output_npol = 4/ndim;
  }

  output->Observation::operator=(*input);

  output->set_state (state);
  output->set_ndim (output_ndim);
  output->set_npol (output_npol);

  int64 output_ndat = input->get_ndat();
  output->resize (output_ndat);

  if (state == Observation::Stokes || state == Observation::Coherence) {
    // double-check the basic assumption of the polarimetry() method

    unsigned block_size = output_ndim * output_ndat;
    
    for (int ichan=0; ichan < output->get_nchan(); ichan ++) {
      float* base = output->get_datptr (ichan, 0);
      
      for (int ipol=1; ipol<output_npol; ipol++)
	if (output->get_datptr (ichan, ipol) != base + ipol*block_size)
	  throw_str ("Detection::resize_output pointer mis-match");
    }
  }
}

void dsp::Detection::square_law ()
{
  if (verbose)
    cerr << "Detection::square_law" << endl;
}

void dsp::Detection::polarimetry ()
{
  if (verbose)
    cerr << "Detection::polarimetry ndim=" << ndim << endl;

  int64 ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  // necessary conditions of this form of detection
  unsigned input_npol = 2;
  unsigned input_ndim = 2;

  bool inplace = (input == output);

  unsigned required_space = 0;
  unsigned copy_bytes = 0;

  float* copyp  = NULL;
  float* copyq = NULL;

  if (inplace && ndim != 2) {
    // only when ndim==2 is this operation really inplace.
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
      r[1] = r[0] + ndat;
      r[2] = r[1] + ndat;
      r[3] = r[2] + ndat;
      break;
    case 2:
      r[1] = r[0] + 1;
      r[2] = r[0] + ndat * 2;
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

