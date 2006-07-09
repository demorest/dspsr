/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/InputBuffering.h"

#include "FTransform.h"

// #define _DEBUG

dsp::PhaseLockedFilterbank::PhaseLockedFilterbank () :
  Transformation <TimeSeries, PhaseSeries> ("PhaseLockedFilterbank",outofplace)
{
  nchan = 0;
  nbin = 0;
  built = false;

  set_buffering_policy (new InputBuffering (this));
}

void dsp::PhaseLockedFilterbank::set_nchan (unsigned _nchan)
{
  nchan = _nchan;
}

void dsp::PhaseLockedFilterbank::set_nbin (unsigned _nbin)
{
  nbin = _nbin;
  divider.set_turns (1.0/double(nbin));
}

template<class T> T sqr (T x) { return x*x; }

void dsp::PhaseLockedFilterbank::prepare ()
{
  if (nchan < 2 && nbin < 2)
    throw Error (InvalidState, "dsp::PhaseLockedFilterbank::prepare",
		 "invalid dimensions.  nchan=%d nbin=%d", nchan, nbin);

  double period = divider.get_polyco()->period(input->get_start_time());

  double samples_per_bin = period * input->get_rate() / nbin;

  unsigned max = (unsigned) pow (2.0, floor( log(samples_per_bin)/log(2.0) ));

  if (nchan < 2)
    nchan = max;
  else if (nchan > max)
    cerr << "dsp::PhaseLockedFilterbank::prepare warning selected nchan="
	 << nchan << " > suggested max=" << max << endl;

  cerr << "dsp::PhaseLockedFilterbank::prepare period=" << period 
       << " nbin=" << nbin << " samples=" << samples_per_bin 
       << " nchan=" << nchan << endl;

  built = true;
}

void dsp::PhaseLockedFilterbank::transformation ()
{
  const uint64 input_ndat = input->get_ndat();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol = input->get_npol();
  const unsigned input_ndim = input->get_ndim();

  if (verbose)
    cerr << "dsp::PhaseLockedFilterbank::transformation input ndat="
	 << input_ndat << " output ndat=" << output->get_ndat() << endl;

  if (!built)
    prepare ();

  // number of time samples in first fft
  unsigned ndat_fft = 0;

  if (input->get_state() == Signal::Nyquist)
    ndat_fft = 2 * nchan;

  else if (input->get_state() == Signal::Analytic)
    ndat_fft = nchan;

  else
    throw Error (InvalidState, "dsp::PhaseLockedFilterbank::transformation",
		 "invalid input data state = " + input->get_state_as_string());

  if (get_output()->get_integration_length() == 0.0) {

    // the integration is currently empty; prepare for integration
    get_output()->Observation::operator = (*input);

    get_output()->set_nchan (nbin);
    get_output()->set_npol (1);
    get_output()->set_ndim (1);
    get_output()->set_state (Signal::Intensity);

    get_output()->resize (nchan * input_nchan);
    get_output()->zero ();

    get_output()->set_hits (1);

  }
  else {
    MJD end_time = std::max (output->get_end_time(), input->get_end_time());
    MJD st_time = std::min (output->get_start_time(), input->get_start_time());

    output->set_end_time (end_time);
    output->set_start_time (st_time);
  }


  if (FTransform::get_norm() == FTransform::nfft)
    output->rescale (nchan);
  
  output->set_rate (input->get_rate() / ndat_fft);

  // complex to complex FFT produces a band swapped result
  if (input->get_state() == Signal::Analytic)
    output->set_swap (true);

  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  if (divider.get_start_time() == MJD::zero)
    divider.set_start_time (input->get_start_time());

  // set up the scratch space
  float* complex_spectrum = float_workingspace (nchan * 2);

  if (verbose)
    cerr << "dsp::PhaseLockedFilterbank::transformation enter main loop " 
	 << endl;

  uint64 idat_start = 0;
  unsigned phase_bin = 0;

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;

  while (more_data) {

    divider.set_bounds( get_input() );

    idat_start = divider.get_idat_start ();

    if (idat_start + ndat_fft > input_ndat)
      break;

    phase_bin = divider.get_phase_bin ();

    // cerr << "phase bin = " << phase_bin << endl;
    get_output()->get_hits()[phase_bin] ++;
  
    for (unsigned ichan=0; ichan < input_nchan; ichan++) {

      float* amps = output->get_datptr (phase_bin, 0) + ichan * nchan;

      for (unsigned ipol=0; ipol < input_npol; ipol++) {

	const float* dat_ptr = input->get_datptr (ichan, ipol);
	dat_ptr += idat_start * input_ndim;
	  
	if (input_ndim == 1)
	  FTransform::frc1d (ndat_fft, complex_spectrum, dat_ptr);
	else
	  FTransform::fcc1d (ndat_fft, complex_spectrum, dat_ptr);

	// square-law detect
	for (unsigned ichan=0; ichan < nchan; ichan++) {
	  amps[ichan] += sqr(complex_spectrum[ichan*2]);
	  amps[ichan] += sqr(complex_spectrum[ichan*2+1]);
	}

      } // for each polarization
    
    } // for each frequency channel

  } // for each big fft (ipart)

  // cerr << "main loop finished" << endl;

  get_buffering_policy()->set_minimum_samples (ndat_fft);
  get_buffering_policy()->set_next_start (idat_start);

}

void dsp::PhaseLockedFilterbank::normalize_output ()
{
  unsigned output_nbin = get_output()->get_nbin();
  unsigned output_nchan = get_output()->get_nchan();

  unsigned* hits = get_output()->get_hits();
  
  for (unsigned ichan=0; ichan < output_nchan; ichan++) {
    float* amps = output->get_datptr (ichan, 0);
    for (unsigned ibin=0; ibin < output_nbin; ibin++)
      amps[ibin] /= hits[ichan];
  }

  get_output()->set_hits(1);
}
