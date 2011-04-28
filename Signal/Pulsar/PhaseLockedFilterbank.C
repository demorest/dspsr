/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"

#include "FTransform.h"

using namespace std;

// #define _DEBUG

dsp::PhaseLockedFilterbank::PhaseLockedFilterbank () :
  Transformation <TimeSeries, PhaseSeries> ("PhaseLockedFilterbank",outofplace)
{
  nchan = 0;
  nbin = 0;
  npol = 1;

  idat_start = ndat_fold = 0;

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
  bin_divider.set_turns (1.0/double(nbin));
}

void dsp::PhaseLockedFilterbank::set_npol (unsigned _npol)
{
  if (_npol!=1 && _npol!=2 && _npol!=4)
    throw Error (InvalidParam, "dsp::PhaseLockedFilterbank::set_npol",
        "Invalid npol (%d)", _npol);
  npol = _npol;
}

/*! sets idat_start to zero and ndat_fold to input->get_ndat() */
void dsp::PhaseLockedFilterbank::set_limits (const Observation* input)
{
  idat_start = 0;
  ndat_fold = input->get_ndat();
}

template<class T> T sqr (T x) { return x*x; }

void dsp::PhaseLockedFilterbank::prepare ()
{
  if (nchan < 2 && nbin < 2)
    throw Error (InvalidState, "dsp::PhaseLockedFilterbank::prepare",
		 "invalid dimensions.  nchan=%d nbin=%d", nchan, nbin);

  MJD epoch = input->get_start_time();
  double period = 1.0/bin_divider.get_predictor()->frequency(epoch);

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
  const uint64_t input_ndat = input->get_ndat();
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
		 "invalid input data state = " + tostring(input->get_state()));

  bool new_integration = false;
  if (get_output()->get_integration_length() == 0.0) 
  {

    if (verbose)
      cerr << "dsp::PhaseLockedFilterbank::transformation"
        << " starting new integration" << endl;

    new_integration = true;

    // the integration is currently empty; prepare for integration
    get_output()->Observation::operator = (*input);

    get_output()->set_nchan (nchan * input_nchan);
    get_output()->set_npol (npol);
    get_output()->set_ndim (1);
    if (npol==1)
      get_output()->set_state (Signal::Intensity);
    else if (npol==2)
      get_output()->set_state (Signal::PPQQ);
    else if (npol==4)
      get_output()->set_state (Signal::Coherence);
    else
      throw Error (InvalidState, "dsp::PhaseLockedFilterbank::transformation", 
          "Invalid npol setting (%d)", npol);

    if (input_npol < 2 && npol > 1)
      throw Error (InvalidState, "dsp::PhaseLockedFilterbank::transformation",
          "Not enough input polns (%d) for output npol (%d)", 
          input_npol, npol);

    if (FTransform::get_norm() == FTransform::unnormalized)
      output->rescale (nchan);

    output->set_rate (input->get_rate() / ndat_fft);

    // complex to complex FFT produces a band swapped result
    output->set_nsub_swap (input_nchan);

    get_output()->resize (nbin);
    get_output()->zero ();

  }

  output->set_rate (input->get_rate() / ndat_fft);

  // complex to complex FFT produces a band swapped result
  output->set_nsub_swap (input_nchan);

  // Does it matter when this is done?
  get_output()->set_folding_predictor( bin_divider.get_predictor() );

  bool first = false;

  // set_limits sets idat_start and ndat_fold
  set_limits (input);
  uint64_t idat_end = ndat_fold==0 ? input_ndat : idat_start + ndat_fold;
  if (idat_end > input_ndat) idat_end = input_ndat;
  if (verbose)
    cerr << "dsp::PhaseLockedFilterbank::transformation"
      << " idat_start=" << idat_start
      << " idat_end=" << idat_end
      << endl;

  // if unspecified, the first TimeSeries to be folded will define the
  // start time from which to begin cutting up the observation
  // XXX do this every time???
  //if (bin_divider.get_start_time() == MJD::zero)  
  {
    //cerr << "First call" << endl;
    first = true;
    bin_divider.set_start_time (input->get_start_time() + 
        (double)idat_start / input->get_rate());
  }

  // set up the scratch space
  unsigned polfac = npol==4 ? 2 : 1;
  float* complex_spectrum_dat = scratch->space<float> (nchan * 2 * polfac);
  float* complex_spectrum[2];
  complex_spectrum[0] = complex_spectrum_dat;
  complex_spectrum[1] = complex_spectrum_dat + (polfac-1)*nchan*2;

  if (verbose)
    cerr << "dsp::PhaseLockedFilterbank::transformation enter main loop " 
	 << endl;

  unsigned phase_bin = 0;

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;

  double time_per_fft = double(ndat_fft) / get_input()->get_rate();
  double total_integrated = 0.0;

  unsigned last_used = 0;

  while (more_data) {

    bin_divider.set_bounds( get_input() );

    idat_start = bin_divider.get_idat_start ();

#if 0 
    if (!first && idat_start != last_used)
      throw Error (InvalidState, "dsp::PhaseLockedFilterbank::transformation",
                   "Sample dropped? last="UI64" start="UI64,
                   last_used, idat_start);
#endif

    first = false;
    last_used = idat_start + bin_divider.get_ndat ();

    if (idat_start + ndat_fft > idat_end) 
    {
      bin_divider.discard_bounds( get_input() );
      break;
    }

    phase_bin = bin_divider.get_phase_bin ();

    // Update totals
    get_output()->get_hits()[phase_bin] ++;
    get_output()->ndat_total ++;
    total_integrated += time_per_fft; 

    // Set times as necessary
    MJD time0 = input->get_start_time() + (double)idat_start/input->get_rate();
    MJD time1 = time0 + (double)ndat_fft/input->get_rate();
    if (new_integration)
    {
      if (verbose)
        cerr << "dsp::PhaseLockedFilterbank::transformation "
          << "new_integration set start time" << endl;
      new_integration = false;
      get_output()->set_start_time(time0);
    }
    else 
    {
      get_output()->set_start_time(std::min(output->get_start_time(), time0));
    }
    get_output()->set_end_time(std::max(output->get_end_time(), time1));

    for (unsigned inchan=0; inchan < input_nchan; inchan++) 
    {

      for (unsigned ipol=0; ipol < input_npol; ipol++) 
      {

	const float* dat_ptr = input->get_datptr (inchan, ipol);
	dat_ptr += idat_start * input_ndim;
	  
	if (input_ndim == 1)
	  FTransform::frc1d (ndat_fft, complex_spectrum[ipol], dat_ptr);
	else
	  FTransform::fcc1d (ndat_fft, complex_spectrum[ipol], dat_ptr);


	// square-law detect
        for (unsigned ichan=0; ichan < nchan; ichan++) 
        {
          unsigned out_ipol = ipol;
          if (npol==1) out_ipol = 0;
          float *amps = output->get_datptr(inchan*nchan + ichan, out_ipol);
          amps[phase_bin] += sqr(complex_spectrum[ipol][ichan*2]);
          amps[phase_bin] += sqr(complex_spectrum[ipol][ichan*2+1]);
        }

      } // for each polarization

      // Compute poln cross terms
      if (npol>2) 
      {
        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          float *amps_re = output->get_datptr(inchan*nchan + ichan, 2);
          float *amps_im = output->get_datptr(inchan*nchan + ichan, 3);
          amps_re[phase_bin] += 
            complex_spectrum[0][ichan*2]*complex_spectrum[1][ichan*2]
            + complex_spectrum[0][ichan*2+1]*complex_spectrum[1][ichan*2+1];
          amps_im[phase_bin] += 
            complex_spectrum[0][ichan*2]*complex_spectrum[1][ichan*2+1]
            - complex_spectrum[0][ichan*2+1]*complex_spectrum[1][ichan*2];
        }
      }
    
    } // for each frequency channel

  } // for each big fft (ipart)

  // cerr << "main loop finished" << endl;

  get_buffering_policy()->set_minimum_samples (ndat_fft);
  get_buffering_policy()->set_next_start (idat_start);

  get_output()->increment_integration_length( total_integrated );

  // Check int times
  if (verbose) 
  {
    MJD span = get_output()->get_end_time() - get_output()->get_start_time();
    cerr << "dsp::PhaseLockedFilterbank transformation end "
      << "span=" << span.in_seconds() 
      << " int=" << get_output()->get_integration_length()
      << endl;
  }

}

void dsp::PhaseLockedFilterbank::normalize_output ()
{
// This is unnecessary if the output data is in the expected order.
#if 0 
  unsigned output_nbin = get_output()->get_nbin();
  unsigned output_nchan = get_output()->get_nchan();

  cerr << "dsp::PhaseLockedFilterbank::normalize_output nbin="
       << output_nbin << " nchan=" << output_nchan << endl;

  unsigned* hits = get_output()->get_hits();
  
  for (unsigned ichan=0; ichan < output_nchan; ichan++) {

    //cerr << "hits[" << ichan << "]=" << hits[ichan] << endl;

    float* amps = output->get_datptr (ichan, 0);
    for (unsigned ibin=0; ibin < output_nbin; ibin++)  {
      //amps[ibin] /= hits[ichan];
      amps[ibin] /= hits[ibin];
#if 0
          cerr << "amps[" << ichan << "," << ibin << "]="
               << amps[ibin] << endl;
#endif
    }
  }

  get_output()->set_hits(1);
#endif
}

void dsp::PhaseLockedFilterbank::reset () 
{
  if (verbose)
  {
    cerr << "dsp::PhaseLockedFilterbank::reset" << endl;
    cerr << "dsp::PhaseLockedFilterbank::reset start_time=" 
      << get_output()->get_start_time().printall() << endl;
    cerr << "dsp::PhaseLockedFilterbank::reset end_time="
      << get_output()->get_end_time().printall() << endl;
  }

  Operation::reset();

  if (output)
    output->zero();
}

void dsp::PhaseLockedFilterbank::finish ()
{
  if (verbose)
    cerr << "dsp::PhaseLockedFilterbank::finish" << endl;

}
