/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/ACFilterbank.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Scratch.h"

#include "dsp/Response.h"
#include "dsp/Apodization.h"

#include "FTransform.h"

#include <string.h>

using namespace std;

// #define _DEBUG 1

dsp::ACFilterbank::ACFilterbank () 
  : Transformation<TimeSeries,TimeSeries> ("ACFilterbank", outofplace, true)
{
  nchan = 0;
  nlag = 0;
  form_acf = false;
}

void dsp::ACFilterbank::set_passband (Response* band)
{
  passband = band;
}

extern "C" int mfilter (const unsigned npts, float* spectrum, float* filter);

void dsp::ACFilterbank::transformation ()
{
  // number of timesamples in input data stream
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();

  if (verbose)
    cerr << "dsp::ACFilterbank::transformation input ndat=" << ndat 
	 << " output ndim=" << get_output()->get_ndim() << endl;

  if (nchan < 2)
    throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		 "invalid number of channels = %d", nchan);

  // number of time samples in first fft
  unsigned nsamp_fft = 0;

  // number of time samples by which big ffts overlap
  unsigned nsamp_overlap = 0;

  if (input->get_state() == Signal::Nyquist) {
    nsamp_fft = 2 * nchan;
    nsamp_overlap = 2 * nlag;
  }

  else if (input->get_state() == Signal::Analytic) {
    nsamp_fft = nchan;
    nsamp_overlap = nlag;
  }

  else
    throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		 "invalid input data state = " + tostring(input->get_state()));

  // if given, test the validity of the window function
  if (apodization) {

    if (apodization->get_ndat() != nsamp_fft)
      throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		   "invalid apodization function ndat=%d"
		   " (nfft=%d)", apodization->get_ndat(), nsamp_fft);

    if (input->get_state() == Signal::Analytic 
	&& apodization->get_ndim() != 2)
      throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		   "Signal::Analytic signal. Real apodization function.");

    if (input->get_state() == Signal::Nyquist 
	&& apodization->get_ndim() != 1)
      throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		   "Signal::Nyquist signal. Complex apodization function.");
  }

  if (passband) {
    passband->resize (input->get_npol(), 1, nchan, 1);
    passband->match (input);
  }

  // number of timesamples over which each lag is integrated
  const unsigned nsamp_step = nsamp_fft - nsamp_overlap;

  // number of FFTs that can fit into data
  const unsigned npart = (ndat-nsamp_overlap)/nsamp_step;

  if (npart == 0)
    throw Error (InvalidState, "dsp::ACFilterbank::transformation",
		 "input.ndat="I64" < nfft=%d",
		 input->get_ndat(), nsamp_fft);

  minimum_samps_can_process = nsamp_fft;

  // prepare the output TimeSeries
  {
    get_output()->copy_configuration ( get_input() );
    get_output()->set_state( Signal::PPQQ );
    get_output()->set_npol( get_input()->get_npol() );

    if (form_acf) {
#ifdef _DEBUG
cerr << "dsp::ACFilterbank ACF nlag=" << nlag << " ndim=2" << endl;
#endif
      get_output()->set_nchan( nlag );
      get_output()->set_ndim( 2 );
    }
    else {
#ifdef _DEBUG
cerr << "dsp::ACFilterbank PSD nchan=1 ndim=" << 2*nchan << endl;
#endif
      get_output()->set_nchan( 1 );
      get_output()->set_ndim( 2*nchan );
    }

  }

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());
  if (weighted_output) {
    weighted_output->convolve_weights (nsamp_fft, nsamp_step);
    weighted_output->scrunch_weights (nsamp_step);
  }

  // resize to new number of valid time samples
  output->resize (npart);
  output->set_rate (input->get_rate() / nsamp_step);

  if (FTransform::get_norm() == FTransform::unnormalized)
    output->rescale (nsamp_fft);
  
  if (verbose) cerr << "dsp::ACFilterbank::transformation scale="
                    << output->get_scale() <<endl;

  // number of floating point numbers in a single complex spectrum
  const unsigned nfloat_fft = nchan * 2;
  const unsigned nbytes_fft = nfloat_fft * sizeof(float);
  const unsigned nbytes_half = nchan * sizeof(float);

  // space required for two complex spectra: one from zero-padded, one not
  unsigned scratch_needed = nfloat_fft * 2 + 8;

  // space required to weight in the time domain
  if (apodization)
    scratch_needed += nfloat_fft;

  // divide up the scratch space
  float* spectrum1 = scratch->space<float> (scratch_needed);
  float* spectrum2 = spectrum1 + nfloat_fft + 4;
  float* windowed_time_domain = spectrum2 + nfloat_fft + 4;

  if (verbose)
    cerr << "dsp::ACFilterbank::transformation enter main loop" <<
      " npart=" << npart << " npol=" << npol << endl;

  // number of floats to step between input to filterbank
  const unsigned long in_step = nsamp_step * input->get_ndim();

  // number of floats to step between output from filterbank
  const unsigned long out_step = nchan * 2;
    
  // counters
  unsigned ipol, ipart, idat, ilag;

  // some temporary pointers
  const float* input_ptr = NULL;
  const float* input_datptr = NULL;

  float* output_ptr = NULL;
  float* output_datptr = NULL;

  for (ipol=0; ipol < npol; ipol++) {

    input_datptr  = input->get_datptr (0, ipol);
    if (!form_acf)
      output_datptr = output->get_datptr (0, ipol);

    for (ipart=0; ipart<npart; ipart++) {

#ifdef _DEBUG
cerr << "dsp::ACFilterbank ipol=" << ipol << " ipart=" << ipart << endl;
#endif

      input_ptr = input_datptr;
      input_datptr += in_step;

      output_ptr = output_datptr;
      output_datptr += out_step;

      if (apodization) {
#ifdef _DEBUG
cerr << "dsp::ACFilterbank apodizing" << endl;
#endif
	apodization -> operate (const_cast<float*>(input_ptr), 
                                windowed_time_domain);
	input_ptr = windowed_time_domain;
      }

      // calculate the zero-padded transform

#ifdef _DEBUG
cerr << "dsp::ACFilterbank copy half" << endl;
#endif

      // copy half of the data
      memcpy (spectrum1, input_ptr, nbytes_half);

#ifdef _DEBUG
cerr << "dsp::ACFilterbank zero pad" << endl;
#endif

      // zero pad the rest
      for (idat=nchan; idat<out_step; idat++)
	spectrum1[idat] = 0.0;

#ifdef _DEBUG
cerr << "dsp::ACFilterbank FFT zero-padded nfft=" << nsamp_fft << endl;
#endif

      if (input->get_state() == Signal::Nyquist)
	FTransform::frc1d (nsamp_fft, spectrum2, spectrum1);
      else
	FTransform::fcc1d (nsamp_fft, spectrum2, spectrum1);

#ifdef _DEBUG
cerr << "dsp::ACFilterbank conjugate result" << endl;
#endif

      // complex conjugate
      for (idat=1; idat<out_step; idat+=2)
	spectrum2[idat] = -spectrum2[idat];

#ifdef _DEBUG
cerr << "dsp::ACFilterbank FFT data" << endl;
#endif
 
      // calculate the normal transform
      if (input->get_state() == Signal::Nyquist)
	FTransform::frc1d (nsamp_fft, spectrum1, input_ptr);
      else
	FTransform::fcc1d (nsamp_fft, spectrum1, input_ptr);

      if (passband) {
#ifdef _DEBUG
cerr << "dsp::ACFilterbank integrate passband" << endl;
#endif
	passband->integrate (spectrum1, ipol);
      }

#ifdef _DEBUG
cerr << "dsp::ACFilterbank form PSD" << endl;
#endif

      // multiply the two complex spectra
      mfilter (nchan, spectrum1, spectrum2);

      if (!form_acf)
	memcpy (output_ptr, spectrum1, nbytes_fft);
      else {

#ifdef _DEBUG
cerr << "dsp::ACFilterbank form ACF" << endl;
#endif

	// if forming lags, do the inverse fft and multiplex the lag data
	FTransform::bcc1d (nchan, spectrum2, spectrum1);
	input_ptr = spectrum2;

	for (ilag=0; ilag < nlag; ilag++) {
	  output_ptr = output->get_datptr (ilag, ipol) + ipart * 2;
	  output_ptr[0] = *input_ptr; input_ptr++;
	  output_ptr[1] = *input_ptr; input_ptr++;
	}
      }

    } // for each fft (ipart)

  } // for each polarization

}




