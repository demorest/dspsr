#include "dsp/ACFilterbank.h"
#include "dsp/WeightedTimeSeries.h"

#include "fftm.h"
#include "spectra.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"

// #define _DEBUG

dsp::ACFilterbank::ACFilterbank () 
  : Transformation<TimeSeries,TimeSeries> ("ACFilterbank", outofplace, true)
{
  nchan = 0;
  nlag = 0;
  form_acf = false;
}

void dsp::ACFilterbank::transformation ()
{
  // number of timesamples in input data stream
  const uint64 ndat = input->get_ndat();
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
		 "invalid input data state = " + input->get_state_as_string());

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
    passband->resize (input->get_npol(), nchan, 1, 1);
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

    if (form_acf) {
      get_output()->set_nchan( nlag );
      get_output()->set_ndim( 2 );
    }
    else {
      get_output()->set_nchan( 1 );
      get_output()->set_ndim( 2*nchan );
    }

    get_output()->set_state( Signal::Analytic );
    get_output()->set_npol( get_input()->get_npol() );
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

  if (fft::get_normalization() == fft::nfft)
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
  float* spectrum1 = float_workingspace (scratch_needed);
  float* spectrum2 = spectrum1 + nfloat_fft + 4;
  float* windowed_time_domain = spectrum2 + nfloat_fft + 4;

  if (verbose)
    cerr << "dsp::ACFilterbank::transformation enter main loop " <<
      " npart:" << npart <<
      " npol:" << input->get_npol() << endl;

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

      input_ptr = input_datptr;
      input_datptr += in_step;

      output_ptr = output_datptr;
      output_datptr += out_step;

      if (apodization) {
	apodization -> operate (const_cast<float*>(input_ptr), 
                                windowed_time_domain);
	input_ptr = windowed_time_domain;
      }

      // calculate the zero-padded transform

      // copy half of the data
      memcpy (spectrum1, input_ptr, nbytes_half);

      // complex conjugate
      for (idat=1; idat<nchan; idat+=2)
	spectrum1[idat] = -spectrum1[idat];

      // zero pad the rest
      for (idat=0; idat<nchan; idat++)
	spectrum1[idat+nchan] = 0.0;

      if (input->get_state() == Signal::Nyquist)
	fft::frc1d (nsamp_fft, spectrum2, spectrum1);
      else
	fft::fcc1d (nsamp_fft, spectrum2, spectrum1);
 
      // calculate the normal transform
      if (input->get_state() == Signal::Nyquist)
	fft::frc1d (nsamp_fft, spectrum1, input_ptr);
      else
	fft::fcc1d (nsamp_fft, spectrum1, input_ptr);

      // integrate the normal transform
      if (passband)
	passband->integrate (spectrum1, ipol);
      
      // multiply the two complex spectra
      mfilter (nchan, spectrum1, spectrum2);

      if (!form_acf)
	memcpy (output_ptr, spectrum1, nbytes_fft);
      else {
	// if forming lags, do the inverse fft and multiplex the lag data
	fft::bcc1d (nchan, spectrum2, spectrum1);
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




