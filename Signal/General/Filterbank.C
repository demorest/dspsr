#include "dsp/Filterbank.h"
#include "dsp/TimeSeries.h"

#include "fftm.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"

#include "genutil.h"

// #define _DEBUG

dsp::Filterbank::Filterbank () : Convolution ("Filterbank", outofplace)
{
  nchan = 0;
  time_res = 1;
  freq_res = 1;
}


void dsp::Filterbank::transformation ()
{
  if (nchan < 2)
    throw Error (InvalidState, "dsp::Filterbank::transformation",
		 "invalid number of channels = %d", nchan);

  //! Complex samples dropped from beginning of cyclical convolution result
  unsigned nfilt_pos = 0;

  //! Complex samples dropped from end of cyclical convolution result
  unsigned nfilt_neg = 0;

  if (response) {

    // convolve the data with a frequency response function during
    // filterbank construction...

    response -> match (input, nchan);
    if (response->get_nchan() != nchan)
      throw Error (InvalidState, "dsp::Filterbank::transformation",
		   "Response.nchan=%d != nchan=%d",
		   response->get_nchan(), nchan);

    nfilt_pos = response->get_impulse_pos ();
    nfilt_neg = response->get_impulse_neg ();

    freq_res = response->get_ndat();

  }

  // number of complex values in the result of the first fft
  unsigned n_fft = nchan * freq_res;

  // number of complex samples invalid in result of small ffts
  unsigned n_filt = nfilt_pos + nfilt_neg;

  // number of time samples in first fft
  unsigned nsamp_fft = 0;
  // number of time samples by which big ffts overlap
  unsigned nsamp_overlap = 0;

  if (input->get_state() == Signal::Nyquist) {
    nsamp_fft = 2 * n_fft;
    nsamp_overlap = 2 * n_filt * nchan;
  }

  else if (input->get_state() == Signal::Analytic) {
    nsamp_fft = n_fft;
    nsamp_overlap = n_filt * nchan;
  }

  else
    throw Error (InvalidState, "dsp::Filterbank::transformation",
		 "invalid input data state = " + input->get_state_as_string());

  // if given, test the validity of the window function
  if (apodization) {

    if (apodization->get_ndat() != nsamp_fft)
      throw Error (InvalidState, "dsp::Filterbank::transformation",
		   "invalid apodization function ndat=%d"
		   " (nfft=%d)", apodization->get_ndat(), nsamp_fft);

    if (input->get_state() == Signal::Analytic 
	&& apodization->get_ndim() != 2)
      throw Error (InvalidState, "dsp::Filterbank::transformation",
		   "Signal::Analytic signal. Real apodization function.");

    if (input->get_state() == Signal::Nyquist 
	&& apodization->get_ndim() != 1)
      throw Error (InvalidState, "dsp::Filterbank::transformation",
		   "Signal::Nyquist signal. Complex apodization function.");
  }

  // number of timesamples between start of each big fft
  int nsamp_step = nsamp_fft - nsamp_overlap;

  // matrix convolution
  bool matrix_convolution = false;

  if (response) {

    // if the response has 8 dimensions, then perform matrix convolution
    matrix_convolution = (response->get_ndim() == 8);

    if (verbose)
      fprintf (stderr, "dsp::Filterbank::transformation with %s convolution\n",
	       (matrix_convolution)?"matrix":"complex");

    if (matrix_convolution && input->get_npol() != 2)
	throw Error (InvalidState, "dsp::Filterbank::transformation",
		     "matrix convolution and input.npol != 2");
  }

  if (passband) {

    if (response)
      passband -> match (response);

    else {
      passband->resize (input->get_npol(), nchan, freq_res, 1);
      passband->match (input);
    }

  }

  // if the time_res is greater than 1, the ffts must overlap by ntimesamp.
  // this may be in addition to any overlap necessary due to deconvolution.
  // nsamp_step is analogous to ngood in Convolution::transformation
  int nsamp_tres = nchan / time_res;
  if (nsamp_tres < 1)
    throw Error (InvalidState, "dsp::Filterbank::transformation",
		 "time resolution:%d > no.channels:%d\n", time_res, nchan);

  unsigned ndat = input->get_ndat();

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  unsigned npart = (ndat-(nchan-nsamp_tres)-nsamp_overlap)/nsamp_step;
  // points kept from each small fft
  unsigned nkeep = freq_res - n_filt;

  if (npart == 0)
    throw Error (InvalidState, "dsp::Filterbank::transformation",
		 "input.ndat="I64" < nfft=%d",
		 input->get_ndat(), nsamp_fft);

  // prepare the output TimeSeries
  output->Observation::operator= (*input);

  // output data will be complex
  output->set_state (Signal::Analytic);

  // output data will be multi-channel
  output->set_nchan (nchan);

  // resize to new number of valid time samples
  output->resize (npart * nkeep * time_res);

  double scalefac = 1.0;

  if (verbose) {
    cerr << "dsp::Filterbank::transformation\n"
      "  n_fft="<< n_fft <<" and freq_res="<< freq_res << "\n"
      "  fft::normalization=" <<
      (fft::get_normalization() == fft::nfft?"fft::nfft":
      fft::get_normalization() == fft::normal?"fft::normal":
      "unknown") << endl;
  }

  if (fft::get_normalization() == fft::nfft)
    scalefac = double(n_fft) * double(freq_res);

  else if (fft::get_normalization() == fft::normal)
    scalefac = double(n_fft) / double(freq_res);

  output->rescale (scalefac);
  
  if (verbose)
    cerr << "dsp::Filterbank::transformation scale="<< output->get_scale() <<endl;

  // output data will have new sampling rate
  // NOTE: that nsamp_fft already contains the extra factor of two required
  // when the input TimeSeries is Signal::Nyquist (real) sampled
  double ratechange = double(freq_res * time_res) / double (nsamp_fft);
  output->set_rate (input->get_rate() * ratechange);

  // complex to complex FFT produces a band swapped result
  if (input->get_state() == Signal::Analytic)
    output->set_swap (true);

  // if freq_res is even, then each sub-band will be centred on a frequency
  // that lies on a spectral bin *edge* - not the centre of the spectral bin
  output->set_dc_centred (freq_res%2);

  // increment the start time by the number of samples dropped from the fft
  output->change_start_time (nfilt_pos);

  // enable the Response to record its effect on the output Timeseries
  if (response)
    response->mark (output);

  // initialize scratch space for FFTs
  unsigned bigfftsize = nchan * freq_res * 2;
  // also need space to hold backward FFTs
  unsigned scratch_needed = bigfftsize + 2 * freq_res;

  if (apodization)
    scratch_needed += bigfftsize;

  if (matrix_convolution)
    scratch_needed += bigfftsize;

  // divide up the scratch space
  float* complex_spectrum[2];
  complex_spectrum[0] = float_workingspace (scratch_needed);
  complex_spectrum[1] = complex_spectrum[0];
  if (matrix_convolution)
    complex_spectrum[1] += bigfftsize;

  float* complex_time = complex_spectrum[1] + bigfftsize;
  float* windowed_time_domain = complex_time + 2 * freq_res;

  // the number of floats skipped between the end of a point and beginning
  // of next point (complex)
  int tres_skip = (time_res - 1) * 2;

  unsigned cross_pol = 1;
  if (matrix_convolution)
    cross_pol = 2;

  if (verbose)
    cerr << "dsp::Filterbank::transformation enter main loop " <<
      " npart:" << npart <<
      " cpol:" << cross_pol <<
      " npol:" << input->get_npol() << endl;

  // number of floats to step between input to filterbank
  unsigned long in_step = nsamp_step * input->get_ndim();

  // number of floats to step between output from filterbank
  unsigned long out_step = nkeep * time_res * 2;

  // number of floats to step between additional time resolution
  unsigned long tres_step = nsamp_tres * input->get_ndim();

  // counters
  unsigned ipt, itres, ipol, jpol, ichan, ipart;

  unsigned npol = input->get_npol();

  // offsets into input and output
  unsigned long in_offset, tres_offset, out_offset;

  // some temporary pointers
  float* time_dom_ptr = NULL;  
  float* freq_dom_ptr = NULL;
  float* data_into = NULL;
  float* data_from = NULL;

  for (ipart=0; ipart<npart; ipart++) {

    in_offset = ipart * in_step;
    out_offset = ipart * out_step;

    for (ipol=0; ipol < npol; ipol++) {

      for (itres=0; itres < time_res; itres ++) {

	tres_offset = itres * tres_step;

	for (jpol=0; jpol<cross_pol; jpol++) {
	  if (matrix_convolution)
	    ipol = jpol;
	  
	  time_dom_ptr = const_cast<float*>(input->get_datptr (0, ipol));
	  time_dom_ptr += in_offset + tres_offset;
	  
	  if (apodization) {
	    apodization -> operate (time_dom_ptr, windowed_time_domain);
	    time_dom_ptr = windowed_time_domain;
	  }

	  if (input->get_state() == Signal::Nyquist)
	    fft::frc1d (nsamp_fft, complex_spectrum[ipol], time_dom_ptr);

	  else
	    fft::fcc1d (nsamp_fft, complex_spectrum[ipol], time_dom_ptr);

	}

	if (matrix_convolution) {

	  if (passband && itres==0)
	    passband->integrate (complex_spectrum[0], complex_spectrum[1]);

	  // cross filt can be set only if there is a response
	  response->operate (complex_spectrum[0], complex_spectrum[1]);

	}
	
	else {

	  if (passband && itres==0)
	    passband->integrate (complex_spectrum[ipol], ipol);

	  if (response)
	    response->operate (complex_spectrum[ipol], ipol);

	}
	
	for (jpol=0; jpol<cross_pol; jpol++) {
	  if (matrix_convolution)
	    ipol = jpol;
	  
	  freq_dom_ptr = complex_spectrum[ipol];
	  
	  if (freq_res == 1) {
	    for (ichan=0; ichan < nchan; ichan++) {
	      data_into = output->get_datptr (ichan, ipol) + out_offset + itres*2;

	      *data_into = *freq_dom_ptr;     // copy the Re[z]
	      data_into++; freq_dom_ptr ++;
	      *data_into = *freq_dom_ptr;     // copy the Im[z]
	      freq_dom_ptr ++;
	    }
	    continue;
	  }
	  
	  // freq_res > 1 requires a backward fft into the time domain
	  // for each channel
	  
	  for (ichan=0; ichan < nchan; ichan++) {

	    fft::bcc1d (freq_res, complex_time, freq_dom_ptr);

	    freq_dom_ptr += freq_res*2;

	    data_into = output->get_datptr (ichan, ipol) + out_offset+itres*2;
	    data_from = complex_time + nfilt_pos*2;  // complex nos.

	    for (ipt=0; ipt < nkeep; ipt++) {
	      *data_into = *data_from;     // copy the Re[z]
	      data_into ++; data_from ++;
	      *data_into = *data_from;     // copy the Im[z]
	      data_into ++; data_from ++;
	      data_into += tres_skip;      // leave space for the in-betweeners
	    }
	    
	  } // for each channel

	} // for each cross poln

      } // for each element of finer time resolution
      
    } // for each polarization
    
  } // for each big fft (ipart)

  if (verbose)
    cerr << "dsp::Filterbank::transformation exit." << endl;
}

#if 0

void filterbank::scattered_power_correct (float_Stream& dispersed_power,
					  const a2d_correct& digitization)
{
  if (!ppweight)
    throw string ("filterbank::scattered_power_correct "
	       "ERROR: no time weights");
  
  // check the validity of this transformation
  if (dispersed_power.get_state() != Detected)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power must be detected");

  if (dispersed_power.rate != rate)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power must have same sampling rate");

  if (dispersed_power.start_time > start_time)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power does not start early enough");

  if (dispersed_power.end_time < end_time)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power ends too soon");

  if (verbose)
    cerr << "filterbank::scattered_power_correct " << endl
	 << " start:" << start_time << " end:" << end_time << endl
	 << " dp.start:" << dispersed_power.start_time
	 << " dp.end:" << dispersed_power.end_time  << endl;

  double offset_time = (start_time - dispersed_power.start_time).in_seconds();
  unsigned offset_samples = (unsigned) floor (offset_time * rate + 0.5);

  if (verbose)
    cerr << "filterbank::scattered_power_correct "
	 << "offset (us):" << offset_time * 1e6 
	 << " offset (samples):" << offset_samples << endl;

  // sanity check
  if (dispersed_power.ndat - offset_samples < ndat)
    throw Error (InvalidState, "filterbank::scattered_power_correct "
	       " dp.ndat="I64" < ndat="I64" + offset="I64,
	       dispersed_power.ndat, ndat, offset_samples);
  
  // only PP and QQ are corrected...
  int cpol = 2;
  if (npol == 1)
    // ...unless only Signal::Stokes I remains
    cpol = 1;

  if (dispersed_power.npol != cpol)
    throw Error (InvalidState, "filterbank::scattered_power_correct "
	       "dispersed power must have npol=%d", cpol);

  if (!( (get_state() == Detected) || (get_state() == Signal::Coherence) ))
    throw string ("filterbank::scattered_power_correct invalid state="
		  + state_str());

  double normalize = scale / dispersed_power.scale;

  if (verbose)
    cerr << "filterbank::scattered_power_correct "
	 << "scale:" << scale << " dp.scale:" << dispersed_power.scale
	 << " normalize:" << normalize << endl
	 << " correct " << cpol
	 << " polns by " << nchan << " chans by " << ndat << " "
	 << state_str() << " pts" << endl;

  int vincr = 0;    // steps between subsequent time samples in dispersed power
  float* vptr = 0;  // points to ipol-T0 in the dispersed power

  double cfac = 0.0;

  int ipol;
  Int64 ipt, endpt;
    
  for (ipol=0; ipol < cpol; ipol++) {

    vptr = dispersed_power.datptr (0, ipol, vincr);
    vptr += offset_samples * vincr;

    ipt = 0;
    endpt = ppweight;

    for (unsigned iwt=0; iwt<nweights; iwt++) {

      if (weights[ipol][iwt] == 0)
	cfac = 0;
      else
	// the nchan denominator is absorbed in the dispersed_power scale
	cfac = (1.0 - digitization.spc_factor(weights[ipol][iwt])) * normalize;
      
      if (endpt > ndat)
	endpt = ndat;
      
      // set the float_Stream to equal the scattered power correction
      for (; ipt<endpt; ipt++) {
	*vptr *= cfac;
	vptr += vincr;
      }

      endpt += ppweight;

    } // for each weight
  
    // sanity check
    if (ipt != ndat)
      throw Error (InvalidState, "filterbank::scattered_power_correct\n"
		 " sanity check ipt="I64" should equal ndat="I64, ipt, ndat);

  } // for each polarization
  
  register float* vp = 0;
  register float* fp = 0; // points to F0-ipol-T0 in the filterbank
  register int fincr = 0;   // step between subsequent time samples in filterbank

  for (ipol=0; ipol < cpol; ipol++) {

    vptr = dispersed_power.datptr (0, ipol, vincr);
    vptr += offset_samples * vincr;

    for (int ichan=0; ichan < nchan; ichan++) {

      fp = datptr (ichan, ipol, fincr);
      vp = vptr;

      for (ipt=0; ipt<ndat; ipt++)  {
	*fp -= *vp;
	fp += fincr;
	vp += vincr;
      }
 
    } // for each channel
    
  } // for each polarization

}

void filterbank::Hanning (int degree)
{
  if (state < Detected)
    throw string ("filterbank::Hanning called on undetected data");

  SignalProcessing::Window triangle;

  // construct the parzen window triangle for real data
  triangle.Parzen ((degree-1) * 2 + 1, false);
  triangle.normalize();

  scrunch (triangle);
}

#endif
