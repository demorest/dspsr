#include "dsp/Convolution.h"
#include "dsp/Apodization.h"
#include "dsp/Response.h"

#include "fftm.h"
#include "genutil.h"
#include "Reference.h"

//#define DEBUG

dsp::Convolution::Convolution (const char* _name, Behaviour _type)
  : Transformation<TimeSeries,TimeSeries> (_name, _type)
{
}

dsp::Convolution::~Convolution ()
{
}

//! Set the frequency response function
void dsp::Convolution::set_response (Response* _response)
{
  response = _response;
}

const dsp::Response* dsp::Convolution::get_response() const{
  return response.get();
}

//! Set the apodization function
void dsp::Convolution::set_apodization (Apodization* _function)
{
  apodization = _function; 
}

//! Set the bandpass integrator
void dsp::Convolution::set_bandpass (Response* _bandpass)
{
  bandpass = _bandpass; 
}

/*!
  \pre input TimeSeries must contain phase coherent (undetected) data
  \post output TimeSeries will contain complex (Analytic) data
    
  \post IMPORTANT!! Most backward complex FFT functions expect
  frequency components organized with f0+bw/2 -> f0, f0-bw/2 -> f0.
  The forward real-to-complex FFT produces f0-bw/2 -> f0+bw/2.  To
  save CPU cycles, convolve() does not re-sort the ouput array, and
  therefore introduces a frequency shift in the output data.  This
  results in a phase gradient in the time domain.  Since only
  relative phases matter when calculating the Stokes parameters,
  this effect is basically ignorable for our purposes.
*/
void dsp::Convolution::transformation ()
{
  if (!response) {
    if (verbose)
      cerr << "Convolution::transformation no frequency response" << endl;
    return;
  }

  response->match (input);

  if (bandpass)
    bandpass->match (response);

  // response must have at least two points in it
  if (response->get_ndat() < 2)
    throw_str ("Convolution::transformation invalid response size");

  // if the response has 8 dimensions, then perform matrix convolution
  bool matrix_convolution = response->get_ndim() == 8;

  Signal::State state = input->get_state();
  unsigned npol  = input->get_npol();
  unsigned nchan = input->get_nchan();
  unsigned ndim  = input->get_ndim();

  // if matrix convolution, then there must be two polns
  if (matrix_convolution && npol != 2)
    throw_str ("Convolution::transformation matrix response and input.npol != 2");

  // response must contain a unique kernel for each channel
  if (response->get_nchan() != nchan)
    throw_str ("Convolution::transformation invalid response nsub=%d != nchan=%d",
	       response->get_nchan(), nchan);

  // number of points after first fft
  unsigned n_fft = response->get_ndat();

  //! Complex samples dropped from beginning of cyclical convolution result
  unsigned nfilt_pos = response->get_impulse_pos ();

  //! Complex samples dropped from end of cyclical convolution result
  unsigned nfilt_neg = response->get_impulse_neg ();

  unsigned n_overlap = nfilt_pos + nfilt_neg;

  if (verbose)
    cerr << "Convolution::transformation filt=" << n_fft 
	 << " smear=" << n_overlap << endl;

  // 2 arrays needed: one for each of the forward and backward FFT results
  // 2 floats per complex number
  unsigned pts_reqd = n_fft * 2 * 2;

  if (matrix_convolution)
    // need space for one more complex spectrum
    pts_reqd += n_fft * 2;

  // number of time samples in forward fft and overlap region
  unsigned nsamp_fft = 0;
  unsigned nsamp_overlap = 0;

  if (input->get_state() == Signal::Nyquist) {
    nsamp_fft = n_fft * 2;
    nsamp_overlap = n_overlap * 2;
    pts_reqd += 4;
  }
  else if (input->get_state() == Signal::Analytic) {
    nsamp_fft = n_fft;
    nsamp_overlap = n_overlap;
  }
  else
    throw_str ("Convolution::transformation Invalid state:"
	       + input->get_state_as_string());


#ifdef DEBUG
  fprintf (stderr, "%d:: X:%d NDAT="I64" NFFT=%d NOVERLAP: %d\n", 
	   mpi_rank, (int)matrix_convolution, ndat, nsamp_fft, nsamp_overlap);
  fflush (stderr);
#endif

  // there must be at least enough data for one FFT
  if (input->get_ndat() < nsamp_fft)
    throw_str ("Convolution::transformation error ndat="I64" < nfft=%d",
	       input->get_ndat(), nsamp_fft);

  // the FFT size must be greater than the number of discarded points
  if (nsamp_fft < nsamp_overlap)
    throw_str ("Convolution::transformation error nfft=%d < nfilt=%d",
	       nsamp_fft, nsamp_overlap);

  // valid time samples per FFT
  unsigned nsamp_good = nsamp_fft-nsamp_overlap;

  // number of FFTs for this data block
  unsigned long npart = (input->get_ndat()-nsamp_overlap)/nsamp_good;

  if (verbose)
    cerr << "Convolution::transformation npart=" << npart << endl;

  float* spectrum[2];
  spectrum[0] = float_workingspace (pts_reqd);
  spectrum[1] = spectrum[0];
  if (matrix_convolution)
    spectrum[1] += n_fft * 2;

  float* complex_time  = spectrum[1] + n_fft * 2;

  if (input->get_state() == Signal::Nyquist)
    complex_time += 2;

  // prepare the output TimeSeries
  output->Observation::operator= (*input);

  // valid time samples convolved
  if (input.get() == output.get())
    output->set_ndat (npart * nsamp_good);
  else
    output->resize (npart * nsamp_good);

  // output data is complex
  // notice that nsamp_good is the number of input time samples.
  // therefore, the state must be changed after resize
  output->change_state (Signal::Analytic);

  // nfilt_pos complex points are dropped from the start of the first FFT
  output->change_start_time (nfilt_pos);

  // data will be scaled by the FFT
  if (fft::get_normalization() == fft::nfft)
    // after performing forward and backward FFTs the data will be scaled
    output->rescale (double(nsamp_fft) * double(n_fft));

  if (verbose)
    cerr << "Convolution::transformation scale="<< output->get_scale() <<endl;

  response->mark (output);

  unsigned nbytes_good = nsamp_good * ndim * sizeof(float);
  
  unsigned cross_pol = 1;
  if (matrix_convolution)
    cross_pol = 2;
 
  // temporary things that should not go in and out of scope
  float* ptr = 0;
  unsigned jpol=0;

  unsigned long offset;
  // number of floats to step between each FFT
  unsigned long step = nsamp_good * ndim;

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned ipol=0; ipol < npol; ipol++)
      for (unsigned ipart=0; ipart < npart; ipart++)  {
	
	offset = ipart * step;
		
	for (jpol=0; jpol<cross_pol; jpol++) {
	  
	  if (matrix_convolution)
	    ipol = jpol;
	  
	  ptr = const_cast<float*>(input->get_datptr (ichan, ipol)) + offset;
	  
	  if (apodization) {
	    apodization -> operate (ptr, complex_time);
	    ptr = complex_time;
	  }
	  
	  if (state == Signal::Nyquist)
	    fft::frc1d (nsamp_fft, spectrum[ipol], ptr);

	  else if (state == Signal::Analytic)
	    fft::fcc1d (nsamp_fft, spectrum[ipol], ptr);
	  
	}
	
	if (matrix_convolution) {

	  response->operate (spectrum[0], spectrum[1], ichan);

	  if (bandpass)
	    bandpass->integrate (spectrum[0], spectrum[1], ichan);

	}
	
	else {

	  response->operate (spectrum[ipol], ipol, ichan);

	  if (bandpass)
	    bandpass->integrate (spectrum[ipol], ipol, ichan);

	}
	
	for (jpol=0; jpol<cross_pol; jpol++) {
	  
	  if (matrix_convolution)
	    ipol = jpol;
	  
#ifdef DEBUG
	  fprintf (stderr, "%d:: %d:%d:%ld backward FFT.\n",
		   mpi_rank, ichan, ipol, ipart);
	  fflush (stderr);
#endif
	  // fft back to the complex time domain
	  fft::bcc1d (n_fft, complex_time, spectrum[ipol]);
	  
	  // copy the good (complex) data back into the time stream
	  ptr = output -> get_datptr (ichan, ipol) + offset;
	  memcpy (ptr, complex_time + nfilt_pos*2, nbytes_good);

	}  // for each poln, if matrix convolution
      }  // for each part of the time series
  // for each poln
  // for each channel
}
