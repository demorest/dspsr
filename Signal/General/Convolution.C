#include "Convolution.h"
#include "Timeseries.h"
#include "fftm.h"
#include "genutil.h"

//#define DEBUG

dsp::Convolution::Convolution() : Operation ("Convolution", true)
{
  apodizing = 0;
  bandpass = 0;

  nfilt_pos = nfilt_neg = 0;
}

void dsp::Convolution::operation ()
{
  if (!input)
    throw_str ("Convolution::operate no input");

  if (!output)
    throw_str ("Convolution::operate no output");

  // response must have at least two points in it
  if (response.get_ndat() < 2 * response.get_nsub())
    throw_str ("Convolution::operate invalid response size");

  // number of points in response must be an even multiple of sub-divisions
  if (response.get_ndat() % response.get_nsub())
    throw_str ("Convolution::operate invalid number of response channels");

  // if the response has 4 dimensions, then perform matrix convolution
  bool matrix_convolution = response.get_npol() == 4;

  Timeseries::State state = input->get_state();
  int npol  = input->get_npol();
  int nchan = input->get_nchan();
  int ndim  = input->get_ndim();

  // if matrix convolution, then there must be two polns
  if (matrix_convolution && npol != 2)
    throw_str ("Convolution::operate matrix response and input.npol != 2");

  // response must contain a unique kernel for each channel
  if (response.get_nsub() != nchan)
    throw_str ("Convolution::operate invalid response nsub=%d != nchan=%d",
	       response.get_nsub(), nchan);

  // number of points after first fft
  int n_fft = response.get_ndat()/response.get_nsub();
  int n_overlap = nfilt_pos + nfilt_neg;

  if (verbose)
    cerr << "Convolution::operate filt=" << n_fft 
	 << " smear=" << n_overlap << endl;

  // 2 arrays needed: one for each of the forward and backward FFT results
  // 2 floats per complex number
  int pts_reqd = n_fft * 2 * 2;

  if (matrix_convolution)
    // need space for one more complex spectrum
    pts_reqd += n_fft * 2;

  // number of time samples in forward fft and overlap region
  int nsamp_fft = 0;
  int nsamp_overlap = 0;

  if (input->get_state() == Timeseries::Nyquist) {
    nsamp_fft = n_fft * 2;
    nsamp_overlap = n_overlap * 2;
    pts_reqd += n_fft * 2 + 4;
  }
  else if (input->get_state() == Timeseries::Analytic) {
    nsamp_fft = n_fft;
    nsamp_overlap = n_overlap;
  }
  else
    throw_str ("Convolution::operate Invalid state:" + input->get_state_str());

#ifdef DEBUG
  fprintf (stderr, "%d:: X:%d NDAT="I64" NFFT=%d NOVERLAP: %d\n", 
	   mpi_rank, (int)matrix_convolution, ndat, nsamp_fft, nsamp_overlap);
  fflush (stderr);
#endif

  int nsamp_good = nsamp_fft-nsamp_overlap;   // valid time samples per FFT
  if (nsamp_good < 0)
    throw_str ("Convolution::operate invalid nfft=%d nfilt=%d",
	       nsamp_fft, n_overlap);

  // number of FFTs for this data block
  unsigned long nparts = (input->get_ndat()-nsamp_overlap)/nsamp_good;
  if (nparts == 0)
    throw_str ("Convolution::operate invalid ndat="I64" nfilt=%d ngood=%d",
	       input->get_ndat(), nsamp_overlap, nsamp_good);

  float* complex_spectrum[2];
  complex_spectrum[0] = float_workingspace (pts_reqd);
  complex_spectrum[1] = complex_spectrum[0];
  if (matrix_convolution)
    complex_spectrum[1] += n_fft * 2;

  float* complex_time  = complex_spectrum[1] + n_fft * 2;
  float* complex_sort  = NULL;

  if (input->get_state() == Timeseries::Nyquist) {
    complex_time += 2;
    complex_sort = complex_time + n_fft * 2;
  }

#ifdef DEBUG
  cerr << mpi_rank << ":: convolve " << ndat << " points in " << nsamp_fft 
       << " point chunks " << nparts << " times" << endl;
  if (state == Nyquist)
    cerr << mpi_rank << ":: Nyquist sampled data stream" << endl;
  else if (state == Analytic)
    cerr << mpi_rank << ":: Quadrature sampled data stream" << endl;
#endif
 

  int nbytes = nsamp_good * ndim * sizeof(float);
  
  int cross_pol = 1;
  if (matrix_convolution)
    cross_pol = 2;
 
  dsp::filter filt;
  dsp::filter bpass;

  // temporary things that should not go in and out of scope
  float* ptr = 0;
  int ipol=0, jpol=0;
  unsigned long ipart=0;

  unsigned long index;
  // number of floats to step between each FFT
  unsigned long step = nsamp_good * ndim;

  for (int ichan=0; ichan < nchan; ichan++) {

    // these quickly point to the subsets that apply to each channel
    filt.external (response, ichan);
    if (bandpass)
      bpass.external (*bandpass, ichan);
    
    for (ipol=0; ipol<npol; ipol++)  {
      
      for (ipart=0; ipart<nparts; ipart++)  {
	
	index = ipart * step;
		
	for (jpol=0; jpol<cross_pol; jpol++) {
	  
	  if (matrix_convolution)
	    ipol = jpol;
	  
	  ptr = const_cast<float*>(input->get_datptr (ichan, ipol)) + index;
	  
	  if (apodizing) {
	    apodizing -> operate (ptr, complex_time);
	    ptr = complex_time;
	  }
	  
	  if (state == Timeseries::Nyquist) {
	    fft::frc1d (nsamp_fft, complex_spectrum[ipol], ptr,
			0, complex_sort);
	  }
	  else if (state == Timeseries::Analytic) {
	    /* complex sampled data coming in */
	    fft::fcc1d (nsamp_fft, complex_spectrum[ipol], ptr);
	  }
	  
	}
	
#ifdef DEBUG
	fprintf (stderr, "%d:: %d:%d:%ld filter.\n",
		 mpi_rank, ichan, ipol, ipart);
	fflush (stderr);
#endif
	
	if (matrix_convolution) {
	  response.operate (complex_spectrum[0], complex_spectrum[1]);
	  if (bandpass) {
#ifdef DEBUG
	    fprintf (stderr, "%d:: %d:%d:%ld bandpass.\n",
		     mpi_rank, ichan, ipol, ipart);
	    fflush (stderr);
#endif
	    bpass.integrate (complex_spectrum[0], complex_spectrum[1]);
	  }
	}
	
	else {
	  filt.operate (ipol, complex_spectrum[ipol]);
	  if (bandpass) {
#ifdef DEBUG
	    fprintf (stderr, "%d:: %d:%d:%ld bandpass.\n",
		     mpi_rank, ichan, ipol, ipart);
	    fflush (stderr);
#endif
	    bpass.integrate (ipol, complex_spectrum[ipol]);
	  }
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
	  fft::bcc1d (n_fft, complex_time, complex_spectrum[ipol]);
	  
	  // copy the good (complex) data back into the time stream
	  ptr = output -> get_datptr (ichan, ipol) + index;
	  memcpy (ptr, complex_time + nfilt_pos*2, nbytes);

	}  // for each poln, if matrix convolution
      }  // for each part of the time series
    }  // for each poln, if simple convolution
  }  // for each channel of filterbank


  // valid time samples convolved
  output->set_ndat (nparts * nsamp_good);

  // output data is complex
  output->change_state (Timeseries::Analytic);

  // nfilt_pos complex points were dropped from the start of the first FFT
  output->change_start_time (nfilt_pos);

  // data will be scaled by the FFT
  if (fft::get_normalization() == fft::nfft)
    // after performing forward and backward FFTs the data will be scaled
    output->rescale (double(nsamp_fft) * double(n_fft));

}
