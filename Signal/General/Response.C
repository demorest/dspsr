#include <assert.h>
#include <math.h>

#include "dsp/Response.h"
#include "dsp/Observation.h"

#include "spectra.h"
#include "Jones.h"
#include "genutil.h"
#include "cross_detect.h"

//#define _DEBUG

/*! If specified, this attribute restricts the value for ndat chosen by 
  the set_optimal_ndat method, enabling the amount of RAM used by the calling
  process to be limited. */
unsigned dsp::Response::ndat_max = 0;

dsp::Response::Response ()
{
  impulse_pos = impulse_neg = 0;

  whole_swapped = chan_swapped = dc_centred = false;

  npol = 2;
  ndim = 1;
  nchan = 1;
}

/*! The ordering of frequency channels in the response function depends 
  upon:
  <UL>
  <LI> the state of the input Observation (real or complex); and </LI>
  <LI> Operations to be performed upon the Observation 
       (e.g. simultaneous filterbank) </LI>
  </UL>
  As well, sub-classes of Response may need to dynamically check, refine, or
  define their frequency response function based on the state of the
  input Observation or the number of channels into which it will be divided.

  \param input Observation to which the frequency response is to be matched

  \param channels If specified, the number of filterbank channels into
  which the input Observation will be divided.  Response::match does not
  use this parameter, but sub-classes may find it useful.

 */
void dsp::Response::match (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Response::match input.nchan=" << input->get_nchan()
	 << " channels=" << channels << endl;

  if ( input->get_nchan() == 1 ) {

    // if the input Observation is single-channel, complex sampled
    // data, then the first forward FFT performed on this data will
    // result in a swapped spectrum
    if ( input->get_state() == Signal::Analytic && !whole_swapped ) {
      if (verbose)
	cerr << "dsp::Response::match swap whole" << endl;
      swap (false);
    }
  }      
  else  {

    // if the filterbank channels are centred on DC
    if ( input->get_dc_centred() && !dc_centred ) {
      if (verbose)
	cerr << "dsp::Response::match rotate half channel" << endl;

      if ( chan_swapped )
        swap (true);

      rotate (-int(ndat/2));
      dc_centred = true;
    }

    // if the input Observation is multi-channel, complex sampled data,
    // then each FFT performed will result in little swapped spectra
    if ( input->get_state() == Signal::Analytic && !chan_swapped ) {
      if (verbose)
	cerr << "dsp::Response::match swap channels (nchan=" << nchan << ")"
	     << endl;
      swap (true);
    }

    // the ordering of the filterbank channels may be swapped
    if ( input->get_swap() && !whole_swapped ) {
      if (verbose)
	cerr << "dsp::Response::match swap whole (nchan=" << nchan << ")"
	     << endl;
      swap (false);
    }
  }
}

//! Returns true if the dimension and ordering match
bool dsp::Response::matches (const Response* response)
{
  return
    whole_swapped == response->whole_swapped &&
    chan_swapped == response->chan_swapped &&
    dc_centred == response->dc_centred &&

    nchan == response->get_nchan() &&
    ndat == response->get_ndat();
    // Shape::matches (response);

}

//! Match the frequency response to another Response
void dsp::Response::match (const Response* response)
{
  if (matches (response))
    return;

  cerr << "dsp::Response::match Response" << endl;
  resize (npol, response->get_nchan(),
	  response->get_ndat(), ndim);
  
  whole_swapped = response->whole_swapped;
  chan_swapped = response->chan_swapped;
  dc_centred = response->dc_centred;
  
  zero();
}

void dsp::Response::mark (Observation* output)
{
  
}

//! Set the flag for a bin-centred spectrum
void dsp::Response::set_dc_centred (bool _dc_centred)
{
  dc_centred = _dc_centred;
}

void dsp::Response::naturalize ()
{
  if (verbose)
    cerr << "dsp::Response::naturalize" << endl;

  if ( whole_swapped ) {
    if (verbose)
      cerr << "dsp::Response::naturalize whole bandpass swap" << endl;
    swap (false);
  }

  if ( chan_swapped ) {
    if (verbose)
      cerr << "dsp::Response::naturalize sub-bandpass swap" << endl;
    swap (true);
  }
  
  if ( dc_centred ) {
    if (verbose)
      cerr << "dsp::Response::naturalize rotation" << endl;
    rotate (ndat/2);
    dc_centred = false;
  }
}

/*!  Using the impulse_pos and impulse_neg attributes, this method
  determines the minimum acceptable ndat for use in convolution.  This
  is given by the smallest power of two greater than or equal to the
  twice the sum of impulse_pos and impulse_neg. */
unsigned dsp::Response::get_minimum_ndat () const
{
  unsigned nsmear = impulse_pos + impulse_neg;
  
  if (nsmear == 0)
    return 0;
  cerr << "Altered minimum n_dat to remove extra power of two" << endl;
  // return (unsigned) pow (2.0, 1.0 + ceil( log((double)nsmear)/log(2.0) ));
  return (unsigned) pow (2.0, ceil( log((double)nsmear)/log(2.0) ));
}

/*!  Using the get_minimum_ndat method and the max_ndat static attribute,
  this method determines the optimal ndat for use in convolution. */
void dsp::Response::set_optimal_ndat ()
{
  unsigned ndat_min = get_minimum_ndat ();

  if (verbose) 
    cerr << "Response::set_optimal_ndat minimum ndat=" << ndat_min << endl;
  
  if (ndat_max && ndat_max < ndat_min)
    throw_str ("Response::set_optimal_ndat specified maximum ndat (%d)" 
	       " < required minimum ndat (%d)", ndat_max, ndat_min);

  int optimal_ndat = optimal_fft_length (impulse_pos+impulse_neg,
					 ndat_max, verbose);
  if (optimal_ndat < 0)
    throw_str ("Response::set_optimal_ndat optimal_fft_length failed");

  resize (npol, nchan, optimal_ndat, ndim);
}


void dsp::Response::check_ndat () const
{
  if (ndat_max && ndat > ndat_max)
    throw_str ("Response::check_ndat specified maximum ndat (%d)" 
	       " < specified ndat (%d)", ndat_max, ndat);

  unsigned ndat_min = get_minimum_ndat ();

  if (verbose) 
    cerr << "Response::check_ndat minimum ndat=" << ndat_min << endl;
  
  if (ndat < ndat_min)
    throw_str ("Response::check_ndat specified ndat (%d)" 
	       " < required minimum ndat (%d)", ndat, ndat_min);
}
  

// /////////////////////////////////////////////////////////////////////////

/*! Multiplies an array of complex points by the complex response

  \param ipol the polarization of the data (Response may optionally
  contain a different frequency response function for each polarization)
  
  \param data an array of nchan*ndat complex numbers */

void dsp::Response::operate (float* data, unsigned ipol, int ichan)
{
  assert (ndim == 2);

  // one filter may apply to two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }
    
  register float* d_from = data;
  register float* d_to = data;
  register float* f_p = buffer + offset * ipol + ichan * ndat * ndim;

#ifdef _DEBUG
  cerr << "dsp::Response::operate nchan=" << nchan << " ipol=" << ipol 
       << " buf=" << buffer << " f_p=" << f_p
       << " off=" << offset(ipol) << endl;
#endif

  // the idea is that by explicitly calling the values from the
  // arrays into local stack space, the routine should run faster
  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;
  
  for (unsigned ipt=0; ipt<npts; ipt++) {
    d_r = *d_from; d_from ++;
    d_i = *d_from; d_from ++;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d_to = f_r * d_r - f_i * d_i; d_to ++;
    *d_to = f_i * d_r + f_r * d_i; d_to ++;
  }
}


// /////////////////////////////////////////////////////////////////////////

/*! Adds the square of each complex point to the current power spectrum

  \param data an array of nchan*ndat complex numbers 

  \param ipol the polarization of the data (Response may optionally
  integrate a different power spectrum for each polarization)
  
*/
void dsp::Response::integrate (float* data, unsigned ipol, int ichan)
{
  assert (ndim == 1);
  assert (npol != 4);

  // may be used to integrate total intensity from two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }
   
  register float* d_p = data;
  register float* f_p = buffer + offset * ipol + ichan * ndat * ndim;

#ifdef _DEBUG
  cerr << "dsp::Response::integrate ipol=" << ipol 
       << " buf=" << buffer << " f_p=" << f_p
       << "off=" << offset(ipol) << endl;
#endif

  register float d;
  register float t;

  for (unsigned ipt=0; ipt<npts; ipt++) {
    d = *d_p; d_p ++; // Re
    t = d*d;
    d = *d_p; d_p ++; // Im

    *f_p += t + d*d;
    f_p ++;
  }
}

void dsp::Response::set (const vector<complex<float> >& filt)
{
  // one poln, one channel, complex
  resize (1, 1, filt.size(), 2);
  float* f = buffer;

  for (unsigned idat=0; idat<filt.size(); idat++) {
    // Re
    *f = filt[idat].real();
    f++;
    // Im
    *f = filt[idat].imag();
    f++;
  }
}

// /////////////////////////////////////////////////////////////////////////
//
// Response::operate - multiplies two complex arrays by complex matrix Response 
// ndat = number of complex points
//
void dsp::Response::operate (float* data1, float* data2, int ichan)
{
  assert (ndim == 8);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* d1_rp = data1;
  float* d1_ip = data1 + 1;
  float* d2_rp = data2;
  float* d2_ip = data2 + 1;

  float* f_p = buffer + ichan * ndat * ndim;

  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  register float r1_r;
  register float r1_i;
  register float r2_r;
  register float r2_i;

  for (unsigned ipt=0; ipt<npts; ipt++) {

    // ///////////////////////
    // multiply: r1 = f11 * d1
    d_r = *d1_rp; 
    d_i = *d1_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r1_r = f_r * d_r - f_i * d_i; 
    r1_i = f_i * d_r + f_r * d_i;

    // ///////////////////////
    // multiply: r2 = f21 * d1
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r2_r = f_r * d_r - f_i * d_i; 
    r2_i = f_i * d_r + f_r * d_i;

    // ////////////////////////////
    // multiply: d2 = r2 + f22 * d2
    d_r = *d2_rp;
    d_i = *d2_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d2_rp = r2_r + f_r * d_r - f_i * d_i;
    d2_rp += 2;
    *d2_ip = r2_i + f_i * d_r + f_r * d_i;
    d2_ip += 2;

    // ////////////////////////////
    // multiply: d1 = r1 + f12 * d2
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d1_rp = r1_r + f_r * d_r - f_i * d_i; 
    d1_rp += 2;
    *d1_ip = r1_i + f_i * d_r + f_r * d_i;
    d1_ip += 2;
  }
}

void dsp::Response::integrate (float* data1, float* data2, int ichan)
{
  assert (ndim == 1);
  assert (npol == 4);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* data = buffer + ichan * ndat * ndim;

  cross_detect_int (npts, data1, data2,
		    data, data + offset, 
		    data + 2*offset, data + 3*offset, 1);
}

void dsp::Response::set (const vector<Jones<float> >& response)
{
  // one poln, one channel, Jones
  resize (1, 1, response.size(), 8);

  float* f = buffer;

  for (unsigned idat=0; idat<response.size(); idat++) {

    // for efficiency, the elements of a Jones matrix Response 
    // are ordered as: f11, f21, f22, f12

    for (int j=0; j<2; j++)
      for (int i=0; i<2; i++) {
	complex<double> element = response[idat].j( (i+j)%2, j );
	// Re
	*f = element.real();
	f++;
	// Im
	*f = element.imag();
	f++;
      }
  }
}

// ////////////////////////////////////////////////////////////////
//
// dsp::Response::swap swaps the passband(s)
//
// If 'each_chan' is true, then the nchan units (channels) into which
// the Response is logically divided will be swapped individually
//
void dsp::Response::swap (bool each_chan)
{
  if (nchan == 0)
   throw_str ("dsp::Response::swap invalid nchan=%d", nchan);

  unsigned half_npts = (ndat * ndim) / 2;

  if (!each_chan)
    half_npts *= nchan;

  if (half_npts < 2)
    throw_str ("dsp::Response::swap invalid npts=%d", half_npts);

  unsigned ndiv = 1;
  if (each_chan)
    ndiv = nchan;

#ifdef _DEBUG
  cerr << "dsp::Response::swap"
    " nchan=" << nchan <<
    " ndat=" << ndat <<
    " ndim=" << ndim <<
    " npts=" << half_npts
       << endl;
#endif

  float* ptr1 = 0;
  float* ptr2 = 0;
  float  temp = 0;

  for (unsigned ipol=0; ipol<npol; ipol++) {

    ptr1 = buffer + offset * ipol;
    ptr2 = ptr1 + half_npts;

    for (unsigned idiv=0; idiv<ndiv; idiv++) {

      for (unsigned ipt=0; ipt<half_npts; ipt++) {
	temp = *ptr1;
	*ptr1 = *ptr2; ptr1++;
	*ptr2 = temp; ptr2++;
      }

      ptr1+=half_npts;
      ptr2+=half_npts;
    }
  }

  if (each_chan)
    chan_swapped = !chan_swapped;
  else
    whole_swapped = !whole_swapped;
}
