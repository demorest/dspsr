/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Dedispersion.h"
#include "dsp/Observation.h"

#include "ThreadContext.h"
#include "Error.h"
#include <complex>

using namespace std;

/*! Although the value:

  \f$ DM\,({\rm pc\,cm^{-3}})=2.410331(2)\times10^{-4}D\,({\rm s\,MHz^{2}}) \f$

  has been derived from "fundamental and primary physical and
  astronomical constants" (section 3 of Backer, Hama, van Hook and
  Foster 1993. ApJ 404, 636-642), the rounded value is in standard 
  use by pulsar astronomers (page 129 of Manchester and Taylor 1977).
*/
const double dsp::Dedispersion::dm_dispersion = 2.41e-4;

float dsp::Dedispersion::smearing_buffer = 0.1;

dsp::Dedispersion::Dedispersion ()
{
  centre_frequency = -1.0;
  bandwidth = 0.0;
  dispersion_measure = 0.0;

  Doppler_shift = 1.0;
  fractional_delay = false;
  dc_centred = false;

  frequency_resolution_set = false;
  times_minimum_nfft = 0;

  smearing_samples_set = false;

  build_delays = false;

  built = false;
  context = 0;
}

//! Set the dimensions of the data and update the built attribute
void dsp::Dedispersion::resize (unsigned _npol, unsigned _nchan,
				unsigned _ndat, unsigned _ndim)
{
  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
    built = false;

  Shape::resize (_npol, _nchan, _ndat, _ndim);
}

//! Set the centre frequency of the band-limited signal in MHz
void dsp::Dedispersion::set_centre_frequency (double _centre_frequency)
{
  if (centre_frequency != _centre_frequency)
    built = false;

  centre_frequency = _centre_frequency;
}

//! Returns the centre frequency of the specified channel in MHz
double dsp::Dedispersion::get_centre_frequency (int ichan) const
{
  cerr << "Dedispersion::get_centre_frequency (ichan) not implemented" << endl;
  return -1.0;
}

//! Set the bandwidth of the signal in MHz
void dsp::Dedispersion::set_bandwidth (double _bandwidth)
{
  if (bandwidth != _bandwidth)
    built = false;

  bandwidth = _bandwidth;
}

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Dedispersion::set_dispersion_measure (double _dispersion_measure)
{
  if (dispersion_measure != _dispersion_measure)
    built = false;

  dispersion_measure = _dispersion_measure;
}

//! Set the Doppler shift due to the Earth's motion
void dsp::Dedispersion::set_Doppler_shift (double _Doppler_shift)
{
  if (Doppler_shift != _Doppler_shift)
    built = false;

  Doppler_shift = _Doppler_shift;
}

//! Set the flag for a bin-centred spectrum
void dsp::Dedispersion::set_dc_centred (bool _dc_centred)
{
  if (dc_centred != _dc_centred)
    built = false;

  dc_centred = _dc_centred;
}

//! Set the blah
void dsp::Dedispersion::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    built = false;

  nchan = _nchan;
}

//! Set the flag to add fractional inter-channel delay
void dsp::Dedispersion::set_fractional_delay (bool _fractional_delay)
{
  if (fractional_delay != _fractional_delay)
    built = false;

  fractional_delay = _fractional_delay;
}

void dsp::Dedispersion::set_frequency_resolution (unsigned nfft)
{
  if (verbose)
    cerr << "dsp::Dedispersion::set_frequency_resolution ("<<nfft<<")"<<endl;
  resize (npol, nchan, nfft, ndim);

  frequency_resolution_set = true;
}

void dsp::Dedispersion::set_smearing_samples (unsigned tot)
{
  unsigned pos = tot / 2;
  unsigned neg = pos;
  if (tot % 2)
    neg ++;
  set_smearing_samples (pos, neg);
}

void dsp::Dedispersion::set_smearing_samples (unsigned pos, unsigned neg)
{
  impulse_pos = pos;
  impulse_neg = neg;
  smearing_samples_set = true;
}

void dsp::Dedispersion::set_times_minimum_nfft (unsigned times)
{
  if (verbose)
    cerr << "dsp::Dedispersion::set_times_minimum_nfft ("<<times<<")"<<endl;

  times_minimum_nfft = times;
  frequency_resolution_set = true;
}

void dsp::Dedispersion::prepare (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Dedispersion::prepare input.nchan=" << input->get_nchan()
	 << " channels=" << channels << "\n\t"
      " centre frequency=" << input->get_centre_frequency() <<
      " bandwidth=" << input->get_bandwidth () <<
      " dispersion measure=" << input->get_dispersion_measure() << endl;
  
  set_centre_frequency ( input->get_centre_frequency() );
  set_bandwidth ( input->get_bandwidth() );
  set_dispersion_measure ( input->get_dispersion_measure() );
  set_dc_centred ( input->get_dc_centred() );

  if (!channels)
    channels = input->get_nchan();

  set_nchan (channels);

  if (!built)
    prepare ();
}

/*! The signal at sky frequencies lower than the centre frequency
  arrives later.  So the finite impulse response (FIR) of the
  dispersion relation, d(t), should have non-zero values for t>0 up to
  the smearing time in the lower half of the band.  However, this
  class represents the inverse, or dedispersion, frequency response
  function, the FIR of which is given by h(t)=d^*(-t).  Therefore,
  h(t) has non-zero values for t>0 up to the smearing time in the
  upper half of the band.

  Noting that the first impulse_pos complex time samples are discarded
  from each cyclical convolution result, it may also be helpful to
  note that each time sample depends upon the preceding impulse_pos
  points.
*/
void dsp::Dedispersion::prepare ()
{
  if (!smearing_samples_set)
  {
    impulse_pos = smearing_samples (1);
    impulse_neg = smearing_samples (-1);
  }

  if (psrdisp_compatible)
  {
    cerr << "dsp::Dedispersion::prepare psrdisp compatibility\n"
      "   using symmetric impulse response function" << endl;
    impulse_pos = impulse_neg;
  }

#if 0
  // test the effect of a possibly common error in the interpretation of HR75
  impulse_pos += impulse_neg;
  impulse_neg = 0;
#endif

}


/*! Builds a frequency response function (kernel) suitable for phase-coherent
  dispersion removal, based on the centre frequency, bandwidth, and number
  of channels in the input Observation. 

  \param input Observation for which a dedispersion kernel will be built.

  \param channels If specified, over-rides the number of channels of the
  input Observation.  This parameter is useful if the Observation is to be
  simultaneously divided into filterbank channels during convolution.
 */
void dsp::Dedispersion::match (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Dedispersion::match before lock" << endl;

  ThreadContext::Lock lock (context);

  if (verbose)
    cerr << "dsp::Dedispersion::match after lock" << endl;

  prepare (input, channels);

  if (!built)
    build ();

  Response::match (input, channels);

  if (verbose)
    cerr << "dsp::Dedispersion::match exit" << endl;

}

void dsp::Dedispersion::mark (Observation* output)
{
  if (verbose)
    cerr << "dsp::Dedispersion::mark no longer changing DM" << endl;
}

void dsp::Dedispersion::build ()
{
  if (built)
    return;

  // prepare internal variables
  if (frequency_resolution_set)
  {
    if (times_minimum_nfft)
      set_frequency_resolution( times_minimum_nfft * get_minimum_ndat() );
    check_ndat ();
  }
  else 
    set_optimal_ndat ();

  // calculate the complex frequency response function
  vector<float> phases (ndat * nchan);

  build (phases, ndat, nchan);

  resize (1, nchan, ndat, 2);
  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = polar (float(1.0), phases[ipt]);

  whole_swapped = chan_swapped = false;

  built = true;

  changed.send (*this);
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

/*
  \param cfreq centre frequency, in MHz
  \param bw bandwidth, in MHz
  \retval dispersion smearing time across the specified band, in seconds
*/
double dsp::Dedispersion::smearing_time (double cfreq, double bw) const
{
  return delay_time (cfreq - fabs(0.5*bw), cfreq + fabs(0.5*bw));
}

double dsp::Dedispersion::delay_time (double freq1, double freq2) const
{
  if (verbose)
    cerr << "dsp::Dedispersion::delay_time DM=" << dispersion_measure
         << " f1=" << freq1 << " f2=" << freq2 << endl;

  double dispersion = dispersion_measure/dm_dispersion;
  return dispersion * ( 1.0/sqr(freq1) - 1.0/sqr(freq2) );
}

double dsp::Dedispersion::delay_time (double freq) const
{
  double dispersion = dispersion_measure/dm_dispersion;
  return dispersion * 1.0/sqr(freq);
}

double dsp::Dedispersion::get_effective_smearing_time () const
{
  return smearing_time (0);
}

//! Return the effective number of smearing samples
unsigned dsp::Dedispersion::get_effective_smearing_samples () const
{
  return smearing_samples (0);
}

/*!
  Calculate the smearing time over the band (or the sub-band with
  the lowest centre frequency) in seconds.  This will determine the
  number of points "nsmear" that must be thrown away for each FFT.
*/
double dsp::Dedispersion::smearing_time (int half) const
{  
  if ( ! (half==0 || half==-1 || half == 1) )
    throw Error (InvalidParam, "dsp::Dedispersion::smearing_time",
		 "invalid half=%d", half);

  double abs_bw = fabs (bandwidth);
  double ch_abs_bw = abs_bw / double(nchan);
  double lower_ch_cfreq = centre_frequency - (abs_bw - ch_abs_bw) / 2.0;

  // calculate the smearing (in the specified half of the band)
  if (half)
  {
    ch_abs_bw /= 2.0;
    lower_ch_cfreq += double(half) * ch_abs_bw;
  }

  if (verbose)
    cerr << "dsp::Dedispersion::smearing_time freq=" << lower_ch_cfreq
	 << " bw=" << ch_abs_bw << endl;

  double tsmear = smearing_time (lower_ch_cfreq, ch_abs_bw);
  
  if (verbose)
  {
    string band = "band";
    if (nchan>1)
      band = "worst channel";

    string side;
    if (half == 1)
      side = "upper half of the ";
    else if (half == -1)
      side = "lower half of the ";

    cerr << "dsp::Dedispersion::smearing_time in the " << side << band << ": "
	 << float(tsmear*1e3) << " ms" << endl;
  }

  return tsmear;
}

unsigned dsp::Dedispersion::smearing_samples (int half) const
{
  double tsmear = smearing_time (half);

  // the sampling rate of the resulting complex time samples
  double ch_abs_bw = fabs (bandwidth) / double (nchan);
  double sampling_rate = ch_abs_bw * 1e6;

  if (verbose)
    cerr << "dsp::Dedispersion::smearing_samples = "
	 << int(tsmear * sampling_rate) << endl;

  // add another ten percent, just to be sure that the pollution due
  // to the cyclical convolution effect is minimized
  if (psrdisp_compatible)
  {
     cerr << "dsp::Dedispersion::prepare psrdisp compatibility\n"
       "   increasing smearing time by 5 instead of " 
          << smearing_buffer*100.0 << " percent" << endl;
    tsmear *= 1.05;
  }
  else
    tsmear *= (1.0 + smearing_buffer);
  
  // smear across one channel in number of time samples.
  unsigned nsmear = unsigned (ceil(tsmear * sampling_rate));
 
  if (psrdisp_compatible)
  {
    cerr << "dsp::Dedispersion::prepare psrdisp compatibility\n"
      "   rounding smear samples down instead of up" << endl;
     nsmear = unsigned (tsmear * sampling_rate);
  }

  if (verbose)
  {
    // recalculate the smearing time simply for display of new value
    tsmear = double (nsmear) / sampling_rate;
    cerr << "dsp::Dedispersion::smearing_samples effective smear time: "
	 << tsmear*1e3 << " ms (" << nsmear << " pts)." << endl;
  }

  return nsmear;
}


void dsp::Dedispersion::build (vector<float>& phases,
			       unsigned _ndat, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::Dedispersion::build"
      "\n  centre frequency = " << centre_frequency <<
      "\n  bandwidth = " << bandwidth <<
      "\n  dispersion measure = " << dispersion_measure <<
      "\n  Doppler shift = " << Doppler_shift <<
      "\n  ndat = " << ndat <<
      "\n  nchan = " << _nchan <<
      "\n  centred on DC = " << dc_centred <<
      "\n  fractional delay compensation = " << fractional_delay << endl;

  double centrefreq = centre_frequency / Doppler_shift;
  double bw = bandwidth / Doppler_shift;

  double sign = bw / fabs (bw);
  double chanwidth = bw / double(_nchan);
  double binwidth = chanwidth / double(_ndat);

  double lower_cfreq = centrefreq - 0.5*bw;
  if (!dc_centred)
    lower_cfreq += 0.5*chanwidth;

  double highest_freq = centrefreq + 0.5*fabs(bw-chanwidth);

  double samp_int = 1.0/chanwidth; // sampint in microseconds, for
                                   // quadrature nyquist data eg fb.
  double delay = 0.0;

  // when divided by MHz, yields a dimensionless value
  double dispersion_per_MHz = 1e6 * dispersion_measure / dm_dispersion;

  phases.resize (_ndat * _nchan);

  for (unsigned ichan = 0; ichan < _nchan; ichan++)
  {
    double chan_cfreq = lower_cfreq + double(ichan) * chanwidth;

    // cerr << "chan_cfreq=" << chan_cfreq << endl;

    if (fractional_delay)
    {
      // Compute the DM delay in microseconds; when multiplied by the
      // frequency in MHz, the powers of ten cancel each other
      delay = dispersion_per_MHz * ( 1.0/sqr(chan_cfreq) -
				     1.0/sqr(highest_freq) );
      // Modulo one sample and invert it
      delay = - fmod(delay, samp_int);
    }

    double coeff = -sign * 2*M_PI * dispersion_per_MHz / sqr(chan_cfreq);

    unsigned spt = ichan * _ndat;
    for (unsigned ipt = 0; ipt < _ndat; ipt++)
    {
      // frequency offset from centre frequency of channel
      double freq = double(ipt)*binwidth - 0.5*chanwidth;

      // additional phase turn for fractional dispersion delay shift
      double delay_phase = -2.0*M_PI * freq * delay;

      phases[spt+ipt] = coeff*sqr(freq)/(chan_cfreq+freq) + delay_phase;

      if (build_delays && freq != 0.0)
	phases[spt+ipt] /= (2.0*M_PI * freq);

#ifdef _DEBUG
      cerr << ichan*_ndat + ipt << " " << chan_cfreq+freq << " " 
	   << phases[spt+ipt] << endl;
#endif
    }
  }
}

//! Build delays in microseconds instead of phases
void dsp::Dedispersion::set_build_delays (bool delays)
{
  build_delays = delays;
}

