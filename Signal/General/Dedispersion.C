#include <complex>

#include "Dedispersion.h"
#include "Timeseries.h"

dsp::Dedispersion::Dedispersion ()
{
  centre_frequency = -1.0;
  bandwidth = 0.0;
  dispersion_measure = 0.0;

  Doppler_shift = 1.0;
  fractional_delay = false;

  built = false;
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

//! Set the bandwidth of signal in MHz
void dsp::Dedispersion::set_bandwidth (double _bandwidth)
{
  if (bandwidth != _bandwidth)
    built = false;

  bandwidth = _bandwidth;
}

//! Set the dispersion measure (in \f${\rm pc cm}^{-3}\f$)
void dsp::Dedispersion::set_dispersion_measure (double _dispersion_measure)
{
  if (dispersion_measure != _dispersion_measure)
    built = false;

  dispersion_measure = _dispersion_measure;
}

//! Set the Doppler shift due to the Earths' motion
void dsp::Dedispersion::set_Doppler_shift (double _Doppler_shift)
{
  if (Doppler_shift != _Doppler_shift)
    built = false;

  Doppler_shift = _Doppler_shift;
}

//! Set the flag to add fractional inter-channel delay
void dsp::Dedispersion::set_fractional_delay (bool _fractional_delay)
{
  if (fractional_delay != _fractional_delay)
    built = false;

  fractional_delay = _fractional_delay;
}


//! Build the dedispersion frequency response kernel
void dsp::Dedispersion::match (const Timeseries* input, unsigned _nchan)
{
  set_centre_frequency ( input->get_centre_frequency() );
  set_bandwidth ( input->get_bandwidth() );

  if (_nchan)
    resize (npol, _nchan, ndat, ndim);

  build ();

  Response::match (input, _nchan);
}


void dsp::Dedispersion::build ()
{
  if (built)
    return;

  vector<float> phases (ndat * nchan);

  build (phases, centre_frequency, bandwidth, 
	 dispersion_measure, Doppler_shift,
	 ndat, nchan, fractional_delay);

  vector<complex<float> > phasors (ndat * nchan);
  for (unsigned ipt=0; ipt<phases.size(); ipt++)
    phasors[ipt] = complex<float>(polar (float(1.0), phases[ipt]));

  set (phasors);
}

void dsp::Dedispersion::build (vector<float>& phases,
			       double centrefreq, double bw, 
			       float dm, double doppler,
			       unsigned npts, unsigned nchan, bool dmcorr)
{
  centrefreq /= doppler;
  bw /= doppler;

  double sign = bw / fabs (bw);
  double chanwidth = bw / double(nchan);
  double binwidth = chanwidth / double(npts);

  double lower_cfreq = centrefreq - 0.5*(bw-chanwidth);

  double highest_freq = centrefreq + 0.5*fabs(bw) - 0.5*chanwidth;

  double samp_int = 1.0/chanwidth; // sampint in microseconds, for
                                   // quadrature nyquist data eg fb.
  double delay = 0.0;

  double DM = dm/2.41e-10;

  phases.resize (npts * nchan);

  for (unsigned ichan = 0; ichan < nchan; ichan++) {

    double chan_cfreq = lower_cfreq + double(ichan) * chanwidth;
   
    // Compute the DM delay in microseconds
    if (dmcorr) {
      delay = DM * ( 1.0/(chan_cfreq*chan_cfreq)
		     -1.0/(highest_freq*highest_freq));
      // Modulo one sample and invert it
      delay = - fmod(delay, samp_int);
    }

    double coeff = -sign * 2*M_PI * DM / (chan_cfreq * chan_cfreq);

    unsigned spt = ichan * npts;
    for (unsigned ipt = 0; ipt < npts; ipt++) {
      // FFTX - not the mean of the bin, but the offset from the DC bin
      double freq = double(ipt) * binwidth - 0.5*chanwidth;
      phases[spt+ipt] = coeff*freq*freq/(chan_cfreq+freq)-2.0*M_PI*freq*delay;
    }
  }
}
