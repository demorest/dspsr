#include "Dedispersion.h"

dsp::Dedispersion::Dedispersion ()
{
  centre_frequency = -1.0;
  bandwidth = 0.0;
  dispersion_measure = 0.0;

  Doppler_shift = 1.0;
  fractional_delay = false;

  built = false;
}

//! Set the dimensions of the data
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

