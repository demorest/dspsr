//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Dedispersion.h,v $
   $Revision: 1.1 $
   $Date: 2002/08/20 05:14:22 $
   $Author: wvanstra $ */

#ifndef __Dedispersion_h
#define __Dedispersion_h

#include "Response.h"

namespace dsp {
  
  //! Phase-coherent dedispersion frequency response function
  /* This class implements the phase-coherent dedispersion kernel, or the
     frequency response of the interstellar medium. */

  class Dedispersion: public Response {

  public:

    //! Null constructor
    Dedispersion ();

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the dimensions of the data
    virtual void resize (unsigned npol, unsigned nchan,
			 unsigned ndat, unsigned ndim);

    //! Set the centre frequency of the band-limited signal in MHz
    void set_centre_frequency (double centre_frequency);

    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const { return centre_frequency; }

    //! Returns the centre frequency of the specified channel in MHz
    double get_centre_frequency (int ichan) const;

    //! Set the bandwidth of signal in MHz
    void set_bandwidth (double bandwidth);

    //! Return the bandwidth of signal in MHz
    double get_bandwidth () const { return bandwidth; }

    //! Set the dispersion measure (in \f${\rm pc cm}^{-3}\f$)
    void set_dispersion_measure (double dm);

    //! Get the dispersion measure (in \f${\rm pc cm}^{-3}\f$)
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the Doppler shift due to the Earths' motion
    void set_Doppler_shift (double Doppler_shift);

    //! Return the doppler shift due to the Earth's motion
    double get_Doppler_shift () const { return Doppler_shift; }

    //! Set the flag to add fractional inter-channel delay
    void set_fractional_delay (bool fractional_delay);

    //! Get the flag to add fractional inter-channel delay
    bool get_fractional_delay () const { return fractional_delay; }


  protected:

    //! Perform the convolution operation on the input Timeseries
    virtual void operation ();

    //! Build the dedispersion frequency response kernel
    virtual void match (const Timeseries* input, unsigned nchan=0);

    //! Centre frequency of the band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz
    double bandwidth;

    //! Dispersion measure (in \f${\rm pc cm}^{-3}\f$)
    double dispersion_measure;

    //! Doppler shift due to the Earths' motion
    double Doppler_shift;

    //! Flag to add fractional inter-channel delay
    bool fractional_delay;

    //! Flag that the response and bandpass attributes reflect the state
    bool built;

  };
  
}

#endif
