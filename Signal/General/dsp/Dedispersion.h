//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Dedispersion.h,v $
   $Revision: 1.14 $
   $Date: 2003/01/07 16:25:12 $
   $Author: wvanstra $ */

#ifndef __Dedispersion_h
#define __Dedispersion_h

#include "dsp/Response.h"

namespace dsp {
  
  //! Phase-coherent dedispersion frequency response function
  /* This class implements the phase-coherent dedispersion kernel, or the
     frequency response of the interstellar medium.  Not tested. */

  class Dedispersion: public Response {

  public:

    //! Conversion factor between dispersion measure, DM, and dispersion, D
    static const double dm_dispersion;

    //! Null constructor
    Dedispersion ();

    //! Match the dedispersion kernel to the input Observation
    virtual void match (const Observation* input, unsigned channels=0);

    //! Set the dispersion measure attribute in the output Observation
    virtual void mark (Observation* output);

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the dimensions of the data
    virtual void resize (unsigned npol, unsigned nchan,
			 unsigned ndat, unsigned ndim);

    //! Set the flag for a bin-centred spectrum
    virtual void set_dc_centred (bool dc_centred);

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

    //! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    void set_dispersion_measure (double dm);

    //! Get the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the Doppler shift due to the Earth's motion
    void set_Doppler_shift (double Doppler_shift);

    //! Return the doppler shift due to the Earth's motion
    double get_Doppler_shift () const { return Doppler_shift; }

    //! Set the flag to add fractional inter-channel delay
    void set_fractional_delay (bool fractional_delay);

    //! Get the flag to add fractional inter-channel delay
    bool get_fractional_delay () const { return fractional_delay; }

    //! Set the frequency resolution in each channel of the kernel
    void set_frequency_resolution (unsigned nfft);

    //! Get the frequency resolution in each channel of the kernel
    unsigned get_frequency_resolution () const { return ndat; }

    //! Return the dispersion delay between freq1 and freq2
    /*! If freq2 is higher than freq1, delay_time is positive */
    double delay_time (double freq1, double freq2) const;

    //! Return the smearing time, given the centre frequency and bandwidth
    double smearing_time (double centre_frequency, double bandwidth) const;

    //! Return the smearing time in seconds
    double smearing_time () const {
      return smearing_time (centre_frequency, bandwidth);
    }

    //! Return the number of complex samples of smearing in the specified half
    unsigned smearing_samples (int half = -1) const;

    //! Compute the phases for a dedispersion kernel
    void build (vector<float>& phases, unsigned npts, unsigned nchan);

  protected:

    //! Build the dedispersion frequency response kernel
    virtual void build ();

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

    //! Flag set when set_frequency_resolution() method is called
    bool frequency_resolution_set;

    //! Flag that the response and bandpass attributes reflect the state
    bool built;

  };
  
}

#endif
