//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Convolution.h,v $
   $Revision: 1.10 $
   $Date: 2002/11/09 15:55:25 $
   $Author: wvanstra $ */

#ifndef __Convolution_h
#define __Convolution_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {
  
  class Apodization;
  class Response;

  //! Convolves a TimeSeries with a frequency response function
  /*! This class implements the overlap-save method of discrete
    (cyclical) convolution, as performed by multiplication in the
    frequency domain.

    The algorithm implements both scalar and matrix convolution
    techniques, and is highly suited to perform both phase-coherent
    dispersion removal and phase-coherent polarimetric calibration.
    
    If g(t) is the impulse response function with which the data
    stream will be convolved, then the Convolution::response member
    represents G(w), the FFT of g(t).  Convolution::response may
    contain an array of filters, one for each frequency channel.
    
    In order to improve the spectral leakage characteristics, an
    apodization function may be applied to the data in the time domain
    by calling the Convolution::set_Apodization() method.
    
    Convolution::nfilt_pos+Convolution::nfilt_neg is effectively the
    duration of g(t), or the the number of complex time samples in
    the result of each backward FFT that are polluted by the cyclical
    convolution transformation.  In order to allow for assymetry of g(t)
    around t=0, Convolution::nfilt_pos and Convolution::nfilt_neg
    complex samples are dropped from the beginning and end,
    respectively, of the result of each backward FFT; neighbouring
    FFTs will overlap by the appropriate number of points to make up
    for this loss.
  */

  class Convolution: public Transformation <TimeSeries, TimeSeries> {

  public:

    //! Null constructor
    Convolution (const char* name = "Convolution", Behaviour type = anyplace);

    //! Destructor
    ~Convolution ();

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the frequency response function
    virtual void set_response (Response* response);

    //! Set the apodization function
    virtual void set_apodization (Apodization* function);

    //! Set the bandpass integrator
    virtual void set_bandpass (Response* bandpass);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Frequency response (convolution kernel)
    Reference::To<Response> response;

    //! Apodization function (time domain window)
    Reference::To<Apodization> apodization;

    //! Integrated bandpass
    Reference::To<Response> bandpass;

  };
  
}

#endif
