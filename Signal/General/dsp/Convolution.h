//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Convolution.h,v $
   $Revision: 1.13 $
   $Date: 2003/04/28 06:28:55 $
   $Author: pulsar $ */

#ifndef __Convolution_h
#define __Convolution_h

#include "dsp/Response.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {
  
  class Apodization;

  //! Convolves a TimeSeries using a frequency response function
  /*! This class implements the overlap-save method of discrete
    convolution with a finite impulse response (FIR) function, as
    described in Chapter 13.1 of Numerical Recipes.

    The algorithm can perform both scalar and matrix convolution
    methods, and is highly suited to phase-coherent dispersion removal
    and phase-coherent polarimetric calibration.
    
    If g(t) is the finite impulse response function with which the
    data stream will be convolved, then the Convolution::response
    attribute represents G(w), the FFT of g(t).  Convolution::response
    may contain an array of filters, one for each frequency channel.
    
    In order to improve the spectral leakage characteristics, an
    apodization function may be applied to the data in the time domain
    by setting the Convolution::apodization attribute.
    
    Referring to Figure 13.1.3 in Numerical Recipes,
    \f$m_+\f$=response->get_impulse_pos() and
    \f$m_-\f$=response->get_impulse_neg(), so that the duration,
    M=\f$m_+ + m_-\f$, of g(t) corresponds to the number of complex
    time samples in the result of each backward FFT that are polluted
    by the cyclical convolution transformation.  Consequently,
    \f$m_+\f$ and \f$m_-\f$ complex samples are dropped from the
    beginning and end, respectively, of the result of each backward
    FFT; neighbouring FFTs will overlap by the appropriate number of
    points to make up for this loss.  */

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

    //! Return a pointer to the frequency response function
    virtual const Response* get_response() const;

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
