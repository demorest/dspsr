//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Convolution.h,v $
   $Revision: 1.1 $
   $Date: 2002/06/30 05:09:01 $
   $Author: pulsar $ */

#ifndef __Convolution_h
#define __Convolution_h

#include "Operation.h"
#include "filter.h"
#include "window.h"

namespace dsp {
  
  //! Convolves a Timeseries with a frequency response function
  /* This class implements the overlap-save method of discrete
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
     apodizing function may be applied to the data in the time domain
     by calling the Convolution::set_apodizing() method.

     Convolution::nfilt_pos+Convolution::nfilt_neg is effectively the
     duration of g(t), or the the number of complex time samples in
     the result of each backward FFT that are polluted by the cyclical
     convolution operation.  In order to allow for assymetry of g(t)
     around t=0, Convolution::nfilt_pos and Convolution::nfilt_neg
     complex samples are dropped from the beginning and end,
     respectively, of the result of each backward FFT; neighbouring
     FFTs will overlap by the appropriate number of points to make up
     for this loss.

     \pre input Timeseries must contain phase coherent (undetected) data
     \post output Timeseries will contain complex (observation::Analytic) data

     \post IMPORTANT!! Most backward complex FFT functions expect
     frequency components organized with f0+bw/2 -> f0, f0-bw/2 -> f0.
     The forward real-to-complex FFT produces f0-bw/2 -> f0+bw/2.  To
     save CPU cycles, convolve() does not re-sort the ouput array, and
     therefore introduces a frequency shift in the output data.  This
     results in a phase gradient in the time domain.  Since only
     relative phases matter when calculating the Stokes parameters,
     this effect is basically ignorable for our purposes.
  */

  class Convolution: public Operation {

  public:

    //! Null constructor
    Convolution ();

    //! Perform the onvolution operation on the input Timeseries
    virtual void operation ();

    //! Return a descriptive string
    virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    virtual void initialize (const string& descriptor);

    //! Set the apodizing function
    virtual void set_apodizing (window* function);


  protected:

    //! Frequency response (convolution kernel)
    filter response;

    //! Apodizing function (time domain window)
    window* apodizing;

    //! Integrated bandpass
    filter* bandpass;

    //! Complex samples dropped from beginning of cyclical convolution result
    int nfilt_pos;

    //! Complex samples dropped from end of cyclical convolution result
    int nfilt_neg;
  };
  
}

#endif
