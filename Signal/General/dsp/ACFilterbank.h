//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/ACFilterbank.h,v $
   $Revision: 1.1 $
   $Date: 2005/03/25 05:54:02 $
   $Author: wvanstra $ */

#ifndef __ACFilterbank_h
#define __ACFilterbank_h

#include "dsp/Transformation.h"

namespace dsp {
  
  //! Calculates Power Spectral Density or Auto-Correlation Function 
  /* This class can be used to form the power spectral density (PSD)
     or auto-correlation fuction (ACF) as a function of time.  There
     are two modes of operation:

     1) the PSD are stored in frequency-major order
     2) the ACF are mutliplexed into time-major order

     In order to compute the ACF (and not the cyclic ACF), the
     TimeSeries data must be zero padded before each Fourier
     transform.  For each good lag in the ACF, a zero is padded and,
     for each zero padded, the neighbouring FFTs will overlap.
     Therefore, in mode 1, the output will be larger than the input.
     However, in mode 2, an inverse FFT is required.
  */

  class ACFilterbank: public Transformation <TimeSeries, TimeSeries> {

  public:

    //! Null constructor
    ACFilterbank ();

    //! Set the number of channels into which the input will be divided
    void set_nchan (unsigned _nchan) { nchan = _nchan; }

    //! Get the number of channels into which the input will be divided
    unsigned get_nchan () const { return nchan; }

    //! Set the time resolution factor
    void set_nlag (unsigned _nlag) { nlag = _nlag; }

    //! Get the time resolution factor
    unsigned get_nlag () const { return nlag; } 

    //! Set flag to calculate auto-correlation function
    void set_form_acf (bool _flag) { form_acf = _flag; }

    //! Get flag to calculate auto-correlation function
    bool get_form_acf () const { return form_acf; }

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Number of valid lags in the ACF
    unsigned nlag;

    //! Flag to calculate auto-correlation function
    form_acf;

  };
  
}

#endif
