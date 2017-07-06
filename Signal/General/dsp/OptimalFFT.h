//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/OptimalFFT.h

#ifndef __OptimalFFT_h
#define __OptimalFFT_h

#include "FTransformBench.h"

namespace dsp
{  
  //! Chooses the optimal FFT length for Filterbank and/or Convolution
  class OptimalFFT : public Reference::Able
  {
  public:

    static bool verbose;

    OptimalFFT ();

    //! Set true when convolution is performed during filterbank synthesis
    void set_simultaneous (bool flag);

    unsigned get_nfft (unsigned nfilt) const;

    //! Set the number of channels into which the data will be divided
    virtual void set_nchan (unsigned nchan);

    //! Return the time required to execute in microseconds
    virtual double compute_cost (unsigned nfft, unsigned nfilt) const;

    //! Get the name of the FFT library to use for the given FFT length
    virtual std::string get_library (unsigned nfft);

  protected:

    mutable Reference::To<FTransform::Bench> bench;

    virtual FTransform::Bench* new_bench () const;

    unsigned nchan;
    bool simultaneous;
  };

}

#endif
