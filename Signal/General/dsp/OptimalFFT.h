//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/OptimalFFT.h,v $
   $Revision: 1.2 $
   $Date: 2010/03/24 04:17:45 $
   $Author: straten $ */

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

    //! Set the number of channels into which the data will be divided
    void set_nchan (unsigned nchan);

    //! Set true when convolution is performed during filterbank synthesis
    void set_simultaneous (bool flag);

    unsigned get_nfft (unsigned nfilt) const;

    double compute_cost (unsigned nfft, unsigned nfilt) const;

    //! Get the name of the FFT library to use for the given FFT length
    std::string get_library (unsigned nfft);

  protected:

    mutable Reference::To<FTransform::Bench> bench;
    unsigned nchan;
    bool simultaneous;
  };

}

#endif
