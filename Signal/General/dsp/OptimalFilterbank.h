//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/OptimalFilterbank.h

#ifndef __OptimalFilterbank_h
#define __OptimalFilterbank_h

#include "OptimalFFT.h"
#include "FilterbankBench.h"

namespace dsp
{  
  //! Chooses the optimal FFT length for Filterbank and/or Convolution
  class OptimalFilterbank : public OptimalFFT
  {
  public:

    static bool verbose;

    OptimalFilterbank (const std::string& library);

    void set_nchan (unsigned nchan);
    double compute_cost (unsigned nfft, unsigned nfilt) const;
    std::string get_library (unsigned nfft);

  protected:

    mutable Reference::To<FilterbankBench> fbench;
    FTransform::Bench* new_bench () const;

    std::string library;
    unsigned nchan;
  };

}

#endif
