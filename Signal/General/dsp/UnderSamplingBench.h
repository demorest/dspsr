//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UnderSamplingBench_h_
#define __UnderSamplingBench_h_

#include "FTransformBench.h"

namespace dsp {

  //! Stores UnderSampling benchmark data
  class UnderSamplingBench : public FTransform::Bench
  {
  public:

    static bool verbose;

    //! Construct from installed benchmarks
    UnderSamplingBench (const std::string& library);

    //! Set the number of channels
    void set_nchan (unsigned);

  protected:

    unsigned nchan;
    std::string library;

    void load () const;
    void load (const std::string& library, const std::string& filename) const;
  };
}

#endif
