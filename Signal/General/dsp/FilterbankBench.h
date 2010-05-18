//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FilterbankBench_h_
#define __FilterbankBench_h_

#include "FTransformBench.h"

namespace dsp {

  //! Stores Filterbank benchmark data
  class FilterbankBench : public FTransform::Bench
  {
  public:

    static bool verbose;

    //! Construct from installed benchmarks
    FilterbankBench (const std::string& library);

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
