//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Reserve_h
#define __Reserve_h

#include "ReferenceAble.h"
#include <inttypes.h>

namespace dsp {

  class TimeSeries;

  //! Remembers how much has been reserved and increases it if necessary
  class Reserve : public Reference::Able
  {
  public:
    
    Reserve ();

    //! Set the minimum number of samples that can be processed
    void reserve (const TimeSeries*, uint64_t samples);

    uint64_t get_reserved () const { return reserved; }

  private:
    
    //! The requested number of samples to be reserved
    uint64_t reserved;

    //! Used to double-check the sanity of the user
    const TimeSeries* sanity_check;

  };

}

#endif // !defined(__Reserve_h)
