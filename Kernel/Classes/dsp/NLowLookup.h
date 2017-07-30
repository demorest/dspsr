//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/NLowLookup.h

#ifndef __NLowLookup_h
#define __NLowLookup_h

#include "dsp/BitTable.h"

namespace dsp
{
  //! Manage now lookup table for ExcisionUnpacker derived classes
  class NLowLookup
  {

  public:

    //! Default constructor
    NLowLookup (BitTable*);

    //! Return the pointer to the bit table
    const BitTable* get_table () const;

  protected:

    //! The bit table passed on construction
    Reference::To<BitTable> bit_table;

    char nlow_lookup [256];
    const float* lookup;

  };

}

#endif
