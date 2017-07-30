//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __EDAFourBit_h
#define __EDAFourBit_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  //! Converts single-dish EDA data from 4-bit to floating point values
  class EDAFourBit: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    EDAFourBit ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

    //! Over-ride the default BitUnpacker::unpack method
    void unpack ();

    //! Over-ride the default FourBitUnpacker::get_histogram method
    void get_histogram (std::vector<unsigned long>&, unsigned idig) const;

  };
}

#endif
