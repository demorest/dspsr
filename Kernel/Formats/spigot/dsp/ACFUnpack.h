//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/spigot/dsp/ACFUnpack.h

#ifndef __ACFUnpack_h
#define __ACFUnpack_h

#include "dsp/Unpacker.h"

namespace dsp {

  class ACFUnpack: public Unpacker {

  public:

    //! Null constructor
    ACFUnpack (const char* name = "ACFUnpack");

    //! Virtual destructor
    ~ACFUnpack ();

    //! Return true if ACFUnpack can convert the Observation
    bool matches (const Observation* observation);

  protected:

    //! Unpack the ACFs into the output TimeSeries
    void unpack ();

  };
  
}

#endif
