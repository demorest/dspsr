//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/spigot/dsp/ACFUnpack.h,v $
   $Revision: 1.2 $
   $Date: 2004/01/24 03:00:59 $
   $Author: wvanstra $ */

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
