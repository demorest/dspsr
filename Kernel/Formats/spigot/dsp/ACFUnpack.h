//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/spigot/dsp/ACFUnpack.h,v $
   $Revision: 1.1 $
   $Date: 2004/01/23 23:00:59 $
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
    virtual ~ACFUnpack ();

  protected:

    //! Perform the bit conversion transformation on the input TimeSeries
    virtual void transformation ();

  };
  
}

#endif
