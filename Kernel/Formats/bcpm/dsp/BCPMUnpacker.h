//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 - 2011 by Haydon Knight and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BCPMUnpacker_h
#define __BCPMUnpacker_h

#include "dsp/Unpacker.h"
#include "dsp/BCPMFile.h"

namespace dsp {

  class BCPMUnpacker : public Unpacker {

    typedef struct {
      float data[512];
    } float512;

  public:

    //! Null constructor
    BCPMUnpacker (const char* name = "BCPMUnpacker");

    //! Return true if BCPMUnpacker can convert the Observation
    bool matches (const Observation* observation);

  protected:

    //! Unpack the BCPM data into the output TimeSeries
    void unpack ();

    //! Generates the lookup table
    float512 get_lookup();

    //! Used to pass information from BCPMFile to BCPMUnpacker
    Reference::To<const BCPMFile> file;
  };
  
}

#endif
