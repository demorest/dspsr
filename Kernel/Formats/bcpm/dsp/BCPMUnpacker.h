//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BCPMUnpacker_h
#define __BCPMUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  class BCPMExtension;

  class BCPMUnpacker : public Unpacker {

    typedef struct {
      float data[512];
    } float512;

  public:

    //! Null constructor
    BCPMUnpacker (const char* name = "BCPMUnpacker");

    //! Destructor
    ~BCPMUnpacker ();

    //! Return true if BCPMUnpacker can convert the Observation
    bool matches (const Observation* observation);

    //! Takes the BCPMExtension added by BCPMFile
    void add_extensions (Extensions* ext);

  protected:

    //! Unpack the BCPM data into the output TimeSeries
    void unpack ();

    //! Generates the lookup table
    float512 get_lookup();

    //! Extension used to pass information from BCPMFile to BCPMUnpacker
    Reference::To<BCPMExtension> extension;
  };
  
}

#endif
