//-*-C++-*-

#ifndef __BCPMUnpacker_h
#define __BCPMUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  class BCPMUnpacker : public Unpacker {

  public:

    //! Null constructor
    BCPMUnpacker (const char* name = "BCPMUnpacker");

    //! Destructor
    ~BCPMUnpacker ();

    //! Return true if BCPMUnpacker can convert the Observation
    bool matches (const Observation* observation);

  protected:

    //! Unpack the BCPM data into the output TimeSeries
    void unpack ();

  };
  
}

#endif
