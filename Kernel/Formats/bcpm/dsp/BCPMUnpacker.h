//-*-C++-*-

#ifndef __BCPMUnpacker_h
#define __BCPMUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

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

  protected:

    //! Unpack the BCPM data into the output TimeSeries
    void unpack ();

    //! Generates the lookup table
    float512 get_lookup();

  };
  
}

#endif
