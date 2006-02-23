//-*-C++-*-

#ifndef __Mark5Unpacker_h
#define __Mark5Unpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Interface to Walter Brisken's upacker for Mark5 VLBA/Mark4 files
  class Mark5Unpacker : public Unpacker {
	
  public:
	
    //! Constructor
    Mark5Unpacker (const char* name = "Mark5Unpacker");
	
  protected:
	
    //! The unpacking routine
    void unpack ();
	
    //! Return true if we can convert the obsservation
    bool matches (const Observation* observation);
	
  };


}

#endif // !defined(__Mark5Unpacker_h)
