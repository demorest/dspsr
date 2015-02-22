//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Stuart Weston and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Mark5bUnpacker_h
#define __Mark5bUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Interface to Walter Brisken's upacker for Mark5b VLBA/Mark4 files
  class Mark5bUnpacker : public Unpacker {
	
  public:
	
    //! Constructor
    Mark5bUnpacker (const char* name = "Mark5bUnpacker");

    //! Get the number of independent digitizers
    unsigned get_ndig () const;

  protected:
	
    //! The unpacking routine
    void unpack ();
	
    //! Return true if we can convert the obsservation
    bool matches (const Observation* observation);
	
  };


}

#endif // !defined(__Mark5bUnpacker_h)
