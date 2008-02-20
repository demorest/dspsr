//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Mark5Unpacker_h
#define __Mark5Unpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Interface to Walter Brisken's upacker for Mark5 VLBA/Mark4 files
  class Mark5Unpacker : public Unpacker {
	
  public:
	
    //! Constructor
    Mark5Unpacker (const char* name = "Mark5Unpacker");

    //! Get the number of independent digitizers
    unsigned get_ndig () const;

  protected:
	
    //! The unpacking routine
    void unpack ();
	
    //! Return true if we can convert the obsservation
    bool matches (const Observation* observation);
	
  };


}

#endif // !defined(__Mark5Unpacker_h)
