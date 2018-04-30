//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FadcUnpacker_h
#define __FadcUnpacker_h

#include "dsp/HistUnpacker.h"
//#include "HistUnpacker.h"

namespace dsp {

  //! 2-bit to float unpacker for the FADC files
  class FadcUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    FadcUnpacker (const char* name = "FadcUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);
    
    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);

  };

}

#endif // !defined(__FadcUnpacker_h)
