//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma2/dsp/PuMa2Unpacker.h,v $
   $Revision: 1.1 $
   $Date: 2005/03/12 13:24:10 $
   $Author: wvanstra $ */

#ifndef __PuMa2Unpacker_h
#define __PuMa2Unpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the PuMa2 files
  class PuMa2Unpacker : public Unpacker {

  public:
    
    //! Constructor
    PuMa2Unpacker (const char* name = "PuMa2Unpacker") : Unpacker (name) {}

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__PuMa2Unpacker_h)
