//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma2/dsp/PuMa2Unpacker.h,v $
   $Revision: 1.2 $
   $Date: 2005/04/26 13:07:00 $
   $Author: wvanstra $ */

#ifndef __PuMa2Unpacker_h
#define __PuMa2Unpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the PuMa2 files
  class PuMa2Unpacker : public HistUnpacker {

  public:
    
    //! Constructor
    PuMa2Unpacker (const char* name = "PuMa2Unpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__PuMa2Unpacker_h)
