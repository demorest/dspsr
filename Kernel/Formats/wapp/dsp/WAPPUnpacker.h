//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/wapp/dsp/WAPPUnpacker.h

#ifndef __WAPPUnpacker_h
#define __WAPPUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the WAPP files
  class WAPPUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    WAPPUnpacker (const char* name = "WAPPUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    //unsigned get_output_offset (unsigned idig) const;
    //unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__WAPPUnpacker_h)
