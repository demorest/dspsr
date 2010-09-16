//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIUnpacker_h
#define __GUPPIUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for GUPPI baseband files
  class GUPPIUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    GUPPIUnpacker (const char* name = "GUPPIUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;
    unsigned get_output_ichan (unsigned idig) const;

  };

}

#endif // !defined(__GUPPIUnpacker_h)
