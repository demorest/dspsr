//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __LWAUnpacker_h
#define __LWAUnpacker_h

#include "dsp/HistUnpacker.h"
#include "dsp/BitTable.h"

namespace dsp {

  //! Simple 4-bit complex to float unpacker for the LWA files
  class LWAUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    LWAUnpacker (const char* name = "LWAUnpacker");

   protected:

    BitTable *table;

    //! Unpack
    virtual void unpack();
    
    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__LWAUnpacker_h)
