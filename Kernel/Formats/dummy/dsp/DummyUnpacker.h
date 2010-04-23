//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DummyUnpacker_h
#define __DummyUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Fake 8-bit unpacker
  class DummyUnpacker : public Unpacker {

  public:
    
    //! Constructor
    DummyUnpacker (const char* name = "DummyUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__DummyUnpacker_h)
