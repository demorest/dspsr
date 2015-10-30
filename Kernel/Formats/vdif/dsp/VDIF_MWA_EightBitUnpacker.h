//-*-C++-*-
/***************************************************************************
 *   Copyright (C) 2015 by Stephen Ord
 *   Heavily built on:
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIF_MWA_EightBitUnpacker_h
#define __VDIF_MWA_EightBitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for VDIF baseband files
  class VDIF_MWA_EightBitUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    VDIF_MWA_EightBitUnpacker (const char* name = "VDIF_MWA_EightBitUnpacker");

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

#endif // !defined(__VDIF_MWA_EightBitUnpacker_h)
