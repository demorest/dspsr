//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/vdif/dsp/VDIFnByteUnpacker.h

#ifndef __VDIFnByteUnpacker_h
#define __VDIFnByteUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp
{

  //! Simple N-byte to float unpacker for VDIF files
  class VDIFnByteUnpacker : public Unpacker 
  {

  public:
    
    //! Constructor
    VDIFnByteUnpacker (const char* name = "VDIFnByteUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_ichan (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;


  };

}

#endif // !defined(__VDIFnByteUnpacker_h)
