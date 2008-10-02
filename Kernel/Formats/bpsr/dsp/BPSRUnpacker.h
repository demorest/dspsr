//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/bpsr/dsp/BPSRUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2008/10/02 23:56:57 $
   $Author: straten $ */

#ifndef __BPSRUnpacker_h
#define __BPSRUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR files
  class BPSRUnpacker : public HistUnpacker 
  {

  public:
    
    //! Constructor
    BPSRUnpacker (const char* name = "BPSRUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_ichan (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__BPSRUnpacker_h)
