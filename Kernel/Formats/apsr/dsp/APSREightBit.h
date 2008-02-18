//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/apsr/dsp/APSREightBit.h,v $
   $Revision: 1.1 $
   $Date: 2008/02/18 11:04:37 $
   $Author: straten $ */

#ifndef __APSREightBit_h
#define __APSREightBit_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for APSR files
  class APSREightBit : public HistUnpacker
  {

  public:
    
    //! Constructor
    APSREightBit (const char* name = "APSREightBit");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__APSREightBit_h)
