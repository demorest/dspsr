//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/gmrt/dsp/GMRTUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2008/04/17 04:47:43 $
   $Author: jayantaroy $ */

#ifndef __GMRTUnpacker_h
#define __GMRTUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the GMRT files
  class GMRTUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    GMRTUnpacker (const char* name = "GMRTUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__GMRTUnpacker_h)
