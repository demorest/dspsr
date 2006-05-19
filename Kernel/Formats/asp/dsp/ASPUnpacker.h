//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/asp/dsp/ASPUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2006/05/19 01:54:24 $
   $Author: wvanstra $ */

#ifndef __ASPUnpacker_h
#define __ASPUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for the ASP files
  class ASPUnpacker : public HistUnpacker {

  public:
    
    //! Constructor
    ASPUnpacker (const char* name = "ASPUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;

  };

}

#endif // !defined(__ASPUnpacker_h)
