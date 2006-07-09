//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/maxim/dsp/MaximUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2006/07/09 13:27:08 $
   $Author: wvanstra $ */


#ifndef __MaximUnpacker_h
#define __MaximUnpacker_h

#include "dsp/Unpacker.h"

#include "Registry.h"

namespace dsp {

  //! Very simple 8-bit to float unpacker for the Hobart 14m Vela system 

  class MaximUnpacker : public Unpacker {

  public:
    
    //! Constructor
    MaximUnpacker (const char* name = "MaximUnpacker")
      : Unpacker (name) {}

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

  };

}

#endif // !defined(__MaximUnpacker_h)
