//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/EightBitUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2008/02/20 21:49:46 $
   $Author: straten $ */

#ifndef __EightBitUnpacker_h
#define __EightBitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  class BitTable;

  //! Converts 4-bit digitised samples to floating point
  class EightBitUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    EightBitUnpacker (const char* name = "EightBitUnpacker");

    //! Virtual destructor
    virtual ~EightBitUnpacker ();

    //! Set the digitisation convention
    void set_table (BitTable* table);

    //! Get the digitisation convention
    const BitTable* get_table () const;
			     
  protected:

    //! The four bit table generator  
    Reference::To<BitTable> table;
    
    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();
	
  };
}
#endif
