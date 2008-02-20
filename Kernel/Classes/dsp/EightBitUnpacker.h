//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/EightBitUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2008/02/20 21:45:40 $
   $Author: straten $ */

#ifndef __EightBitUnpacker_h
#define __EightBitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  class EightBitTable;

  //! Converts 4-bit digitised samples to floating point
  class EightBitUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    EightBitUnpacker (const char* name = "EightBitUnpacker");

    //! Virtual destructor
    virtual ~EightBitUnpacker ();

    //! Set the digitisation convention
    void set_table (EightBitTable* table);

    //! Get the digitisation convention
    const EightBitTable* get_table () const;
			     
  protected:

    //! The four bit table generator  
    Reference::To<EightBitTable> table;
    
    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();
	
  };
}
#endif
