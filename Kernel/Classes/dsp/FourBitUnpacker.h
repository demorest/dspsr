//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/FourBitUnpacker.h,v $
   $Revision: 1.5 $
   $Date: 2008/02/18 13:27:16 $
   $Author: straten $ */

#ifndef __FourBitUnpacker_h
#define __FourBitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  class FourBitTable;

  //! Converts 4-bit digitised samples to floating point
  class FourBitUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    FourBitUnpacker (const char* name = "FourBitUnpacker");

    //! Virtual destructor
    virtual ~FourBitUnpacker ();

    //! Set the digitisation convention
    void set_table (FourBitTable* table);

    //! Get the digitisation convention
    const FourBitTable* get_table () const;
			     
  protected:

    //! The four bit table generator  
    Reference::To<FourBitTable> table;
    
    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();
	
  };
}
#endif
