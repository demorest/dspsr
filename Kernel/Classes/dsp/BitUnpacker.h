//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/BitUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2008/02/21 04:56:06 $
   $Author: straten $ */

#ifndef __BitUnpacker_h
#define __BitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  class BitTable;

  //! Converts N-bit digitised samples to floating point using a BitTable
  class BitUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    BitUnpacker (const char* name = "BitUnpacker");

    //! Virtual destructor
    virtual ~BitUnpacker ();

    //! Set the digitisation convention
    void set_table (BitTable* table);

    //! Get the digitisation convention
    const BitTable* get_table () const;
			     
  protected:

    //! The four bit table generator  
    Reference::To<BitTable> table;
    
    //! Unpack all channels, polarizations, real/imag, etc.
    virtual void unpack ();

    //! Unpack a single digitizer output
    virtual void unpack (uint64 ndat, const unsigned char* from,
			 const unsigned nskip,
			 float* into, unsigned long* hist) = 0;

  };
}
#endif
