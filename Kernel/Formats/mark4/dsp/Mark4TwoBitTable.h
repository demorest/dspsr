//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/mark4/dsp/Mark4TwoBitTable.h


#ifndef __Mark4TwoBitTable_h
#define __Mark4TwoBitTable_h

#include "dsp/TwoBitTable.h"

namespace dsp {

  //! Look-up tables for conversion from Mark4 two-bit to floating point numbers
  /*! 
    The conversion scheme is specific to the ordering of bits in Mark4 data

  */
  class Mark4TwoBitTable : public TwoBitTable {

  public:

    //! Constructor
    Mark4TwoBitTable (Type type) : TwoBitTable (type) { }
    
    //! Destructor
    ~Mark4TwoBitTable () { }
 
    //! Return the 2-bit number from byte corresponding to sample
    virtual unsigned twobit (unsigned byte, unsigned sample) const;

  };

}

#endif // !defined(__Mark4TwoBitTable_h)
