//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFTwoBitTable_h
#define __VDIFTwoBitTable_h

#include "dsp/TwoBitTable.h"

namespace dsp {

  //! Look-up tables for conversion from VDIF two-bit to floating point numbers
  /*! VDIF defines bits to run in time order from LSB to MSB, this is
   * the opposite of the standard dspsr TwoBitTable convention so
   * we need to override the 'extract' function here.
  */
  class VDIFTwoBitTable : public TwoBitTable {

  public:

    //! Constructor
    VDIFTwoBitTable (Type type) : TwoBitTable (type) { }
    
    //! Destructor
    ~VDIFTwoBitTable () { }
 
    //! Return the 2-bit number from byte corresponding to sample
    virtual unsigned extract (unsigned byte, unsigned sample) const;

  };

}

#endif // !defined(__VDIFTwoBitTable_h)
