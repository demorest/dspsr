//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __EmerlinTwoBitTable_h
#define __EmerlinTwoBitTable_h

#include "dsp/TwoBitTable.h"

namespace dsp {

  //! Look-up tables for conversion from Emerlin two-bit to floating point numbers
  /*! Emerlin defines bits to run in time order from LSB to MSB, this is
   * the opposite of the standard dspsr TwoBitTable convention so
   * we need to override the 'extract' function here.
  */
  class EmerlinTwoBitTable : public TwoBitTable {

  public:

    //! Constructor
    EmerlinTwoBitTable () : TwoBitTable (TwoBitTable::OffsetBinary) { 
    destroy();
    build();
    }
    
    //! Destructor
    ~EmerlinTwoBitTable () { }
 
    //! Return the 2-bit number from byte corresponding to sample
    virtual unsigned extract (unsigned byte, unsigned sample) const;

  };

}

#endif // !defined(__EmerlinTwoBitTable_h)
