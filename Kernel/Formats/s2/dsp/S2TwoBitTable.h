//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2TwoBitTable.h,v $
   $Revision: 1.4 $
   $Date: 2006/07/09 13:27:09 $
   $Author: wvanstra $ */


#ifndef __S2TwoBitTable_h
#define __S2TwoBitTable_h

#include "dsp/TwoBitTable.h"

namespace dsp {

  //! Look-up tables for conversion from S2 two-bit to floating point numbers
  /*! 
    The conversion scheme is specific to the ordering of bits in S2 data

  */
  class S2TwoBitTable : public TwoBitTable {

  public:

    //! Constructor
    S2TwoBitTable (Type type) : TwoBitTable (type) { }
    
    //! Destructor
    ~S2TwoBitTable () { }
 
    //! Return the 2-bit number from byte corresponding to sample
    virtual unsigned twobit (unsigned byte, unsigned sample) const;

  };

}

#endif // !defined(__S2TwoBitTable_h)
