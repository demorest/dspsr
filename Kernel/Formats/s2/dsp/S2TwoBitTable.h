//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2TwoBitTable.h,v $
   $Revision: 1.3 $
   $Date: 2002/11/03 21:51:49 $
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
