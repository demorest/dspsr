//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/FourBitTwo.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/13 00:41:38 $
   $Author: straten $ */

#ifndef __FourBitTwo_h
#define __FourBitTwo_h

#include "dsp/NLowLookup.h"

namespace dsp
{
  //! Unpack two 4-bit samples per byte from an array of bytes
  class FourBitTwo : public NLowLookup
  {

  public:

    FourBitTwo (BitTable* table) : NLowLookup (table) { }

    template<class Iterator>
    inline void prepare (const Iterator& input, unsigned ndat) { }
    
    template<class Iterator>
    inline void unpack (Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& nlow)
    {
      const unsigned ndat2 = ndat/2;
      nlow = 0;

      for (unsigned idat = 0; idat < ndat2; idat++)
      {
	unsigned index = *input;
	++ input;

	*output = lookup[ index*2 ];     output += output_incr;
	*output = lookup[ index*2 + 1 ]; output += output_incr;
	
	nlow += nlow_lookup [index];
      }
    }
  };

}

#endif
