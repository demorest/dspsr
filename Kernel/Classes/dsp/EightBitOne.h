//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/EightBitOne.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/13 00:38:53 $
   $Author: straten $ */

#ifndef __EightBitOne_h
#define __EightBitOne_h

#include "dsp/NLowLookup.h"

namespace dsp
{
  //! Unpack one 8-bit samples per byte from an array of bytes
  class EightBitOne : public NLowLookup
  {

  public:

    EightBitOne (BitTable* table) : NLowLookup (table) { }

    template<class Iterator>
    inline void prepare (const Iterator& input, unsigned ndat) { }
    
    template<class Iterator>
    inline void unpack (Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& nlow)
    {
      nlow = 0;

      for (unsigned idat = 0; idat < ndat; idat++)
      {
	output[idat * output_incr] = lookup[ *input ];
	nlow += nlow_lookup[ *input ];
	++ input;
      }
    }
  };

}

#endif
