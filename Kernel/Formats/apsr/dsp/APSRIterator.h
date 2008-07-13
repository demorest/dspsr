//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/apsr/dsp/APSRIterator.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/13 00:38:54 $
   $Author: straten $ */

#ifndef __APSRIterator_h
#define __APSRIterator_h

#include "dsp/BlockIterator.h"

namespace dsp
{
  class Input;

  class APSRIterator : public BlockIterator<const unsigned char>
  {
   public:

    typedef BlockIterator<const unsigned char> Base;

    //! Construct from base pointer
    inline APSRIterator () : Base (0) { }
  
    inline APSRIterator (const APSRIterator& copy) : Base (copy) { }

    inline const APSRIterator& operator = (const APSRIterator& copy)
    {
      Base::operator = (copy);
      return *this;
    }

    inline void set_base (const unsigned char* base)
    {
      current = base;
      if (data_size > 1)
        end_of_data = current + data_size;
    }

    void init (const Input* input);
  };
}

#endif
