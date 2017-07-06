//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TwoBitMask.h

#ifndef __TwoBitMask_h
#define __TwoBitMask_h

#include <iostream>

namespace dsp {

  //! Shift and mask two contiguous bits
  template<unsigned N>
  class ShiftMask {

  public:

    unsigned shift[N];

    //! Return the shifted 2-bit number
    inline unsigned char operator() (unsigned char data, unsigned isamp)
    { return (data >> shift[isamp]) & 0x03; }

    //! Return the shifted 2-bit number
    inline unsigned char operator() (unsigned char data)
    { return (data >> shift[0]) & 0x03; }

  };

  //! Gathers the two bits from separate locations
  template<unsigned N>
  class GatherMask {

  public:

    unsigned shift0[N];
    unsigned shift1[N];

    //! Return the shifted 2-bit number
    inline unsigned char operator() (unsigned char data, unsigned isamp)
    { return ((data>>shift0[isamp]) & 0x01)|((data>>shift1[isamp]) & 0x02); }

  };

  template<unsigned N>
  std::ostream& operator<< (std::ostream& ostr, const GatherMask<N>& mask)
  {
    for (unsigned i=0; i<N; i++)
      ostr << mask.shift0[i] << ":" << mask.shift1[i] << " ";
    return ostr;
  }
  
}

#endif // !defined(__TwoBitMask_h)

