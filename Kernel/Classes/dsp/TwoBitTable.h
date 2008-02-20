//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitTable.h,v $
   $Revision: 1.12 $
   $Date: 2008/02/20 21:44:11 $
   $Author: straten $ */


#ifndef __TwoBitTable_h
#define __TwoBitTable_h

#include "dsp/BitTable.h"

namespace dsp {

  //! Look-up table for converting 2-bit digitized to floating point numbers
  class TwoBitTable : public BitTable
  {

  public:

    //! Constructor
    TwoBitTable (Type type);
    
    //! Set the value of the low voltage state
    void set_lo_val (float lo_val);

    //! Return the value of the low voltage state
    float get_lo_val () const { return lo_val; }

    //! Set the value of the high voltage state
    void set_hi_val (float hi_val);

    //! Return the value of the high voltage state
    float get_hi_val () const { return hi_val; }

    //! Set the flip value to be true or false
    void set_flip (bool flipped);

    //! Get the flip value
    bool get_flip () const { return flip; }

    //! Build a two-bit table with the current attributes
    void rebuild ();

    //! Generate a look-up table for 2-bit to floating point conversion
    void generate_unique_values (float* vals) const;

  protected:

    //! Value of voltage in low state
    float lo_val;

    //! Value of voltage in high state
    float hi_val;

    //! Flip the two data bits - that is SignMag becomes MagSign
    bool flip;

  };

}

#endif // !defined(__TwoBitTable_h)
