/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBitTable.h"

dsp::TwoBitTable::TwoBitTable (Type _type, bool _reverse)
  : BitTable (2, _type, _reverse)
{
  lo_val = 0.25;
  hi_val = 0.75;
  reverse_2bit = false;

  build ();
}

//! Set the value of the low voltage state
void dsp::TwoBitTable::set_lo_val (float _lo_val)
{
  lo_val = _lo_val;
}

//! Set the value of the high voltage state
void dsp::TwoBitTable::set_hi_val (float _hi_val)
{
  hi_val = _hi_val;
}

void dsp::TwoBitTable::set_reverse_2bit (bool flag)
{
  reverse_2bit = flag;
}

void dsp::TwoBitTable::rebuild ()
{
  build ();
}

void dsp::TwoBitTable::generate_unique_values (float* vals) const
{
  switch (type) {

  case OffsetBinary:
    vals[0] = -hi_val;
    vals[1] = -lo_val;
    vals[2] = lo_val;
    vals[3] = hi_val;
    break;
    
  case SignMagnitude:
    vals[0] = lo_val;
    vals[1] = hi_val;
    vals[2] = -lo_val;
    vals[3] = -hi_val;
    break;
    
  case TwosComplement:
    vals[0] = lo_val;
    vals[1] = hi_val;
    vals[2] = -hi_val;
    vals[3] = -lo_val;
    break;

  default:
    throw Error (InvalidState, "TwoBitTable::generate_unique_values",
		 "unsupported type");
  }
  
  // Swap the order of the bits, e.g. SignMag becomes MagSign
  if (reverse_2bit)
    std::swap (vals[1], vals[2]);
}
