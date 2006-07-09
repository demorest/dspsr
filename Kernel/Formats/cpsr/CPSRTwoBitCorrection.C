/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/CPSRTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

#include <assert.h>

bool dsp::CPSRTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "CPSR";
}

//! Null constructor
dsp::CPSRTwoBitCorrection::CPSRTwoBitCorrection ()
  : SubByteTwoBitCorrection ("CPSRTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  set_ndig (4);
}

/*! Each 2-bit sample from each digitizer is packed into one byte */
unsigned dsp::CPSRTwoBitCorrection::get_ndig_per_byte () const
{ 
  return 4;
}


/*! The quadrature components must be offset by one */
unsigned dsp::CPSRTwoBitCorrection::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The in-phase and quadrature components must be interleaved */
unsigned dsp::CPSRTwoBitCorrection::get_output_incr () const
{
  return 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::CPSRTwoBitCorrection::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}


/*! MSB xQ xI yQ yI LSB

     idig   poln
      0   0re = xI
      1   0im = xQ
      2   1re = yI
      3   1im = yQ 
 */
unsigned
dsp::CPSRTwoBitCorrection::get_shift (unsigned idig, unsigned isamp) const
{
  unsigned shift[4] = { 4, 6, 0, 2 };

  assert (isamp == 0);
  assert (idig < 4);

  return shift[idig];
}

