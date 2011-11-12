/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitCorrectionMulti.h"
#include "dsp/TwoBitTable.h"

bool dsp::VDIFTwoBitCorrectionMulti::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" 
      && observation->get_nbit() == 2
      && observation->get_npol() == 2;
}

//! Null constructor
dsp::VDIFTwoBitCorrectionMulti::VDIFTwoBitCorrectionMulti ()
  : SubByteTwoBitCorrection ("VDIFTwoBitCorrectionMulti")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  set_ndig(2);
}

#if 0 
unsigned
dsp::VDIFTwoBitCorrectionMulti::get_shift (unsigned idig, unsigned isamp) const
{
  unsigned shift[4] = { 4, 6, 0, 2 };

  assert (isamp < 2);
  assert (idig < 2);

  return shift[idig];
}
#endif
