#include "CPSR2TwoBitCorrection.h"
#include "TwoBitTable.h"

//! Null constructor
dsp::CPSR2TwoBitCorrection::CPSR2TwoBitCorrection ()
  : TwoBitCorrection ("CPSR2TwoBitCorrection")
{
  nchannel = 2;
  channels_per_byte = 1;

  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

