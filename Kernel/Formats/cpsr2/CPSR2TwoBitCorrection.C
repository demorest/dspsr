#include "dsp/CPSR2TwoBitCorrection.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

bool dsp::CPSR2TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "CPSR2";
}

//! Null constructor
dsp::CPSR2TwoBitCorrection::CPSR2TwoBitCorrection ()
  : TwoBitCorrection ("CPSR2TwoBitCorrection")
{
  nchannel = 2;
  channels_per_byte = 1;

  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

