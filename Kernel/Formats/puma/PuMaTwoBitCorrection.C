#include "dsp/PuMaTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::PuMaTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "PuMa";
}

//! Null constructor
dsp::PuMaTwoBitCorrection::PuMaTwoBitCorrection ()
  : SubByteTwoBitCorrection ("PuMaTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

