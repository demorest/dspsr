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
  threshold = 1.5;
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

