#include "CPSR2TwoBitCorrection.h"
#include "Observation.h"
#include "TwoBitTable.h"

// Register the CPSR2TwoBitCorrection class with the Unpacker::registry
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> entry;

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

