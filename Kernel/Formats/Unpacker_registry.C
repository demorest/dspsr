#include "CPSR2TwoBitCorrection.h"
#include "CPSRTwoBitCorrection.h"
#include "S2TwoBitCorrection.h"

Registry::List<dsp::Unpacker> dsp::Unpacker::registry;

static Registry::List<dsp::Unpacker>::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
static Registry::List<dsp::Unpacker>::Enter<dsp::CPSRTwoBitCorrection>  cpsr;
static Registry::List<dsp::Unpacker>::Enter<dsp::S2TwoBitCorrection>    s2;

