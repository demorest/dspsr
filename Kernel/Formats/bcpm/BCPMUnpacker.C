#include "Error.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "dsp/BCPMUnpacker.h"

//! Null constructor
dsp::BCPMUnpacker::BCPMUnpacker (const char* _name) : Unpacker (_name){ }

//! Destructor
dsp::BCPMUnpacker::~BCPMUnpacker (){ }

bool dsp::BCPMUnpacker::matches (const Observation* observation){
  return observation->get_machine() == "BCPM";
}

//! Does the work
void dsp::BCPMUnpacker::unpack (){


}
