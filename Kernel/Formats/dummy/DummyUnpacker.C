/***************************************************************************
 *
 *   Copyright (C) 2015 by Erik Madsen (based on GMRTUnpacker)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DummyUnpacker.h"
#include "dsp/BitTable.h"

using namespace std;

//! Constructor
dsp::DummyUnpacker::DummyUnpacker (const char* name)
  : EightBitUnpacker ("DummyEightBit")
{
  set_table( new BitTable (8, BitTable::TwosComplement) );
}

bool dsp::DummyUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::DummyUnpacker::matches machine=" << observation->get_machine()
         << " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "Dummy" 
    && observation->get_nbit() == 8;
}

