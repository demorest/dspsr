/***************************************************************************
 *
 *   Copyright (C) 2015 by Erik Madsen (from GUPPIFourBit by P. Demorest)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DummyFourBit.h"
#include "dsp/BitTable.h"

#include <iostream>
using namespace std;

dsp::DummyFourBit::DummyFourBit ()
  : FourBitUnpacker ("DummyFourBit")
{
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::DummyFourBit::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::DummyFourBit::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "Dummy"
    && observation->get_nbit() == 4;
}
