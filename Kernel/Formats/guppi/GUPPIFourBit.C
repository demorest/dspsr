/***************************************************************************
 *
 *   Copyright (C) 2015 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GUPPIFourBit.h"
#include "dsp/BitTable.h"

#include <iostream>
using namespace std;

dsp::GUPPIFourBit::GUPPIFourBit ()
  : FourBitUnpacker ("GUPPIFourBit")
{
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::GUPPIFourBit::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GUPPIFourBit::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine().substr(1) == "UPPI"
    && observation->get_nbit() == 4;
}
