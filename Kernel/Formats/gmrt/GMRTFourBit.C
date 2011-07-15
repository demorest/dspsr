/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTFourBit.h"
#include "dsp/BitTable.h"

#include <iostream>
using namespace std;

dsp::GMRTFourBit::GMRTFourBit ()
  : FourBitUnpacker ("GMRTFourBit")
{
  BitTable* table = new BitTable (4, BitTable::TwosComplement);
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::GMRTFourBit::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GMRTUnpacker::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "GMRT" 
    && observation->get_nbit() == 4;
}
