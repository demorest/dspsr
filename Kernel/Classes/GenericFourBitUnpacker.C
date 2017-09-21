/***************************************************************************
 *
 *   Copyright (C) 2017 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericFourBitUnpacker.h"
#include "dsp/BitTable.h"

#include <iostream>
using namespace std;

dsp::GenericFourBitUnpacker::GenericFourBitUnpacker ()
  : FourBitUnpacker ("GenericFourBitUnpacker")
{
#define ASSUME_TWOS_COMPLEMENT 1
#if ASSUME_TWOS_COMPLEMENT
  BitTable* table = new BitTable (4, BitTable::TwosComplement);
#else
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
#endif
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::GenericFourBitUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GenericUnpacker::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_nbit() == 4;
}
