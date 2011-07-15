/***************************************************************************
 *
 *   Copyright (C) 2008 by Jayanta Roy and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTUnpacker.h"
#include "dsp/BitTable.h"

using namespace std;

//! Constructor
dsp::GMRTUnpacker::GMRTUnpacker (const char* name)
  : EightBitUnpacker ("GMRTEightBit")
{
  set_table( new BitTable (8, BitTable::TwosComplement) );
}

bool dsp::GMRTUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GMRTUnpacker::matches machine=" << observation->get_machine() 
         << " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "GMRT" 
    && observation->get_nbit() == 8;
}

