/***************************************************************************
 *
 *   Copyright (C) 2008 by Jayanta Roy and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/VDIFEightBitUnpacker.h"
#include "dsp/BitTable.h"

using namespace std;

//! Constructor
dsp::VDIFEightBitUnpacker::VDIFEightBitUnpacker (const char* name)
  : EightBitUnpacker ("VDIFEightBit")
{
  set_table( new BitTable (8, BitTable::OffsetBinary) );
}

bool dsp::VDIFEightBitUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::VDIFEightBitUnpacker::matches machine=" 
         << observation->get_machine() 
         << " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "VDIF" 
    && observation->get_nbit() == 8;
}

