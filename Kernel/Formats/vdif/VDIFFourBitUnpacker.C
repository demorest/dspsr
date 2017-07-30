/***************************************************************************
 *
 *   Copyright (C) 2008 by Jayanta Roy and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/VDIFFourBitUnpacker.h"
#include "dsp/BitTable.h"

using namespace std;

//! Constructor
dsp::VDIFFourBitUnpacker::VDIFFourBitUnpacker (const char* name)
  : FourBitUnpacker ("VDIFFourBit")
{
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::VDIFFourBitUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::VDIFFourBitUnpacker::matches machine=" 
         << observation->get_machine() 
         << " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "VDIF" 
    && observation->get_nbit() == 4
    && observation->get_npol() == 1;
}

