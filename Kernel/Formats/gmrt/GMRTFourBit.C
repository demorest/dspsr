/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTFourBit.h"
#include "dsp/BitTable.h"

dsp::GMRTFourBit::GMRTFourBit ()
  : FourBitUnpacker ("GMRTFourBit")
{
  set_table( new BitTable (4, BitTable::TwosComplement) );
}

