/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/EightBitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::EightBitUnpacker::EightBitUnpacker (const char* _name)
  : BitUnpacker (_name)
{
}

void dsp::EightBitUnpacker::unpack (uint64 ndat, 
				    const unsigned char* from,
				    const unsigned nskip,
				    float* into, 
				    unsigned long* hist)
{
  const unsigned ndim = input->get_ndim();
  const float* lookup = table->get_values ();

  for (uint64 idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];

    from += nskip;
    into += ndim;
  }
}
