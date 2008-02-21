/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FourBitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::FourBitUnpacker::FourBitUnpacker (const char* _name)
  : BitUnpacker (_name)
{
}

void dsp::FourBitUnpacker::unpack (uint64 ndat, 
				   const unsigned char* from,
				   const unsigned nskip,
				   float* into, 
				   unsigned long* hist)
{
  const uint64 ndat2  = ndat/2;
  const unsigned ndim = input->get_ndim();
  const float* lookup = table->get_values ();

  for (uint64 idat = 0; idat < ndat2; idat++)
  {
    into[0]    = lookup[ *from * 2 ];
    into[ndim] = lookup[ *from * 2 + 1 ];
    
    hist[ *from ] ++;

    from += nskip;
    into += ndim * 2;
  }
}
