/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"

#include "dsp/FourBitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"
#include <assert.h>

using namespace std;

//! Null constructor
dsp::FourBitUnpacker::FourBitUnpacker (const char* _name)
  : BitUnpacker (_name)
{
  // with 4 bits, there are 16 output states
  set_nstate (16);

  // internally, count the frequency of 8-bit bytes
  set_nstate_internal (256);
}

void dsp::FourBitUnpacker::get_histogram (std::vector<unsigned long>& hist,
					  unsigned idig) const
{
  assert( get_nstate() == 16 );
  assert( get_nstate_internal() == 256 );
  assert( idig < get_ndig() );

  hist.resize( get_nstate() );

  unsigned mask = 0x0f;

  const unsigned long* hist_internal = HistUnpacker::get_histogram (idig);

  for (unsigned i=0; i<get_nstate_internal(); i++)
  {
    unsigned s0 = i & mask;
    unsigned s1 = (i >> 4) & mask;

    hist[s0] += hist_internal[i];
    hist[s1] += hist_internal[i];
  }
}


void dsp::FourBitUnpacker::unpack (uint64_t ndat, 
				   const unsigned char* from,
				   const unsigned nskip,
				   float* into, const unsigned fskip,
				   unsigned long* hist)
{
  const uint64_t ndat2  = ndat/2;
  const float* lookup = table->get_values ();

  if (ndat % 2)
    throw Error (InvalidParam, "dsp::FourBitUnpacker::unpack",
                 "invalid ndat="UI64, ndat);

  for (uint64_t idat = 0; idat < ndat2; idat++)
  {
    into[0]    = lookup[ *from * 2 ];
    into[fskip] = lookup[ *from * 2 + 1 ];
    
    hist[ *from ] ++;

    from += nskip;
    into += fskip * 2;
  }
}
