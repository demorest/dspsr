/***************************************************************************
 *
 *   Copyright (C) 2017 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/EDAFourBit.h"
#include "dsp/BitTable.h"

#include <assert.h>
#include <iostream>
using namespace std;

dsp::EDAFourBit::EDAFourBit ()
  : FourBitUnpacker ("EDAFourBit")
{
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::EDAFourBit::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::EDAUnpacker::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "EDA" 
    && observation->get_nbit() == 4
    && observation->get_npol() == 2
    && observation->get_ndim() == 1;
}

void dsp::EDAFourBit::unpack ()
{
  cerr << "dsp::EDAFourBit::unpack" << endl;
  
  const uint64_t ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  assert (nchan == 1);
  assert (npol == 2);
  assert (ndim == 1);
  
  const unsigned char* from = input->get_rawptr();

  float* pol0 = output->get_datptr (0,0);
  float* pol1 = output->get_datptr (0,1);

  unsigned long* hist = BitUnpacker::get_histogram (0);

  const float* lookup = table->get_values ();

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    pol0[idat] = lookup[ from[idat] * 2 ];
    pol1[idat] = lookup[ from[idat] * 2 + 1 ];
    
    hist[ from[idat] ] ++;
  }
}

void dsp::EDAFourBit::get_histogram (std::vector<unsigned long>& hist,
				     unsigned idig) const
{
  assert( get_nstate() == 16 );
  assert( get_nstate_internal() == 256 );
  assert( get_ndig() == 2 );
  assert( idig < 2 );
  
  hist.resize( get_nstate() );
  for (unsigned i=0; i<hist.size(); i++)
    hist[i] = 0;
  
  unsigned mask = 0x0f;

  const unsigned long* hist_internal = HistUnpacker::get_histogram (0);

  for (unsigned i=0; i<get_nstate_internal(); i++)
  {
    unsigned s0;
    if (idig == 0)
      s0 = i & mask;
    else
      s0 = (i >> 4) & mask;

    hist[s0] += hist_internal[i];
  }
}
