/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FourBitTable.h"
#include "dsp/FourBitUnpacker.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::FourBitUnpacker::FourBitUnpacker (const char* _name)
  : HistUnpacker (_name)
{
  set_nsample (256);
}

dsp::FourBitUnpacker::~FourBitUnpacker ()
{
}

void dsp::FourBitUnpacker::set_table (FourBitTable* _table)
{
  if (verbose)
    cerr << "dsp::FourBitUnpacker::set_table" << endl;

  table = _table;
}

const dsp::FourBitTable* dsp::FourBitUnpacker::get_table () const
{ 
  return table;
}

void dsp::FourBitUnpacker::unpack ()
{
  const uint64   ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = npol * nchan * ndim;

  const uint64 ndat2 = ndat/2;
  if (ndat % 2)
    throw Error (InvalidParam, "dsp::FourBitUnpacker::unpack",
		 "invalid ndat="UI64, ndat);
		 
  unsigned offset = 0;

  const float* lookup = table->get_values ();

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
	const unsigned char* from = input->get_rawptr() + offset;
	float* into = output->get_datptr (ichan, ipol) + idim;
	unsigned long* hist = get_histogram (offset);
  
	offset ++;

	for (uint64 idat = 0; idat < ndat2; idat++)
	{
	  into[0]    = lookup[ *from * 2 ];
	  into[ndim] = lookup[ *from * 2 + 1 ];

	  hist[ *from ] ++;

	  from += nskip;
	  into += ndim * 2;
	}
      }
    }
  }
}


