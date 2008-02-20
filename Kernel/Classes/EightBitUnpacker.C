/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/EightBitTable.h"
#include "dsp/EightBitUnpacker.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::EightBitUnpacker::EightBitUnpacker (const char* _name)
  : HistUnpacker (_name)
{
  set_nsample (256);
}

dsp::EightBitUnpacker::~EightBitUnpacker ()
{
}

void dsp::EightBitUnpacker::set_table (EightBitTable* _table)
{
  if (verbose)
    cerr << "dsp::EightBitUnpacker::set_table" << endl;

  table = _table;
}

const dsp::EightBitTable* dsp::EightBitUnpacker::get_table () const
{ 
  return table;
}

void dsp::EightBitUnpacker::unpack ()
{
  const uint64   ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = npol * nchan * ndim;

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

	for (uint64 idat = 0; idat < ndat; idat++)
	{
	  into[0] = lookup[ *from ];
	  hist[ *from ] ++;

	  from += nskip;
	  into += ndim;
	}
      }
    }
  }
}


