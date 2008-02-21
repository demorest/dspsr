/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::BitUnpacker::BitUnpacker (const char* _name)
  : HistUnpacker (_name)
{
  set_nsample (256);
}

dsp::BitUnpacker::~BitUnpacker ()
{
}

void dsp::BitUnpacker::set_table (BitTable* _table)
{
  if (verbose)
    cerr << "dsp::BitUnpacker::set_table" << endl;

  table = _table;
}

const dsp::BitTable* dsp::BitUnpacker::get_table () const
{ 
  return table;
}

void dsp::BitUnpacker::unpack ()
{
  const uint64   ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = npol * nchan * ndim;

  if (ndat % 2)
    throw Error (InvalidParam, "dsp::BitUnpacker::unpack",
		 "invalid ndat="UI64, ndat);	 

  unsigned offset = 0;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
	const unsigned char* from = input->get_rawptr() + offset;
	float* into = output->get_datptr (ichan, ipol) + idim;
	unsigned long* hist = get_histogram (offset);
  
	unpack (ndat, from, nskip, into, hist);

	offset ++;


      }
    }
  }
}

