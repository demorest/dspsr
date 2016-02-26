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
  set_nstate (256);
}

dsp::BitUnpacker::~BitUnpacker ()
{
}

double dsp::BitUnpacker::get_optimal_variance ()
{
  if (!table)
    throw Error (InvalidState, "dsp::BitUnpacker::get_optimal_variance",
                 "BitTable not set");

  return table->get_optimal_variance();
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
  const uint64_t ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  const unsigned nbit  = input->get_nbit();

  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;

  // Step through the array in small block sizes so that the matrix
  // transpose (for nchan>1 case) remains cache-friendly.
  const unsigned blockdat = npol*nchan*ndim*8/nbit > 32 ? npol*nchan*ndim*8/nbit : 32;
  const unsigned blockbytes = blockdat*nbit/8;

  const unsigned char* iptr = input->get_rawptr();

  for (uint64_t idat=0; idat<ndat; idat+=blockdat, iptr+=blockbytes)
  {
    unsigned offset = 0;
    const unsigned ndatblock = (blockdat>(ndat-idat)) ? ndat-idat : blockdat/nskip;
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned idim=0; idim<ndim; idim++)
        {
          const unsigned char* from = iptr + offset;
          float* into = output->get_datptr (ichan, ipol) + ndim*idat + idim;
          unsigned long* hist = get_histogram (offset);

#ifdef _DEBUG
          cerr << "c=" << ichan << " p=" << ipol << " d=" << idim << endl;
#endif
 
          unpack (ndatblock, from, nskip, into, fskip, hist);
          offset ++;
        }
      }
    }
  }
}

