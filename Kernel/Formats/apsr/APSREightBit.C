/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSREightBit.h"
#include "dsp/Observation.h"
#include "dsp/Input.h"

#include "dsp/BitTable.h"

#include <assert.h>
#include <iostream>
using namespace std;

bool dsp::APSREightBit::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Analytic;
}

//! Null constructor
dsp::APSREightBit::APSREightBit ()
  : EightBitUnpacker ("APSREightBit")
{
  bool reverse_bits = true;
  table = new BitTable (8, BitTable::TwosComplement, reverse_bits);
}

/*!
  The real and imaginary components of the complex polyphase
  filterbank outputs are decimated together
*/
unsigned dsp::APSREightBit::get_ndim_per_digitizer () const
{
  return 2;
}

void dsp::APSREightBit::unpack ()
{
  const uint64   ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = 1;
  const unsigned fskip = 1;

  const unsigned sample_resolution = input->get_loader()->get_resolution();

  // unpack real and imaginary components at the same time
  const unsigned byte_resolution = sample_resolution * ndim;

  const unsigned npack = ndat / sample_resolution;

  unsigned offset = 0;

  // cerr << "npack=" << npack << " res=" << sample_resolution << " nfloat=" << ndat*ndim << endl;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      float* into = output->get_datptr (ichan, ipol);
      float* backup = into;

      const unsigned char* from = input->get_rawptr() + ipol * byte_resolution;
      const unsigned char* also = from;

      for (unsigned ipack=0; ipack<npack; ipack++)
      {

        // cerr << "ipack=" << ipack << " offset=" << offset << " end=" << into+byte_resolution - backup << endl;

        unsigned long* hist = get_histogram (offset);

        EightBitUnpacker::unpack (byte_resolution, from, nskip, into, fskip, hist);
        from += byte_resolution * 2;
        into += byte_resolution;
      }

      offset ++;
    }
  }
}

