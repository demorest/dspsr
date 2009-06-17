/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "apsr_unpack.h"
#include "dsp/Input.h"

#include <iostream>
using namespace std;

void apsr_unpack (const dsp::BitSeries* input, dsp::TimeSeries* output,
                  dsp::BitUnpacker* unpacker)
{
  const uint64_t   ndat  = input->get_ndat();

  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = 1;
  const unsigned fskip = 1;

  const unsigned sample_resolution = input->get_loader()->get_resolution();

  // unpack real and imaginary components at the same time
  const unsigned nfloat = sample_resolution * ndim;

  // the number of bytes corresponding to nfloat floats
  const unsigned nbyte = nfloat * input->get_nbit() / 8;

  const unsigned npack = ndat / sample_resolution;

  unsigned offset = 0;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      float* into = output->get_datptr (ichan, ipol);
      const unsigned char* from = input->get_rawptr() + ipol * nbyte;

      for (unsigned ipack=0; ipack<npack; ipack++)
      {
        /*
        cerr << "ipack=" << ipack << " offset=" << offset 
             << " end=" << into+nfloat - backup << endl;
        */

        bool all_zero = true;
        for (unsigned ibyte = 0; ibyte < nbyte; ibyte++)
          if (from[ibyte] != 0)
          {
            all_zero = false;
            break;
          }

        if (all_zero)
          for (unsigned ifloat = 0; ifloat < nfloat; ifloat++)
            into[ifloat] = 0.0;
        else
        {
          unsigned long* hist = unpacker->get_histogram (offset);
          unpacker->unpack (nfloat, from, nskip, into, fskip, hist);
        }

        from += nbyte * npol;
        into += nfloat;
      }

      offset ++;
    }
  }
}

