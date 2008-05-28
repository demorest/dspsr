/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BPSRUnpacker.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::BPSRUnpacker::BPSRUnpacker (const char* name) : HistUnpacker (name)
{
}

bool dsp::BPSRUnpacker::matches (const Observation* observation)
{
  cerr << "dsp::BPSRUnpacker::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;

  return observation->get_machine() == "BPSR" 
    && observation->get_nbit() == 8
    && observation->get_ndim() == 1;
}

/*! The first nchan digitizer channels are poln0, the next nchan are poln1 */
unsigned dsp::BPSRUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / input->get_nchan();
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::BPSRUnpacker::get_output_ichan (unsigned idig) const
{
  return idig % input->get_nchan();
}

void dsp::BPSRUnpacker::unpack ()
{
  const uint64 ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned step = npol * nchan;

  // data are organized: p0c0 p0c1 p1c0 p1c1, p0c2 p0c3 p1c2 p1c3, ...

  for (unsigned ichan=0; ichan<nchan; ichan++) 
  {
    unsigned chan_off = (ichan/2) * 4 + ichan%2;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      unsigned pol_off = ipol * 2;

      const unsigned char* from = input->get_rawptr() + chan_off + pol_off;
      float* into = output->get_datptr (ichan, ipol);

      // unsigned long* hist = get_histogram (off);

      for (unsigned bt = 0; bt < ndat; bt++)
      {
        // hist[ *from ] ++;
        into[bt] = float( *from );
        from += step;
      }
    }
  }
}

