/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GUPPIUnpacker.h"
#include "Error.h"

using namespace std;

//! Constructor
dsp::GUPPIUnpacker::GUPPIUnpacker (const char* name) : HistUnpacker (name)
{
  set_nstate (256); // XXX only allow 8-bit for now.
}

bool dsp::GUPPIUnpacker::matches (const Observation* observation)
{
  // TODO make a better matching scheme here
  return observation->get_machine() == "GUPPI" 
    && observation->get_nbit() == 8;
}

/*! The quadrature components are offset by one */
unsigned dsp::GUPPIUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::GUPPIUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

/*! Each chan has 4 values (quadrature, dual pol) */
unsigned dsp::GUPPIUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::GUPPIUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nchan = input->get_nchan();
  const unsigned nskip = npol * ndim;

  //cerr << "npol=" << npol << " ndim=" << ndim << endl;
  cerr << "ndat=" << ndat << endl;

  // TODO how to deal with structure of the file where large chunks of
  // each channel come in one at a time...

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      for (unsigned idim=0; idim<ndim; idim++) {

        unsigned idig = ichan*ndim*npol + ipol*ndim + idim;
        // XXX this only works if ndat always equals 1 data block:
        unsigned off = ichan*ndim*npol*ndat + ipol*ndim + idim;

        const char* from = 
          reinterpret_cast<const char*>(input->get_rawptr()+off);
        float* into = output->get_datptr(ichan,ipol) + idim;
        unsigned long* hist = get_histogram(idig);
  
        for (unsigned bt = 0; bt < ndat; bt++) {
          hist[ (unsigned char) *from ] ++;
          *into = float(int( (signed char) *from ));
          from += nskip;
          into += ndim;
        }
      }
    }
  }
}

