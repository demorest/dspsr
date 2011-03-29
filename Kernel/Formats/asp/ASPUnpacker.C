/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASPUnpacker.h"
#include "Error.h"

//! Constructor
dsp::ASPUnpacker::ASPUnpacker (const char* name) : HistUnpacker (name)
{
  set_ndig (4);
  set_nstate (256);
}

bool dsp::ASPUnpacker::matches (const Observation* observation)
{
  // Mock spectrometer data happens to be in the same format...
  return (observation->get_machine() == "ASP" 
      || observation->get_machine() == "Mock")
    && observation->get_nbit() == 8;
}

/*! The quadrature components must be offset by one */
unsigned dsp::ASPUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::ASPUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

void dsp::ASPUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nskip = npol * ndim;

  //cerr << "npol=" << npol << " ndim=" << ndim << endl;

  for (unsigned ipol=0; ipol<npol; ipol++) {
    for (unsigned idim=0; idim<ndim; idim++) {

      unsigned off = ipol * ndim + idim;

      //cerr << "ipol=" << ipol << " idim=" << idim << " off=" << off << endl;

      const char* from = reinterpret_cast<const char*>(input->get_rawptr()+off);
      float* into = output->get_datptr (0, ipol) + idim;
      unsigned long* hist = get_histogram (off);
  
      for (unsigned bt = 0; bt < ndat; bt++) {
        hist[ (unsigned char) *from ] ++;
        *into = float(int( (signed char) *from ));
        from += nskip;
        into += ndim;
      }
    }
  }
}

