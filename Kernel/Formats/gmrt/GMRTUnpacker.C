/***************************************************************************
 *
 *   Copyright (C) 2008 by Jayanta Roy
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTUnpacker.h"
#include "Error.h"

using namespace std;

//! Constructor
dsp::GMRTUnpacker::GMRTUnpacker (const char* name) : HistUnpacker (name)
{
  set_ndig (4);
}

bool dsp::GMRTUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GMRTUnpacker::matches machine=" << observation->get_machine() 
         << " nbit=" << observation->get_nbit() << endl;

  return observation->get_machine() == "GMRT" 
    && observation->get_nbit() == 8;
}

/*! The quadrature components must be offset by one */
unsigned dsp::GMRTUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::GMRTUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

void dsp::GMRTUnpacker::unpack ()
{
  const uint64 ndat = input->get_ndat();
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
        *into = float(int( *from ));
        from += nskip;
        into += ndim;
      }
    }
  }
}

