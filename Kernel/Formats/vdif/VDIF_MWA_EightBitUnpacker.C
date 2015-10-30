/***************************************************************************
 *   Copyright (C) 2015 by Stephen Ord
 *   Heavily built on:    
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/VDIF_MWA_EightBitUnpacker.h"
#include "Error.h"

using namespace std;

//! Constructor
dsp::VDIF_MWA_EightBitUnpacker::VDIF_MWA_EightBitUnpacker (const char* name) : HistUnpacker (name)
{
  set_nstate (256); // XXX only allow 8-bit for now.

}


bool dsp::VDIF_MWA_EightBitUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" 
    && observation->get_nbit() == 8
    && observation->get_telescope() == "MWA";
}

/*! The quadrature components are offset by one */
unsigned dsp::VDIF_MWA_EightBitUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::VDIF_MWA_EightBitUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

/*! Each chan has 4 values (quadrature, dual pol) */
unsigned dsp::VDIF_MWA_EightBitUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::VDIF_MWA_EightBitUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nchan = input->get_nchan();
  //const unsigned nskip = npol * ndim;
  const unsigned nskip = npol * ndim * nchan;


  if (ndat==0) return;
 
 
  cerr << "npol=" << npol << " ndim=" << ndim << endl;
  cerr << "ndat=" << ndat << endl;
  cerr << "input_sample" << input->get_input_sample();

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      for (unsigned idim=0; idim<ndim; idim++) {

        unsigned idig = ichan*ndim*npol + ipol*ndim + idim;
        unsigned off = ichan*ndim*npol + ipol*ndim + idim;

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

