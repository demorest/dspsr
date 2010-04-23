/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DummyUnpacker.h"
#include "Error.h"

//! Constructor
dsp::DummyUnpacker::DummyUnpacker (const char* name) : Unpacker (name)
{
}

bool dsp::DummyUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Dummy" 
    && observation->get_nbit() == 8;
}

/*! The quadrature components must be offset by one */
unsigned dsp::DummyUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::DummyUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

void dsp::DummyUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const unsigned nskip = npol * ndim;

  // Simple loop over data, ignoring ordering.
  // If the incoming data needs to be reordered this
  // will overestimate the processing speed..
  const char* from = reinterpret_cast<const char*>(input->get_rawptr());
  float* into = output->get_datptr(0,0);
  const uint64_t tot_dat = ndat * npol * ndim;
  for (unsigned bt=0; bt<tot_dat; bt++) {
    *into = (float)((signed char) *from);
    from++;
    into++;
  }

}

