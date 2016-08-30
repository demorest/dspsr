/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/BitTable.h"
#include "dsp/LWAUnpacker.h"

#include "Error.h"

//! Constructor
dsp::LWAUnpacker::LWAUnpacker (const char* name) : HistUnpacker (name)
{
  table = new BitTable (4, BitTable::TwosComplement);
  table->set_order( BitTable::MostToLeast );
  set_ndig(2);
  set_nstate(16);
}

bool dsp::LWAUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "LWA" 
    && observation->get_nbit() == 4;
}

/*! The quadrature components must be offset by one */
unsigned dsp::LWAUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! Then the two polns */
unsigned dsp::LWAUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

void dsp::LWAUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const float *lookup = table->get_values();

  for (unsigned ipol=0; ipol<npol; ipol++) {
    const unsigned char *from = 
      reinterpret_cast<const unsigned char*>(input->get_rawptr()) + ipol;
    float *into = output->get_datptr(0,ipol);
    // TODO histogram?
    for (unsigned bt=0; bt<ndat; bt++) {
      into[0] = lookup[*from * 2];
      into[1] = lookup[*from * 2 + 1];
      into+=2;
      from+=npol; // If polns are interleaved, or single-pol
    }
  }
}
