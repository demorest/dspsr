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
  //BitTable* table = new BitTable (4, BitTable::TwosComplement);
  table = new BitTable (4, BitTable::TwosComplement);
  table->set_order( BitTable::MostToLeast );
  //set_table( table );
  set_ndig(2);
  set_nstate(16);
}

bool dsp::LWAUnpacker::matches (const Observation* observation)
{
  // Mock spectrometer data happens to be in the same format...
  return observation->get_machine() == "LWA" 
    && observation->get_nbit() == 4 && observation->get_npol() == 1;
}

/*! The quadrature components must be offset by one */
unsigned dsp::LWAUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! This only works for 1 pol */
unsigned dsp::LWAUnpacker::get_output_ipol (unsigned idig) const
{
  return 0;
}

void dsp::LWAUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const float *lookup = table->get_values();

  for (unsigned ipol=0; ipol<npol; ipol++) {
    const unsigned char *from = 
      reinterpret_cast<const unsigned char*>(input->get_rawptr());
    float *into = output->get_datptr(0,ipol);
    // TODO histogram
    for (unsigned bt=0; bt<ndat; bt++) {
      into[0] = lookup[*from * 2];
      into[1] = lookup[*from * 2 + 1];
      into+=2;
      from++;
    }
  }
}
