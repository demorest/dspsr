/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GUPPIFITSUnpacker.h"
#include "Error.h"

dsp::GUPPIFITSUnpacker::GUPPIFITSUnpacker(const char* name) : Unpacker(name) {}

void dsp::GUPPIFITSUnpacker::unpack()
{
  if (verbose) {
    std::cerr << "dsp::GUPPIFITSUnpacker::unpack" << std::endl;
  }

  // Allocate mapping method to use depending on how many bits per value.
  const unsigned nbit = input->get_nbit();

  switch (nbit) {
    case 8:
      break;
    default:
      throw Error(InvalidState, "GUPPIFITSUnpacker::unpack",
          "invalid nbit=%d", nbit);
  }

  const unsigned npol  = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndat  = input->get_ndat();

  const unsigned char* from = input->get_rawptr();

  // NOTE Only 8-bit case here so far

  // Figure out which polarizations have signed values
  std::vector<bool> pol_signed (npol, false);
  if (npol==4) 
  {
    pol_signed[2] = true;
    pol_signed[3] = true;
    if (input->get_state() == Signal::Coherence)
      pol_signed[1] = false;
    if (input->get_state() == Signal::Stokes)
      pol_signed[1] = true;
  }

  // Loop thru data
  for (unsigned idat=0; idat<ndat; idat++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        float *into = output->get_datptr(ichan, ipol) + idat;
        if (pol_signed[ipol])
          *into = (signed char)(*from);
        else
          *into = (unsigned char)(*from);
        // Fix GUPPI cross-terms
        if (ipol>1 && input->get_state()==Signal::Stokes) { *into += 0.5; }
        from++;
      }
    }
  }

}

bool dsp::GUPPIFITSUnpacker::matches(const Observation* observation)
{
  return observation->get_machine() == "GUPPIFITS" && 
    observation->get_nbit() == 8;
}
