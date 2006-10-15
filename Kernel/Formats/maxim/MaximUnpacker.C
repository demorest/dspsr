/***************************************************************************
 *
 *   Copyright (C) 2004 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MaximUnpacker.h"
#include "Error.h"

using namespace std;

bool dsp::MaximUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Maxim"
    && observation->get_nbit() == 4
    && observation->get_state() == Signal::Nyquist;
}

void dsp::MaximUnpacker::unpack ()
{
  unsigned samples_per_byte = 1;

  if (input->get_state() != Signal::Nyquist && input->get_state() 
      != Signal::Analytic)
    throw Error (InvalidParam, "dsp::MaximUnpacker::check_input",
		 "input is detected");

  const unsigned char* from = input->get_rawptr();

  float* into = output->get_datptr (0, 0);

  unsigned bytes = input->get_ndat()/samples_per_byte;

  for (unsigned bt = 0; bt < bytes; bt++) {
    *into = float(int(*from)-128);
    into += 1;
    from += 1;
  }

  if (verbose)
    cerr << "dsp::MaximUnpacker::unpack out of simple unpack";
}
