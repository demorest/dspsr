#include <iostream>
#include <assert.h>
#include <math.h>

#include "dsp/MaximUnpacker.h"

#include "Error.h"

bool dsp::MaximUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Maxim"
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Nyquist;
}

void dsp::MaximUnpacker::unpack ()
{
  uint64 ndat = input->get_ndat();

  unsigned samples_per_byte = 1;

  if (input->get_state() != Signal::Nyquist && input->get_state() 
      != Signal::Analytic)
    throw Error (InvalidParam, "dsp::MaximUnpacker::check_input",
		 "input is detected");

  const unsigned char* from = input->get_rawptr();

  float* into_0 = output->get_datptr (0, 0);
  float* into_1 = output->get_datptr (0, 1); 
 
  unsigned bytes = input->get_ndat()/samples_per_byte;

  for (unsigned bt = 0; bt < bytes; bt++) {
    *into_0 = float(int(*from)-128);
    *into_1 = float(int(*from)-128);
    into_0 += 1;
    into_1 += 1;
    from += 1;
  }

  if (verbose)
    cerr << "dsp::MaximUnpacker::unpack out of simple unpack";
}
