#include "PuMa2Unpacker.h"
#include "Error.h"

bool dsp::PuMa2Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "PuMa2" 
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Nyquist;
}

void dsp::PuMa2Unpacker::unpack ()
{
  const unsigned npol = input->get_npol();
  const uint64 ndat = input->get_ndat();
  const unsigned char mask = 0x7f;

  for (unsigned ipol=0; ipol<npol; ipol++) {

    const unsigned char* from = input->get_rawptr() +ipol;
    float* into = output->get_datptr (0, ipol);

    for (unsigned bt = 0; bt < ndat; bt++) {
      into[bt] = float(int( *from & mask ) - 128);
      from += npol;
    }

  }
}

