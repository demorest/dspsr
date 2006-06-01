#include "PuMa2Unpacker.h"
#include "Error.h"

//! Constructor
dsp::PuMa2Unpacker::PuMa2Unpacker (const char* name) : HistUnpacker (name)
{
  set_nsample (256);
}

bool dsp::PuMa2Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "PuMa2" 
    && observation->get_nbit() == 8
    && observation->get_state() == Signal::Nyquist;
}

void dsp::PuMa2Unpacker::unpack ()
{
  const uint64 ndat = input->get_ndat();
  const unsigned npol = input->get_npol();

  for (unsigned ipol=0; ipol<npol; ipol++) {

    const unsigned char* from = input->get_rawptr() + ipol;
    float* into = output->get_datptr (0, ipol);
    unsigned long* hist = get_histogram (ipol);

    for (unsigned bt = 0; bt < ndat; bt++) {
      hist[ *from ] ++;
      into[bt] = float(int( (char)*from ));
      from += npol;
    }

  }
}

