
#include "dsp/ACFUnpack.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "Error.h"

//! Null constructor
dsp::ACFUnpacker::ACFUnpacker (const char* _name) : Unpacker (_name)
{

}

dsp::ACFUnpacker::~ACFUnpacker ()
{
}


//! Initialize and resize the output before calling unpack
void dsp::ACFUnpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::ACFUnpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  uint64 ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  // resize the output 
  output->resize (ndat);

  const uint16* input16 = reinterpret_cast<const uint16*>(input->get_rawptr());

  float* input_fft = float_workingspace (nchan * 4 + 2);
  float* output_fft = input + nchan * 2;

  unsigned ichan;

  for (unsigned idat=0; idat < ndat; idat++) {

    for (ichan=0; ichan<nchan; ichan++)
      input_fft[ichan] = (float) input16[ichan];

    input_fft[nchan] = 0.0;

    for (ichan=1; ichan<nchan; ichan++)
      input_fft[nchan+ichan] = input_fft[nchan-ichan];

    fft::frc1d (nchan*2, output_fft, input_fft);

    for (unsigned ichan=0; ichan<nchan; ichan++)
      output->get_datptr (ichan) [idat] = output_fft[ichan*2];

  }

  if (verbose)
    cerr << "dsp::ACFUnpacker::transformation exit" << endl;
}
