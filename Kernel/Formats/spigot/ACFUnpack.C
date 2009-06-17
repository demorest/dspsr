/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ACFUnpack.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/Scratch.h"

#include "FTransform.h"
#include "Error.h"

using namespace std;

//! Null constructor
dsp::ACFUnpack::ACFUnpack (const char* _name) : Unpacker (_name)
{
}

dsp::ACFUnpack::~ACFUnpack ()
{
}

bool dsp::ACFUnpack::matches (const Observation* observation)
{
  return observation->get_machine() == "Spigot";
}


//! Initialize and resize the output before calling unpack
void dsp::ACFUnpack::unpack ()
{
  if (verbose)
    cerr << "dsp::ACFUnpack::unpack" << endl;;

  uint64_t ndat = input->get_ndat();
  unsigned nchan = input->get_nchan();

  const uint16_t* input16 = reinterpret_cast<const uint16_t*>(input->get_rawptr());

  float* input_fft = scratch->space<float> (nchan * 4 + 2);
  float* output_fft = input_fft + nchan * 2;

  unsigned ichan;

  for (unsigned idat=0; idat < ndat; idat++) {

    for (ichan=0; ichan<nchan; ichan++)
      input_fft[ichan] = (float) input16[ichan];

    input_fft[nchan] = 0.0;

    for (ichan=1; ichan<nchan; ichan++)
      input_fft[nchan+ichan] = input_fft[nchan-ichan];

    FTransform::frc1d (nchan*2, output_fft, input_fft);

    for (unsigned ichan=0; ichan<nchan; ichan++)
      output->get_datptr (ichan) [idat] = output_fft[ichan*2];

  }

  if (verbose)
    cerr << "dsp::ACFUnpack::unpack exit" << endl;
}
