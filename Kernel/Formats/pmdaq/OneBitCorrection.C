#include <iostream>
#include <assert.h>
#include <math.h>

#include "environ.h"
#include "OneBitCorrection.h"
#include "TimeSeries.h"
#include "Error.h"

#include "genutil.h"

//! Null constructor
dsp::OneBitCorrection::OneBitCorrection (const char* _name)
  : Unpacker (_name)
{
}

dsp::OneBitCorrection::~OneBitCorrection ()
{
}

//! Initialize and resize the output before calling unpack
void dsp::OneBitCorrection::transformation ()
{
  if (input->get_nbit() != 1)
    throw_str ("OneBitCorrection::transformation input not 1-bit digitized");

  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));  // MXB K?

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();
}

// MXB - this is where it all happens.
// Unpack the data into a 2D array with nfreq channels and ndat
// time samples.

void dsp::OneBitCorrection::unpack ()
{
  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack" << endl;

  if (input->get_state() != Signal::Intensity)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "input not total intensity");

  if (input->get_nbit() != 1)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "input not 1-bit sampled");

  int64 ndat = input->get_ndat();
  unsigned n_freq = input->get_nchan();

  if (n_freq % 32)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "nchan=%d is not a multiple of 32", n_freq);

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack ndat="
	 << ndat << " n_freq=" << n_freq << endl;
  
  const uint32 * rawptr = (const uint32 *) input->get_rawptr();
  const uint32 mask = 0x01;

  unsigned nskip32 = n_freq/32;

  for (unsigned ichan=0; ichan < n_freq; ichan++) {

    unsigned shift32 = ichan % 32;

    const uint32* from = rawptr + ichan / 32;
    float* into = output -> get_datptr (ichan, 0);

    for (unsigned idat=0; idat < ndat; idat++) {
      *into = float (((*from)>>shift32) & mask);
      from += nskip32;
      into ++;
    }
  }
}

bool dsp::OneBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "PMDAQ"
    && observation->get_nbit() == 1;
}
