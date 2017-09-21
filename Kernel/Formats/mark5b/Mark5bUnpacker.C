/***************************************************************************
 *
 *   Copyright (C) 2016 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mark5bUnpacker.h"
#include "dsp/Mark5bFile.h"

#include <mark5access.h>

using namespace std;

//! Constructor
dsp::Mark5bUnpacker::Mark5bUnpacker (const char* _name) : Unpacker (_name)
{
}

/*! Returns true if the machine name is Mark5b */
bool dsp::Mark5bUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Mark5b";
}

unsigned dsp::Mark5bUnpacker::get_ndig () const
{
  return input->get_nchan() * input->get_npol();
}

void dsp::Mark5bUnpacker::unpack()
{
  // Bit Stream in input?
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();      
  const unsigned npol = input->get_npol();

  if (verbose)
    cerr << "Mark5bUnpacker::unpack ndat=" << ndat << " npol=" << npol
	 << " nchan=" << nchan << endl;
	
  const Mark5bFile* file = get_Input<Mark5bFile>();
  if (!file)
    throw Error (InvalidState, "dsp::Mark5bUnpacker::unpack",
		 "Input is not a Mark5bFile");

  struct mark5_stream* m5stream = (struct mark5_stream*) file->stream;

  float* data [npol * nchan];

  /* Stuart: this is the place in the code where we rearrange channels */
  for (unsigned ipol = 0 ; ipol < npol ; ipol++)
    for (unsigned ichan=0; ichan < nchan; ichan++)
      data[ipol + npol*ichan] = output->get_datptr(ichan,ipol);

  if (mark5_stream_decode(m5stream, ndat, data) < 0)
    throw Error (InvalidState, "dsp::Mark5bUnpacker::unpack",
                 "error mark5_stream_decode (most likely EOD)");

}

