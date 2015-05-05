/***************************************************************************
 *
 *   Copyright (C) 2015 by Stuart Weston and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mark5bUnpacker.h"
#include "dsp/Mark5bFile.h"
#include "vlba_stream.h"

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

  struct VLBA_stream* vlba_stream = (struct VLBA_stream*) file->stream;

  float* data [npol * nchan];

  for (unsigned ipol = 0 ; ipol < npol ; ipol++)
    for (unsigned ichan=0; ichan < nchan; ichan++)
      data[ipol + 2*ichan] = output->get_datptr(ichan,ipol);

  if (VLBA_stream_get_data (vlba_stream, ndat, data) < 0)
    throw Error (InvalidState, "dsp::Mark5bUnpacker::unpack",
                 "error VLBA_stream_get_data (most likely EOD)");

}

