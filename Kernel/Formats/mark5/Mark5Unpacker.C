/***************************************************************************
 *
 *   Copyright (C) 2005-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mark5Unpacker.h"
#include "dsp/Mark5File.h"
#include "dsp/Mark5TwoBitCorrection.h"
#include "vlba_stream.h"

using namespace std;

//! Constructor
dsp::Mark5Unpacker::Mark5Unpacker (const char* _name) : Unpacker (_name)
{
}

/*! Returns true if the machine name is Mark5 and Mark5TwoBitCorrection
  can't do it */
bool dsp::Mark5Unpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Mark5" &&
    ! dsp::Mark5TwoBitCorrection::can_do( observation );
}

unsigned dsp::Mark5Unpacker::get_ndig () const
{
  return input->get_nchan() * input->get_npol();
}

void dsp::Mark5Unpacker::unpack()
{
  // Bit Stream in input?
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();      
  const unsigned npol = input->get_npol();

  if (verbose)
    cerr << "Mark5Unpacker::unpack ndat=" << ndat << " npol=" << npol
	 << " nchan=" << nchan << endl;
	
  const Mark5File* file = get_Input<Mark5File>();
  if (!file)
    throw Error (InvalidState, "dsp::Mark5Unpacker::unpack",
		 "Input is not a Mark5File");

  struct VLBA_stream* vlba_stream = (struct VLBA_stream*) file->stream;

  float* data [npol * nchan];

  for (unsigned ipol = 0 ; ipol < npol ; ipol++)
    for (unsigned ichan=0; ichan < nchan; ichan++)
      data[ipol + 2*ichan] = output->get_datptr(ichan,ipol);

  if (VLBA_stream_get_data (vlba_stream, ndat, data) < 0)
    throw Error (InvalidState, "dsp::Mark5Unpacker::unpack",
                 "error VLBA_stream_get_data (most likely EOD)");

}

