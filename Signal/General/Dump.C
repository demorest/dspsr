/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Dump.h"

#include <fstream>
using namespace std;

dsp::Dump::Dump (const char* name) : Sink<TimeSeries> (name)
{
  binary = false;
}

    //! Set the ostream to which data will be dumped
void dsp::Dump::set_output (FILE* fptr)
{
  output = fptr;
}

//! Set the flag to output binary data
void dsp::Dump::set_output_binary (bool flag)
{
  binary = flag;
}

//! Adds to the totals
void dsp::Dump::calculation ()
{
  if (!output)
    throw Error (InvalidState, "dsp::Dump::calculation",
		 "output FILE* not set");

  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const uint64_t ndat = input->get_ndat();

  const int64_t istart = input->get_input_sample();

  for (unsigned ichan = 0; ichan < nchan; ichan ++)
  {
    for (unsigned ipol = 0; ipol < npol; ipol++)
    {
      const float* data = input->get_datptr (ichan, ipol);
      if (binary)
      {
	fwrite (&ichan, sizeof(unsigned), 1, output);
	fwrite (&ipol,  sizeof(unsigned), 1, output);
	fwrite (&istart, sizeof(int64_t), 1, output);
	fwrite (&ndat,  sizeof(uint64_t), 1, output);
	fwrite (data,   sizeof(float), ndat, output);
      }
      else
      {
	for (uint64_t idat = 0; idat < ndat; idat++)
	  fprintf (output, "%"PRIu64" %u %u %f",
		   istart+idat, ichan, ipol, data[idat]);
      }
    }
  }
}

