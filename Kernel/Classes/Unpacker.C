/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Unpacker.h"
#include "Error.h"

using namespace std;

void dsp::Unpacker::prepare ()
{
  if (verbose)
    cerr << "dsp::Unpacker::prepare" << endl;

  // set the Observation information
  output->Observation::operator=(*input);
}

void dsp::Unpacker::resize_output ()
{
  // resize the output
  output->resize (input->get_ndat());
}

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::Unpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  resize_output ();

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "dsp::Unpacker::tranformation TimeSeries book-keeping\n"
      "  input_sample=" << input->input_sample <<
      "  seek=" << input->get_request_offset() <<
      "  ndat=" << input->get_request_ndat() << endl;;

  // Set the input_sample attribute
  output->input_sample = input->input_sample;

  // The following lines deal with time sample resolution of the data source
  output->seek (input->get_request_offset());

  output->decrease_ndat (input->get_request_ndat());

  if (verbose)
    cerr << "dsp::Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Unpacker::match" << endl;
}

//! Constructor
dsp::Unpacker::Unpacker (const char* name)
  : Transformation <BitSeries, TimeSeries> (name, outofplace, true) 
{
}
