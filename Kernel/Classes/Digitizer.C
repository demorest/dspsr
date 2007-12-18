/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Digitizer.h"
#include "Error.h"

using namespace std;

void dsp::Digitizer::prepare ()
{
  if (verbose)
    cerr << "dsp::Digitizer::prepare" << endl;

  // set the Observation information
  output->Observation::operator=(*input);
}

void dsp::Digitizer::resize_output ()
{
  if (verbose)
    cerr << "dsp::Digitizer::resize_output" << endl;

  // resize the output
  output->resize (input->get_ndat());
}

//! Initialize and resize the output before calling unpack
void dsp::Digitizer::transformation ()
{
  if (verbose)
    cerr << "dsp::Digitizer::transformation" << endl;;

  prepare ();
  resize_output ();
  pack ();

  if (verbose)
    cerr << "dsp::Digitizer::transformation exit" << endl;;
}



//! Constructor
dsp::Digitizer::Digitizer (const char* name)
  : Transformation <TimeSeries, BitSeries> (name, outofplace, true) 
{
}
