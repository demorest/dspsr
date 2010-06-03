/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdlib.h>
#include "dsp/Digitizer.h"
#include "Error.h"

using namespace std;

void dsp::Digitizer::prepare ()
{
  if (verbose)
    cerr << "dsp::Digitizer::prepare" << endl;

  // set the Observation information
  output->Observation::operator=(*input);

  // nbit may equal -32 for float (FITS BITPIX convention)
  output->set_nbit( abs(nbit) );
}

void dsp::Digitizer::set_nbit (int n)
{
  nbit = n;
}

int dsp::Digitizer::get_nbit () const
{
  return nbit;
}

void dsp::Digitizer::reserve ()
{
  if (verbose)
    cerr << "dsp::Digitizer::reserve" << endl;

  // resize the output
  output->resize (input->get_ndat());
}

//! Initialize and resize the output before calling unpack
void dsp::Digitizer::transformation ()
{
  if (verbose)
    cerr << "dsp::Digitizer::transformation" << endl;;

  prepare ();
  reserve ();
  pack ();

  if (verbose)
    cerr << "dsp::Digitizer::transformation exit" << endl;;
}



//! Constructor
dsp::Digitizer::Digitizer (const char* name)
  : Transformation <TimeSeries, BitSeries> (name, outofplace) 
{
}
