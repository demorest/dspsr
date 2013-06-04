/***************************************************************************
 *
 *   Copyright (C) 2002-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Accumulation.h"
#include "dsp/Observation.h"
#include "dsp/Scratch.h"

#include "Error.h"

#include <memory>

#include <string.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


using namespace std;

//! Constructor
dsp::Accumulation::Accumulation ( unsigned _tscrunch)
  : Transformation <TimeSeries,TimeSeries> ("Accumulation", outofplace)
{
  tscrunch = _tscrunch;
  stride = 512;
}

void dsp::Accumulation::set_engine (Engine* _engine)
{
  engine = _engine;
}

void dsp::Accumulation::prepare ()
{
  if (verbose)
    cerr << "dsp::Accumulation::prepare()" << endl;
  resize_output ();
}

void dsp::Accumulation::reserve()
{
  output->Observation::operator=(*input);
  output->set_npol (1);
  output->set_ndim (1);
  output->set_nchan (input->get_nchan());
  output->resize (input->get_ndat() / tscrunch);
}

//! Detect the input data
void dsp::Accumulation::transformation () try
{
  if (verbose)
    cerr << "dsp::Accumulation::transformation()" << endl;

  // check for real, detected input
  if (get_input()->get_state() != Signal::Intensity)
    throw (InvalidState, "dsp::Accumulation::transformation",
     "dsp::Accumulation cannot integrate non-detected input data");

  // check for inplace operation
  if (input.get() == output.get())
   throw (InvalidState, "dsp::Accumulation::transformation",
     "dsp::Accumulation cannot integrate inplace");

  if (verbose)
    cerr << "dsp::Accumulation::transformation input ndat=" << input->get_ndat()
         << " state=" << Signal::state_string(get_input()->get_state()) << endl;

  if (input->get_ndat() == 0)
    return;

  const uint64_t ndat = input->get_ndat();
  uint64_t output_ndat = ndat / tscrunch; 
  if (verbose)
    cerr << "dsp::Accumulation::transformation output_ndat=" << output_ndat << " tscrunch=" << tscrunch << endl;


  output->set_ndat (output_ndat);
  output->set_npol (input->get_npol());
  output->set_ndim (1);
  output->set_state (Signal::Intensity);
  output->resize (output_ndat);
  //output->set_order (TimeSeries::OrderTFP);

  if (engine)
  {
    if (verbose)
      cerr << "dsp::Accumulation::transformation using Engine" << endl;

    engine->integrate (input, output, tscrunch, stride);
    return;
  }

  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const unsigned nfloat = input->get_ndim() * input->get_ndat();

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      // TODO implement
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::Accumulation::transformation";
}

void dsp::Accumulation::resize_output ()
{
  if (verbose)
    cerr << "dsp::Accumulation::resize_output" << endl;

  if (input.get() == output.get())
   throw (InvalidState, "dsp::Accumulation::resize_output",
     "dsp::Accumulation cannot integrate inplace");

  get_output()->copy_configuration( get_input() );

  uint64_t input_ndat = input->get_ndat();
  if (input_ndat % (tscrunch * stride) != 0)
    throw Error (InvalidState, "dsp::Accumulation::resize_output",
        "Input ndat must be divisible by tscrunch factor");
  get_output()->resize( input_ndat / tscrunch );
}

