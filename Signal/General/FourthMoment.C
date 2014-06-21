/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FourthMoment.h"
#include "dsp/Observation.h"
#include "dsp/Scratch.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"

#include <memory>

#include <string.h>

using namespace std;

//! Constructor
dsp::FourthMoment::FourthMoment () 
  : Transformation <TimeSeries,TimeSeries> ("FourthMoment", outofplace)
{
}

void dsp::FourthMoment::prepare ()
{
  if (input->get_state() != Signal::Stokes)
    throw Error (InvalidState, "dsp::FourthMoment::transformation",
		 "input does not contain Stokes parameters");

  if (input->get_ndim() != 4)
    throw Error (InvalidState, "dsp::FourthMoment::transformation",
		 "Stokes parameters were not detected with ndim == 4");

  output->copy_configuration( input );
  output->set_state( Signal::FourthMoment );
  output->set_ndim( 14 );
  output->set_npol( 1 );
  output->resize( input->get_ndat() );
}

//! Detect the input data
void dsp::FourthMoment::transformation () try
{
  if (input->get_ndat() == 0)
    return;

  prepare ();

  const unsigned nchan = output->get_nchan();
  const unsigned ndat = output->get_ndat();

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    const float* in = input->get_datptr (ichan, 0);
    float* out= output->get_datptr(ichan,0);

    for (unsigned idat=0; idat<ndat; idat++)
    {
      memcpy (out, in, 4 * sizeof(float));
      out += 4;

      for (unsigned i=0; i<4; i++)
	for (unsigned j=i; j<4; j++)
	{
	  *out = in[i] * in[j];
	  out ++;
	}

      in += 4;
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::FourthMoment::transformation";
}
