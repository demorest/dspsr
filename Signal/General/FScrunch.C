/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FScrunch.h"
#include "Error.h"

using namespace std;

dsp::FScrunch::FScrunch (Behaviour place) 
  : Transformation <TimeSeries, TimeSeries> ("FScrunch", place, true)
{
  factor = 0;
  frequency_resolution = 0;
  use_fres = false;
}

void dsp::FScrunch::set_factor( unsigned samples )
{
  factor = samples;
  use_fres = false;
}

void dsp::FScrunch::set_frequency_resolution( double microseconds )
{
  frequency_resolution = microseconds;
  use_fres = true;
}

unsigned dsp::FScrunch::get_factor() const
{
  if (use_fres)
  {
    if (frequency_resolution <= 0.0)
      throw Error (InvalidState,"dsp::Fscrunch::get_factor",
		   "invalid frequency resolution:%lf", frequency_resolution);

    double in_fres = input->get_bandwidth() / input->get_nchan();
    factor = unsigned(in_fres / frequency_resolution + 0.00001);
    
    if ( factor<1 )
      factor = 1;

    use_fres = false;
    frequency_resolution = 0.0;
  }
  
  return factor;
}

double dsp::FScrunch::get_frequency_resolution() const
{
  if (!frequency_resolution)
    frequency_resolution = factor*input->get_bandwidth()/input->get_nchan();

  return frequency_resolution;
}

void dsp::FScrunch::transformation ()
{
  sfactor = get_factor();

  if (!sfactor)
    throw Error (InvalidState, "dsp::FScrunch::get_factor",
		   "scrunch factor not set");

  if (sfactor==1)
  {
    if( input.get() != output.get() )
      output->operator=( *input );
    return;
  }

  if( !input->get_detected() )
    throw Error(InvalidState,"dsp::FScrunch::transformation()",
		"invalid input state: " + tostring(input->get_state()));

  output_nchan = input->get_nchan()/sfactor;

  if (verbose)
    cerr << "dsp::FScrunch::transformation input nchan=" << input->get_nchan()
         << " output nchan=" << output_nchan << " sfactor=" << sfactor << endl;

  if (input.get() != output.get())
  {
    get_output()->copy_configuration (get_input());
    get_output()->set_nchan (output_nchan);
    get_output()->resize (input->get_ndat());
  }

  output->rescale( sfactor );

  switch (input->get_order())
  {
    case TimeSeries::OrderFPT:
      fpt_fscrunch ();
      break;

    case TimeSeries::OrderTFP:
      tfp_fscrunch ();
      break;
  }

  if( input.get() == output.get() )
    output->set_nchan( output_nchan );
}

void dsp::FScrunch::fpt_fscrunch ()
{
  const unsigned input_nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const uint64_t nfloat = input->get_ndat() * input->get_ndim();

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    unsigned input_chan = 0;

    for (unsigned ichan=0; ichan<output_nchan; ichan++)
    {
      float* out = output->get_datptr (ichan, ipol);
      const float* in = input->get_datptr (input_chan, ipol); input_chan ++;

      for (unsigned ifloat=0; ifloat<nfloat; ifloat++)
	out[ifloat] = in[ifloat];

      for (unsigned ifactor=1; ifactor<sfactor; ifactor++)
      {
	assert( input_chan < input_nchan );
	in = input->get_datptr (input_chan, ipol); input_chan ++;
	for (unsigned ifloat=0; ifloat<nfloat; ifloat++)
	  out[ifloat] += in[ifloat];
      }
    }
  } // for each ipol
}

void dsp::FScrunch::tfp_fscrunch ()
{
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
  const uint64_t ndat = input->get_ndat();

  unsigned nfloat = npol * ndim;

  const float* indat = input->get_dattfp ();
  float* outdat = output->get_dattfp ();
  
  for (unsigned idat=0; idat<ndat; idat++)
  {
    for (unsigned ifloat=0; ifloat<nfloat; ifloat++)
      outdat[ifloat] = indat[ifloat];

    for (unsigned ifactor=1; ifactor < sfactor; ifactor++)
    {
      indat += nfloat;
      for (unsigned ifloat=0; ifloat < nfloat; ifloat++)
        outdat[ifloat] += indat[ifloat];
    }

    indat += nfloat;
    outdat += nfloat;
  }
}
