/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunch.h"
#include "dsp/InputBuffering.h"

#include "Error.h"

using namespace std;

dsp::TScrunch::TScrunch (Behaviour place) 
  : Transformation <TimeSeries, TimeSeries> ("TScrunch", place, true)
{
  factor = 0;
  time_resolution = 0;
  use_tres = false;

  if (preserve_data)
    set_buffering_policy (new InputBuffering (this));
}

void dsp::TScrunch::set_factor( unsigned samples )
{
  factor = samples;
  use_tres = false;
}

void dsp::TScrunch::set_time_resolution( double microseconds )
{
  time_resolution = microseconds;
  use_tres = true;
}

unsigned dsp::TScrunch::get_factor() const
{
  if (use_tres)
  {
    if( time_resolution <= 0.0 )
      throw Error(InvalidState,"dsp::Tscrunch::get_factor",
		  "invalid time resolution:%lf", time_resolution);
    double in_tsamp = 1.0e6/input->get_rate();  // in microseconds
    factor = unsigned(time_resolution/in_tsamp + 0.00001);
    
    if ( factor<1 )
      factor = 1;

    use_tres = false;
    time_resolution = 0.0;
  }
  
  return factor;
}

double dsp::TScrunch::get_time_resolution() const
{
  if (!time_resolution)
    time_resolution = 1.0e6/(input->get_rate()*double(factor));

  return time_resolution;
}

void dsp::TScrunch::prepare ()
{
  if (!has_buffering_policy())
    return;

  unsigned sfactor = get_factor();

  if (verbose)
    cerr << "dsp::TScrunch::prepare factor=" << sfactor << endl;

  get_buffering_policy()->set_minimum_samples ( sfactor );
}

void dsp::TScrunch::transformation ()
{
  sfactor = get_factor();

  if (!sfactor)
    throw Error (InvalidState, "dsp::TScrunch::get_factor",
		   "scrunch factor not set");

  if (sfactor==1)
  {
    if( input.get() != output.get() )
      output->operator=( *input );
    return;
  }

  if( !input->get_detected() )
    throw Error(InvalidState,"dsp::TScrunch::transformation()",
		"invalid input state: " + tostring(input->get_state()));

  output_ndat = input->get_ndat()/sfactor;

  if (verbose)
    cerr << "dsp::TScrunch::transformation input ndat=" << input->get_ndat()
         << " output ndat=" << output_ndat << " sfactor=" << sfactor << endl;

  prepare ();

  if (has_buffering_policy())
    get_buffering_policy()->set_next_start (output_ndat * sfactor);

  if (input.get() != output.get())
  {
    get_output()->copy_configuration( get_input() );
    get_output()->resize( output_ndat );
  }

  output->rescale( sfactor );
  output->set_rate( input->get_rate()/sfactor );

  switch (input->get_order())
  {
    case TimeSeries::OrderFPT:
      fpt_tscrunch ();
      break;

    case TimeSeries::OrderTFP:
      tfp_tscrunch ();
      break;
  }

  if( input.get() == output.get() )
    output->set_ndat( input->get_ndat()/sfactor );
}

void dsp::TScrunch::fpt_tscrunch ()
{
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol = input->get_npol();

  for (unsigned ichan=0; ichan<input_nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<input_npol; ipol++)
    {
      const float* in = input->get_datptr(ichan, ipol);
      float* out = output->get_datptr(ichan, ipol);
      
      unsigned input_idat=0;

      for( unsigned output_idat=0; output_idat<output_ndat; ++output_idat)
      {
	unsigned stop = input_idat + sfactor;
	
	out[output_idat] = in[input_idat]; 	++input_idat;
	
	for( ; input_idat<stop; ++input_idat)
	  out[output_idat] += in[input_idat];
      }
    } // for each ipol
  } // for each ichan
}

void dsp::TScrunch::tfp_tscrunch ()
{
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol = input->get_npol();

  const unsigned nfloat = input_nchan * input_npol;

  const float* indat = input->get_dattfp ();
  float* outdat = output->get_dattfp ();
  
  for( unsigned output_idat=0; output_idat<output_ndat; ++output_idat)
  {
    for (unsigned ifloat=0; ifloat < nfloat; ifloat++)
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
