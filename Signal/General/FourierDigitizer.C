/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FourierDigitizer.h"
#include <iostream>

//! Default constructor
dsp::FourierDigitizer::FourierDigitizer () : Digitizer ("FourierDigitizer")
{
  nbit = 8;
}

void dsp::FourierDigitizer::set_engine (Engine* _engine)
{
  engine = _engine;
}

//! Set the number of bits per sample
void dsp::FourierDigitizer::set_nbit (int _nbit)
{
  switch(_nbit)
  {
    case 8:
      nbit = _nbit;
      break;
    default:
      throw Error (InvalidState, "dsp::FourierDigitizer::set_nbit",
                  "nbit=%i not understood", _nbit);
      break;
  }
}

/*! 
  This method must tranpose the data from frequency major order to
  time major order.  It is assumed that ndat > 4 * nchan, and therefore
  stride in output time is smaller than stride in input frequency.

  If this condition isn't true, then the nesting of the loops should
  be inverted.
*/
void dsp::FourierDigitizer::pack ()
{
  if (verbose)
    cerr << "dsp::FourierDigitizer::pack" << std::endl;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::FourierDigitizer::pack using Engine with nbit=" << nbit << std::endl;
    engine->pack (nbit, input, output);
    engine->finish();
    return;
  }

  output->set_swap( false );
  output->set_nsub_swap( 0 );

  // the number of polarizations
  const unsigned npol = input->get_npol();

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  float digi_mean=0;
  float digi_sigma=6;
  float digi_scale=0;
  int digi_max=0;
  int digi_min=0;
  int samp_per_byte = 8/nbit;

  switch (nbit)
  {
    case 8:
      digi_mean=127.5;
      digi_scale= digi_mean / digi_sigma;
      digi_min = 0;
      digi_max = 255;
      break;
  }

  int value;

  const unsigned in_step = npol * nchan;

  //only support TPF mode for now
  switch (input->get_order())
  {
/*
    case TimeSeries::OrderTPF:
    {
      for (uint64_t idat=0; idat < ndat; idat++)
      {
        for (unsigned ipol=0; ipol < npol; ipol++)
        {
          const float* inptr = input->get_dattpf() + idat * in_step;
          unsigned char* outptr = output->get_rawptr() + idat*nchan/samp_per_byte;

          // TODO implmenent GPU engine
          for (unsigned ichan=0; ichan < nchan; ichan++)
          {
            printf("%f\t",(*inptr));
            int result = (int)(((*(inptr+ichan)) * digi_scale) + digi_mean +0.5 );
            printf("%d\n",result);

            // clip the result at the limits
            result = std::max(result,digi_min);
            result = std::min(result,digi_max);

            switch (nbit)
            {
              case 8:
                outptr++;
                (*outptr) = (unsigned char) result;
                break;
            }
          }
        }
      }
      return;
    }
*/
    default:
      throw Error (InvalidState, "dsp::FourierDigitizer::operate",
                   "Can only operate on data ordered FTP or PFT.");
  }
}
