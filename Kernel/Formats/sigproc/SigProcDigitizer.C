/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcDigitizer.h"

    //! Default constructor
dsp::SigProcDigitizer::SigProcDigitizer ()
{
  nbit = 8;
}

//! Set the number of bits per sample
void dsp::SigProcDigitizer::set_nbit (unsigned _nbit)
{
  if (_nbit != 8)
    throw Error (InvalidState, "dsp::SigProcDigitizer::set_nbit",
		 "only 8 bit sampling implemented");
}

unsigned dsp::SigProcDigitizer::get_nbit () const
{
  return nbit;
}

/*! 
  This method must tranpose the data from frequency major order to
  time major order.  It is assumed that ndat > 4 * nchan, and therefore
  stride in output time is smaller than stride in input frequency.

  If this condition isn't true, then the nesting of the loops should
  be inverted.
*/
void dsp::SigProcDigitizer::pack ()
{
  if (input->get_npol() != 1)
    throw Error (InvalidState, "dsp::SigProcDigitizer::pack",
		 "cannot handle npol=%d", input->get_npol());

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64 ndat = input->get_ndat();

  unsigned char* outptr = output->get_rawptr();

  const float bit8_mean = 127.5;
  const float cutoff_sigma = 6.0;
  const float bit8_scale = bit8_mean / cutoff_sigma;

  bool flip_band = input->get_bandwidth() > 0;
  if (flip_band)
    output->set_bandwidth(-input->get_bandwidth());
  
  output->set_nbit(nbit);

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    const float* inptr;
    if (flip_band)
      inptr = input->get_datptr (nchan-ichan-1);
    else
      inptr = input->get_datptr (ichan);

    for (uint64 idat=0; idat < ndat; idat++)
    {
      int result = int( (inptr[idat] * bit8_scale) + bit8_mean );

      // clip the result at the limits
      if (result < 0)
	result = 0;

      if (result > 255)
	result = 255;

      outptr[idat*nchan + ichan] = (unsigned char) result;
    }
  }
}

