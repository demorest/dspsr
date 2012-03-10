/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcDigitizer.h"

//! Default constructor
dsp::SigProcDigitizer::SigProcDigitizer () : Digitizer ("SigProcDigitizer")
{
  nbit = 8;
}

//! Set the number of bits per sample
void dsp::SigProcDigitizer::set_nbit (int _nbit)
{
  switch(_nbit)
  {
  case 1:
  case 2:
  case 4:
  case 8:
  case -32:
    nbit=_nbit;
    break;
  default:
    throw Error (InvalidState, "dsp::SigProcDigitizer::set_nbit",
		 "nbit=%i not understood", _nbit);
    break;
  }
}

class ChannelSort
{
  const bool flip_band;
  const bool swap_band;
  const unsigned nchan;
  const unsigned half_chan;
  const dsp::Observation* input;

public:

  ChannelSort (const dsp::Observation* input) :
    flip_band ( input->get_bandwidth() > 0 ),
    swap_band ( input->get_swap() ),
    nchan ( input->get_nchan() ),
    half_chan ( nchan / 2 ),
    input ( input ) { }

  //! Return the mapping from output channel to input channel
  inline unsigned operator () (unsigned out_chan)
  {
    unsigned in_chan = out_chan;
    if (flip_band)
      in_chan = (nchan-in_chan-1);
    if (input->get_nsub_swap() > 1) 
      in_chan = input->get_unswapped_ichan(out_chan);
    else if (swap_band)
      in_chan = (in_chan+half_chan)%nchan;
    return in_chan;
  }
};

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

  // ChannelSort will re-organize the frequency channels in the output
  output->set_bandwidth( -fabs(input->get_bandwidth()) );
  output->set_swap( false );

  if (nbit == -32)
  {
    pack_float ();
    return;
  }

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  ChannelSort channel (input);

  float digi_mean=0;
  float digi_sigma=6;
  float digi_scale=0;
  int digi_max=0;
  int digi_min=0;
  int samp_per_byte = 8/nbit;

  switch (nbit){
  case 1:
    digi_mean=0.5;
    digi_scale=1;
    digi_min = 0;
    digi_max = 1;
    break;
  case 2:
    digi_mean=1.5;
    digi_scale=1;
    digi_min = 0;
    digi_max = 3;
    break;
  case 4:
    digi_mean=7.5;
    digi_scale= digi_mean / digi_sigma;
    digi_min = 0;
    digi_max = 15;
    break;
  case 8:
    digi_mean=127.5;
    digi_scale= digi_mean / digi_sigma;
    digi_min = 0;
    digi_max = 255;
    break;
  }


  /*
    TFP mode
  */

  switch (input->get_order())
  {

  case TimeSeries::OrderTFP:
  {
#pragma omp parallel for
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      unsigned char* outptr = output->get_rawptr() + idat*nchan/samp_per_byte;
      const float* inptr = input->get_dattfp() + idat*nchan;

      // The following line is important: this function increments
      // the pointer at the start of each byte. MJK2008.
      outptr--;

      int bit_counter=0;
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
	unsigned inChan = channel (ichan);

	//printf("%f\t",(*inptr));
	int result = (int)(((*(inptr+inChan)) * digi_scale) + digi_mean +0.5 );
	//printf("%d\n",result);

	// clip the result at the limits
	//if (result < digi_min)
	//	result = digi_min;
	result = std::max(result,digi_min);

	//if (result > digi_max)
	//	result = digi_max;
	result = std::min(result,digi_max);

	switch (nbit){
	case 1:
	case 2:
	case 4:
	  bit_counter = ichan % (samp_per_byte);
	  
	  //	if(bit_counter==0)outptr[idat*(int)(nchan/samp_per_byte)
	  //		+ (int)(ichan/samp_per_byte)]=(unsigned char)0;
	  //	outptr[idat*(int)(nchan/samp_per_byte)
	  //		+ (int)(ichan/samp_per_byte)] += ((unsigned char) (result)) << (bit_counter*nbit);
	  
	  if(bit_counter==0){
	    outptr++;
	    (*outptr) = (unsigned char)0;
	  }
	  //fprintf(stderr,"%d %d\n",outptr - output->get_rawptr(), idat*(int)(nchan/samp_per_byte) + (int)(ichan/samp_per_byte));
	  (*outptr) |= ((unsigned char) (result)) << (bit_counter*nbit);

	  break;
	case 8:
	  outptr++;
	  (*outptr) = (unsigned char) result;
	  break;
	}
	
	
      }
    }
    
    return;
  }
  case TimeSeries::OrderFPT:
  {
    unsigned char* outptr = output->get_rawptr();

    int bit_counter=0;
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      const float* inptr = input->get_datptr( channel (ichan) );

      for (uint64_t idat=0; idat < ndat; idat++)
      {
	int result = int( (inptr[idat] * digi_scale) + digi_mean +0.5 );

	// clip the result at the limits
	if (result < digi_min)
	  result = digi_min;
	
	if (result > digi_max)
	  result = digi_max;

	switch (nbit){
	case 1:
	case 2:
	case 4:
	  bit_counter = ichan % (samp_per_byte);
	  
	  
	  if(bit_counter==0)outptr[idat*(int)(nchan/samp_per_byte)
				   + (int)(ichan/samp_per_byte)]=(unsigned char)0;
	  outptr[idat*(int)(nchan/samp_per_byte)
		 + (int)(ichan/samp_per_byte)] += ((unsigned char) (result)) << (bit_counter*nbit);
	  
	  break;
	case 8:
	  outptr[idat*nchan + ichan] = (unsigned char) result;
	  break;
	}
      }
    }
    return;
  }
  default:
    throw Error (InvalidState, "dsp::SigProcDigitizer::operate",
		 "Can only operate on data ordered FTP or PFT.");
  }
}




void dsp::SigProcDigitizer::pack_float () try
{
  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  ChannelSort channel (input);

  float* outptr = reinterpret_cast<float*>( output->get_rawptr() );

  switch (input->get_order())
  {
  case TimeSeries::OrderTFP:
  {
    const float* inptr = input->get_dattfp();

    for (uint64_t idat=0; idat < ndat; idat++)
    {
      for (unsigned ichan=0; ichan < nchan; ichan++)
	outptr[ichan] = inptr[ channel(ichan) ];

      inptr += nchan;
    }
    return;
  }
  case TimeSeries::OrderFPT:
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      const float* inptr = input->get_datptr( channel(ichan) );

      for (uint64_t idat=0; idat < ndat; idat++)
	outptr[idat*nchan + ichan] = inptr[idat];
    }
    return;
  }

  default:
    throw Error (InvalidState, "dsp::SigProcDigitizer::pack_float",
		 "Can only operate on data ordered FTP or PFT.");
  }
}
catch (Error& error)
{
  throw error += "dsp::SigProcDigitizer::pack_float";
}

