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
  scale_fac = 1.0;
  rescale = true;
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
  case 16:
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
    //if (flip_band)
    //  in_chan = (nchan-in_chan-1);
    if (input->get_nsub_swap() > 1) 
      in_chan = input->get_unswapped_ichan(out_chan);
    else if (swap_band)
      in_chan = (in_chan+half_chan)%nchan;
    // moved from the start of the block 
    if (flip_band)
      in_chan = (nchan-in_chan-1);
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

  // ChannelSort will re-organize the frequency channels in the output
  output->set_bandwidth( -fabs(input->get_bandwidth()) );
  output->set_swap( false );
  output->set_nsub_swap( 0 );

  if (nbit == -32)
  {
    pack_float ();
    return;
  }

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  // number of polns
  const unsigned npol = input->get_npol();

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
  case 16:
    digi_mean=32768.0;
    digi_scale= digi_mean / digi_sigma;
    digi_min = 0;
    digi_max = 65535;
    break;
  }

  // If rescale is false we do not apply the nbit-dependent scaling above.
  // Note that for 4-pol data we still need to offset the (signed) poln
  // cross-terms so that they can be packed into an unsigned value.
  float xpol_offset = 0.0;
  if (rescale==false)
  {
    xpol_offset = digi_mean;
    digi_mean = 0.0;
    digi_scale = 1.0;
  }

  // Also apply any existing scale factors (note, Rescale will set the
  // input scale to 1.0 if it has been applied to the data).
  digi_scale /= input->get_scale() * scale_fac;

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

      for (unsigned ipol=0; ipol < npol; ipol++)
      {

        float mean = digi_mean;
        if (ipol>1) { mean += xpol_offset; }

        unsigned char* outptr;
        if (nbit==16)
          outptr = output->get_rawptr() + 2*(idat*nchan*npol + ipol*nchan);
        else
          outptr = output->get_rawptr() 
            + (idat*nchan*npol + ipol*nchan)/samp_per_byte;

        const float* inptr = input->get_dattfp() + idat*nchan*npol + ipol;

        // The following line is important: this function increments
        // the pointer at the start of each byte. MJK2008.
        outptr--;

        int bit_counter=0;
        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          unsigned inChan = channel (ichan);

          //printf("%f\t",(*inptr));
          int result = (int)(((*(inptr+inChan*npol))*digi_scale)+mean+0.5);
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
          case 16:
            outptr++;
            *((uint16_t*)outptr) = (uint16_t) result;
            outptr++;
            break;
          }
        } // chan
      } // poln
    } // time
    
    return;
  }
  case TimeSeries::OrderFPT:
  {
    unsigned char* outptr = output->get_rawptr();

    int bit_counter=0;
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {

      for (unsigned ipol=0; ipol < npol; ipol++) 
      {

        float mean = digi_mean;
        if (ipol>1) { mean += xpol_offset; }

        const float* inptr = input->get_datptr( channel (ichan), ipol );

        for (uint64_t idat=0; idat < ndat; idat++)
        {
          int result = int( (inptr[idat] * digi_scale) + mean +0.5 );

          // clip the result at the limits
          if (result < digi_min)
            result = digi_min;
          
          if (result > digi_max)
            result = digi_max;

          // output is TPF order:
          uint64_t outidx = idat*nchan*npol + ipol*nchan + ichan;

          switch (nbit){
          case 1:
          case 2:
          case 4:
            bit_counter = outidx % samp_per_byte;

            if (bit_counter==0) 
              outptr[(uint64_t)(outidx/samp_per_byte)] = (unsigned char)0;
            outptr[(uint64_t)(outidx/samp_per_byte)] 
              += ((unsigned char) (result)) << (bit_counter*nbit);
            
            break;
          case 8:
            outptr[outidx] = (unsigned char) result;
            break;
          case 16:
            ((uint16_t*)outptr)[outidx] = (uint16_t) result;
            break;
          }

        } // time
      } // poln
    } // chan
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

  // number of polarizations
  const unsigned npol = input->get_npol();

  // scale factor
  const float scale = input->get_scale();

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
      {
        const unsigned inchan = channel(ichan);
        for (unsigned ipol=0; ipol<npol; ipol++)
          outptr[nchan*ipol+ichan] = inptr[inchan*npol+ipol]/scale;
      }

      inptr += nchan*npol;
      outptr += nchan*npol;
    }
    return;
  }
  case TimeSeries::OrderFPT:
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float* inptr = input->get_datptr( channel(ichan), ipol );

        for (uint64_t idat=0; idat < ndat; idat++)
          outptr[idat*nchan*npol + ipol*nchan + ichan] = inptr[idat]/scale;
      }
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

