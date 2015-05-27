/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// Cribbed from Willem's SigProcDigitizer

#include "dsp/FITSDigitizer.h"
#include <assert.h>

//! Default constructor
dsp::FITSDigitizer::FITSDigitizer (unsigned _nbit)
: Digitizer ("FITSDigitizer")
{
  set_nbit(_nbit);
}

//! Set the number of bits per sample
void dsp::FITSDigitizer::set_nbit (unsigned _nbit)
{
  switch(_nbit)
  {
  case 1:
  case 2:
  case 4:
  case 8:
    nbit = _nbit;
    break;
  default:
    throw Error (InvalidState, "dsp::FITSDigitizer::set_nbit",
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
  PSRFITS search mode data are in TPF order, whereas the native
  buffer class (DataSeries) is in FPT(d).
*/

void dsp::FITSDigitizer::pack ()
{
  if (input->get_ndim() != 1)
    throw Error (InvalidState, "dsp::FITSDigitizer::pack",
  		 "cannot handle ndim=%d", input->get_ndim());

  // ChannelSort will re-organize the frequency channels in the output
  output->set_bandwidth ( -fabs(input->get_bandwidth()) );
  output->set_swap ( false );
  output->set_nsub_swap ( 0 );
  output->set_input_sample ( input->get_input_sample() );

  const unsigned npol = input->get_npol();

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::FITSDigitizer::pack ndat="<<ndat << std::endl;

  // this always puts channels in "lower sideband" order
  // I reckon that's OK
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

  // due to bit packing, the only sane way to do these loops is 
  // with F in inner loop
  case TimeSeries::OrderTFP:
  {
#pragma omp parallel for
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      unsigned char* outptr = output->get_rawptr() + (idat*nchan*npol)/samp_per_byte;
      // TODO -- this needs to account for reserve
      const float* inptr = input->get_dattfp() + idat*nchan*npol;

      // The following line is important: this function increments
      // the pointer at the start of each byte. MJK2008.
      outptr--;

      int bit_counter = 0, bit_shift = 0;
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          unsigned inChan = channel (ichan);
          const float* indat = inptr + inChan*npol + ipol;

          int result = (int)(((*indat) * digi_scale) + digi_mean + 0.5 );

          // clip the result at the limits
          result = std::max(result,digi_min);
          result = std::min(result,digi_max);

          switch (nbit){
            case 1:
            case 2:
            case 4:
              bit_counter = ichan % (samp_per_byte);

              // NB -- this original "sigproc" implementation is such that
              // later samples are shifted to the more significant bits, 
              // backwards to the PSRFITS convention; so reverse it in the
              // bit shift below
              bit_shift = (samp_per_byte-bit_counter-1)*nbit;
            
              if (bit_counter == 0 ) {
                outptr++;
                (*outptr) = (unsigned char)0;
              }
              (*outptr) |= ((unsigned char) (result)) << bit_shift;

              break;
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
  case TimeSeries::OrderFPT:
  {
    unsigned char* outptr = output->get_rawptr();

    int bit_counter=0;
    unsigned inner_stride = nchan * npol;
    unsigned idx = 0, bit_shift = 0; // make gcc happy
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        uint64_t isamp = ipol*nchan + ichan;
        const float* inptr = input->get_datptr( channel (ichan) , ipol );

        for (uint64_t idat=0; idat < ndat; idat++,isamp+=inner_stride)
        {
          int result = int( ((*inptr) * digi_scale) + digi_mean +0.5 );
          inptr++;

          // clip the result at the limits
          result = std::max (std::min (result, digi_max), digi_min);

          switch (nbit) {
          case 1:
          case 2:
          case 4:

            bit_counter = isamp % (samp_per_byte);
            idx = unsigned(isamp / samp_per_byte);

            // NB -- this original "sigproc" implementation is such that
            // later samples are shifted to the more significant bits, 
            // backwards to the PSRFITS convention; so reverse it in the
            // bit shift below
            bit_shift = (samp_per_byte-bit_counter-1)*nbit;

            if (bit_counter==0) 
              outptr[idx]=(unsigned char)0;

            outptr[idx] += ((unsigned char) (result)) << bit_shift;
            
            break;
          case 8:
            outptr[isamp] = (unsigned char) result;
            break;
          }
        }
      }
    }
    return;
  }

  default:
    throw Error (InvalidState, "dsp::FITSDigitizer::operate",
     "Can only operate on data ordered FTP or PFT.");
  }
}


