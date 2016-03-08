/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// Cribbed from Willem's SigProcDigitizer

#include "dsp/FITSDigitizer.h"
#include "dsp/InputBuffering.h"
#include <assert.h>

void dsp::FITSDigitizer::set_digi_scales()
{
  // NB -- the scale here is defined s.t. roughly digi_sigma fit within
  // the re-digitized values, so a smaller value compresses less
  float digi_sigma = 8;
  switch (nbit){
    case 1:
      digi_mean=0.5;
      digi_scale=1;
      digi_min = 0;
      digi_max = 1;
      return;
    case 2:
      digi_mean=1.5;
      digi_scale=1;
      digi_min = 0;
      digi_max = 3;
      return;
    case 4:
      digi_mean=7.5;
      digi_scale= digi_mean / digi_sigma;
      digi_min = 0;
      digi_max = 15;
      return;
    case 8:
      digi_mean=127.5;
      digi_scale= digi_mean / digi_sigma;
      digi_min = 0;
      digi_max = 255;
      return;
  }
}

//! Default constructor
dsp::FITSDigitizer::FITSDigitizer (unsigned _nbit)
: Digitizer ("FITSDigitizer")
{
  set_nbit(_nbit);
  rescale_nsamp = 0;
  rescale_nblock = 1;
  rescale_idx = 0;
  rescale_counter = 0;
  rescale_constant = false;
  freq_total = freq_totalsq = scale = offset = NULL;
  digi_scale = 1;
  digi_mean = 0;
  digi_min = 0;
  digi_max = 0;
}

//! Default destructor
dsp::FITSDigitizer::~FITSDigitizer ()
{
  delete freq_total;
  delete freq_totalsq;
  delete scale;
  delete offset;
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

//! Set the rescaling interval in samples
void dsp::FITSDigitizer::set_rescale_samples (unsigned nsamp)
{
  rescale_nsamp = nsamp;
  if (!has_buffering_policy())
    set_buffering_policy( new InputBuffering (this) );
  get_buffering_policy()->set_minimum_samples (rescale_nsamp);
}

void dsp::FITSDigitizer::set_rescale_nblock (unsigned nblock)
{
  rescale_nblock = nblock;
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();
  delete freq_totalsq;
  delete freq_total;
  freq_totalsq = new double[npol*nchan*nblock];
  freq_total = new double[npol*nchan*nblock];
}

void dsp::FITSDigitizer::set_rescale_constant (bool rconst)
{
  rescale_constant = rconst;
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

void dsp::FITSDigitizer::init ()
{
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();
  freq_totalsq = new double[npol*nchan*rescale_nblock];
  freq_total = new double[npol*nchan*rescale_nblock];
  scale = new double[npol*nchan];
  offset = new double[npol*nchan];
}

/*
//! Override parent implementation if doing rescaling
void dsp::FITSDigitizer::transformation ()
{
  if (rescale_nsamp > 0)
  {
    if (!scale)
      init ();
    if (input->get_ndat() < rescale_nsamp)
    {
      if (verbose)
        cerr << "dsp::FITSDigitizer::transformation waiting for additional samples" << std::endl;
      get_buffering_policy()->set_next_start ( 0 );
      output->set_ndat (0);
      return;
    }
  }
  Digitizer::transformation ();
}
*/

void dsp::FITSDigitizer::measure_scale ()
{

  if (rescale_constant && (rescale_counter > 0)) return;

  unsigned input_nchan = input->get_nchan ();
  unsigned input_npol = input->get_npol ();
  unsigned stride = input_nchan * input_npol;

  // index into current row of storage
  if (rescale_idx == rescale_nblock) rescale_idx = 0;

  double* ft = freq_total + rescale_idx*stride;
  double* fts = freq_totalsq + rescale_idx*stride;

  // zero storage
  for (unsigned i = 0; i < stride; ++i)
  {
    ft[i] = 0;
    fts[i] = 0;
  }

  // accumulate values from data
  switch (input->get_order()) {
  case TimeSeries::OrderTFP:
  {
    const float* in_data = input->get_dattfp();
    for (unsigned idat=0; idat < rescale_nsamp; idat++)
    {
      for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
        for (unsigned ipol=0; ipol < input_npol; ipol++)
        {
          unsigned idx = ipol*input_nchan + ichan;
          ft[idx] += *in_data;
          fts[idx] += (*in_data) * (*in_data);

          in_data++;

        }
      }
    }
    break;
  }
  case TimeSeries::OrderFPT:
  {
    unsigned idx = 0;
    for (unsigned ipol=0; ipol < input_npol; ipol++) 
    {
      for (unsigned ichan=0; ichan < input_nchan; ichan++)
      {
        const float* in_data = input->get_datptr (ichan, ipol);

        double sum = 0.0;
        double sumsq = 0.0;

        for (unsigned idat=0; idat < rescale_nsamp; idat++)
        {
          sum += in_data[idat];
          sumsq += in_data[idat] * in_data[idat];
        }

        ft[idx] += sum;
        fts[idx] += sumsq;
        idx++;
      }
    }
    break;
  }
  default:
    throw Error (InvalidState, "dsp::Rescale::operate",
        "Requires data in TFP or FPT order");
  }

  // if no memory, set scales directly
  if (rescale_nblock==1)
  {
    rescale_counter = 1; // use this to keep track of first entry
    double recip = 1./rescale_nsamp;
    for (unsigned i = 0; i < stride; ++i)
    {
      offset[i] = ft[i]*recip;
      scale[i] = sqrt (fts[i]*recip - offset[i]*offset[i]);
    }
    return;
  }

  // if we haven't fully populated the memory yet, it's easiest just to
  // average the stored values directly; we can use the last rows as cache
  if (rescale_counter < rescale_nblock)
  {
    // increment here -- should be one on first iteration
    rescale_counter += 1;

    // zero out the scale/offset
    for (unsigned i = 0; i < stride; ++i)
    {
      offset[i] = 0;
      scale[i] = 0;
    }

    // now add up the rows so far
    double *ft_copy = freq_total;
    double *fts_copy = freq_totalsq;
    for (unsigned i = 0; i < rescale_counter; ++i)
    {
      for (unsigned j = 0; j < stride; ++j)
      {
        offset[j] += ft_copy[j];
        scale[j] += fts_copy[j];
      }
      ft_copy += stride;
      fts_copy += stride;
    }

    // finally, convert moments to mean/standard deviation
    double recip = (1./rescale_nsamp)*(1./rescale_counter);
    for (unsigned i = 0; i < stride; ++i)
    {
      offset[i] = offset[i]*recip;
      scale[i] = sqrt(scale[i]*recip - offset[i]*offset[i]);
    }

    rescale_idx += 1;
    return;
  }

  // finally, if the memory is fully populated, we can update the values
  // more efficiently by adding/subtracting the contributions from the
  // newest/oldest values, respectively
  unsigned old_idx = rescale_idx + 1;
  if (old_idx == rescale_nblock) old_idx = 0;
  double recip = (1./rescale_nsamp)*(1./rescale_nblock);
  double* old_ft = freq_total + old_idx*stride;
  double* old_fts = freq_totalsq + old_idx*stride;
  for (unsigned i = 0; i < stride; ++i)
  {
    double old_offset = offset[i];
    offset[i] = offset[i] + (ft[i]-old_ft[i])*recip;
    scale[i] = sqrt( scale[i]*scale[i] - offset[i]*offset[i] + old_offset*old_offset + (fts[i]-old_fts[i])*recip );
  }
  rescale_idx += 1;
  
}

/*! 
  PSRFITS search mode data are in TPF order, whereas the native
  buffer class (DataSeries) is in FPT(d).
*/

void dsp::FITSDigitizer::pack ()
{
  if (input->get_ndat() == 0)
    return;

  if (input->get_ndim() != 1)
    throw Error (InvalidState, "dsp::FITSDigitizer::pack",
  		 "cannot handle ndim=%d", input->get_ndim());

  if (rescale_nsamp > 0)
  {
    rescale_pack ();
    return;
  }

  // ChannelSort will re-organize the frequency channels in the output
  output->set_bandwidth ( -fabs(input->get_bandwidth()) );
  output->set_swap ( false );
  output->set_nsub_swap ( 0 );
  output->set_input_sample ( input->get_input_sample() );

  const unsigned npol = input->get_npol();

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  uint64_t ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::FITSDigitizer::pack ndat="<<ndat << std::endl;

  // this always puts channels in "lower sideband" order
  // I reckon that's OK
  ChannelSort channel (input);

  int samp_per_byte = 8/nbit;
  set_digi_scales();

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

void dsp::FITSDigitizer::rescale_pack ()
{
  if (!scale)
    init ();

  if (input->get_ndat() < rescale_nsamp)
  {
    if (verbose)
      cerr << "dsp::FITSDigitizer::transformation waiting for additional samples" << std::endl;
    get_buffering_policy()->set_next_start ( 0 );
    output->set_ndat (0);
    return;
  }

  measure_scale ();

  // ChannelSort will re-organize the frequency channels in the output
  output->set_bandwidth ( -fabs(input->get_bandwidth()) );
  output->set_swap ( false );
  output->set_nsub_swap ( 0 );
  output->set_input_sample ( input->get_input_sample() );

  const unsigned npol = input->get_npol();

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = rescale_nsamp;
  output->set_ndat (ndat);

  if (verbose)
    cerr << "dsp::FITSDigitizer::rescale_pack ndat="<<ndat << std::endl;

  // this always puts channels in "lower sideband" order
  // I reckon that's OK
  ChannelSort channel (input);

  int samp_per_byte = 8/nbit;
  set_digi_scales();

  /*
    TFP mode
  */

  switch (input->get_order())
  {

  // due to bit packing, the only sane way to do these loops is 
  // with F in inner loop
  case TimeSeries::OrderTFP:
  {
//#pragma omp parallel for
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
          unsigned scale_idx = inChan + ipol*nchan;
          const float* indat = inptr + inChan*npol + ipol;
          double dat = (*indat-offset[scale_idx])/scale[scale_idx];

          //int result = (int)(((*indat) * digi_scale) + digi_mean + 0.5 );
          int result = (int)((dat * digi_scale) + digi_mean + 0.5 );

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
    break;
  }
  case TimeSeries::OrderFPT:
  {
    unsigned char* outptr = output->get_rawptr();

    int bit_counter=0;
    unsigned inner_stride = nchan * npol;
    unsigned idx = 0, bit_shift = 0; // make gcc happy
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      unsigned mapped_chan = channel (ichan);
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        uint64_t isamp = ipol*nchan + ichan;
        const float* inptr = input->get_datptr( mapped_chan , ipol );
        //float m_scale = digi_scale/scale[ipol*nchan + mapped_chan];
        //float m_offset = digi_mean - offset[ipol*nchan + mapped_chan]*m_scale;
        float m_scale = 1./scale[ipol*nchan + mapped_chan];
        float m_offset = offset[ipol*nchan + mapped_chan];

        for (uint64_t idat=0; idat < ndat; idat++,isamp+=inner_stride)
        {
          //int result = int( ((*inptr)*m_scale + m_offset) + 0.5 );
          float tmp = ((*inptr) - m_offset)*m_scale;
          int result = int( tmp*digi_scale + digi_mean + 0.5 );
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
    break;
  }

  default:
    throw Error (InvalidState, "dsp::FITSDigitizer::operate",
     "Can only operate on data ordered FTP or PFT.");
  }

  get_buffering_policy () -> set_next_start (rescale_nsamp);

  update (this);

}

void dsp::FITSDigitizer::get_scales(
  std::vector<float>* dat_scl, std::vector<float>* dat_offs)
{
  unsigned nchan = input->get_nchan ();
  unsigned npol = input->get_npol ();
  dat_scl->resize(nchan*npol);
  dat_offs->resize(nchan*npol);
  ChannelSort channel (input);

  for (unsigned ichan=0; ichan < nchan; ++ichan)
  {
    unsigned chan_idx = channel (ichan) ;
    for (unsigned ipol=0; ipol < npol; ++ipol)
    {
      unsigned offs = ipol*nchan;
      // NB -- it is critical to remove the digitizer scale here, in order
      // to preserve the correct mean/variance relationship when the data
      // are decoded.
      (*dat_scl)[offs + ichan] = scale[offs + chan_idx]/digi_scale;
      // On the other hand, the digitizer offset can be corrected
      // based on the ZERO_OFFS encoded in the FITS file.
      (*dat_offs)[offs + ichan] = offset[offs + chan_idx];
    }
  }
}
