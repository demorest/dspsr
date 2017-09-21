//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/MOPSRUnpacker.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/MOPSRUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>
#include <string.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

#ifdef _DEBUG
#define CHECK_ERROR(x) check_error(x)
#define CHECK_ERROR_STREAM(x,y) check_error_stream(x,y)
#else
#define CHECK_ERROR(x)
#define CHECK_ERROR_STREAM(x,y)
#endif

#if HAVE_CUDA
void check_error (const char *);
void check_error_stream (const char *, cudaStream_t);
#endif

dsp::MOPSRUnpacker::MOPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::MOPSRUnpacker ctor" << endl;

  gpu_stream = undefined_stream;

  table = new BitTable (8, BitTable::TwosComplement);
  device_prepared = false;

  // complex input data, 1 polarisation
  set_ndig (2);
  set_nstate (256);

  input_order = NONE;
  input_resolution = 1;

  debugd = 0;
}

dsp::MOPSRUnpacker::~MOPSRUnpacker ()
{
}

/*! The quadrature components must be offset by one */
unsigned dsp::MOPSRUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! */
unsigned dsp::MOPSRUnpacker::get_output_ipol (unsigned idig) const
{
  return 0;
}

/*! */
unsigned dsp::MOPSRUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::MOPSRUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

//! Return true if the unpacker support the specified output order
bool dsp::MOPSRUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return ((order == TimeSeries::OrderFPT)  || (order == TimeSeries::OrderTFP));
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::MOPSRUnpacker::set_output_order (TimeSeries::Order order)
{
  if (verbose)
  {
    if (order == TimeSeries::OrderFPT)
      cerr << "dsp::MOPSRUnpacker::set_output_order (TimeSeries::OrderFPT)" << endl;
    if (order == TimeSeries::OrderTFP)
      cerr << "dsp::MOPSRUnpacker::set_output_order (TimeSeries::OrderTFP)" << endl;
  }
  output_order = order;
  output->set_order (order);
}

dsp::MOPSRUnpacker * dsp::MOPSRUnpacker::clone () const
{
  return new MOPSRUnpacker (*this);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::MOPSRUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::get_device_supported HAVE_CUDA" << endl;
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::MOPSRUnpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    gpu_stream = (void *) gpu_mem->get_stream();
    mopsr_unpack_prepare (gpu_mem->get_stream(), (float) table->get_scale());
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device gpu_stream=" << gpu_stream << endl;
#ifdef USE_TEXTURE_MEMORY
    CUDA::TextureMemory * texture_mem = new CUDA::TextureMemory (gpu_mem->get_stream());
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device using texture memory ptr=" << texture_mem << endl;
    texture_mem->set_format_signed(8, 8, 0, 0);

    cerr << "dsp::MOPSRUnpacker::set_device staging.set_memory (texture_mem)" << endl;
    staging.set_memory( texture_mem );
#else
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device: using gpu memory" << endl;
    staging.set_memory ( gpu_mem );
#endif
  }
  else
  {
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::set_device: using cpu memory" << endl;
    gpu_stream = undefined_stream;
  }
#else
  Unpacker::set_device (memory);
#endif
  device_prepared = true;
}


bool dsp::MOPSRUnpacker::matches (const Observation* observation)
{
  if (verbose)
  {
    if (observation->get_state() == Signal::Analytic)
      cerr << "dsp::MOPSRUnpacker::matches state=Analytic" << endl;
    else if (observation->get_state() == Signal::Intensity)
      cerr << "dsp::MOPSRUnpacker::matches state=Intensity" << endl;
    else 
      cerr << "dsp::MOPSRUnpacker::matches states=" << observation->get_state() << endl;
  }

  return observation->get_machine()== "MOPSR"
    && (observation->get_state() == Signal::Analytic || observation->get_state() == Signal::Intensity)
    && (observation->get_nbit() == 8 || observation->get_nbit() == 32)
    && (observation->get_ndim() == 2 || observation->get_ndim() == 1)
    && (observation->get_npol() == 1 || observation->get_npol() == 2);
}


void dsp::MOPSRUnpacker::match_resolution (const Input* input)
{
  input_resolution = input->get_resolution();
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::match_resolution input_resolution=" << input_resolution << endl;
}

/*! Validate whether the unpacker can handle the combination of input order 
    and output order */
void dsp::MOPSRUnpacker::validate_transformation ()
{
  // see if this unpacker already knows in order of the input data 
  if (input_order == NONE)
  {
    const Input * in = input->get_loader();
    const Observation * obs = in->get_info();
    const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
    if (info)
    {
      char buffer[8];
      if (info->custom_header_get ("ORDER", "%s", buffer) == 1)
      {
        if (strcmp(buffer, "TF") == 0)
        {
          input_order = TF;
        }
        else if (strcmp(buffer, "FT") == 0)
        {
          input_order = FT;
        }
        else if (strcmp(buffer, "T") == 0)
        {
          cerr << "input order=T" << endl;
          input_order = T;
        }
        else
        {
          throw Error (InvalidState, "dsp::MOPSRUnpacker::valid_transformation", "unrecognized input order [%s]", buffer);
        }
      }
    }
    // have an assumed order when it cannot be determined
    else
    {
      cerr << "dsp::MOPSRUnpacker::valid_transformation could not get ASCIIObservation reference" << endl;
      input_order = TF;
    }
  }

  const unsigned nchan = input->get_nchan();
  if ((nchan == 1) && (input_order == TF))
    throw Error (InvalidState, "dsp::MOPSRUnpacker::valid_transformation", "input order not compatible with nchan=%u", nchan);
  if ((nchan != 1) && (input_order == T))
    throw Error (InvalidState, "dsp::MOPSRUnpacker::valid_transformation", "input order not compatible with nchan=%u", nchan);
}

void dsp::MOPSRUnpacker::unpack ()
{
  const unsigned int nbit = input->get_nbit();

  // 32-bit data does not have a digitizer
  if ((nbit == 32) && (get_ndig() != 0))
    set_ndig (0);

  // 8-bit data has a digitizer for each channel
  if ((nbit == 8) && (get_ndig() != input->get_nchan() * input->get_ndim()))
    set_ndig(input->get_nchan() * input->get_ndim());

#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  if (input_order == NONE)
    validate_transformation();

  const unsigned int nchan = input->get_nchan();
  const unsigned int ndim = input->get_ndim();
  const unsigned int npol = input->get_npol();
  const unsigned int ipol = 0;
  const uint64_t ndat = input->get_ndat();

  // input channel stride - distance between successive (temporal) samples from same channel
  unsigned int in_chan_stride = nchan * ndim;
  unsigned int out_chan_stride = ndim;
  const float* lookup = table->get_values ();

  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack in_chan_stride="<< in_chan_stride << " input_resolution=" << input_resolution << endl;

  if (debugd)
    cerr << "ndat=" << ndat << " nchan=" << nchan << " ndim=" << ndim << " nbit=" << nbit << endl;

  // TF order is produced by the beam-former, TFS produced by the AQ engines
  if (input_order == TF)
  {
    if (output->get_order() == TimeSeries::OrderFPT)
    {
      // 32-bit floats are produced by the beam former
      if (nbit == 32)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          const unsigned int in_chan_off =  ndim * ichan;
          const float * from = (float *) (input->get_rawptr()) + in_chan_off;
          float* into = output->get_datptr (ichan, ipol);
          for (uint64_t idat=0; idat < ndat; idat++)
          {
            for (unsigned idim=0; idim < ndim; idim++)
              into[idim] = from[idim];
            from += in_chan_stride;
            into += out_chan_stride;
          }
        } 
      }
      // 8-bit signed integers products by the PFBs
      else if (nbit == 8)
      {
        unsigned long * hists[2];

        // transpose from TF order to FT order
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          const unsigned int in_chan_off =  ndim * ichan;
          const int8_t * from = (int8_t *) (input->get_rawptr() + in_chan_off);
          float* into = output->get_datptr (ichan, ipol);
          const unsigned int nfloat = ndim;

          for (unsigned idim=0; idim < ndim; idim++)
            hists[idim] = get_histogram (ndim*ichan+idim);

          for (uint64_t idat=0; idat < ndat; idat++)
          {
            for (unsigned idim=0; idim < ndim; idim++)
            {
              into[idim] = float ( from[idim] );
              hists[idim][from[idim]+128]++;
            }
            from += in_chan_stride;
            into += out_chan_stride;
          }
        }
      }
      else
      {
        throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "unsupported unpacking bit width input=TF, output=FPT");
      }
    }
    else if (output->get_order() == TimeSeries::OrderTFP)
    {
      // 32-bit floats are produced by the beam former
      if (nbit == 32)
      {
        // direct unpack from TF to TF
        const float * from = (float *) input->get_rawptr();
        float * into = output->get_dattfp();
        const uint64_t nfloat = npol * nchan * ndat * ndim;
        for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        {
          into[ifloat] = from[ifloat];
        }
      }
      // 8-bit input are produced by the PFBs, Ndim == 2
      else if (nbit == 8 && ndim == 2)
      {
        // direct unpack from TF to TF
        const unsigned char* from = input->get_rawptr();
        float* into = output->get_dattfp();
        const uint64_t nfloat = npol * nchan * ndat;
        unsigned long* hist_re;
        unsigned long* hist_im;

        for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        {
          into[0] = lookup[ from[0] ];
          into[1] = lookup[ from[1] ];

          unsigned ichan = ifloat % nchan;

          hist_re = get_histogram (2*ichan);
          hist_im = get_histogram (2*ichan+1);

          int bin_re = int8_t(from[0]) + 128;
          int bin_im = int8_t(from[1]) + 128;

          hist_re[bin_re]++;
          hist_im[bin_im]++;

          into += 2;
          from += 2;
        }
      }
      else if (nbit == 8 && ndim == 1)
      {
        // direct unpack from TF to TF
        const unsigned char* from = input->get_rawptr();
        float* into = output->get_dattfp();
        const uint64_t nfloat = npol * nchan * ndat;
        unsigned long* hist = get_histogram (0);

        for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        {
          into[ifloat] = lookup[ from[ifloat] ];
        }
      }
      else
      {
        throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "unsupported unpacking bit width input=TF, output=TFP");
      }
    }
    else 
    {
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "output order not suitable for input == TS");
    }
  }
  else if (input_order == FT)
  {
    if (output->get_order() == TimeSeries::OrderFPT)
    {
      if (nbit == 32)
      {
        const unsigned nfloat = input_resolution * ndim;
        cerr << "dsp::MOPSRUnpacker::unpack ndat="<<ndat << " nfloat=" << nfloat << 
                " nchan=" << nchan << " output FPT input_resolution=" << input_resolution << endl;
        float * from = (float *) input->get_rawptr();
        unsigned nblock = ndat / input_resolution;
        cerr << "dsp::MOPSRUnpacker::unpack nblock=" << nblock << endl;
        if (ndat % input_resolution != 0)
          cerr << "input block size error" << endl;

        for (unsigned iblock=0; iblock<nblock; iblock++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            float* into = output->get_datptr (ichan, 0) + iblock * input_resolution;
            for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
              into[ifloat] = from[ifloat];
            from += nfloat;
          }
        }
      }
      else
      {
        const unsigned nval = ndat * ndim;
        int8_t * from = (int8_t *) input->get_rawptr();
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          unsigned long* hist_re = get_histogram (2*ichan);
          unsigned long* hist_im = get_histogram (2*ichan+1);
          float* into = output->get_datptr (ichan, 0);
          for (uint64_t ival=0; ival < nval; ival++)
            into[ival] = float (from[ival]);
          from += nval;
        }
      }
    }
    else if (output->get_order() == TimeSeries::OrderTFP)
    {
      if (nbit == 32)
      {
        // transpose from FT to TF
        const unsigned nchandim = nchan * ndim;
        float * from = (float *) input->get_rawptr();
        float * into = output->get_dattfp ();

        cerr << "dsp::MOPSRUnpacker::unpack ndat=" << ndat << " nchan=" << nchan << " ndim=" << ndim << " output TFP" << endl;

        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (uint64_t idat=0; idat < ndat; idat++)
          {
            for (unsigned idim=0; idim<ndim; idim++)
            {
              into[idat*nchandim + ichan*ndim + idim] = from[ndim*idat + idim];
            }
          }
          from += ndat * ndim;
        }
      }
    }
    else 
    {
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "output order not suitable for input == TS");
    }
  }
  // input data is a single time series of complex value samples
  else if (input_order == T)
  {
    if (nchan != 1)
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "input order == T, but nchan=%u", nchan);

    // simple unpack of single chan/ant time series to either FPT or TFP
    const int8_t * from8 = (int8_t *) input->get_rawptr();
    const float * from32 = (float *) input->get_rawptr();
    float * into;

    if (output->get_order() == TimeSeries::OrderFPT)
      into = output->get_datptr (0, 0);
    else if (output->get_order() == TimeSeries::OrderTFP)
      into = output->get_dattfp();
    else
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack", "output order not suitable for input == T");
     
    
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::unpack ndim=" << ndim << endl;
    unsigned long* hist_re = get_histogram (0);
    if (ndim > 1)
      unsigned long* hist_im = get_histogram (1);
    const uint64_t nfloat = ndat * ndim;

    if (nbit == 32)
      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        into[ifloat] = from32[ifloat];
    else
      for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
        into[ifloat] = lookup[ from8[ifloat] ];
  }
  debugd = 0;
}

unsigned dsp::MOPSRUnpacker::get_resolution () const { return 1024; }

#if HAVE_CUDA

void dsp::MOPSRUnpacker::unpack_on_gpu ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim  = input->get_ndim();
  const unsigned npol  = input->get_npol();

  const uint64_t to_copy = ndat * nchan * ndim * npol;

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  int8_t * d_staging = (int8_t *) staging.get_rawptr();
  const unsigned char * from = input->get_rawptr();
  float * into;

  if (ndat == 0)
  {
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::unpack_on_gpu ndat == 0" << endl;
    return;
  }

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
      into = output->get_datptr(0,0);
      break;
    }

    case TimeSeries::OrderTFP:
    {
      into = output->get_dattfp();
      break;
    }

    default:
    {
      throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack_on_gpu", "unrecognized order");
    }
    break;
  }

  cudaStream_t stream = (cudaStream_t) gpu_stream;
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack_on_gpu stream=" << stream << endl;

  cudaError error;
  if (stream)
  {
    error = cudaMemcpyAsync (d_staging, from, to_copy, cudaMemcpyHostToDevice, stream);
    CHECK_ERROR_STREAM ("dsp::MOPSRUnpacker::unpack_on_gpu cudaMemcpyAsync", stream);
  }
  else
  {
    error = cudaMemcpy (d_staging, from, to_copy, cudaMemcpyHostToDevice);
    CHECK_ERROR ("dsp::MOPSRUnpacker::unpack_on_gpu cudaMemcpy");
  }
  

#ifdef USE_TEXTURE_MEMORY
  if (verbose)
    cerr << "dsp::MOPSRUnpacker::unpack_on_gpu binding TextureMemory" << endl;
  CUDA::TextureMemory * gpu_mem = dynamic_cast< CUDA::TextureMemory*>( staging.get_memory() );
  cerr << "dsp::MOPSRUnpacker::unpack_on_gpu textureMemory stream=" << stream << " gpu_mem->get_tex()= " << gpu_mem->get_tex() << endl;
#endif

  if (error != cudaSuccess)
    throw Error (FailedCall, "MOPSRUnpacker::unpack_on_gpu",
                 "cudaMemcpy%s %s", stream?"Async":"", 
                 cudaGetErrorString (error));

#ifdef USE_TEXTURE_MEMORY
  mopsr_unpack (stream, ndat, d_staging, into, gpu_mem->get_tex());
#else
  if (output->get_order() == TimeSeries::OrderFPT)
  {
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_fpt ndat=" << ndat 
           << " d_staging=" << (void *) d_staging << " into=" << (void *) into << endl;
    mopsr_unpack_fpt (stream, ndat, nchan, table->get_scale(), d_staging, into);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      if (stream)
        CHECK_ERROR_STREAM ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_fpt", stream);
      else
        CHECK_ERROR ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_fpt");
  }
  else if (output->get_order() == TimeSeries::OrderTFP)
  {
    if (verbose)
      cerr << "dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_tfp ndat=" << ndat << endl;
    mopsr_unpack_tfp (stream, ndat, nchan, table->get_scale(), d_staging, into);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      if (stream)
        CHECK_ERROR_STREAM ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_tfp", stream);
      else
        CHECK_ERROR ("dsp::MOPSRUnpacker::unpack_on_gpu mopsr_unpack_tfp");
  }
  else
    throw Error (InvalidState, "dsp::MOPSRUnpacker::unpack_on_gpu", "unrecognized order");
#endif
}

#endif
