//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/MeerKATUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/MeerKATUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <string.h>
#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::MeerKATUnpacker::MeerKATUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::MeerKATUnpacker ctor" << endl;

  set_nstate (256);

  table = new BitTable (8, BitTable::TwosComplement);

  device_prepared = false;
  
  engine = 0;

  tfp_buffer = 0;
  tfp_buffer_size = 0;

}

dsp::MeerKATUnpacker::~MeerKATUnpacker ()
{
}

//! Return true if the unpacker support the specified output order
bool dsp::MeerKATUnpacker::get_order_supported (TimeSeries::Order order) const
{
  //return ((order == TimeSeries::OrderFPT) || (order == TimeSeries::OrderTFP));
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::MeerKATUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}


/*! The quadrature components are offset by one */
unsigned dsp::MeerKATUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::MeerKATUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

/*! Each chan has 4 values (quadrature, dual pol) */
unsigned dsp::MeerKATUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

dsp::MeerKATUnpacker * dsp::MeerKATUnpacker::clone () const
{
  return new MeerKATUnpacker (*this);
}

void dsp::MeerKATUnpacker::set_engine (Engine* _engine)
{
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::MeerKATUnpacker::get_device_supported (Memory* memory) const
{
  // create a temporary engine in the default stream
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    CUDA::MeerKATUnpackerEngine * tmp = new CUDA::MeerKATUnpackerEngine(0);
    return tmp->get_device_supported (memory);
  }
  else
#endif
  {
    return false;
  }
}

//! Set the device on which the unpacker will operate
void dsp::MeerKATUnpacker::set_device (Memory* memory)
{
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    set_engine (new CUDA::MeerKATUnpackerEngine(stream));
  }
#endif

  if (verbose)
    cerr << "dsp::MeerKATUnpacker::set_device" << endl;
  if (engine)
  {
    engine->set_device(memory);
    engine->setup();
  }
  else
    Unpacker::set_device (memory);
  device_prepared = true;
}

bool dsp::MeerKATUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "MeerKAT"
    && observation->get_ndim() == 2
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && observation->get_nbit() == 8;
}

void dsp::MeerKATUnpacker::unpack ()
{
  const uint64_t ndat  = input->get_ndat();
  if (ndat == 0)
    return;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::MeerKATUnpacker::unpack using Engine" << endl;
    engine->unpack(table->get_scale(), input, output);
    return;
  }

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  int16_t * from = (int16_t *) input->get_rawptr();
  int16_t from16;
  int8_t * from8 = (int8_t * ) &from16;
  float * into;
  const float scale = table->get_scale();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = 2;
  const unsigned nsamp_per_heap = 256;
  const unsigned nheap = ndat / nsamp_per_heap;
  const float* lookup = table->get_values ();

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = nsamp_per_heap * ndim;

  if (verbose)
    cerr << "dsp::MeerKATUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan 
         << " npol=" << npol << " nval=" << nval << endl;

  unsigned long * digs[2];

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
#ifdef _DEBUG
      cerr << "dsp::MeerKATUnpacker::unpack TimeSeries::OrderFPT" << endl;
#endif
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            unsigned idig = ichan*ndim*npol + ipol*ndim;
            digs[0] = get_histogram (idig+0);
            digs[1] = get_histogram (idig+1);
            into = output->get_datptr (ichan, ipol) + iheap*nsamp_per_heap * ndim; 

            for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
            {
              from16 = from[isamp];
              digs[0][(int) from8[0] + 128]++;
              digs[1][(int) from8[1] + 128]++;
              into[2*isamp+0] = (float(from8[0]) + 0.5) * scale;
              into[2*isamp+1] = (float(from8[1]) + 0.5) * scale;
            }
            from += nsamp_per_heap;
          }
        }
      }
    }
    break;
    case TimeSeries::OrderTFP:
    {
#ifdef _DEBUG
      cerr << "dsp::MeerKATUnpacker::unpack TimeSeries::OrderTFP" << endl;
#endif
      into = output->get_dattfp();
      const unsigned heap_stride = nchan * npol * ndim * nsamp_per_heap;
      const unsigned into_stride = nchan * npol * ndim;
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        // memcpy a heap into a local buffer
        memcpy (tfp_buffer, from, tfp_buffer_size);

        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          for (unsigned ichan=0; ichan<nchan; ichan++)
          {      
            unsigned idig = ichan*ndim*npol + ipol*ndim;
            //digs[0] = get_histogram (idig+0);
            //digs[1] = get_histogram (idig+1);

            float * into_ptr = into + (ichan*npol*ndim) + (ipol*ndim);

            for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
            {
              from16 = from[isamp];
              //digs[0][(int) from8[0] + 127]++;
              //digs[1][(int) from8[1] + 127]++;
              into_ptr[0] = (float(from8[0]) + 0.5) * scale;
              into_ptr[1] = (float(from8[1]) + 0.5) * scale;

              into_ptr += into_stride;
            }
            from += nsamp_per_heap;;
          }
        }
        into += heap_stride;
      }
    }
    break;
    default:
      throw Error (InvalidState, "dsp::MeerKATUnpacker::unpack",
                   "unrecognized output order");
    break;

  }
}
