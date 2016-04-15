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
}

dsp::MeerKATUnpacker::~MeerKATUnpacker ()
{
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
    && observation->get_npol() == 2
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

  int8_t * from = (int8_t *) input->get_rawptr();
  float * into;
  const float scale = table->get_scale();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = 2;
  const unsigned ndim  = 2;
  const float* lookup = table->get_values ();

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = ndat * ndim;

  if (verbose)
    cerr << "dsp::MeerKATUnpacker::unpack ndat=" << ndat << " nchan=" << nchan 
         << " nval=" << nval << endl;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      into = output->get_datptr (ichan, ipol); 
      for (unsigned ival=0; ival<nval; ival++)
      {
        //into[ival] = (float) from[ival] + 0.5;
        into[ival] = (float) from[ival] * scale;
      }
      from += nval;
    }
  }
}
