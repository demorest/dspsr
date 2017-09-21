//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/on_host.h

#ifndef __on_host_h
#define __on_host_h

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/Memory.h"

#if HAVE_CUDA
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif


template<class Container>
void on_host (const Container* input, Reference::To<const Container>& output,
	      bool clone = false)
{
  if (input->get_memory()->on_host())
  {
    if (clone)
    {
      if (!output)
        output = input->clone();
      else
        *const_cast<Container*>(output.ptr()) = *input;
    }
    else
      output = input;

    return;
  }

#if HAVE_CUDA
  CUDA::DeviceMemory * device_memory = dynamic_cast<CUDA::DeviceMemory*>( 
                          const_cast<dsp::Memory *>(input->get_memory()) );
  if (device_memory)
  {
    dsp::TransferCUDA transfer ( device_memory->get_stream() );
    transfer.set_kind( cudaMemcpyDeviceToHost );
    transfer.set_input( input );

    if (!output)
      output = new Container;
    transfer.set_output( const_cast<Container*>(output.ptr()) );
    transfer.operate ();

    return;
  }
#endif

  throw Error (InvalidState, "on_host", "unknown memory manager");
}

template<class Container>
void on_host (const Container* input, Reference::To<Container>& output,
	      bool clone = false)
{
  Reference::To<const Container> const_output;
  if (output)
    const_output = output;

  on_host (input, const_output, clone);

  output = const_cast<Container*>( const_output.ptr() );
}

template<class Container>
Container* on_host (const Container* container, bool clone = false)
{
  Reference::To<const Container> host;
  on_host (container, host, clone);
  return const_cast<Container*>( host.release() );
}

#endif
