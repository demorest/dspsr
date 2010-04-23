//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/on_host.h,v $
   $Revision: 1.1 $
   $Date: 2010/04/23 05:37:58 $
   $Author: straten $ */

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
Container* on_host (Container* container, bool clone = false)
{
  if (container->get_memory()->on_host())
  {
    if (clone)
      return container->clone();
    else
      return container;
  }

#if HAVE_CUDA
  if ( dynamic_cast<const CUDA::DeviceMemory*>( container->get_memory() ) )
  {
    cerr << "dsp::Archiver::unload retrieving from GPU" << endl;
    TransferCUDA transfer;
    transfer.set_kind( cudaMemcpyDeviceToHost );
    transfer.set_input( container );

    Reference::To<Container> host = new Container;
    transfer.set_output( host );
    transfer.operate ();

    return host.release();
  }
#endif

  throw Error (InvalidState, "on_host", "unknown memory manager");
}

#endif
