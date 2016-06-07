//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2013 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __LaunchConfig_h
#define __LaunchConfig_h

#include <cuda_runtime.h>

namespace CUDA
{
  //! Base class of launch configuration helpers
  class LaunchConfig
  {
  protected:
    int device;
    struct cudaDeviceProp device_properties;

  public:
    //! notes that init has not yet been called by setting device = -1
    LaunchConfig () { device = -1; }

    //! gets the current device ID and calls cudaGetDeviceProperties
    void init ();

    size_t get_max_threads_per_block ();

    size_t get_max_shm ();
  };


  //! Simple one-dimensional launch configuration
  class LaunchConfig1D : public LaunchConfig
  {
    unsigned nblock;
    unsigned nthread;
    unsigned block_dim;

  public:

    LaunchConfig1D () { block_dim = 0; }

    //! Set the block dimension to be used
    /*! 
      default: block_dim == 0 and 
      element index = blockIdx.x*blockDim.x + threadIdx.x;
    */
    void set_block_dim (unsigned i) { block_dim = i; }

    //! Set the number of elements to be computed
    void set_nelement (unsigned n);

    //! Return the number of blocks into which jobs is divided
    unsigned get_nblock() { return nblock; }
    //! Return the number of threads per block
    unsigned get_nthread() { return nthread; }
  };
}

#endif
