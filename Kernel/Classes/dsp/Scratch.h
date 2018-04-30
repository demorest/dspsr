//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Scratch.h

#ifndef __dsp_Scratch_h
#define __dsp_Scratch_h

#include "OwnStream.h"
#include "Memory.h"

namespace dsp
{
  //! Scratch space that can be shared between Operations
  /*! This simple class manages a block of memory that can be used
    as a temporary scratch spaceshared by multiple Operations */
  class Scratch : public OwnStream
  {

  public:

    //! Default constructor
    Scratch ();

    //! Destructor
    ~Scratch ();

    //! Return typed pointer to shared memory resource
    template<typename T>
    T* space (size_t ndat)
    { return reinterpret_cast<T*>( space (ndat * sizeof(T)) ); }
    
    //! Return pointer to shared memory resource
    void* space (size_t nbytes);

    //! Default scratch space
    static Scratch* get_default_scratch();

    //! Set the memory manager
    virtual void set_memory (Memory*);

  protected:
    
    //! The memory manager
    Reference::To<Memory> memory;

    char* working_space;
    size_t working_size;

  };
  
}

#endif
