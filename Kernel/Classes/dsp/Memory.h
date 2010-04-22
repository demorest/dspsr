//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Memory_h_
#define __dsp_Memory_h_

#include "Reference.h"

namespace dsp {

  //! Manages memory allocation and destruction
  class Memory : public Reference::Able
  {
  protected:
    static Memory* manager;

  public:
    static void* allocate (unsigned nbytes);
    static void free (void*);
    static void set_manager (Memory*);
    static Memory* get_manager ();

    virtual void* do_allocate (unsigned nbytes);
    virtual void  do_free (void*);
    virtual void  do_zero (void* ptr, unsigned nbytes);
    virtual void  do_copy (void* to, const void* from, size_t bytes);
  };

}

#endif
