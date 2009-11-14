//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Memory_h_
#define __dsp_Memory_h_

namespace dsp {

  //! Manages memory allocation and destruction
  class Memory
  {
  protected:
    virtual void* do_allocate (unsigned nbytes);
    virtual void  do_free (void*);
    static Memory* manager;

  public:
    static void* allocate (unsigned nbytes);
    static void free (void*);
    static void set_manager (Memory*);
  };

}

#endif
