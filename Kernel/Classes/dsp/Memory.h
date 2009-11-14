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
  public:
    virtual void* allocate (unsigned nbytes);
    virtual void free (void*);
  };

}

#endif
