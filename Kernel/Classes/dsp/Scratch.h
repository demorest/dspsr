//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Scratch.h,v $
   $Revision: 1.2 $
   $Date: 2008/01/25 05:22:51 $
   $Author: straten $ */

#ifndef __dsp_Scratch_h
#define __dsp_Scratch_h

#include "Reference.h"

namespace dsp {
  
  //! Scratch space that can be shared between Operations
  /*! This simple class manages a block of memory that can be used
    as a temporary scratch spaceshared by multiple Operations */
  class Scratch : public Reference::Able {

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
    static Scratch default_scratch;

    //! Set verbosity ostream
    virtual void set_ostream (std::ostream& os) const;

  protected:

    char* working_space;
    size_t working_size;

    //! Stream to which verbose messages are sent
    mutable std::ostream cerr;

  };
  
}

#endif
