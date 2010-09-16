//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/HasInput.h,v $
   $Revision: 1.2 $
   $Date: 2010/09/16 17:45:34 $
   $Author: demorest $ */

#ifndef __dsp_HasInput_h
#define __dsp_HasInput_h

#include "ReferenceTo.h"

namespace dsp
{
  //! Attaches to Operations with input
  template <class In>
  class HasInput
  {
  public:

    //! Destructor
    virtual ~HasInput () {}

    //! Set the container from which input data will be read
    virtual void set_input (In* _input) { input = _input; }

    //! Return pointer to the container from which input data will be read
    const In* get_input () const { return input; }
 
    //! Returns true if input is set
    bool has_input() const { return input; }

  protected:

    //! Container from which input data will be read
    Reference::To<const In> input;
  };
}

#endif
