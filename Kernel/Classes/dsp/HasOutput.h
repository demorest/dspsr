//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/HasOutput.h,v $
   $Revision: 1.2 $
   $Date: 2010/04/25 04:56:20 $
   $Author: straten $ */

#ifndef __dsp_HasOutput_h
#define __dsp_HasOutput_h

#include "ReferenceTo.h"

namespace dsp
{
  template <class Out>
  class HasOutput
  {
  public:

    //! Destructor
    virtual ~HasOutput () {}

    //! Set the container into which output data will be written
    virtual void set_output (Out* _output) { output = _output; }

    //! Return pointer to the container into which output data will be written
    virtual Out* get_output () const { return output; }

    //! Returns true if output is set
    bool has_output() const { return output.ptr(); }

  protected:

    //! Container into which output data will be written
    Reference::To<Out> output;
  };
}

#endif
