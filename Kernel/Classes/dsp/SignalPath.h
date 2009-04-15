//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_SignalPath_h_
#define __dsp_SignalPath_h_

#include "dsp/dspExtension.h"

namespace dsp {

  class Operation;

  //! Stores information about the signal path
  class SignalPath : public dspExtension {
	
  public:

    typedef std::vector< Reference::To<Operation> > List;

    //! Default constructor
    SignalPath (List* list = 0);

    //! Clone operator
    dspExtension* clone () const;

    //! Combine information from another signal path
    void combine (const SignalPath*);

    //! Reset all of the components in the signal path
    void reset ();

    //! Set the list of operations
    void set_list (List*);

    //! Get the list of operations
    List* get_list () const;

  protected:

    //! The operations in the signal path
    List* operations;

  };

}

#endif
