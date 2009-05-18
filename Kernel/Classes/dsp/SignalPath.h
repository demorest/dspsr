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

    typedef std::vector< Reference::To<Operation, false> > List;

    //! Default constructor
    SignalPath (const List& list);

    //! Alternative constructor
    SignalPath (const std::vector< Reference::To<Operation> >&);

    //! Clone operator
    dspExtension* clone () const;

    //! Combine information from another signal path
    void combine (const SignalPath*);

    //! Reset all of the components in the signal path
    void reset ();

    //! Set the list of operations
    void set_list (const List&);

    //! Get the list of operations
    const List* get_list () const;

    //! Add a component to the list of operations
    void add (Operation*);

  protected:

    //! The operations in the signal path
    List list;

  };

}

#endif
