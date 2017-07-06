//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Sink.h

#ifndef __dsp_Sink_h
#define __dsp_Sink_h

#include "dsp/Operation.h"
#include "dsp/HasInput.h"

namespace dsp {

  //! Defines the interface to operations that terminate on input
  template <class In>
  class Sink : public Operation, public HasInput<In>
  {

  public:

    //! All sub-classes must specify name and capacity for inplace operation
    Sink (const char* _name) : Operation (_name) { }

    //! Destructor
    ~Sink () { }

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const
    {
      Operation::set_cerr (os);
      if (this->input)
	this->input->set_cerr(os);
    }

  protected:

    //! Define the Operation pure virtual method
    virtual void operation ();

    //! Declare that sub-classes must define a transformation method
    virtual void calculation () = 0;
  };

}

//! Define the Operation pure virtual method
template <class In>
void dsp::Sink<In>::operation () try
{
#if 0
  if (buffering_policy)
    buffering_policy -> pre_transformation ();
#endif

  calculation ();

#if 0
  if (buffering_policy)
    buffering_policy -> post_transformation ();
#endif
}
catch (Error& error)
{
  throw error += "dsp::Sink[" + name + "]::operation";
}

#endif
