//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.4 $
   $Date: 2002/11/08 01:23:00 $
   $Author: hknight $ */


#ifndef __Unpacker_h
#define __Unpacker_h

class Unpacker;

#include "dsp/Timeseries.h"
#include "dsp/Bitseries.h"
#include "dsp/Operation.h"

#include "Registry.h"
#include "Reference.h"

namespace dsp {

  class Observation;

  //! Abstract base class of Operations that convert n-bit to float
  /*! 

  */
  class Unpacker : public Operation {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker") : Operation (name, outofplace) { }
    
    //! Return a pointer to a new instance of the appropriate sub-class
    static Unpacker* create (const Observation* observation);

   protected:
    //! kludge of the century:
    Reference::To<const Bitseries> input;

    //! kludge of the century part 2
    Reference::To<Timeseries> output;

    //! check the input is dsp::Timeseries- called by set_input()
    virtual void check_input();

    //! check the output is dsp::Timeseries- called by set_output()
    virtual void check_output();

    //! The operation unpacks n-bit into floating point Timeseries
    virtual void operation ();
    
    //! The unpacking routine
    virtual void unpack () = 0;

    //! Return true if Unpacker sub-class can convert the Observation
    virtual bool matches (const Observation* observation) = 0;

    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);

    //! List of registered sub-classes
    static Registry::List<Unpacker> registry;

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;

  };

}

#endif // !defined(__Unpacker_h)
