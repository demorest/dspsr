//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.1 $
   $Date: 2002/10/15 13:13:38 $
   $Author: pulsar $ */


#ifndef __Unpacker_h
#define __Unpacker_h

#include "Operation.h"
#include "Registry.h"

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
