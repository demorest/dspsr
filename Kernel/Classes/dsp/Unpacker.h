//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.6 $
   $Date: 2002/11/18 05:45:45 $
   $Author: cwest $ */


#ifndef __Unpacker_h
#define __Unpacker_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "Registry.h"

namespace dsp {

  //! Abstract base class of Transformations that convert n-bit to float
  /*! 

  */
  class Unpacker : public Transformation <const BitSeries, TimeSeries> {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker") 
      : Transformation <const BitSeries, TimeSeries> (name, outofplace) { }
    
    //! Return a pointer to a new instance of the appropriate sub-class
    static Unpacker* create (const Observation* observation);

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;


   protected:
    //! The operation unpacks n-bit into floating point TimeSeries
    virtual void transformation ();
    
    //! The unpacking routine
    virtual void unpack () = 0;

    //! Return true if Unpacker sub-class can convert the Observation
    virtual bool matches (const Observation* observation) = 0;

    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);

    //! List of registered sub-classes
    static Registry::List<Unpacker> registry;


  };

}

#endif // !defined(__Unpacker_h)
