//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Example.h

#ifndef __Example_h
#define __Example_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Simple example of how to inherit the Transformation template base class
  /*! This Example class exists only to demonstrate basic Transformation
    template base class inheritance.  Note that the constructor must define
    the name and behaviour of the transformation.  As well, one pure virtual
    method, transformation, must be defined at least once in the inheritance
    tree.  This example transforms a TimeSeries into a TimeSeries, though
    any classes that fit the Tranformation template could be used.
  */
  class Example : public Transformation <TimeSeries, TimeSeries> {

  public:
    
    //! Constructor
    Example () : Transformation ("Example", inplace) { }
    
    //! Destructor
    ~Example ();
    
  protected:

    //! Define the Transformation template base class pure virtual method
    virtual void transformation ();

  };

}

#endif // !defined(__Example_h)
