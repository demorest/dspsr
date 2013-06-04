//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Accumulation.h,v $
   $Revision: 1.20 $
   $Date: 2010/06/01 09:12:18 $
   $Author: straten $ */


#ifndef __Accumulation_h
#define __Accumulation_h

class Accumulation;

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  /*! Accumulates the specified number of integrations of detected 
  */
  class Accumulation : public Transformation <TimeSeries, TimeSeries> {

  public:
    
    //! Constructor
    Accumulation (unsigned to_integrate);

    //! Reserve the required amount of output space required
    void reserve();

    //! Prepare the output TimeSeries attributes
    void prepare ();

    //! Resize output for this transformation
    void resize_output ();

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:

    //! Detect the input data
    virtual void transformation ();

    //! Dimension of the output data
    unsigned int tscrunch;

    //! Input stride
    unsigned int stride;
      
    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  };

  class Accumulation::Engine : public Reference::Able
  {
  public:
    virtual void integrate (const TimeSeries* in, TimeSeries* out,
                            unsigned tscrunch, unsigned stride) = 0;
  }; 
}

#endif // !defined(__Accumulation_h)
