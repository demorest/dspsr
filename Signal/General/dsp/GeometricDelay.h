//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten et al
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/GeometricDelay.h

#ifndef __Geometric_SampleDelay_h
#define __Geometric_SampleDelay_h

#include "dsp/SampleDelayFunction.h"
#include "dsp/Response.h"

namespace dsp {

  class GeometricDelay : public SampleDelayFunction {
    
  public:
    
    //! Default constructor
    GeometricDelay ();
    
    //! Set up the dispersion delays
    bool match (const Observation* obs);
    
    //! Return the dispersion delay for the given frequency channel
    int64_t get_delay (unsigned ichan, unsigned ipol);
   
    //! Return the fractional delay + fringe response function
    Response* get_response () { return &response; }

    bool get_absolute () const { return true; }

  protected:

    // HERE I would add any additional parameters, such as telescope position

    // the delays applied to the two polarizations
    int64_t delay[2];

    // fractional delay + fringe response function
    Response response;
  };

}

#endif
