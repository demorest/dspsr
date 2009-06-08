//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FourthMoment.h,v $
   $Revision: 1.2 $
   $Date: 2009/06/08 19:45:01 $
   $Author: straten $ */


#ifndef __FourthMoment_h
#define __FourthMoment_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Forms the fourth-order moments of the electric field
  class FourthMoment : public Transformation <TimeSeries, TimeSeries> {

  public:
    
    //! Constructor
    FourthMoment ();
    
    //! Prepare the output TimeSeries attributes
    void prepare ();

  protected:

    //! Detect the input data
    virtual void transformation ();

  };

}

#endif // !defined(__FourthMoment_h)
