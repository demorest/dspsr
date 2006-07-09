//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SLDetect_h
#define __SLDetect_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Simply Square Law detects TimeSeries.  When working 

  class SLDetect : public Transformation <TimeSeries, TimeSeries> {

  public:

    SLDetect(Behaviour _type=anyplace);
       
  protected:
    //! The transformation loads the next block of data
    virtual void transformation ();
  };

}

#endif // !defined(__SLDetect_h)
