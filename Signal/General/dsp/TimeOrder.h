//-*-C++-*-

/*

A TimeOrder converts a Timeseries into a Bitseries

*/

#ifndef __TimeOrder_h
#define __TimeOrder_h

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/Transformation.h"

namespace dsp {

  class TimeOrder : public Transformation<TimeSeries,BitSeries> {

  public:

    //! Null constructor- always outofplace
    TimeOrder();

    //! Destructor
    virtual ~TimeOrder();

  protected:

    virtual void transformation ();
  };

}

#endif
