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

    enum ChangingVariable{ Channel, Polarisation};

    //! Null constructor- always outofplace
    TimeOrder();

    //! Destructor
    virtual ~TimeOrder();

    //! Set the most rapidly varying dimension for the BitSeries, out of channel and polarisation
    void set_rapid_variable( ChangingVariable _rapid){ rapid = _rapid; }

    //! Inquire the most rapidly varying dimension for the BitSeries, out of channel and polarisation
    ChangingVariable get_rapid_variable(){ return rapid; }

  protected:

    virtual void transformation ();

    //! The most rapidly changing variable
    ChangingVariable rapid;

  };

}

#endif
