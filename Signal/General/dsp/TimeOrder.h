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

    //! Set the ndat of the BitSeries.  This must be less than or equal to the TimeSeries' ndat.  negative values indicate just use the TimeSeries ndat
    void set_bitseries_ndat(int64 _bitseries_ndat){ bitseries_ndat = _bitseries_ndat; }

    //! Inquire the desired ndat of the BitSeries.  This must be less than or equal to the TimeSeries' ndat.  negative values indicate just use the TimeSeries ndat
    int64 get_bitseries_ndat(){ return bitseries_ndat; }
    
    //! Set the offset (in time samples) that BitSeries will start at
    void set_offset(uint64 _offset){ offset = _offset; }

    //! Get the offset (in time samples) that BitSeries will start at
    uint64 get_offset(){ return offset; }

  protected:

    virtual void transformation ();

    //! The most rapidly changing variable
    ChangingVariable rapid;

    //! Desired ndat of the bitseries (-ve for timeseries' ndat)
    int64 bitseries_ndat;

    //! How many samples into the timeseries we want to start the bitseries at
    uint64 offset;

  };

}

#endif
