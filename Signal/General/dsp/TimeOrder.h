//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

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
    void set_bitseries_ndat(int64_t _bitseries_ndat){ bitseries_ndat = _bitseries_ndat; }

    //! Inquire the desired ndat of the BitSeries.  This must be less than or equal to the TimeSeries' ndat.  negative values indicate just use the TimeSeries ndat
    int64_t get_bitseries_ndat(){ return bitseries_ndat; }
    
    //! Set the offset (in time samples) that BitSeries will start at
    void set_offset(uint64_t _offset){ offset = _offset; }

    //! Get the offset (in time samples) that BitSeries will start at
    uint64_t get_offset(){ return offset; }

  protected:

    virtual void transformation ();

    //! The most rapidly changing variable
    ChangingVariable rapid;

    //! Desired ndat of the bitseries (-ve for timeseries' ndat)
    int64_t bitseries_ndat;

    //! How many samples into the timeseries we want to start the bitseries at
    uint64_t offset;

  };

}

#endif
