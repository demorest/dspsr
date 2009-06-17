//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Chomper_h
#define __Chomper_h

#include "environ.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  class Chomper : public Transformation<TimeSeries,TimeSeries> {
  public:

    //! Default constructor- always inplace
    Chomper();

    //! Set the new ndat
    void set_new_ndat(uint64_t _new_ndat){ new_ndat = _new_ndat; use_new_ndat = true; }

    //! Inquire the new ndat
    uint64_t get_new_ndat(){ return new_ndat; }

    //! Set the new rounding factor
    void set_rounding(uint64_t _rounding){ rounding = _rounding; }

    //! Inquire the new rounding factor
    uint64_t get_rounding(){ return rounding; }

    //! Don't chomp off the timeseries to new_ndat
    void dont_use_new_ndat(){ use_new_ndat = false; }

    //! Set multiplying facter
    void set_multiplier(float _multiplier){ multiplier = _multiplier; dont_multiply = false; }

    //! Get multiplying facter
    float get_multiplier(){ return multiplier; }

    //! Don't multiply by a previously set multiplying factor
    void dont_use_multiplier(){ dont_multiply = true; }

  protected:

    //! Do stuff
    void transformation();

  private:

    //! The new ndat for the TimeSeries
    uint64_t new_ndat;

    //! After setting the ndat, it is rounded off to divide this number
    uint64_t rounding;

    //! Gets set to true on set_new_ndat
    bool use_new_ndat;

    //! Chomper can also multiply
    float multiplier;

    //! Whether or not to use 'multiplier'
    bool dont_multiply;

  };

}

#endif
