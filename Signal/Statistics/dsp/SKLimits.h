//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SKLimits_h
#define __SKLimits_h

#include "dsp/PearsonIV.h"
#include "dsp/NewtonRaphson.h"

namespace dsp {

  class SKLimits {

  public:

    SKLimits ( unsigned _M, unsigned _std_devs );

    ~SKLimits ();

    int calc_limits ();

    double get_lower_threshold() { return lower_threshold; }

    double get_upper_threshold() { return upper_threshold; }

    void set_M ( unsigned _M) { M = _M; }

    void set_std_devs ( unsigned _std_devs ) { std_devs = _std_devs; }

  private:

    //! Calculate the first four moments of the distribution and ancilliary parameters
    void prepare();

    double lower_threshold;

    double upper_threshold;

    //! Number of integrations in SK Statisic
    unsigned M;

    //! Numer of standard deviations to form limits
    unsigned std_devs;

    unsigned verbose;
  };
}

#endif
