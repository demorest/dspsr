//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TScrunch_h
#define __TScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Decimates a TimeSeries in the time domain

  /*! Input data are buffered to handle block sizes that are not integer
      multiples of the decimation factor (number of samples)
  */

  class TScrunch : public Transformation <TimeSeries, TimeSeries>
  {

  public:

    TScrunch (Behaviour place=anyplace);
    
    void set_factor ( unsigned samples );
    unsigned get_factor () const;

    void set_time_resolution ( double microseconds );
    double get_time_resolution () const;
    
  protected:

    //! Prepare input buffer
    void prepare ();

    //! Perform decimation
    void transformation ();
    void tfp_tscrunch ();
    void fpt_tscrunch ();

    mutable unsigned factor;
    mutable double time_resolution;

    // If true, use the tres parameter, if false use the factor parameter
    mutable bool use_tres; 

    unsigned sfactor;
    uint64_t output_ndat;
  };

}

#endif // !defined(__TScrunch_h)
