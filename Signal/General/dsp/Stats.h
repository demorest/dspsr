//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Stats_h_
#define __dsp_Stats_h_

#include "dsp/Sink.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  class Stats : public Sink<TimeSeries>
  {
  public:

    //! Null constructor
    Stats (const char* name = "Stats");
    
    //! Destructor
    ~Stats ();

    //! Resize arrays and set to zero
    void init ();

    //! Returns mean in given chan,pol
    float get_mean (unsigned ichan, unsigned ipol);
    
    //! Returns standard deviation in given chan,pol
    float get_sigma (unsigned ichan, unsigned ipol);

  protected:

    //! Adds to the totals
    void calculation ();

    //! The sum of each channel and polarization
    std::vector< std::vector<double> > sum;

    //! The sumsq of each channel and polarization
    std::vector< std::vector<double> > sumsq;

    //! The total number of samples in each of the above sums
    uint64_t total;

  };

}

#endif

