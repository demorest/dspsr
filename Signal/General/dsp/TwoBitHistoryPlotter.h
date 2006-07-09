/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __TwoBitHistoryPlotter_h
#define __TwoBitHistoryPlotter_h

#include <sys/time.h>
#include <vector>

#include "dsp/TwoBitHistory.h"
#include "dsp/TwoBitStatsPlotter.h"

namespace dsp {

  /*!
    Uses PGPLOT to plot the history of the digitizer statistics
  */
  class TwoBitHistoryPlotter : public TwoBitHistory {
    
  public:
    
    TwoBitHistoryPlotter () { ndig = 0; keep_minutes = 150; }
    
    virtual ~TwoBitHistoryPlotter() {}

    //! Reset the history
    void reset () { ndig = 0; }
    
    //! Log the statistics of the digitized data in some form
    virtual void log_stats (std::vector<double>& mean,
                            std::vector<double>& variance,
			    TwoBitCorrection* stats);
    
    // the number of minutes to keep in history
    float  keep_minutes;
    
  protected:
    
    //! Two-bit digitization histogram plotter
    TwoBitStatsPlotter stat;
    
    //! Time of the first call to log_stats
    timeval start;
    
    //! Data mean as a function of time and digitizer
    std::vector< std::vector<float> > means;
    //! Data variance as a function of time and digitizer
    std::vector< std::vector<float> > variances;
    //! Time of each index in minutes since start
    std::vector< float > times;
    
    //! Colours of symbol
    std::vector<int> colour;
    //! Shapes of symbols
    std::vector<int> symbol;
    
    //! number of digitizers
    unsigned ndig;

    //! this gets called on the first call
    void init_stats_log (TwoBitCorrection* converter);
    
  };

}

#endif
