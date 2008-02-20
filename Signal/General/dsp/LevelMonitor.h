//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/LevelMonitor.h,v $
   $Revision: 1.1 $
   $Date: 2008/02/20 06:31:44 $
   $Author: straten $ */

#ifndef __LevelMonitor_h
#define __LevelMonitor_h

#include "Reference.h"
#include "environ.h"

#include <vector>

namespace dsp {

  //! Base class for history of digitized data statistics
  class LevelHistory;
  class TimeSeries;
  class IOManager;
  class HistUnpacker;

  //! Base class for monitoring a digitizer
  class LevelMonitor : public Reference::Able
  {
    
  public:

    //! Verbosity flag
    static bool verbose;
  
    //! Actually connect
    static bool connect;

    //! Number of iterations 0 for infinite
    int iterations;
    
    //! Constructor
    LevelMonitor ();
    
    //! Destructor
    virtual ~LevelMonitor ();
    
    //! Set the number of points included in each calculation of thresholds
    void set_integration (uint64 npts);

    //! Using input and converter, calculate sampling thresholds
    virtual void monitor ();
    
    //! Abort monitoring
    virtual void monitor_abort ();
    
    //! Change the gain in the given channel
    virtual int change_gain (int channel, double delta_dBm);
    
    //! Change the level in the given channel
    virtual int change_levels (int channel, double delta_Volt);
    
    //! Accumulate statistics about incoming data stream
    virtual int accumulate_stats (std::vector<double>& mean, 
				  std::vector<double>& variance);
    
    //! Set the sampling thresholds based on mean and variance in each channel
    virtual int set_thresholds (std::vector<double>& mean, 
                                std::vector<double>& variance);

    //! If you are really fast, you might need to sleep a while
    virtual void rest_a_while () {}

    //! Set the number of iterations to perform
    void set_iterations (unsigned);

    //! Set the device to be used to plot/log the digitizer statistics
    void set_history (LevelHistory* history);
    
    //! Set the device to be used to plot/log the digitizer statistics
    void set_input (IOManager* input);
 
  protected:
    
    void init();
    
    //! the number of points to integrate
    unsigned long n_integrate;
    
    //! abort current integration
    bool abort;
    
    //! the optimal variance
    double optimal_variance;
    //! the optimal dBm
    double optimal_dBm;
    
    //! the amount by which the mean may be off zero
    double mean_tolerance;
    
    //! the amount by which the variance may be off optimal
    double var_tolerance;

    //! flag says we need to get on it
    bool far_from_good;

  private:

    //! the converted data
    Reference::To<TimeSeries> data;

    //! the source of data
    Reference::To<IOManager> input;
    
    //! something to log the statistics
    Reference::To<LevelHistory> history;

    //! an unpacker that keeps histograms
    Reference::To<HistUnpacker> unpacker;

    void connect_bits ();
    
  };

}

//! convenience function for conversion
double variance2dBm (double);

#endif
