//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/LevelMonitor.h,v $
   $Revision: 1.5 $
   $Date: 2008/05/09 03:47:39 $
   $Author: straten $ */

#ifndef __LevelMonitor_h
#define __LevelMonitor_h

#include "Reference.h"
#include "environ.h"

#include <vector>

namespace dsp {

  class LevelHistory;
  class TimeSeries;
  class IOManager;
  class HistUnpacker;

  //! Monitors digitizer levels and issues corrective commands
  class LevelMonitor : public Reference::Able
  {
    
  public:

    //! Verbosity flag
    static bool verbose;
  
    //! Actually connect
    static bool connect;

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
    virtual int change_gain (unsigned ichan, unsigned ipol, unsigned idim, double scale);
    
    //! Change the level in the given channel
    virtual int change_levels (unsigned ichan, unsigned ipol, unsigned idim, double offset);
    
    //! Accumulate statistics about incoming data stream
    virtual int accumulate_stats (std::vector<double>& mean, 
				  std::vector<double>& variance);
    
    //! Set the sampling thresholds based on mean and variance in each channel
    virtual int set_thresholds (std::vector<double>& mean, 
                                std::vector<double>& variance);

    //! Set the number of iterations to perform
    void set_max_iterations (unsigned);

    //! Set the device to be used to plot/log the digitizer statistics
    void set_history (LevelHistory* history);
    
    //! Set the device to be used to plot/log the digitizer statistics
    void set_input (IOManager* input);
 
    //! Swap the polarizations
    void set_swap_polarizations (bool swap);

    //! Read data consecutively (do not seek to end on each iteration)
    void set_consecutive (bool swap);

  protected:
    
    void init();
    
    //! the number of points to integrate
    uint64 n_integrate;
    
    //! the number of points to load in one iteration
    uint64 block_size;

    //! Number of iterations 0 for infinite
    unsigned max_iterations;
    
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

    //! Swap polarizations
    bool swap_polarizations;

    //! Read data consecutively
    bool consecutive;

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

    unsigned nchan, npol, ndim, ndig;
    
  };

}

//! convenience function for conversion
double variance2dBm (double);

#endif
