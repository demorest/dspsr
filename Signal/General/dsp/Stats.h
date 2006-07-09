//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Stats_h_
#define __dsp_Stats_h_

#include <vector>

#include "dsp/TimeSeries.h"
#include "dsp/Sink.h"

namespace dsp {

  class Stats : public Sink {
  public:

    //! Stores the last max value
    static float lastmax;

    //! Null constructor
    Stats(const char* name = "Stats");

    //! Virtual destructor
    virtual ~Stats();
    
    //! Returns mean in given chan,pol
    float get_mean(unsigned ichan, unsigned ipol);
    
    //! Returns standard deviation in given chan,pol
    float get_sigma(unsigned ichan, unsigned ipol);

    //! Returns true if a call to operate() has been made and 'mean' and 'sigma' are meaningful
    bool has_stats(){ return stats_calculated; }

    //! Over-ride set_input to check that input is MiniSeries
    void set_input(Observation* _input);

    //! Over-ride get_input to return a TimeSeries
    TimeSeries* get_input(){ return dynamic_cast<TimeSeries*>(input.get()); }

    //! Calculates the maximum sum of any two bins in the given window, and returns the index of the first bin
    unsigned twobinmaxbin(unsigned ichan, unsigned ipol, unsigned samp_start, unsigned nsamps, float& mmax = lastmax);

    //! Returns the maximum bin
    unsigned onebinmaxbin(unsigned ichan, unsigned ipol, unsigned samp_start, unsigned nsamps, float& mmax = lastmax);

    //! Calculates the sum of n bins in the window, and returns the index of the first bin
    unsigned nbinmaxbin(unsigned ichan, unsigned ipol, unsigned samp_start, unsigned nsamps,
			unsigned bins_to_sum, float& mmax = lastmax);

  protected:

    //! Calculates mean and standard deviation of data
    virtual void operation ();

    //! Stores the mean of each ichan,ipol once calculated
    vector<vector<float> > mmeans;

    //! Stores the standard deviation of each ichan,ipol once calculated
    vector<vector<float> > ssigmas;
    
    //! Set to true once operate() is called [false]
    bool stats_calculated;
    
    //! Makes sure 'working_space' is big enough
    void get_working_space(unsigned nfloats);

    //! Frees up working space
    void free_working_space();
    
  private:
    
    //! Working space
    float* working_space;
    unsigned working_space_size;

  };

}

#endif

