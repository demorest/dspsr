//-*-C++-*-

#ifndef __dsp_Stats_h_
#define __dsp_Stats_h_

#include <vector>

#include "dsp/TimeSeries.h"
#include "dsp/Sink.h"

namespace dsp {

  class Stats : public Sink<TimeSeries> {
  public:

    //! Null constructor
    Stats(const char* name = "Stats");

    //! Virtual destructor
    virtual ~Stats();
    
    //! Calculates the maximum sum of any two bins in the given window, and returns the index of the first bin
    unsigned twobinmaxbin(unsigned ichan, unsigned ipol, unsigned samp_start, unsigned nsamps);

    //! Returns mean in given chan,pol
    float get_mean(unsigned ichan, unsigned ipol);
    
    //! Returns standard deviation in given chan,pol
    float get_sigma(unsigned ichan, unsigned ipol);

    //! Returns true if a call to operate() has been made and 'mean' and 'sigma' are meaningful
    bool has_stats(){ return stats_calculated; }

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
