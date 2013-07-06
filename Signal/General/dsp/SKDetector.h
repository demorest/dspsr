//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __SKDetector_h
#define __SKDetector_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;

  //! Apply SKFilterbank results to a weighted timeseries
  class SKDetector: public Transformation<TimeSeries,BitSeries> {

  public:

    //! Default constructor
    SKDetector ();

    //! Destructor
    ~SKDetector ();

    //! Set the RFI thresholds with the specified factor
    void set_thresholds (unsigned _M, unsigned _n_std_devs);

    //! Set the channel range to conduct detection
    void set_channel_range (unsigned start, unsigned end);

    //! Set various options for SKDetector
    void set_options (bool disable_fscr, bool disable_tscr, bool disable_ft);

    //! Set the tsrunched SKFilterbank input
    void set_input_tscr (TimeSeries * _input_tscr);

    //! The number of time samples used to calculate the SK statistic
    unsigned get_M () const { return M; }

    //! The excision threshold in number of standard deviations
    unsigned get_excision_threshold () const { return n_std_devs; }

    //! Total SK statistic for each poln/channel, post filtering
    void get_filtered_sum (std::vector<float>& sum) const
    {  sum = filtered_sum; }

    //! Hits on filtered average for each channel
    void get_filtered_hits (std::vector<uint64_t>& hits) const
    { hits = filtered_hits; }

    //! Total SK statistic for each poln/channel, before filtering
    void get_unfiltered_sum (std::vector<float>& sum) const
    { sum = unfiltered_sum; }

    //! Hits on unfiltered SK statistic, same for each channel
    uint64_t get_unfiltered_hits () const { return unfiltered_hits; }

    //! The arrays will be reset when count_zapped is next called
    void reset_count () { unfiltered_hits = 0; }

  protected:

    //! Reserve the required amount of output space required
    void reserve();

    //! Perform the transformation on the input time series
    void transformation ();

    void reset_mask ();

    void detect_tscr ();

    void detect_skfb ();

    void detect_fscr();

    void count_zapped ();

    //! Total SK statistic for each poln/channel, post filtering
    std::vector<float> filtered_sum;

    //! Hits on filtered average for each channel
    std::vector<uint64_t> filtered_hits;

    //! Total SK statistic for each poln/channel, before filtering
    std::vector<float> unfiltered_sum;

    //! Hits on unfiltered SK statistic, same for each channel
    uint64_t unfiltered_hits;

    //! Tsrunched SK statistic timeseries for the current block
    Reference::To<TimeSeries> input_tscr;
  
    //! Number of time samples integrated into tscr SK estimates
    unsigned tscr_M;

    float tscr_upper;
    float tscr_lower;

    //! The number of adjacent blocks to be used in SK estimator
    unsigned M;

    float one_sigma;

    unsigned n_std_devs;

    float upper_thresh;

    float lower_thresh;

    float mega_upper_thresh;

    float mega_lower_thresh;

    unsigned s_chan;

    unsigned e_chan;

    uint64_t ndat_zapped;

    uint64_t ndat_zapped_skfb;
    
    uint64_t ndat_zapped_mega;

    uint64_t ndat_zapped_fscr;

    uint64_t ndat_zapped_tscr;
    
    uint64_t ndat_total;

    bool disable_fscr;

    bool disable_tscr;

    bool disable_ft;

    unsigned debugd;

  private:

  };
  
}

#endif
