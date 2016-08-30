//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/Memory.h"

#ifndef __SpectralKurtosis_h
#define __SpectralKurtosis_h

#define ZAP_ALL  0
#define ZAP_SKFB 1
#define ZAP_FSCR 2
#define ZAP_TSCR 3

namespace dsp {

  //! Perform Spectral Kurtosis on Input Timeseries, creating output Time Series
  /*! Output will be in time, frequency, polarization order */

  class SpectralKurtosis: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    SpectralKurtosis ();

    //! Destructor
    ~SpectralKurtosis ();

    bool get_order_supported (TimeSeries::Order order) const;

    void set_M (unsigned _M) { M = _M; }

    //! Set the RFI thresholds with the specified factor
    void set_thresholds (unsigned _M, unsigned _std_devs);

    //! Set the channel range to conduct detection
    void set_channel_range (unsigned start, unsigned end);

    //! Set various options for Specral Kurtosis
    void set_options (bool _disable_fscr, bool _disable_tscr, bool _disable_ft);

    void reserve ();

    void prepare ();

    void prepare_output ();

    //! The number of time samples used to calculate the SK statistic
    unsigned get_M () const { return M; }

    //! The excision threshold in number of standard deviations
    unsigned get_excision_threshold () const { return std_devs; }

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


    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    void compute ();

    void detect ();
    void detect_tscr ();
    void detect_skfb ();
    void detect_fscr ();
    void count_zapped ();

    void mask ();
    void reset_mask ();

    void insertsk ();

    unsigned debugd;

    //! number of samples used in each SK estimate
    unsigned M;

    unsigned nchan;

    unsigned npol;

    unsigned ndim;

    uint64_t npart;

    uint64_t output_ndat;

    //! SK Estimates 
    Reference::To<TimeSeries> estimates;

    //! Tscrunched SK Estimate for block
    Reference::To<TimeSeries> estimates_tscr;

    //! Zap mask
    Reference::To<BitSeries> zapmask;

    //! accumulation arrays for S1 and S2 in t scrunch
    std::vector <float> S1_tscr;
    std::vector <float> S2_tscr;

    //! Total SK statistic for each poln/channel, post filtering
    std::vector<float> filtered_sum;

    //! Hits on filtered average for each channel
    std::vector<uint64_t> filtered_hits;

    //! Total SK statistic for each poln/channel, before filtering
    std::vector<float> unfiltered_sum;

    //! Hits on unfiltered SK statistic, same for each channel
    uint64_t unfiltered_hits;

    //! number of std devs used to calculate excision limits
    unsigned std_devs;

    //! lower and upper thresholds of excision limits
    std::vector<float> thresholds;

    float one_sigma;

    //! Number of samples integrated into tscr
    unsigned M_tscr;

    //! exicision thresholds for tscr
    std::vector<float> thresholds_tscr;

    //! channel range to compute and apply SK excisions
    std::vector<unsigned> channels;

    //! samples zapped by type [0:all, 1:sk, 2:fscr, 3:tscr]
    std::vector<uint64_t> zap_counts;

    //! total number of samples processed
    uint64_t npart_total;

    //! flags for detection types [0:fscr, 1:tscr, 2:tscr]
    std::vector<bool> detection_flags;

    bool prepared;

  };

  class SpectralKurtosis::Engine : public Reference::Able
  {
  public:

      virtual void setup () = 0;

      virtual void compute (const TimeSeries* input, TimeSeries* output,
                            TimeSeries *output_tscr, unsigned tscrunch) = 0;

      virtual void reset_mask (BitSeries* output) = 0;

      virtual void detect_ft (const TimeSeries* input, BitSeries* output,
                              float upper_thresh, float lower_thresh) = 0;

      virtual void detect_fscr (const TimeSeries* input, BitSeries* output,
                                const float lower, const float upper,
                                unsigned schan, unsigned echan) = 0;

      virtual void detect_tscr (const TimeSeries* input,
                                const TimeSeries * input_tscr,
                                BitSeries* output,
                                float upper, float lower) = 0;
 
      virtual int count_mask (const BitSeries* output) = 0;

      virtual float * get_estimates (const TimeSeries* input) = 0;

      virtual unsigned char * get_zapmask (const BitSeries* input) = 0;

      virtual void mask (BitSeries* mask, const TimeSeries * in, TimeSeries* out, unsigned M) = 0;

      virtual void insertsk (const TimeSeries* input, TimeSeries* out, unsigned M) = 0;

  };
}

#endif
