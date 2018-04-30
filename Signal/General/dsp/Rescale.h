//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Rescale.h

#ifndef __baseband_dsp_Rescale_h
#define __baseband_dsp_Rescale_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BandpassMonitor.h"

#include <vector>

namespace dsp
{
  //! Rescale all channels and polarizations
  class Rescale : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    Rescale ();

    //! Desctructor
    ~Rescale ();

    void prepare ();

    //! Rescale to zero mean and unit variance
    void transformation ();

    //! Set the rescaling interval in seconds
    void set_interval_seconds (double seconds);

    //! Set the rescaling interval in samples
    void set_interval_samples (uint64_t samples);

    //! If exact, only allow set_interval_samples through each iteration
    void set_exact (bool);

    //! After setting offset and scale, keep them constant
    void set_constant (bool);

    //! Subtract an exponential smooth with specified decay constant
    void set_decay (float);

    //! Do not output any data before the first integration interval has passed
    void set_output_after_interval (bool);

    //! Maintain fscrunched total that can be output
    void set_output_time_total (bool);

    //! Get the epoch of the last scale/offset update
    MJD get_update_epoch () const;

    //! Get the offset bandpass for the given polarization
    const float* get_offset (unsigned ipol) const;

    //! Get the scale bandpass for the given polarization
    const float* get_scale (unsigned ipol) const;

    //! Get the mean bandpass for the given polarization
    const double* get_mean (unsigned ipol) const;

    //! Get the scale bandpass for the given polarization
    const double* get_variance (unsigned ipol) const;

    //! Get the number of samples between updates
    uint64_t get_nsample () const;

    //! Get the total power time series for the given polarization
    const float* get_time (unsigned ipol) const;

    Callback<Rescale*> update;

  private:

    std::vector< std::vector<double> > freq_total;
    std::vector< std::vector<double> > freq_totalsq;

    std::vector< std::vector<float> > time_total;

    std::vector< std::vector<float> > scale;
    std::vector< std::vector<float> > offset;

    std::vector< std::vector<float> > decay_offset;

    bool exact;
    bool output_time_total;
    bool output_after_interval;

    double interval_seconds;
    uint64_t interval_samples;

    float decay_constant;
    bool do_decay;

    uint64_t nsample;
    uint64_t isample;

    MJD update_epoch;

    bool constant_offset_scale;

    void init ();
    void compute_various (bool first_call = false);
  };
}

#endif
