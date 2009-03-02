//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Rescale.h,v $
   $Revision: 1.6 $
   $Date: 2009/03/02 07:15:05 $
   $Author: straten $ */

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

    //! Rescale to zero mean and unit variance
    void transformation ();

    //! Set the rescaling interval in seconds
    void set_interval_seconds (double seconds);

    //! Set the rescaling interval in samples
    void set_interval_samples (uint64 samples);

    //! After setting offset and scale, keep them constant
    void set_constant (bool);

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
    uint64 get_nsample () const;

    //! Get the total power time series for the given polarization
    const float* get_time (unsigned ipol) const;

    Callback<Rescale*> update;

  private:

    std::vector< std::vector<double> > freq_total;
    std::vector< std::vector<double> > freq_totalsq;

    std::vector< std::vector<float> > time_total;

    std::vector< std::vector<float> > scale;
    std::vector< std::vector<float> > offset;

    double interval_seconds;
    uint64 interval_samples;

    uint64 nsample;
    uint64 isample;

    MJD update_epoch;

    bool constant_offset_scale;

    void init ();

  };
}

#endif
