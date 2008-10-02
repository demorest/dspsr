//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Rescale.h,v $
   $Revision: 1.3 $
   $Date: 2008/10/02 06:40:10 $
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

  private:

    Callback<Rescale*> update;

    std::vector< std::vector<double> > freq_total;
    std::vector< std::vector<double> > freq_totalsq;

    std::vector< std::vector<float> > time_total;

    std::vector< std::vector<float> > scale;
    std::vector< std::vector<float> > offset;

    double interval_seconds;
    uint64 interval_samples;

    uint64 nsample;
    uint64 isample;

    void init ();

  };
}

#endif
