//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/BandpassMonitor.h

#ifndef __baseband_dsp_BandpassMonitor_h
#define __baseband_dsp_BandpassMonitor_h

#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  class Rescale;

  //! Rescale all channels and polarizations
  class BandpassMonitor : public Reference::Able
  {
  public:

    BandpassMonitor();

    void output_state (Rescale*);

    void dump (const std::string& timestamp, 
	       unsigned pol, unsigned ndat, 
	       const float* data, const char* ext);

  protected:

    std::vector<float> temp;

  };
}

#endif
