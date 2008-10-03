//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/BandpassMonitor.h,v $
   $Revision: 1.4 $
   $Date: 2008/10/03 04:42:30 $
   $Author: straten $ */

#ifndef __baseband_dsp_BandpassMonitor_h
#define __baseband_dsp_BandpassMonitor_h

#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  class Rescale;

  //! Rescale all channels and polarizations
  class BandpassMonitor
  {
  public:

    BandpassMonitor();

    void output_state (Rescale*);

    void dump (const std::string& timestamp, 
	       unsigned pol, unsigned ndat, 
	       const float* data, const char* ext);

  protected:

    std::vector<float> rms;

  };
}

#endif
