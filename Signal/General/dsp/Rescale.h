//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Rescale.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/01 11:14:23 $
   $Author: straten $ */

#ifndef __baseband_dsp_Rescale_h
#define __baseband_dsp_Rescale_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

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
  };
}

#endif
