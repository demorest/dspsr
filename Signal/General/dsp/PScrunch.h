//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/PScrunch.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/01 12:23:21 $
   $Author: straten $ */

#ifndef __baseband_dsp_PScrunch_h
#define __baseband_dsp_PScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  //! PScrunch all channels and polarizations
  class PScrunch : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PScrunch ();

    //! PScrunch to zero mean and unit variance
    void transformation ();
  };
}

#endif
