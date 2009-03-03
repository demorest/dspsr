//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/PrestoObservation.h,v $
   $Revision: 1.2 $
   $Date: 2009/03/03 05:29:30 $
   $Author: straten $ */

#ifndef __PrestoObservation_h
#define __PrestoObservation_h

#include "dsp/Observation.h"
#include "dsp/infodata.h"

namespace dsp
{ 
  //! Copy Observation attributes from a PRESTO infodata structure
  /*! The infodata structure has been slightly modified to include
    the number of bits per sample and the number of polarizations. */
  class PrestoObservation : public Observation
  {
  public:

    //! Construct from a PRESTO infodata struct
    PrestoObservation (const infodata*);

  };  
}

#endif
