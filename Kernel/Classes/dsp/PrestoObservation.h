//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/PrestoObservation.h,v $
   $Revision: 1.1 $
   $Date: 2009/03/02 17:39:51 $
   $Author: straten $ */

#ifndef __PrestoObservation_h
#define __PrestoObservation_h

#include "dsp/Observation.h"
#include "makeinf.h"

namespace dsp
{ 
  //! General means of constructing Observation attributes from CPSR2 data
  /*! This class parses the ASCII header block used for CPSR2 data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a CPSR2 data file, or from the
    shared memory (data block) of the machines in the CPSR2
    cluster. */
  class PrestoObservation : public Observation
  {
  public:

    //! Construct from a PRESTO infodata struct
    PrestoObservation (const infodata*, unsigned extern_nbit);

  };  
}

#endif
