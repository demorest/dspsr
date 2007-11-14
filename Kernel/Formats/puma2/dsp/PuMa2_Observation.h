//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma2/dsp/PuMa2_Observation.h,v $
   $Revision: 1.3 $
   $Date: 2007/11/14 03:11:02 $
   $Author: straten $ */

#ifndef __PuMa2_Observation_h
#define __PuMa2_Observation_h

#include "dsp/ASCIIObservation.h"

namespace dsp {
 
  //! General means of constructing Observation attributes from PuMa2 data
  /*! This class parses the ASCII header block used for PuMa2 data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a PuMa2 data file, or from the
    shared memory (data block) of the machines in the PuMa2
    cluster. */
  class PuMa2_Observation : public ASCIIObservation {

  public:

    //! Construct from a PuMa2 ASCII header block
    PuMa2_Observation (const char* header);

  };
  
}

#endif
