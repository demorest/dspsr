//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr2/dsp/CPSR2_Observation.h

#ifndef __CPSR2_Observation_h
#define __CPSR2_Observation_h

#include "dsp/ASCIIObservation.h"

namespace dsp {
 
  //! General means of constructing Observation attributes from CPSR2 data
  /*! This class parses the ASCII header block used for CPSR2 data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a CPSR2 data file, or from the
    shared memory (data block) of the machines in the CPSR2
    cluster. */
  class CPSR2_Observation : public ASCIIObservation {

  public:

    //! Construct from a CPSR2 ASCII header block
    CPSR2_Observation (const char* header=0);

    std::string prefix;
  };
  
}

#endif
