//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2_Observation.h,v $
   $Revision: 1.1 $
   $Date: 2002/08/15 07:03:11 $
   $Author: wvanstra $ */

#ifndef __CPSR2_Observation_h
#define __CPSR2_Observation_h

#include "Observation.h"

namespace dsp {
  
  class CPSR2_Observation : public Observation {

  public:
    //! Construct from a CPSR2 ASCII header block
    CPSR2_Observation (const char* header=0);

  };
  
}

#endif
