#ifndef __PMDAQ_Observation_h
#define __PMDAQ_Observation_h

#include "dsp/Observation.h"

namespace dsp {
 
  //! General means of constructing Observation attributes from PMDAQ data
  /*! This class parses the header block used for PMDAQ data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a PMDAQ header file.
  */

  class PMDAQ_Observation : public Observation {

  public:

    //! Construct from a CPSR2 ASCII header block
    PMDAQ_Observation (const char* header=0);

    //! Number of bytes offset from the beginning of acquisition
    uint64 offset_bytes;
  };
  
}

#endif
