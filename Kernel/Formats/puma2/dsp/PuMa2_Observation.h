//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma2/dsp/PuMa2_Observation.h,v $
   $Revision: 1.1 $
   $Date: 2005/03/11 16:59:07 $
   $Author: wvanstra $ */

#ifndef __PuMa2_Observation_h
#define __PuMa2_Observation_h

#include "dsp/Observation.h"

namespace dsp {
 
  //! General means of constructing Observation attributes from PuMa2 data
  /*! This class parses the ASCII header block used for PuMa2 data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a PuMa2 data file, or from the
    shared memory (data block) of the machines in the PuMa2
    cluster. */
  class PuMa2_Observation : public Observation {

  public:

    //! Construct from a PuMa2 ASCII header block
    PuMa2_Observation (const char* header=0);

    //! Number of bytes offset from the beginning of acquisition
    uint64 offset_bytes;

  };
  
}

#endif
