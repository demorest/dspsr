//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/ASCIIObservation.h,v $
   $Revision: 1.2 $
   $Date: 2007/08/13 10:56:27 $
   $Author: straten $ */

#ifndef __ASCIIObservation_h
#define __ASCIIObservation_h

#include "dsp/Observation.h"

namespace dsp {
 
  //! Parses Observation attributes from an ASCII header
  /*! This class parses the ASCII header block used by DADA-based
    instruments such as CPSR2, PuMa2, and APSR.  It initializes all of
    the attributes of the Observation base class.  The header block
    may come from a data file, or from shared memory. */
  class ASCIIObservation : public Observation {

  public:

    //! Construct from an ASCII header block
    ASCIIObservation (const char* header=0);

    //! Number of bytes offset from the beginning of acquisition
    uint64 offset_bytes;

    //! Parse the ASCII header block
    void parse (const char* header);

  protected:

    std::string hdr_version;

  };
  
}

#endif
