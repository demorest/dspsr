//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/sigproc/dsp/SigProcObservation.h

#ifndef __SigProcObservation_h
#define __SigProcObservation_h

#include "dsp/Observation.h"

namespace dsp {
 
  //! Parses Observation attributes from a SigProc header
  class SigProcObservation : public Observation {

  public:

    //! Construct from a sigproc file
    SigProcObservation (const char* filename);

    //! Read the sigproce header from file
    void load (const char* filename);

    //! Construct from an SigProc header block
    SigProcObservation (FILE* header=0);

    //! Read the SigProc header block
    void load (FILE* header);

    //! Write a SigProc header block
    void unload (FILE* header);

    //! Copy parameters from the sigproc global variables
    void load_global ();

    //! Copy parameters to the sigproc global variables
    void unload_global ();

    int header_bytes;
  };
  
}

#endif
