//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/SigProcObservation.h,v $
   $Revision: 1.1 $
   $Date: 2008/05/30 10:33:32 $
   $Author: straten $ */

#ifndef __SigProcObservation_h
#define __SigProcObservation_h

#include "dsp/Observation.h"

namespace dsp {
 
  //! Parses Observation attributes from a SigProc header
  class SigProcObservation : public Observation {

  public:

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

  };
  
}

#endif
