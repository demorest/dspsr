//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/SampleDelayFunction.h,v $
   $Revision: 1.3 $
   $Date: 2006/07/09 13:27:13 $
   $Author: wvanstra $ */

#ifndef __baseband_dsp_SampleDelayFunction_h
#define __baseband_dsp_SampleDelayFunction_h

#include "Reference.h"
#include "environ.h"

namespace dsp {

  //! Virtual base class of sample delay functions
  class SampleDelayFunction : public Reference::Able {

  public:

    //! Destructor
    virtual ~SampleDelayFunction () { }

    //! Compute the delays for the specified Observation
    /*! \retval true if the function has changed */
    virtual bool match (const Observation* obs) = 0;

    //! Return the delay for the specified channel and polarization
    virtual int64 get_delay (unsigned ichan=0, unsigned ipol=0) = 0;

    //! Add to the history of operations performed on the observation
    virtual void mark (Observation* observation) { }

  };


}

#endif
