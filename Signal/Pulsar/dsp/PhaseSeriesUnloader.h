//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.1 $
   $Date: 2003/01/31 16:00:36 $
   $Author: wvanstra $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include "ReferenceAble.h"

namespace dsp {

  class PhaseSeries;

  //! Base class for things that can unload PhaseSeries data somewhere

  class PhaseSeriesUnloader : public Reference::Able {

  public:
    
    //! Constructor
    PhaseSeriesUnloader () {}
    
    //! Destructor
    virtual ~PhaseSeriesUnloader () {}
    
    //! Defined by derived classes
    virtual void unload (const PhaseSeries* data) = 0;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
