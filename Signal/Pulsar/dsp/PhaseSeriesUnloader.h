//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.2 $
   $Date: 2003/01/31 16:30:12 $
   $Author: wvanstra $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include <string>

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

    //! Creates a good filename for the PhaseSeries data archive
    virtual string get_filename (const PhaseSeries* data) const;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
