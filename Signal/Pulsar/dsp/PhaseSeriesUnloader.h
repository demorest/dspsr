//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.3 $
   $Date: 2003/06/16 21:47:08 $
   $Author: wvanstra $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include <string>

#include "Reference.h"

namespace dsp {

  class PhaseSeries;

  //! Base class for things that can unload PhaseSeries data somewhere

  class PhaseSeriesUnloader : public Reference::Able {

  public:
    
    //! Constructor
    PhaseSeriesUnloader () {}
    
    //! Destructor
    virtual ~PhaseSeriesUnloader () {}
    
    //! Set the PhaseSeries from which Profile data will be constructed
    void set_profiles (const PhaseSeries* profiles);

    //! Defined by derived classes
    virtual void unload () = 0;

    //! Creates a good filename for the PhaseSeries data archive
    virtual string get_filename (const PhaseSeries* data) const;


  protected:
    //! PhaseSeries from which Profile data will be constructed
    Reference::To<const PhaseSeries> profiles;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
