//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.4 $
   $Date: 2003/09/22 05:34:18 $
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
    PhaseSeriesUnloader ();
    
    //! Destructor
    virtual ~PhaseSeriesUnloader ();
    
    //! Set the PhaseSeries from which Profile data will be constructed
    void set_profiles (const PhaseSeries* profiles);

    //! Defined by derived classes
    virtual void unload () = 0;

    //! Creates a good filename for the PhaseSeries data archive
    virtual string get_filename (const PhaseSeries* data) const;

    //! Set the filename (pattern) to be used by get_filename
    virtual void set_filename (const char* filename);
    void set_filename (const string& filename)
    { set_filename (filename.c_str()); }

    //! Set the extension to be used by get_filename
    virtual void set_extension (const char* extension);
    void set_extension (const string& extension)
    { set_extension (extension.c_str()); }

  protected:
    //! PhaseSeries from which Profile data will be constructed
    Reference::To<const PhaseSeries> profiles;

    //! The filename pattern
    string filename_pattern;

    //! The filename extension;
    string filename_extension;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
