//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.8 $
   $Date: 2005/08/28 05:08:46 $
   $Author: hknight $ */

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
    virtual std::string get_filename (const PhaseSeries* data) const;

    //! Set the filename (pattern) to be used by get_filename
    virtual void set_filename (const char* filename);
    void set_filename (const std::string& filename)
    { set_filename (filename.c_str()); }

    //! Set the extension to be used by get_filename
    virtual void set_extension (const char* extension);
    void set_extension (const std::string& extension)
    { set_extension (extension.c_str()); }

    //! Set whether you want to allow the archive filename to be
    //! over-ridden by a pulse number
    void set_force_filename(bool _force_filename)
    { force_filename = _force_filename; }

    //! Inquire whether it is possible for the archive filename to be
    //! over-ridden by a pulse number
    bool get_force_filename(){ return force_filename; }

  protected:

    //! Helper function that makes sure a given filename is unique
    std::string make_unique(const std::string& filename, const std::string& fname_extension,
			    const PhaseSeries* data) const;

    //! PhaseSeries from which Profile data will be constructed
    Reference::To<const PhaseSeries> profiles;

    //! The filename pattern
    std::string filename_pattern;

    //! The filename extension;
    std::string filename_extension;

    //! Force make_unique() to return 'filename' [false]
    bool force_filename;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
