
/***************************************************************************
 *
 *   Copyright (C) 2003-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.16 $
   $Date: 2008/10/04 11:50:37 $
   $Author: straten $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include <string>

#include "Reference.h"

namespace dsp {

  class PhaseSeries;
  class FilenameConvention;

  //! Base class for things that can unload PhaseSeries data somewhere
  class PhaseSeriesUnloader : public Reference::Able
  {

  public:
    
    //! Constructor
    PhaseSeriesUnloader ();
    
    //! Destructor
    virtual ~PhaseSeriesUnloader ();
    
    //! Unload the PhaseSeries data
    virtual void unload (const PhaseSeries*) = 0;

    //! Handle partially completed PhaseSeries data
    virtual void partial (const PhaseSeries*);

    //! Perform any clean up tasks before completion
    virtual void finish ();

    //! Generate a filename using the current convention
    virtual std::string get_filename (const PhaseSeries* data) const;

    //! Set the filename convention
    virtual void set_convention (FilenameConvention*);
    virtual FilenameConvention* get_convention ();

    //! Set the path to which output data will be written
    virtual void set_path (const std::string&);
    virtual std::string get_path () const;
    
    //! place output files in a sub-directory named by source
    virtual void set_path_add_source (bool);
    virtual bool get_path_add_source () const;

    //! Set the prefix to be added to the front of filenames
    virtual void set_prefix (const std::string&);
    virtual std::string get_prefix () const;

    //! Set the extension to be added to the end of filenames
    virtual void set_extension (const std::string&);
    virtual std::string get_extension () const;


  protected:

    //! The filename convention
    Reference::To<FilenameConvention> convention;

    //! The filename path
    std::string path;

    //! The filename prefix
    std::string prefix;

    //! The filename extension
    std::string extension;

    //! Put each output file in a sub-directory named by source
    bool path_add_source;

  };

  //! Defines the output filename convention
  class FilenameConvention : public Reference::Able
  {
  public:
    virtual std::string get_filename (const PhaseSeries* data) const = 0;
  };

  //! Defines the output filename convention
  class FilenameEpoch : public FilenameConvention
  {
  public:
    FilenameEpoch ();
    void set_datestr_pattern (const std::string&);
    void set_integer_seconds (unsigned);
    std::string get_filename (const PhaseSeries* data) const;

  protected:
    std::string datestr_pattern;
    unsigned integer_seconds;
  };

  //! Defines the output filename convention
  class FilenamePulse : public FilenameConvention
  {
  public:
    std::string get_filename (const PhaseSeries* data) const;
  };
}

#endif // !defined(__PhaseSeriesUnloader_h)
