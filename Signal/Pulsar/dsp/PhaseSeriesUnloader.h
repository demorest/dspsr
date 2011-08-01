
/***************************************************************************
 *
 *   Copyright (C) 2003-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.23 $
   $Date: 2010/01/15 11:55:32 $
   $Author: straten $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include "OwnStream.h"
#include "Reference.h"

namespace dsp {

  class PhaseSeries;
  class Operation;
  class FilenameConvention;

  //! Base class for things that can unload PhaseSeries data somewhere
  class PhaseSeriesUnloader : public OwnStream
  {

  public:
    
    //! Constructor
    PhaseSeriesUnloader ();
    
    //! Destructor
    virtual ~PhaseSeriesUnloader ();

    //! Clone operator
    virtual PhaseSeriesUnloader* clone () const = 0;

    //! Unload the PhaseSeries data
    virtual void unload (const PhaseSeries*) = 0;

    //! Handle partially completed PhaseSeries data
    virtual void partial (const PhaseSeries*);

    //! After unload, a different PhaseSeries may be available for use
    virtual PhaseSeries* recycle () { return 0; }

    //! Perform any clean up tasks before completion
    virtual void finish ();

    //! Generate a filename using the current convention
    virtual std::string get_filename (const PhaseSeries* data) const;

    //! Set the filename convention
    virtual void set_convention (FilenameConvention*);
    virtual FilenameConvention* get_convention ();

    //! Set the directory to which output data will be written
    virtual void set_directory (const std::string&);
    virtual std::string get_directory () const;
    
    //! place output files in a sub-directory named by source
    virtual void set_path_add_source (bool);
    virtual bool get_path_add_source () const;

    //! Set the prefix to be added to the front of filenames
    virtual void set_prefix (const std::string&);
    virtual std::string get_prefix () const;

    //! Set the extension to be added to the end of filenames
    virtual void set_extension (const std::string&);
    virtual std::string get_extension () const;

    //! Set the minimum integration length required to unload data
    virtual void set_minimum_integration_length (double seconds) = 0;

  protected:

    //! The filename convention
    Reference::To<FilenameConvention> convention;

    //! The filename directory
    std::string directory;

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

    bool report_unload;

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

  //! Defines the output filename convention
  class FilenameOriginal : public FilenameEpoch
  {
    public:
      FilenameOriginal ();
      FilenameOriginal (const std::string&);
      std::string get_filename (const PhaseSeries* data) const;

      bool report_unload;

    protected:
      std::string input_filename;
  };
}

#endif // !defined(__PhaseSeriesUnloader_h)
