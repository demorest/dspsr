//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.27 $
   $Date: 2009/05/04 02:14:30 $
   $Author: straten $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "dsp/PhaseSeriesUnloader.h"
#include "Pulsar/Archive.h"

namespace Pulsar {
  class Interpreter;
  class Integration;
  class Profile;

  class dspReduction;
  class TwoBitStats;
  class Passband;
}

namespace dsp {

  class Response;
  class Operation;
  class ExcisionUnpacker;

  //! Class to unload PhaseSeries data in a Pulsar::Archive
  /*! 

  */
  class Archiver : public PhaseSeriesUnloader {

  public:

    //! Verbose flag
    static bool verbose;

    //! Constructor
    Archiver ();
    
    //! Destructor
    virtual ~Archiver ();

    //! Set the name of the Pulsar::Archive class used to create new instances
    void set_archive_class (const std::string& archive_class_name);

    //! Set the post-processing script
    void set_script (const std::vector<std::string>& jobs);

    //! Set the Pulsar::Archive instance to which data will be added
    void set_archive (Pulsar::Archive* archive);

    //! Set the minimum integration length required to unload data
    void set_minimum_integration_length (double seconds);

    //! Get the Pulsar::Archive instance to which all data were added
    Pulsar::Archive* get_archive ();

    //! Add a Pulsar::Archive::Extension to those added to the output archive
    void add_extension (Pulsar::Archive::Extension* extension);

    //! Unloads all available data to a Pulsar::Archive instance
    void unload (const PhaseSeries*);

    //! Perform any clean up tasks before completion
    void finish ();

    //! Set the Pulsar::Archive with the PhaseSeries data
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    //! Add the PhaseSeries data to the Pulsar::Archive instance
    void add (Pulsar::Archive* archive, const PhaseSeries* phase);

    void set_archive_dedispersed (bool _archive_dedispersed)
    { archive_dedispersed = _archive_dedispersed; }
    
    bool get_archive_dedispersed() const
    { return archive_dedispersed; }
 
    //! A dspReduction extension is added to the archive with this string
    void set_archive_software(std::string _archive_software)
    { archive_software = _archive_software; }

    //! A dspReduction extension is added to the archive with this string
    std::string get_archive_software()
    { return archive_software; }

  protected:
    
    //! Used only internally
    const PhaseSeries* profiles;

    //! Minimum integration length required to unload data
    double minimum_integration_length;

    //! Name of the Pulsar::Archive class used to create new instances
    std::string archive_class_name;

    //! The Pulsar::Archive instance to which data will be added
    Reference::To<Pulsar::Archive> single_archive;

    //! The Pulsar::Archive instance to which data will be unloaded
    Reference::To<Pulsar::Archive> archive;

    //! Commands used to process Archive data before unloading
    std::vector<std::string> script;

    //! The script interpreter used to process Archive data before unloading
    Reference::To<Pulsar::Interpreter> interpreter;

    //! Response from which Passband Extension will be constructed
    Reference::To<const Response> passband;

    //! ExcisionUnpacker from which TwoBitStats Extension will be constructed
    Reference::To<const ExcisionUnpacker> excision_unpacker;

    //! The Pulsar::Archive::Extension classes to be added to the output
    std::vector< Reference::To<Pulsar::Archive::Extension> > extensions;

    //! Set the Pulsar::Integration with the PhaseSeries data
    void set (Pulsar::Integration* integration, const PhaseSeries* phase,
	      unsigned isub=0, unsigned nsub=1);

    //! Set the Pulsar::Profile with the specified subset of PhaseSeries data
    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
	      unsigned ichan, unsigned ipol, unsigned idim);

    //! Set the Pulsar::dspReduction Extension with the dsp::Operation
    void set (Pulsar::dspReduction* dspR);

    //! Set the Pulsar::TwoBitStats Extension with the dsp::TwoBitCorrection
    void set (Pulsar::TwoBitStats* twobit);

    //! Set the Pulsar::Passband Extension with the dsp::Response
    void set (Pulsar::Passband* pband);

  private:

    //! Hack attribute that indicates whether the archive is dedispersed already [false]
    bool archive_dedispersed;

    //! String to go in the dspReduction Extension of output archive ["Software Unknown"]
    std::string archive_software;
  };

}

#endif // !defined(__Archiver_h)


