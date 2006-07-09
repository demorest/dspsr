//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.15 $
   $Date: 2006/07/09 13:27:13 $
   $Author: wvanstra $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "dsp/PhaseSeriesUnloader.h"
#include "Pulsar/Archive.h"

namespace Pulsar {
  class Integration;
  class Profile;

  class dspReduction;
  class TwoBitStats;
  class Passband;
}

namespace dsp {

  class Response;
  class Operation;
  class TwoBitCorrection;

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

    //! Set the Response from which Passband Extension will be constructed
    void set_passband (const Response* passband);

    //! Set the Operation instances for the dspReduction Extension
    void set_operations (const vector<Operation*>& operations);

    //! Set the name of the Pulsar::Archive class used to create new instances
    void set_archive_class (const string& archive_class_name);
    
    //! Set the Pulsar::Archive instance to which data will be added
    void set_archive (Pulsar::Archive* archive);

    //! Add a Pulsar::Archive::Extension to those added to the output archive
    void add_extension (Pulsar::Archive::Extension* extension);

    //! Unloads all available data to a Pulsar::Archive instance
    void unload ();

    //! Set the Pulsar::Archive with the PhaseSeries data
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    //! Add the PhaseSeries data to the Pulsar::Archive instance
    void add (Pulsar::Archive* archive, const PhaseSeries* phase);

    //! Quick hack until a dsp::Observation can store _properly_ the ChannelSum history
    //! In particular it will need to store whether the 'channel_align' flag was enabled
    //! and therefore whether the archive was dedispersed
    void set_archive_dedispersed(bool _archive_dedispersed){ archive_dedispersed = _archive_dedispersed; }
    
    //! Retrieves this hack attribute that indicates whether the archive is dedispersed already
    bool get_archive_dedispersed(){ return archive_dedispersed; }
 
    //! If true, a dspReduction extension is added to the archive with this string
    void set_archive_software(string _archive_software)
    { archive_software = _archive_software; }

    //! If true, a dspReduction extension is added to the archive with this string
    string get_archive_software()
    { return archive_software; }

  protected:

    //! Name of the Pulsar::Archive class used to create new instances
    string archive_class_name;

    //! The Pulsar::Archive instance to which data will be added
    Reference::To<Pulsar::Archive> single_archive;

    //! Response from which Passband Extension will be constructed
    Reference::To<const Response> passband;

    //! TwoBitCorrection from which TwoBitStats Extension will be constructed
    Reference::To<const TwoBitCorrection> twobit;

    //! The Operation instances for the dspReduction Extension
    vector< Reference::To<Operation> > operations;

    //! The Pulsar::Archive::Extension classes to be added to the output
    vector< Reference::To<Pulsar::Archive::Extension> > extensions;

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
    string archive_software;
  };

}

#endif // !defined(__Archiver_h)


