//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.8 $
   $Date: 2003/08/22 15:49:09 $
   $Author: wvanstra $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "dsp/PhaseSeriesUnloader.h"

namespace Pulsar {
  class Archive;
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
    void set_archive_class (const char* archive_class_name);

    //! Set the Pulsar::Archive instance to which data will be added
    void set_archive (Pulsar::Archive* archive);

    //! Unloads all available data to a Pulsar::Archive instance
    void unload ();

    //! Set the Pulsar::Archive with the PhaseSeries data
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    //! Add the PhaseSeries data to the Pulsar::Archive instance
    void add (Pulsar::Archive* archive, const PhaseSeries* phase);

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

    void set (Pulsar::Integration* integration, const PhaseSeries* phase);

    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
	      unsigned ichan, unsigned ipol, unsigned idim);

    //! Set the Pulsar::dspReduction Extension with the dsp::Operation
    void set (Pulsar::dspReduction* dspR);

    //! Set the Pulsar::TwoBitStats Extension with the dsp::TwoBitCorrection
    void set (Pulsar::TwoBitStats* twobit);

    //! Set the Pulsar::Passband Extension with the dsp::Response
    void set (Pulsar::Passband* pband);

  };

}

#endif // !defined(__Archiver_h)
